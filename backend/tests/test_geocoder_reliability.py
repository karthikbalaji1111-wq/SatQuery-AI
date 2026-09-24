"""Geocoder reliability against a rate-limited public service.

Production (2026-09-24): Render's outbound IPs are shared by every service in
the region, Nominatim limits per IP, and for about forty minutes most geocodes
from Render were refused with HTTP 429 while this application sent a few
requests a minute. The client retried one second later and every new question
probed again, so it spent the shared budget exactly when there was none.

These tests pin the replacement, against a FAKE clock that advances instead of
sleeping, so every wait is asserted exactly and none is paid for:

* the cache: hits, repeats, provenance, expiry, successes only;
* coalescing: concurrent callers for one place share ONE upstream sequence -
  its answer, or its failure;
* 429: Retry-After honoured (seconds or HTTP-date), exponential backoff when
  absent, a process-wide cooldown during which no request is sent, and a
  structured ``geocoding_unavailable`` (503 + Retry-After) when waiting would
  exceed the caller's budget;
* bounded retry for timeouts, connection failures and 5xx; no retry for a
  malformed answer or a 4xx;
* no fabricated coordinates, ever: every failure is an error, nothing is cached;
* the optional warm-up goes through the same path and caches only real answers.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest
from app.core.config import Settings
from app.core.errors import (
    GeocodingUnavailableError,
    NotFoundError,
    UpstreamServiceError,
)
from app.services.geospatial import GeospatialService, ResolveRequest, nominatim
from app.services.geospatial.nominatim import (
    geocode,
    geocoder_status,
    reset_geocoder_state,
    warm_geocoder_cache,
    warm_places,
)

CUBBON = [
    {
        "lat": "12.9763",
        "lon": "77.5929",
        "boundingbox": ["12.9700", "12.9830", "77.5870", "77.5990"],
        "display_name": "Cubbon Park, Bengaluru, Karnataka, India",
        "class": "leisure",
        "type": "park",
    }
]


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class FakeClock:
    """Monotonic time that moves only when the geocoder pauses."""

    def __init__(self) -> None:
        self.t = 1_000.0
        self.pauses: list[float] = []

    def now(self) -> float:
        return self.t

    async def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.t += seconds
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    reset_geocoder_state()
    fake = FakeClock()
    monkeypatch.setattr(nominatim, "_now", fake.now)
    monkeypatch.setattr(nominatim, "_pause", fake.pause)
    return fake


def settings(**overrides: object) -> Settings:
    """The production defaults, except the one-second spacing.

    Backoff is REAL here (2 s base, doubling, 300 s cap, 15 s budget, 3
    attempts) - the fake clock makes it free.
    """

    base: dict[str, object] = {
        "geocoder_min_interval_seconds": 0.0,
        "geocoder_backoff_base_seconds": 2.0,
        "geocoder_max_cooldown_seconds": 300.0,
        "geocoder_retry_budget_seconds": 15.0,
        "geocoder_max_attempts": 3,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class Upstream:
    """A scripted Nominatim: replays responses, records every request.

    Each response takes a few event-loop turns, as a real network round trip
    does - otherwise a request would complete before any concurrent caller had
    even started, and coalescing could never be observed.
    """

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        async def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            for _ in range(5):
                await asyncio.sleep(0)
            item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
            if isinstance(item, Exception):
                raise item
            return item

        return httpx.MockTransport(handle)


def ok(payload: object = CUBBON) -> httpx.Response:
    return httpx.Response(200, json=payload)


def throttled(retry_after: str | None = None) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(429, text="Too Many Requests", headers=headers)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# =========================================================================== #
# Cache
# =========================================================================== #


def test_a_successful_geocode_returns_the_services_own_answer() -> None:
    upstream = Upstream(ok())
    place = run(geocode("Cubbon Park, Bengaluru", settings=settings(),
                        transport=upstream.transport()))

    assert place.display_name.startswith("Cubbon Park")
    assert (place.bbox.south, place.bbox.north) == (12.97, 12.983)
    assert (place.bbox.west, place.bbox.east) == (77.587, 77.599)
    assert place.place_class == "leisure" and place.place_type == "park"
    assert len(upstream.requests) == 1
    assert upstream.requests[0].url.params["q"] == "Cubbon Park, Bengaluru"


def test_a_cache_entry_records_query_source_and_time() -> None:
    upstream = Upstream(ok())
    config = settings()
    before = datetime.now(UTC)
    run(geocode("  Cubbon PARK,   Bengaluru ", settings=config,
                transport=upstream.transport()))

    [entry] = nominatim._CACHE.values()
    assert entry.query == "cubbon park, bengaluru"
    assert entry.source == f"nominatim {config.nominatim_base_url}"
    assert entry.place.display_name.startswith("Cubbon Park")
    assert before <= entry.resolved_at <= datetime.now(UTC)


def test_a_cache_hit_sends_nothing_upstream() -> None:
    upstream = Upstream(ok())
    config = settings()

    async def scenario() -> None:
        first = await geocode("Cubbon Park, Bengaluru", settings=config,
                              transport=upstream.transport())
        second = await geocode("cubbon park,  bengaluru", settings=config,
                               transport=upstream.transport())
        assert second == first

    run(scenario())
    assert len(upstream.requests) == 1
    assert geocoder_status().cache_hits == 1
    assert geocoder_status().upstream_requests == 1


def test_a_repeated_query_costs_one_request_however_often_it_is_asked() -> None:
    upstream = Upstream(ok())
    config = settings()

    async def scenario() -> None:
        for _ in range(25):
            await geocode("Cubbon Park, Bengaluru", settings=config,
                          transport=upstream.transport())

    run(scenario())
    assert len(upstream.requests) == 1
    assert geocoder_status().cache_hits == 24


def test_the_default_cache_outlives_a_demo_but_not_a_day() -> None:
    config = Settings()
    assert config.geocoder_cache_ttl_seconds == 86_400
    assert 1 <= config.geocoder_cache_entries <= 10_000


def test_an_entry_expires_after_its_ttl(clock: FakeClock) -> None:
    upstream = Upstream(ok())
    config = settings(geocoder_cache_ttl_seconds=600.0)

    async def scenario() -> None:
        await geocode("Cubbon Park, Bengaluru", settings=config,
                      transport=upstream.transport())
        clock.t += 599
        await geocode("Cubbon Park, Bengaluru", settings=config,
                      transport=upstream.transport())
        assert len(upstream.requests) == 1  # still fresh
        clock.t += 2
        await geocode("Cubbon Park, Bengaluru", settings=config,
                      transport=upstream.transport())

    run(scenario())
    assert len(upstream.requests) == 2


def test_the_cache_is_bounded_and_evicts_the_least_recently_used() -> None:
    upstream = Upstream(ok())
    config = settings(geocoder_cache_entries=3)

    async def scenario() -> None:
        for name in ("a", "b", "c"):
            await geocode(name, settings=config, transport=upstream.transport())
        await geocode("a", settings=config, transport=upstream.transport())  # a is fresh
        await geocode("d", settings=config, transport=upstream.transport())  # evicts b

    run(scenario())
    keys = [entry.query for entry in nominatim._CACHE.values()]
    assert keys == ["c", "a", "d"]


# =========================================================================== #
# Coalescing
# =========================================================================== #


def test_concurrent_identical_requests_make_one_upstream_request() -> None:
    upstream = Upstream(ok())
    config = settings()

    async def scenario() -> list[Any]:
        return await asyncio.gather(*[
            geocode("Cubbon Park, Bengaluru", settings=config,
                    transport=upstream.transport())
            for _ in range(8)
        ])

    places = run(scenario())
    assert len(upstream.requests) == 1
    assert len({place.display_name for place in places}) == 1
    assert geocoder_status().coalesced == 7


def test_concurrent_identical_requests_share_one_failure_not_eight() -> None:
    """During an outage N identical callers must not become N retry sequences."""

    upstream = Upstream(throttled("60"))
    config = settings()

    async def scenario() -> list[Any]:
        return await asyncio.gather(
            *[
                geocode("Cubbon Park, Bengaluru", settings=config,
                        transport=upstream.transport())
                for _ in range(8)
            ],
            return_exceptions=True,
        )

    outcomes = run(scenario())
    assert len(upstream.requests) == 1
    assert all(isinstance(o, GeocodingUnavailableError) for o in outcomes)
    # They shared the ONE sequence in flight; none started its own.
    assert geocoder_status().coalesced == 7
    assert geocoder_status().refused_during_cooldown == 0


def test_different_places_are_not_coalesced() -> None:
    upstream = Upstream(ok())
    config = settings()

    async def scenario() -> None:
        await asyncio.gather(
            geocode("Cubbon Park, Bengaluru", settings=config,
                    transport=upstream.transport()),
            geocode("Ameerpet, Hyderabad", settings=config,
                    transport=upstream.transport()),
        )

    run(scenario())
    assert sorted(r.url.params["q"] for r in upstream.requests) == [
        "Ameerpet, Hyderabad", "Cubbon Park, Bengaluru",
    ]


# =========================================================================== #
# HTTP 429
# =========================================================================== #


def test_retry_after_in_seconds_is_waited_exactly_then_retried(clock: FakeClock) -> None:
    upstream = Upstream(throttled("7"), ok())
    place = run(geocode("Cubbon Park, Bengaluru", settings=settings(),
                        transport=upstream.transport()))

    assert place.display_name.startswith("Cubbon Park")
    assert clock.pauses == [7.0]
    assert len(upstream.requests) == 2
    # A success ends the cooldown.
    assert geocoder_status().cooldown_remaining_seconds == 0


def test_retry_after_as_an_http_date_is_honoured(clock: FakeClock) -> None:
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=9), usegmt=True)
    upstream = Upstream(throttled(when), ok())
    run(geocode("Cubbon Park, Bengaluru", settings=settings(),
                transport=upstream.transport()))

    [waited] = clock.pauses
    assert 7.0 <= waited <= 9.0  # the date has one-second resolution


def test_retry_after_beyond_the_budget_fails_at_once_with_the_upstreams_ask(
    clock: FakeClock,
) -> None:
    upstream = Upstream(throttled("120"), ok())

    async def scenario() -> GeocodingUnavailableError:
        with pytest.raises(GeocodingUnavailableError) as raised:
            await geocode("Cubbon Park, Bengaluru", settings=settings(),
                          transport=upstream.transport())
        return raised.value

    error = run(scenario())
    assert len(upstream.requests) == 1
    assert clock.pauses == []  # nobody sat through two minutes
    assert error.retry_after_seconds == pytest.approx(120.0)
    assert error.headers == {"Retry-After": "120"}
    assert error.status_code == 503
    assert error.code == "geocoding_unavailable"
    assert "429" in error.message and "120 seconds" in error.message


def test_without_retry_after_the_backoff_is_exponential_and_bounded(
    clock: FakeClock,
) -> None:
    upstream = Upstream(throttled())

    async def scenario() -> None:
        with pytest.raises(GeocodingUnavailableError):
            await geocode("Cubbon Park, Bengaluru", settings=settings(),
                          transport=upstream.transport())

    run(scenario())
    assert clock.pauses == [2.0, 4.0]
    assert len(upstream.requests) == 3


def test_no_request_is_sent_during_a_cooldown(clock: FakeClock) -> None:
    """The retry storm this replaces: every new question probed the limit."""

    upstream = Upstream(throttled("60"))
    config = settings()

    async def scenario() -> None:
        with pytest.raises(GeocodingUnavailableError):
            await geocode("Cubbon Park, Bengaluru", settings=config,
                          transport=upstream.transport())
        for place in ("Ameerpet, Hyderabad", "Dal Lake", "Chennai", "Marina Beach"):
            with pytest.raises(GeocodingUnavailableError) as raised:
                await geocode(place, settings=config, transport=upstream.transport())
            assert 0 < raised.value.retry_after_seconds <= 60

    run(scenario())
    assert len(upstream.requests) == 1
    assert geocoder_status().refused_during_cooldown == 4


def test_a_short_cooldown_is_waited_out_rather_than_refused(clock: FakeClock) -> None:
    upstream = Upstream(throttled("5"), ok())
    config = settings()

    async def scenario() -> None:
        with pytest.raises(GeocodingUnavailableError):
            # One attempt only, so the 429 ends this call.
            await geocode("Dal Lake", settings=settings(geocoder_max_attempts=1),
                          transport=upstream.transport())
        # The next caller finds 5 s of cooldown - within its budget.
        await geocode("Cubbon Park, Bengaluru", settings=config,
                      transport=upstream.transport())

    run(scenario())
    assert clock.pauses == [5.0]
    assert len(upstream.requests) == 2


def test_the_cooldown_ends_and_the_next_request_is_sent(clock: FakeClock) -> None:
    upstream = Upstream(throttled("60"), ok())
    config = settings()

    async def scenario() -> None:
        with pytest.raises(GeocodingUnavailableError):
            await geocode("Cubbon Park, Bengaluru", settings=config,
                          transport=upstream.transport())
        clock.t += 61
        await geocode("Cubbon Park, Bengaluru", settings=config,
                      transport=upstream.transport())

    run(scenario())
    assert len(upstream.requests) == 2
    assert geocoder_status().cooldown_remaining_seconds == 0


def test_consecutive_throttles_grow_the_cooldown_until_it_is_capped(
    clock: FakeClock,
) -> None:
    upstream = Upstream(throttled())
    config = settings(geocoder_max_attempts=1, geocoder_max_cooldown_seconds=30.0)
    waits: list[float] = []

    async def scenario() -> None:
        for _ in range(6):
            with pytest.raises(GeocodingUnavailableError) as raised:
                await geocode("Cubbon Park, Bengaluru", settings=config,
                              transport=upstream.transport())
            waits.append(raised.value.retry_after_seconds)
            clock.t += raised.value.retry_after_seconds + 0.01  # wait it out

    run(scenario())
    assert waits == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_an_absurd_retry_after_is_capped_at_an_hour() -> None:
    upstream = Upstream(throttled("99999999"))

    async def scenario() -> GeocodingUnavailableError:
        with pytest.raises(GeocodingUnavailableError) as raised:
            await geocode("x", settings=settings(), transport=upstream.transport())
        return raised.value

    assert run(scenario()).retry_after_seconds == 3_600.0


def test_an_unparseable_retry_after_falls_back_to_the_backoff(clock: FakeClock) -> None:
    upstream = Upstream(throttled("soon, maybe"), ok())
    run(geocode("x", settings=settings(), transport=upstream.transport()))
    assert clock.pauses == [2.0]


# =========================================================================== #
# Bounded retry for everything else
# =========================================================================== #


@pytest.mark.parametrize("attempts", [1, 2, 3, 5])
def test_no_geocode_sends_more_than_max_attempts(attempts: int) -> None:
    upstream = Upstream(httpx.Response(503))

    async def scenario() -> None:
        with pytest.raises(UpstreamServiceError):
            await geocode("x", settings=settings(
                geocoder_max_attempts=attempts, geocoder_retry_budget_seconds=120.0,
                geocoder_backoff_base_seconds=0.5,
            ), transport=upstream.transport())

    run(scenario())
    assert len(upstream.requests) == attempts


def test_retry_exhaustion_reports_the_last_failure_in_its_own_words(
    clock: FakeClock,
) -> None:
    upstream = Upstream(httpx.Response(502))

    async def scenario() -> UpstreamServiceError:
        with pytest.raises(UpstreamServiceError) as raised:
            await geocode("x", settings=settings(), transport=upstream.transport())
        return raised.value

    error = run(scenario())
    assert error.message == "The geocoding service responded with status 502."
    assert not isinstance(error, GeocodingUnavailableError)
    assert clock.pauses == [2.0, 4.0]
    assert len(upstream.requests) == 3


def test_the_wait_budget_stops_retries_before_the_attempt_cap(clock: FakeClock) -> None:
    upstream = Upstream(httpx.Response(503))

    async def scenario() -> None:
        with pytest.raises(UpstreamServiceError):
            await geocode("x", settings=settings(
                geocoder_max_attempts=5, geocoder_retry_budget_seconds=5.0,
            ), transport=upstream.transport())

    run(scenario())
    # 2 s fits; 2 + 4 would exceed 5 s, so the third attempt is never made.
    assert clock.pauses == [2.0]
    assert len(upstream.requests) == 2


def test_a_timeout_is_retried_with_backoff_then_reported(clock: FakeClock) -> None:
    upstream = Upstream(httpx.ReadTimeout("slow"))

    async def scenario() -> UpstreamServiceError:
        with pytest.raises(UpstreamServiceError) as raised:
            await geocode("x", settings=settings(), transport=upstream.transport())
        return raised.value

    assert run(scenario()).message == "The geocoding service timed out."
    assert len(upstream.requests) == 3


def test_a_timeout_then_recovery_succeeds() -> None:
    upstream = Upstream(httpx.ReadTimeout("slow"), ok())
    place = run(geocode("Cubbon Park, Bengaluru", settings=settings(),
                        transport=upstream.transport()))
    assert place.display_name.startswith("Cubbon Park")
    assert len(upstream.requests) == 2


def test_an_unreachable_provider_is_reported_as_unavailable() -> None:
    upstream = Upstream(httpx.ConnectError("refused"))

    async def scenario() -> UpstreamServiceError:
        with pytest.raises(UpstreamServiceError) as raised:
            await geocode("x", settings=settings(), transport=upstream.transport())
        return raised.value

    assert run(scenario()).message == "The geocoding service is unavailable."
    assert len(upstream.requests) == 3


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json=[{"lat": "12.9"}]),
        httpx.Response(200, json=[{"lat": "x", "lon": "y", "boundingbox": [1, 2, 3, 4]}]),
        httpx.Response(200, json=[{"lat": "1", "lon": "1",
                                   "boundingbox": ["5", "1", "1", "5"]}]),
        httpx.Response(200, json={"not": "a list"}),
    ],
    ids=["not-json", "missing-fields", "non-numeric", "inverted-bbox", "not-a-list"],
)
def test_a_malformed_answer_is_an_error_and_is_not_retried(
    response: httpx.Response,
) -> None:
    upstream = Upstream(response)

    async def scenario() -> None:
        with pytest.raises((UpstreamServiceError, NotFoundError)):
            await geocode("x", settings=settings(), transport=upstream.transport())

    run(scenario())
    assert len(upstream.requests) == 1
    assert nominatim._CACHE == {}


# =========================================================================== #
# No fabricated coordinates
# =========================================================================== #


@pytest.mark.parametrize(
    "failure",
    [
        throttled("60"),
        throttled(),
        httpx.Response(503),
        httpx.Response(400),
        httpx.Response(200, json=[]),
        httpx.ConnectError("refused"),
    ],
    ids=["429-retry-after", "429", "503", "400", "no-match", "unreachable"],
)
def test_a_failure_never_yields_a_location(failure: httpx.Response | Exception) -> None:
    """No fallback coordinate, no nearby city, no stale guess - an error."""

    upstream = Upstream(failure)
    service = GeospatialService(settings=settings(), transport=upstream.transport())

    async def scenario() -> None:
        with pytest.raises((UpstreamServiceError, NotFoundError)):
            await service.resolve(ResolveRequest(place="Cubbon Park, Bengaluru"))
        # And the next caller is not handed one from the cache either.
        with pytest.raises((UpstreamServiceError, NotFoundError)):
            await service.resolve(ResolveRequest(place="Cubbon Park, Bengaluru"))

    run(scenario())
    assert nominatim._CACHE == {}


def test_a_cached_answer_is_the_services_answer_byte_for_byte() -> None:
    upstream = Upstream(ok())
    service = GeospatialService(settings=settings(), transport=upstream.transport())

    async def scenario() -> tuple[Any, Any]:
        first = await service.resolve(ResolveRequest(place="Cubbon Park, Bengaluru"))
        second = await service.resolve(ResolveRequest(place="Cubbon Park, Bengaluru"))
        return first, second

    first, second = run(scenario())
    assert first.model_dump() == second.model_dump()
    assert second.source == "nominatim"


# =========================================================================== #
# Over HTTP and through the agent
# =========================================================================== #


def test_the_resolve_route_answers_503_with_retry_after_during_a_throttle() -> None:
    from app.api.routes.geospatial import get_geospatial_service
    from app.main import create_app
    from fastapi.testclient import TestClient

    upstream = Upstream(throttled("45"))
    app = create_app()
    app.dependency_overrides[get_geospatial_service] = lambda: GeospatialService(
        settings=settings(), transport=upstream.transport()
    )
    response = TestClient(app).post(
        "/api/v1/geospatial/resolve", json={"place": "Cubbon Park, Bengaluru"}
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "45"
    body = response.json()["error"]
    assert body["code"] == "geocoding_unavailable"
    assert "temporarily unavailable" in body["message"]
    assert len(upstream.requests) == 1


def test_an_agent_run_reports_the_throttle_and_searches_nothing() -> None:
    """Honest end to end: no scene searched, no number, the reason stated."""

    from app.services.agent.schemas import AgentQuestionRequest
    from app.services.query import QueryService
    from app.services.query.execution import QueryExecutionService

    from tests.test_standard_workflow import standard_service

    class Tripwire:
        searched = 0

        async def search(self, request: Any) -> Any:
            Tripwire.searched += 1
            raise AssertionError("a catalog search ran without a location")

    upstream = Upstream(throttled("60"))
    execution = QueryExecutionService(
        query_service=QueryService(
            GeospatialService(settings=settings(), transport=upstream.transport())
        ),
        satellite_service=Tripwire(),  # type: ignore[arg-type]
    )
    service, _, analysis = standard_service(query=execution)
    result = run(service.answer(AgentQuestionRequest(
        question="Show vegetation around Cubbon Park, Bengaluru in December 2024"
    )))

    assert Tripwire.searched == 0
    assert analysis.calls == []
    assert result.status != "needs_clarification"  # an outage is not a question
    [failure] = [i for i in result.evidence.items
                 if i.id == "execution.discovery_failure"]
    assert "temporarily unavailable" in (failure.text or "")
    assert not any(i.measurement for i in result.evidence.items)
    assert len(upstream.requests) == 1


def test_readiness_reports_what_the_geocoder_did_without_contacting_it() -> None:
    from app.main import create_app
    from fastapi.testclient import TestClient

    upstream = Upstream(ok())
    config = settings()

    async def scenario() -> None:
        await geocode("Cubbon Park, Bengaluru", settings=config,
                      transport=upstream.transport())
        await geocode("Cubbon Park, Bengaluru", settings=config,
                      transport=upstream.transport())

    run(scenario())
    body = TestClient(create_app()).get("/ready").json()
    detail = {c["name"]: c for c in body["capabilities"]}["geocoder"]["detail"]
    assert "1 upstream requests (0 refused with HTTP 429), 1 cache hits" in detail
    assert "1 places cached" in detail
    assert len(upstream.requests) == 1


# =========================================================================== #
# Optional warm-up
# =========================================================================== #


def test_warm_places_are_parsed_trimmed_and_deduplicated() -> None:
    config = settings(geocoder_warm_places=(
        " Cubbon Park, Bengaluru ;Ameerpet, Hyderabad;; cubbon park,  bengaluru "
    ))
    assert warm_places(config) == ["Cubbon Park, Bengaluru", "Ameerpet, Hyderabad"]
    assert warm_places(settings()) == []


def test_warm_up_caches_real_answers_through_the_normal_path() -> None:
    upstream = Upstream(ok())
    config = settings(geocoder_warm_places="Cubbon Park, Bengaluru;Ameerpet, Hyderabad")

    async def scenario() -> list[str]:
        left = await warm_geocoder_cache(config, transport=upstream.transport())
        await geocode("Cubbon Park, Bengaluru", settings=config,
                      transport=upstream.transport())
        return left

    assert run(scenario()) == []
    assert len(upstream.requests) == 2  # the later question was a cache hit
    assert geocoder_status().cache_hits == 1


def test_warm_up_waits_out_a_throttle_and_tries_again(clock: FakeClock) -> None:
    upstream = Upstream(throttled("90"), ok())
    config = settings(geocoder_warm_places="Cubbon Park, Bengaluru")

    left = run(warm_geocoder_cache(config, transport=upstream.transport()))

    assert left == []
    assert len(upstream.requests) == 2
    assert clock.pauses and clock.pauses[0] == pytest.approx(90.0)


def test_warm_up_is_bounded_and_caches_nothing_it_could_not_resolve(
    clock: FakeClock,
) -> None:
    upstream = Upstream(throttled("30"))
    config = settings(geocoder_warm_places="Cubbon Park, Bengaluru;Dal Lake")

    left = run(warm_geocoder_cache(config, transport=upstream.transport(), rounds=3))

    assert left == ["Cubbon Park, Bengaluru", "Dal Lake"]
    # One request per round at most: the cooldown refuses the rest unsent.
    assert len(upstream.requests) == 3
    assert nominatim._CACHE == {}


def test_warm_up_does_not_retry_a_place_that_does_not_exist() -> None:
    upstream = Upstream(ok(payload=[]))
    config = settings(geocoder_warm_places="Qwxzt Nowhere")

    assert run(warm_geocoder_cache(config, transport=upstream.transport())) == []
    assert len(upstream.requests) == 1


def test_the_app_starts_the_warm_up_only_when_places_are_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main
    from fastapi.testclient import TestClient

    started: list[list[str]] = []

    async def record(config: Settings) -> list[str]:
        started.append(warm_places(config))
        return []

    monkeypatch.setattr(main, "warm_geocoder_cache", record)

    with TestClient(main.create_app(settings())):
        pass
    assert started == []

    with TestClient(main.create_app(settings(geocoder_warm_places="Dal Lake"))):
        pass
    assert started == [["Dal Lake"]]
