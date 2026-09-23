"""The application's budget for the public Nominatim instance.

Its operator asks for at most one request per second from an application, a
genuine User-Agent, and caching of repeated queries. Only the User-Agent was
ever sent. On the execution path a single query geocodes once per window per
modality, so a three-window two-modality request issued six geocodes as fast as
the event loop could send them.

These tests pin the three parts of the policy - spacing, serialisation and
caching - plus the bounded retry, and they pin that a failure is never cached.
The suite sets the interval to zero (see ``conftest``); the tests that are
ABOUT the spacing configure a real one here.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from app.core.config import Settings
from app.core.errors import NotFoundError, UpstreamServiceError
from app.services.geospatial.nominatim import geocode, reset_geocoder_state

CHENNAI = [
    {
        "lat": "13.0837",
        "lon": "80.2702",
        "boundingbox": ["12.9", "13.2", "80.1", "80.3"],
        "display_name": "Chennai, Tamil Nadu, India",
        "class": "place",
        "type": "city",
    }
]


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"geocoder_min_interval_seconds": 0.0}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class Recorder:
    """A transport that records every request and replays canned responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if len(self._responses) > 1:
                return self._responses.pop(0)
            return self._responses[0]

        return httpx.MockTransport(handle)


def ok(payload: object = CHENNAI) -> httpx.Response:
    return httpx.Response(200, json=payload)


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    reset_geocoder_state()


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_a_repeated_query_costs_nothing_upstream() -> None:
    recorder = Recorder(ok())
    config = settings()

    async def scenario() -> None:
        first = await geocode("Chennai", settings=config, transport=recorder.transport())
        second = await geocode(
            "Chennai", settings=config, transport=recorder.transport()
        )
        assert first == second

    asyncio.run(scenario())
    assert len(recorder.requests) == 1


def test_the_cache_ignores_case_and_surrounding_space() -> None:
    recorder = Recorder(ok())
    config = settings()

    async def scenario() -> None:
        await geocode("Chennai", settings=config, transport=recorder.transport())
        await geocode("  chennai  ", settings=config, transport=recorder.transport())

    asyncio.run(scenario())
    assert len(recorder.requests) == 1


def test_a_different_place_is_not_served_from_the_cache() -> None:
    """Non-vacuity: caching everything would be indistinguishable here."""

    recorder = Recorder(ok())
    config = settings()

    async def scenario() -> None:
        await geocode("Chennai", settings=config, transport=recorder.transport())
        await geocode("Rotterdam", settings=config, transport=recorder.transport())

    asyncio.run(scenario())
    assert len(recorder.requests) == 2


def test_a_cached_answer_from_one_service_never_satisfies_another() -> None:
    recorder = Recorder(ok())

    async def scenario() -> None:
        await geocode(
            "Chennai",
            settings=settings(nominatim_base_url="https://a.example"),
            transport=recorder.transport(),
        )
        await geocode(
            "Chennai",
            settings=settings(nominatim_base_url="https://b.example"),
            transport=recorder.transport(),
        )

    asyncio.run(scenario())
    assert len(recorder.requests) == 2


def test_a_failure_is_never_cached() -> None:
    """An outage cached for the TTL would outlive the outage itself."""

    recorder = Recorder(httpx.Response(500), httpx.Response(500), ok())
    config = settings()

    async def scenario() -> None:
        with pytest.raises(UpstreamServiceError):
            await geocode("Chennai", settings=config, transport=recorder.transport())
        # The service has recovered; the next caller must reach it.
        place = await geocode(
            "Chennai", settings=config, transport=recorder.transport()
        )
        assert place.display_name.startswith("Chennai")

    asyncio.run(scenario())


def test_an_expired_entry_is_refetched() -> None:
    recorder = Recorder(ok())
    config = settings(geocoder_cache_ttl_seconds=0.0)

    async def scenario() -> None:
        await geocode("Chennai", settings=config, transport=recorder.transport())
        await asyncio.sleep(0.01)
        await geocode("Chennai", settings=config, transport=recorder.transport())

    asyncio.run(scenario())
    assert len(recorder.requests) == 2


def test_the_cache_is_bounded() -> None:
    recorder = Recorder(ok())
    config = settings(geocoder_cache_entries=4)

    async def scenario() -> None:
        for index in range(20):
            await geocode(
                f"place-{index}", settings=config, transport=recorder.transport()
            )

    asyncio.run(scenario())

    from app.services.geospatial.nominatim import _CACHE

    assert len(_CACHE) <= 4


# --------------------------------------------------------------------------- #
# Spacing and serialisation
# --------------------------------------------------------------------------- #


def test_two_requests_are_spaced_by_the_configured_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wait is asserted, not slept through.

    Sleeping a real second per test would pay the politeness budget to nobody.
    """

    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def record(delay: float) -> None:
        slept.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("app.services.geospatial.nominatim.asyncio.sleep", record)

    recorder = Recorder(ok())
    config = settings(geocoder_min_interval_seconds=1.0)

    async def scenario() -> None:
        await geocode("Chennai", settings=config, transport=recorder.transport())
        await geocode("Rotterdam", settings=config, transport=recorder.transport())

    asyncio.run(scenario())

    assert len(recorder.requests) == 2
    assert slept, "the second request was not spaced at all"
    assert 0 < slept[0] <= 1.0


def test_concurrent_callers_are_serialised_not_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ten concurrent queries must not become ten simultaneous requests."""

    in_flight = 0
    peak = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        in_flight -= 1
        return ok()

    config = settings()

    async def scenario() -> None:
        await asyncio.gather(
            *[
                geocode(
                    f"place-{index}",
                    settings=config,
                    transport=httpx.MockTransport(handle),
                )
                for index in range(10)
            ]
        )

    asyncio.run(scenario())
    assert peak == 1


def test_concurrent_callers_for_one_place_make_one_request() -> None:
    """The second caller finds the cache filled while it waited its turn."""

    recorder = Recorder(ok())
    config = settings()

    async def scenario() -> None:
        await asyncio.gather(
            *[
                geocode(
                    "Chennai", settings=config, transport=recorder.transport()
                )
                for _ in range(5)
            ]
        )

    asyncio.run(scenario())
    assert len(recorder.requests) == 1


# --------------------------------------------------------------------------- #
# Bounded retry
# --------------------------------------------------------------------------- #


def test_a_transient_failure_is_retried_once() -> None:
    recorder = Recorder(httpx.Response(503), ok())
    config = settings()

    async def scenario() -> None:
        place = await geocode(
            "Chennai", settings=config, transport=recorder.transport()
        )
        assert place.display_name.startswith("Chennai")

    asyncio.run(scenario())
    assert len(recorder.requests) == 2


def test_retries_are_bounded() -> None:
    """An unbounded retry loop against a rate limiter is how one gets blocked."""

    recorder = Recorder(httpx.Response(429))
    config = settings()

    async def scenario() -> None:
        with pytest.raises(UpstreamServiceError) as raised:
            await geocode("Chennai", settings=config, transport=recorder.transport())
        assert "429" in str(raised.value)

    asyncio.run(scenario())
    assert len(recorder.requests) == 2


def test_a_client_error_is_not_retried() -> None:
    """Repeating a request the service already rejected is rude and useless."""

    recorder = Recorder(httpx.Response(400))
    config = settings()

    async def scenario() -> None:
        with pytest.raises(UpstreamServiceError):
            await geocode("Chennai", settings=config, transport=recorder.transport())

    asyncio.run(scenario())
    assert len(recorder.requests) == 1


def test_an_empty_result_is_not_retried() -> None:
    recorder = Recorder(ok(payload=[]))
    config = settings()

    async def scenario() -> None:
        with pytest.raises(NotFoundError):
            await geocode("nowhere-xyz", settings=config, transport=recorder.transport())

    asyncio.run(scenario())
    assert len(recorder.requests) == 1
