"""Admission control: rate, concurrency and payload size.

Every endpoint here is unauthenticated, and one request can cost a catalog
search per window, a windowed COG read per band, a base64 PNG held in memory
and a metered provider call. The per-request contracts bound the SHAPE of one
request; nothing bounded how many could be in flight or how often one caller
could ask.

The three refusals are deliberately distinguishable and are tested as such:
429 is about the caller, 503 is about this process, 413 is about this body.
Collapsing them would tell a client to back off when the honest answer is
"try again now".
"""

from __future__ import annotations

import asyncio

import pytest
from app.core.config import Settings
from app.core.errors import (
    RateLimitedError,
    ServiceOverloadedError,
    register_exception_handlers,
)
from app.core.limits import (
    ConcurrencyGate,
    FixedWindowRateLimiter,
    install_request_limits,
    rate_limited,
    workflow_slot,
)
from app.main import create_app
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tests.test_app_factory import iter_routes

# --------------------------------------------------------------------------- #
# The limiter itself
# --------------------------------------------------------------------------- #


def test_requests_within_the_allowance_are_admitted() -> None:
    limiter = FixedWindowRateLimiter(limit=3, window_seconds=60.0)

    for moment in (0.0, 1.0, 2.0):
        limiter.admit("client-a", now=moment)  # must not raise


def test_the_request_after_the_allowance_is_refused() -> None:
    limiter = FixedWindowRateLimiter(limit=2, window_seconds=60.0)
    limiter.admit("client-a", now=0.0)
    limiter.admit("client-a", now=1.0)

    with pytest.raises(RateLimitedError) as raised:
        limiter.admit("client-a", now=2.0)

    # Actionable: how long to wait, in the response itself.
    assert raised.value.headers is not None
    assert raised.value.headers["Retry-After"] == "58"
    assert raised.value.status_code == 429


def test_the_window_slides_rather_than_resetting() -> None:
    """A fixed calendar window admits twice the limit across its boundary.

    That burst is the thing this exists to stop, so the oldest hit must age out
    individually rather than the whole window clearing at once.
    """

    limiter = FixedWindowRateLimiter(limit=2, window_seconds=10.0)
    limiter.admit("client-a", now=0.0)
    limiter.admit("client-a", now=9.0)

    with pytest.raises(RateLimitedError):
        limiter.admit("client-a", now=9.5)

    # The first hit has aged out by 10.1; the second (t=9.0) has not.
    limiter.admit("client-a", now=10.1)
    with pytest.raises(RateLimitedError):
        limiter.admit("client-a", now=10.2)


def test_clients_are_counted_separately() -> None:
    limiter = FixedWindowRateLimiter(limit=1, window_seconds=60.0)
    limiter.admit("client-a", now=0.0)

    limiter.admit("client-b", now=0.0)  # unaffected by client-a
    with pytest.raises(RateLimitedError):
        limiter.admit("client-a", now=0.1)


def test_the_bookkeeping_is_itself_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A defence against unbounded memory must not be an unbounded dictionary.

    The endpoint is addressed by anyone, so the number of distinct clients is
    attacker-controlled.
    """

    monkeypatch.setattr("app.core.limits._MAX_TRACKED_CLIENTS", 8)
    limiter = FixedWindowRateLimiter(limit=5, window_seconds=60.0)

    for index in range(200):
        limiter.admit(f"client-{index}", now=float(index))

    assert len(limiter._hits) <= 8


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_a_gate_admits_up_to_its_limit_and_then_refuses() -> None:
    async def scenario() -> None:
        gate = ConcurrencyGate(2, name="test")
        async with gate.slot(wait=0.01), gate.slot(wait=0.01):
            # Both slots are held; the third caller waits briefly, then is
            # refused rather than queueing without bound.
            with pytest.raises(ServiceOverloadedError) as raised:
                async with gate.slot(wait=0.01):
                    pass
            assert raised.value.status_code == 503

    asyncio.run(scenario())


def test_a_released_slot_is_reusable() -> None:
    async def scenario() -> None:
        gate = ConcurrencyGate(1, name="test")
        async with gate.slot(wait=0.01):
            pass
        async with gate.slot(wait=0.01):
            pass  # the first holder released, so this must succeed

    asyncio.run(scenario())


def test_a_waiting_caller_is_admitted_when_a_slot_frees() -> None:
    """A passing burst should queue, not fail - that is what the wait is for."""

    async def scenario() -> None:
        gate = ConcurrencyGate(1, name="test")
        admitted: list[str] = []

        async def hold() -> None:
            async with gate.slot(wait=0.01):
                admitted.append("first")
                await asyncio.sleep(0.02)

        async def follow() -> None:
            await asyncio.sleep(0.005)
            async with gate.slot(wait=1.0):
                admitted.append("second")

        await asyncio.gather(hold(), follow())
        assert admitted == ["first", "second"]

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Over HTTP
# --------------------------------------------------------------------------- #


def probe_app(settings: Settings) -> FastAPI:
    """A minimal app carrying the real dependencies and nothing else.

    The mechanism is tested here rather than through a real route so that no
    network call, provider or raster read is involved; that the real routes
    actually carry these dependencies is asserted separately below.
    """

    app = FastAPI()
    install_request_limits(app, settings)
    register_exception_handlers(app)

    @app.get("/probe", dependencies=[Depends(rate_limited)])
    async def probe() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/slow", dependencies=[Depends(workflow_slot)])
    async def slow() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_an_exceeded_allowance_answers_429_in_the_one_envelope() -> None:
    client = TestClient(
        probe_app(Settings(rate_limit_requests=2, rate_limit_window_seconds=60.0)),
        raise_server_exceptions=False,
    )

    assert client.get("/probe").status_code == 200
    assert client.get("/probe").status_code == 200

    refused = client.get("/probe")
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "rate_limited"
    # Actionable rather than merely negative.
    assert int(refused.headers["Retry-After"]) >= 1
    # The same envelope every other failure uses.
    assert "detail" not in refused.json()


def test_an_oversized_body_is_refused_before_it_is_read() -> None:
    app = create_app(Settings(max_request_bytes=500))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/query/parse", json={"prompt": "x" * 5000}
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_a_refusal_still_carries_cors_headers() -> None:
    """A 413 without CORS headers reads to a browser as a network failure.

    The size guard is installed INSIDE the CORS middleware for this reason, and
    the ordering is easy to reverse by accident.
    """

    origin = "http://localhost:5173"
    app = create_app(
        Settings(max_request_bytes=500, cors_origins=[origin])
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/query/parse",
        json={"prompt": "x" * 5000},
        headers={"Origin": origin},
    )

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == origin


def test_a_body_within_the_limit_is_not_refused() -> None:
    """Non-vacuity: the guard must not simply reject everything."""

    app = create_app(Settings(max_request_bytes=1_000_000))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/api/v1/query/parse", json={"prompt": "show Chennai"})

    assert response.status_code != 413


# --------------------------------------------------------------------------- #
# The real routes carry the real dependencies
# --------------------------------------------------------------------------- #


def _dependency_calls(route: object) -> set[object]:
    """Every dependency callable reachable from one route."""

    found: set[object] = set()
    stack = list(getattr(route, "dependant", None).dependencies)  # type: ignore[union-attr]
    while stack:
        dependency = stack.pop()
        if dependency.call is not None:
            found.add(dependency.call)
        stack.extend(dependency.dependencies)
    return found


def _route(app: FastAPI, path: str) -> object:
    """The one route at ``path``, which is the ROUTER-LOCAL path.

    This FastAPI version stores an included router's routes unprefixed and
    applies the prefix when matching, so the route objects carry ``/agent``
    rather than ``/api/v1/query/agent``. The published URLs are asserted in
    ``test_app_factory`` from the OpenAPI document; what matters here is which
    dependencies a route carries.

    Ambiguity fails loudly: if two routers ever share a local path, this must
    not quietly assert against whichever one it happened to find first.
    """

    matches = [
        route for route in iter_routes(app) if getattr(route, "path", None) == path
    ]
    if not matches:
        raise AssertionError(f"no route at {path}")
    if len(matches) > 1:
        raise AssertionError(f"{path} is ambiguous across routers")
    return matches[0]


#: Router-local paths - see `_route`. Published as `/api/v1/<router>/<path>`.
EXPENSIVE = [
    "/parse",
    "/build-plan",
    "/execute",
    "/analyze",
    "/agent",
    "/search",
    "/imagery",
    "/resolve",
]

#: Every route that does real work holds a workflow slot. `build-plan` and
#: `search` make one bounded outbound call each and are rate-limited only.
SLOT_HOLDERS = [
    "/parse",
    "/execute",
    "/analyze",
    "/agent",
    "/imagery",
]


@pytest.mark.parametrize("path", EXPENSIVE)
def test_every_expensive_route_is_rate_limited(path: str) -> None:
    app = create_app()
    assert rate_limited in _dependency_calls(_route(app, path))


@pytest.mark.parametrize("path", SLOT_HOLDERS)
def test_every_working_route_holds_a_workflow_slot(path: str) -> None:
    app = create_app()
    assert workflow_slot in _dependency_calls(_route(app, path))


def test_health_is_never_limited() -> None:
    """A probe that queues behind four analyses reports an outage that is not
    happening, and a rate-limited health check turns monitoring into an
    attacker."""

    app = create_app()
    calls = _dependency_calls(_route(app, "/health"))

    assert rate_limited not in calls
    assert workflow_slot not in calls


def test_the_model_catalog_is_never_limited() -> None:
    """The selector polls this on focus; queueing it behind analyses would make
    the UI report providers as unavailable while the server is merely busy."""

    app = create_app()
    calls = _dependency_calls(_route(app, "/models"))

    assert workflow_slot not in calls
