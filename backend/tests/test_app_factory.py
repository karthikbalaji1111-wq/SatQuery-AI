"""The application factory honours the configuration it is handed.

``create_app(settings=...)`` accepted a ``Settings`` that could not reach the
routes: ``api/router.py`` called ``get_settings()`` at IMPORT time and mounted
every router under whatever that cached singleton said. A deployment that set
``SATQUERY_API_V1_PREFIX`` therefore served its API at the default path while
its own configuration said otherwise - a 404 at a URL the configuration
promises, which is the kind of failure that looks like a routing bug for a long
time before anyone suspects the factory.
"""

from __future__ import annotations

from collections.abc import Iterator

from app.core.config import Settings
from app.main import create_app
from fastapi import FastAPI
from fastapi.testclient import TestClient


def iter_routes(app: FastAPI) -> Iterator[object]:
    """Every route, descending into included routers.

    This FastAPI version keeps an included router as ONE entry in ``app.routes``
    rather than flattening its routes into it, so a naive scan finds only the
    documentation endpoints and concludes - wrongly - that the API is not
    mounted. Shared with ``test_limits`` so both introspect the real tree.
    """

    stack = list(app.routes)
    while stack:
        route = stack.pop()
        # An included router appears as one `_IncludedRouter`, which keeps the
        # real thing on `original_router` rather than exposing `.routes`.
        included = getattr(route, "original_router", None)
        nested = getattr(included, "routes", None) or getattr(route, "routes", None)
        if nested:
            stack.extend(nested)
        else:
            yield route


def paths(app: FastAPI) -> set[str]:
    """Every path the application actually publishes.

    Read from the OpenAPI document rather than by walking the route tree: this
    FastAPI version keeps an included router's routes under their ROUTER-LOCAL
    paths and applies the prefix when matching, so a walk sees ``/agent`` where
    a client sees ``/api/v1/query/agent``. The document is the client's view,
    which is exactly what a test about mounting should assert.
    """

    return set(app.openapi()["paths"])


def test_the_configured_prefix_is_where_the_routes_are_mounted() -> None:
    app = create_app(Settings(api_v1_prefix="/api/v2"))

    assert "/api/v2/query/agent" in paths(app)
    assert "/api/v2/satellite/search" in paths(app)
    # The default prefix must be GONE, not merely also present.
    assert "/api/v1/query/agent" not in paths(app)


def test_the_default_prefix_still_applies_when_none_is_given() -> None:
    """Non-vacuity: the ordinary path is unchanged."""

    assert "/api/v1/query/agent" in paths(create_app())


def test_liveness_is_not_under_the_versioned_prefix() -> None:
    """A probe URL that moves with an API version is not a probe URL."""

    app = create_app(Settings(api_v1_prefix="/api/v2"))

    assert "/health" in paths(app)
    assert TestClient(app).get("/health").status_code == 200


def test_each_application_gets_its_own_admission_state() -> None:
    """Two apps must not share counters.

    A limiter shared across a process would make one test's requests another's
    failures - and, in production, one tenant's burst another's rejection.
    """

    first = create_app(Settings(rate_limit_requests=1))
    second = create_app(Settings(rate_limit_requests=1))

    assert first.state.limits is not second.state.limits
    assert first.state.limits.limiter is not second.state.limits.limiter


def test_the_configured_limits_reach_the_application() -> None:
    app = create_app(Settings(rate_limit_requests=7, max_concurrent_workflows=3))

    assert app.state.limits.limiter.limit == 7
    assert app.state.limits.settings.max_concurrent_workflows == 3
