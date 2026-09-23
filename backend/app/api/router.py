"""Top-level API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import ai, geospatial, health, query, satellite
from app.core.config import Settings, get_settings


def build_api_router(settings: Settings | None = None) -> APIRouter:
    """Assemble the API router for one application's configuration.

    Built per application rather than once at import.

    The module previously held ``settings = get_settings()`` at import time and
    baked the prefixes from that cached singleton. ``create_app(settings=...)``
    therefore accepted a configuration it could not honour: the routes had
    already been mounted under whatever the FIRST import saw, so a production
    ``api_v1_prefix`` silently did not apply, and the failure would have been a
    404 at a URL the configuration says should exist.

    ``settings`` defaults to the cached instance, so the ordinary path is
    unchanged.
    """

    settings = settings or get_settings()
    prefix = settings.api_v1_prefix

    router = APIRouter()
    router.include_router(health.router, tags=["health"])
    router.include_router(
        geospatial.router, prefix=f"{prefix}/geospatial", tags=["geospatial"]
    )
    router.include_router(
        satellite.router, prefix=f"{prefix}/satellite", tags=["satellite"]
    )
    router.include_router(query.router, prefix=f"{prefix}/query", tags=["query"])
    router.include_router(ai.router, prefix=f"{prefix}/ai", tags=["ai"])
    return router


# Future domain routers (multimodal, temporal, map) will be registered here as
# they are implemented.
