"""Top-level API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import geospatial, health, query, satellite
from app.core.config import get_settings

settings = get_settings()

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(
    geospatial.router,
    prefix=f"{settings.api_v1_prefix}/geospatial",
    tags=["geospatial"],
)
api_router.include_router(
    satellite.router,
    prefix=f"{settings.api_v1_prefix}/satellite",
    tags=["satellite"],
)
api_router.include_router(
    query.router,
    prefix=f"{settings.api_v1_prefix}/query",
    tags=["query"],
)

# Future domain routers (multimodal, temporal, ai, map) will be registered
# here as they are implemented.
