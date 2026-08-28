"""Top-level API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import geospatial, health
from app.core.config import get_settings

settings = get_settings()

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(
    geospatial.router,
    prefix=f"{settings.api_v1_prefix}/geospatial",
    tags=["geospatial"],
)

# Future domain routers (query, satellite, multimodal, temporal, ai, map) will be
# registered here as they are implemented.
