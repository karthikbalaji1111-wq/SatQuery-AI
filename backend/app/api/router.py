"""Top-level API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])

# Future domain routers (query, satellite, multimodal, temporal, geospatial, ai,
# map) will be registered here as they are implemented.
