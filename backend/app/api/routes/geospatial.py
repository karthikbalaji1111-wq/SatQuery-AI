"""Geospatial grounding endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.geospatial import GeospatialService, ResolveRequest, ResolveResponse

router = APIRouter()


def get_geospatial_service() -> GeospatialService:
    """Provider for :class:`GeospatialService`; overridden in tests."""

    return GeospatialService()


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_location(
    request: ResolveRequest,
    service: GeospatialService = Depends(get_geospatial_service),
) -> ResolveResponse:
    """Resolve a place name (via OpenStreetMap Nominatim) or a bounding box into
    a validated geographic representation (center point + bounding box)."""

    return await service.resolve(request)
