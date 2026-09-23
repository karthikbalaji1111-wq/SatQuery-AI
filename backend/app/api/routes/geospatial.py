"""Geospatial grounding endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.limits import rate_limited
from app.services.geospatial import GeospatialService, ResolveRequest, ResolveResponse

router = APIRouter()


def get_geospatial_service() -> GeospatialService:
    """Provider for :class:`GeospatialService`; overridden in tests."""

    return GeospatialService()


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    # Reaches a third party whose usage policy this application must honour.
    # The geocoder's own budget is application-wide; this bounds one caller.
    dependencies=[Depends(rate_limited)],
)
async def resolve_location(
    request: ResolveRequest,
    service: GeospatialService = Depends(get_geospatial_service),
) -> ResolveResponse:
    """Resolve a place name (via OpenStreetMap Nominatim) or a bounding box into
    a validated geographic representation (center point + bounding box)."""

    return await service.resolve(request)
