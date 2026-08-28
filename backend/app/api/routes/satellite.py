"""Sentinel-2 scene-discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.satellite import (
    SatelliteService,
    SceneSearchRequest,
    SceneSearchResponse,
)

router = APIRouter()


def get_satellite_service() -> SatelliteService:
    """Provider for :class:`SatelliteService`; overridden in tests."""

    return SatelliteService()


@router.post("/search", response_model=SceneSearchResponse)
async def search_scenes(
    request: SceneSearchRequest,
    service: SatelliteService = Depends(get_satellite_service),
) -> SceneSearchResponse:
    """Discover Sentinel-2 L2A scenes (STAC metadata only) for a bounding box
    and date range, with optional cloud-cover and result limits."""

    return await service.search(request)
