"""Sentinel-2 scene-discovery and bounded-imagery endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.satellite import (
    ImageryRequest,
    ImageryResponse,
    ImageryService,
    SatelliteService,
    SceneSearchRequest,
    SceneSearchResponse,
)

router = APIRouter()


def get_satellite_service() -> SatelliteService:
    """Provider for :class:`SatelliteService`; overridden in tests."""

    return SatelliteService()


def get_imagery_service() -> ImageryService:
    """Provider for :class:`ImageryService`; overridden in tests."""

    return ImageryService()


@router.post("/search", response_model=SceneSearchResponse)
async def search_scenes(
    request: SceneSearchRequest,
    service: SatelliteService = Depends(get_satellite_service),
) -> SceneSearchResponse:
    """Discover Sentinel-2 L2A scenes (STAC metadata only) for a bounding box
    and date range, with optional cloud-cover and result limits."""

    return await service.search(request)


@router.post("/imagery", response_model=ImageryResponse)
def retrieve_imagery(
    request: ImageryRequest,
    service: ImageryService = Depends(get_imagery_service),
) -> ImageryResponse:
    """Retrieve a bounded RGB window for an already-discovered scene.

    Uses only the requested spatial window (windowed COG read) - the full scene
    is never downloaded. No natural-language location search is performed.

    Declared ``def`` (not ``async``): the STAC item lookup and windowed raster
    read are blocking, so Starlette runs this in a threadpool."""

    return service.retrieve(request)
