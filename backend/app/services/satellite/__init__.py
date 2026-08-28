"""Sentinel-1 SAR and Sentinel-2 optical imagery retrieval.

Implemented so far:
- Sentinel-2 optical scene *discovery* (STAC metadata).
- Bounded Sentinel-2 RGB *imagery retrieval* (windowed COG reads).
"""

from app.services.satellite.imagery import ImageryService
from app.services.satellite.schemas import (
    ImageryRequest,
    ImageryResponse,
    Scene,
    SceneAsset,
    SceneSearchRequest,
    SceneSearchResponse,
)
from app.services.satellite.service import SatelliteService

__all__ = [
    "ImageryRequest",
    "ImageryResponse",
    "ImageryService",
    "Scene",
    "SceneAsset",
    "SceneSearchRequest",
    "SceneSearchResponse",
    "SatelliteService",
]
