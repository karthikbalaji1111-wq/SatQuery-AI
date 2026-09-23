"""Sentinel-1 SAR and Sentinel-2 optical imagery retrieval.

Implemented so far:
- Sentinel-2 optical scene *discovery* (Earth Search STAC metadata).
- Bounded Sentinel-2 RGB *imagery retrieval* (windowed COG reads).
- Sentinel-1 RTC scene *discovery* (Planetary Computer STAC metadata),
  including polarizations, instrument mode and orbit direction.
- Bounded Sentinel-1 RTC VV/VH *imagery retrieval*: the provider's
  terrain-corrected gamma-naught COG, opened through an anonymously signed
  read URL and display-stretched to grayscale.

Deliberately NOT implemented: SAR radiometric calibration, dB-domain
measurement, speckle filtering, terrain correction (the provider does it) and
any quantitative backscatter analysis. No SAR asset is in
``ANALYSIS_BAND_ASSETS``, so the quantitative read path refuses one.
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
