"""Request/response models for Sentinel-2 scene discovery.

The geospatial :class:`BoundingBox` is reused verbatim - this phase does not
define a competing geometry schema.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.services.geospatial.schemas import BoundingBox

DEFAULT_LIMIT = 10
MAX_LIMIT = 100

# Windowed-readable remote rasters (COGs) supported for bounded retrieval:
#   "visual" - Sentinel-2 true-colour 8-bit RGB, used as-is.
#   "vv"     - Sentinel-1 GRD VV backscatter, single-band Float32, display-only
#              normalized to 8-bit grayscale (see raster._normalize_sar_band).
DEFAULT_IMAGERY_ASSET = "visual"
SAR_IMAGERY_ASSET = "vv"
SUPPORTED_IMAGERY_ASSETS = (DEFAULT_IMAGERY_ASSET, SAR_IMAGERY_ASSET)

# Assets readable QUANTITATIVELY (raw values, native resolution) for analysis.
# Deliberately a separate allowlist from the display whitelist above: display
# and analysis are different concerns and must not be merged. These are Earth
# Search *STAC asset keys* (common names), not Sentinel-2 band identifiers -
# "green" is the key for band B03, "nir" for B08, "red" for B04. 20 m assets
# ("swir16", "scl") are excluded: mixing them with 10 m bands would require
# resampling, which this phase does not do.
ANALYSIS_BAND_ASSETS = ("green", "nir", "red")


class SceneSearchRequest(BaseModel):
    """Validated input for a satellite scene search.

    ``collection`` optionally overrides the configured default STAC collection
    (e.g. to target Sentinel-1 rather than Sentinel-2). ``None`` preserves the
    existing Sentinel-2 behaviour.
    """

    bbox: BoundingBox
    start_date: date
    end_date: date
    collection: str | None = None
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)

    @model_validator(mode="after")
    def _check_dates(self) -> Self:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        return self


class SceneAsset(BaseModel):
    """Minimal pointer to a scene asset - metadata only, never downloaded."""

    key: str
    href: str
    type: str | None = None
    title: str | None = None
    roles: list[str] | None = None


class Scene(BaseModel):
    """A normalised Sentinel-2 scene. Never the raw STAC item."""

    id: str
    datetime: str | None
    bbox: BoundingBox | None
    geometry: dict[str, Any] | None
    cloud_cover: float | None
    collection: str | None
    platform: str | None
    processing_level: str | None
    thumbnail_url: str | None
    assets: list[SceneAsset]


class QueryEcho(BaseModel):
    """The exact parameters sent to the STAC API."""

    collections: list[str]
    bbox: list[float]
    datetime: str
    max_cloud_cover: float | None
    limit: int
    filter: dict[str, Any] | None


class SceneSearchResponse(BaseModel):
    """Normalised search result returned to the client."""

    query: QueryEcho
    scene_count: int
    scenes: list[Scene]
    catalog: str


class ImageryRequest(BaseModel):
    """Bounded imagery request for an already-discovered scene.

    No natural-language search happens here - the scene must already be known
    from the discovery phase.

    ``collection`` names the STAC collection the scene belongs to (e.g.
    ``sentinel-1-grd``); ``None`` preserves the existing Sentinel-2 default.
    """

    scene_id: str = Field(min_length=1, max_length=200)
    bbox: BoundingBox
    asset: str = Field(default=DEFAULT_IMAGERY_ASSET, min_length=1, max_length=50)
    collection: str | None = None
    max_dimension: int | None = Field(default=None, ge=16, le=4096)


class WindowInfo(BaseModel):
    """The pixel window actually read from the source raster."""

    col_off: int
    row_off: int
    width: int
    height: int


class ImageryResponse(BaseModel):
    """A bounded RGB representation of a scene, suitable for a later VLM phase.

    The raw STAC item is never exposed.
    """

    scene_id: str
    bbox: BoundingBox
    asset: str
    asset_href: str
    width: int
    height: int
    format: Literal["png"]
    media_type: Literal["image/png"]
    bands: list[str]
    crs: str | None
    resolution: float | None
    normalization: str
    window: WindowInfo
    source_shape: list[int]
    #: Affine coefficients ``[a, b, c, d, e, f]`` of the window actually read,
    #: in ``crs``. Optional for backward compatibility. This - never ``bbox`` -
    #: is the georeferencing source: ``bbox`` echoes the REQUEST, while the read
    #: window is floor/ceil clamped onto the source grid and so covers more.
    transform: list[float] | None = Field(
        default=None, min_length=6, max_length=6
    )
    image_base64: str
