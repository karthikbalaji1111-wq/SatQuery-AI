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

# Assets that are true 3-band RGB, windowed-readable remote rasters (COGs).
SUPPORTED_IMAGERY_ASSETS = ("visual",)
DEFAULT_IMAGERY_ASSET = "visual"


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
    """

    scene_id: str = Field(min_length=1, max_length=200)
    bbox: BoundingBox
    asset: str = Field(default=DEFAULT_IMAGERY_ASSET, min_length=1, max_length=50)
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
    image_base64: str
