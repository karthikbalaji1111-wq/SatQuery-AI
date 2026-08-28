"""Request/response models for Sentinel-2 scene discovery.

The geospatial :class:`BoundingBox` is reused verbatim - this phase does not
define a competing geometry schema.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from app.services.geospatial.schemas import BoundingBox

DEFAULT_LIMIT = 10
MAX_LIMIT = 100


class SceneSearchRequest(BaseModel):
    """Validated input for a Sentinel-2 scene search."""

    bbox: BoundingBox
    start_date: date
    end_date: date
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
