"""Request/response models for geospatial grounding."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Coordinate(BaseModel):
    """A single WGS84 point."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class BoundingBox(BaseModel):
    """Axis-aligned geographic bounding box in WGS84 degrees."""

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        return self

    @property
    def center(self) -> Coordinate:
        return Coordinate(
            lat=(self.south + self.north) / 2,
            lon=(self.west + self.east) / 2,
        )


class ResolveRequest(BaseModel):
    """Resolve a place name *or* a bounding box - exactly one is required."""

    place: str | None = Field(default=None, max_length=300)
    bbox: BoundingBox | None = None

    @field_validator("place", mode="before")
    @classmethod
    def _clean_place(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.place is None) == (self.bbox is None):
            raise ValueError("provide exactly one of 'place' or 'bbox'")
        return self


class ResolveResponse(BaseModel):
    """A validated geographic representation of the resolved location."""

    query_type: Literal["place", "bbox"]
    display_name: str | None
    center: Coordinate
    bbox: BoundingBox
    source: Literal["nominatim", "input"]
