"""Structured query-intent contracts.

This layer is deliberately deterministic: no NLP, no LLM. It only defines the
shape of "what the user asks for" (:class:`SatQueryIntent`) and the shape of a
grounded plan (:class:`ResolvedQueryPlan`). Location grounding is delegated to
the existing Geospatial Service; the :class:`BoundingBox` type is reused verbatim.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.geospatial.schemas import BoundingBox

TemporalMode = Literal["single", "compare", "timeseries"]
Modality = Literal["sentinel-2-optical", "sentinel-1-sar"]
QueryTask = Literal["visualize", "change_detection", "object_identification"]


class TimeRange(BaseModel):
    """A closed date interval."""

    start_date: date
    end_date: date

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        return self


class TemporalComparison(BaseModel):
    """A baseline window compared against a target window."""

    baseline: TimeRange
    target: TimeRange


class SatQueryIntent(BaseModel):
    """What the user asks for, before any location grounding."""

    location_query: str = Field(min_length=1, max_length=300)
    temporal_mode: TemporalMode
    time_windows: TemporalComparison | list[TimeRange]
    modalities: list[Modality] = Field(min_length=1)
    task: QueryTask

    @field_validator("location_query", mode="before")
    @classmethod
    def _strip_location(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("modalities")
    @classmethod
    def _no_duplicate_modalities(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("modalities must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _check_temporal_structure(self) -> Self:
        windows = self.time_windows
        is_comparison = isinstance(windows, TemporalComparison)
        count = 0 if is_comparison else len(windows)

        if self.temporal_mode == "compare" and not is_comparison:
            raise ValueError(
                "temporal_mode 'compare' requires a baseline/target comparison"
            )
        if self.temporal_mode == "single" and (is_comparison or count != 1):
            raise ValueError(
                "temporal_mode 'single' requires exactly one time range"
            )
        if self.temporal_mode == "timeseries" and (is_comparison or count < 2):
            raise ValueError(
                "temporal_mode 'timeseries' requires at least two time ranges"
            )
        return self


class ResolvedQueryPlan(BaseModel):
    """An intent whose location has been grounded to a bounding box."""

    intent: SatQueryIntent
    bbox: BoundingBox
