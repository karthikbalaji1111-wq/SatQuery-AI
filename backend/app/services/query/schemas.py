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
from app.services.satellite.schemas import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    ImageryResponse,
    Scene,
)

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


# --------------------------------------------------------------------------- #
# Query execution orchestration
#
# These models describe the *composition* of the existing grounding, discovery
# and bounded-imagery contracts. No new geometry, scene, or imagery shape is
# introduced - ``ResolvedQueryPlan``, ``Scene`` and ``ImageryResponse`` are
# reused verbatim.
# --------------------------------------------------------------------------- #


class SkippedModality(BaseModel):
    """A requested modality that this phase deliberately does not execute."""

    modality: Modality
    reason: str


class QueryExecutionRequest(BaseModel):
    """Input to end-to-end query execution.

    ``intent`` is the same contract that ``/query/parse`` produces and
    ``/query/build-plan`` consumes. ``max_cloud_cover`` and ``limit`` are
    optional pass-throughs to the existing Sentinel-2 discovery contract; their
    bounds mirror :class:`SceneSearchRequest`.
    """

    intent: SatQueryIntent
    include_imagery: bool = False
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)


class ExecutedWindow(BaseModel):
    """Discovery and selection outcome for one (modality, temporal window) pair."""

    modality: Modality
    label: str
    time_range: TimeRange
    scene_count: int
    scenes: list[Scene]
    selected_scene_id: str | None
    imagery: ImageryResponse | None = None
    imagery_error: str | None = None


class QueryExecutionResult(BaseModel):
    """Structured, deterministic result of executing a :class:`SatQueryIntent`."""

    plan: ResolvedQueryPlan
    executed_modalities: list[Modality]
    skipped_modalities: list[SkippedModality]
    windows: list[ExecutedWindow]
    catalog: str
