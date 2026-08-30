"""Structured query-intent contracts.

This layer is deliberately deterministic: no NLP, no LLM. It only defines the
shape of "what the user asks for" (:class:`SatQueryIntent`) and the shape of a
grounded plan (:class:`ResolvedQueryPlan`). Location grounding is delegated to
the existing Geospatial Service; the :class:`BoundingBox` type is reused verbatim.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

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


# --------------------------------------------------------------------------- #
# Temporal observation model
#
# A temporal *window* and an *observation* are deliberately different things:
#
#   TimeRange / ExecutedWindow.time_range -> what the USER ASKED FOR. A request.
#   Observation                           -> what was ACTUALLY ACQUIRED. Data.
#
# One requested window yields zero or one observation per modality: zero when
# discovery found nothing to select, one when a scene was selected. The two must
# never be conflated - a window with no observation is a legitimate outcome.
#
# Nothing here assumes observations are comparable. They may differ in CRS,
# native resolution, pixel grid, acquisition time and sensor, and they are NOT
# co-registered. The model exists to carry enough metadata for a later phase to
# establish alignment explicitly; it performs no alignment, no resampling and no
# comparison of its own.
# --------------------------------------------------------------------------- #


class Observation(BaseModel):
    """One actual satellite acquisition selected for one requested window.

    The acquired scene is embedded verbatim rather than copied field by field,
    so ``Scene`` stays the single canonical description of a scene (id,
    acquisition datetime, collection, footprint, geometry, platform, cloud
    cover, processing level, assets).
    """

    modality: Modality
    #: Label of the requested window this observation answers ("single",
    #: "baseline", "target", "series[0]", ...).
    window_label: str
    #: The window that was REQUESTED. The actual acquisition time is
    #: ``scene.datetime`` / :attr:`acquired_at` and will differ.
    requested_window: TimeRange
    #: The acquisition itself, exactly as discovery normalised it.
    scene: Scene
    #: Bounded imagery for this observation, when it was retrieved. Display
    #: rendering only - never raw raster arrays.
    imagery: ImageryResponse | None = None

    @property
    def scene_id(self) -> str:
        return self.scene.id

    @property
    def collection(self) -> str | None:
        """STAC collection, straight from the acquired scene."""

        return self.scene.collection

    @property
    def acquired_at(self) -> datetime | None:
        """``scene.datetime`` parsed, for ordering. ``None`` if absent/unparseable.

        A typed accessor over the canonical string - the string remains the
        stored representation, this is not a second copy of it.
        """

        raw = self.scene.datetime
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None


class ObservationSet(BaseModel):
    """The observations produced by one execution, over one requested AOI.

    ``requested_bbox`` is the AOI that was *asked for*. It is emphatically not a
    claim that every observation covers exactly that extent, shares a grid, or
    is co-registered - see the module note above.
    """

    requested_bbox: BoundingBox
    observations: list[Observation] = Field(default_factory=list)

    @classmethod
    def from_windows(
        cls, requested_bbox: BoundingBox, windows: list[ExecutedWindow]
    ) -> ObservationSet:
        """Derive observations from executed windows.

        A window contributes an observation only when it actually selected a
        scene and that scene is present in its discovery results; windows that
        found nothing contribute nothing. Window order is preserved.
        """

        observations: list[Observation] = []
        for window in windows:
            if window.selected_scene_id is None:
                continue
            scene = next(
                (s for s in window.scenes if s.id == window.selected_scene_id), None
            )
            if scene is None:
                continue
            observations.append(
                Observation(
                    modality=window.modality,
                    window_label=window.label,
                    requested_window=window.time_range,
                    scene=scene,
                    imagery=window.imagery,
                )
            )
        return cls(requested_bbox=requested_bbox, observations=observations)

    def for_modality(self, modality: Modality) -> list[Observation]:
        return [o for o in self.observations if o.modality == modality]

    def for_window_label(self, label: str) -> list[Observation]:
        """All observations answering one requested window, across modalities."""

        return [o for o in self.observations if o.window_label == label]

    def ordered_by_acquisition(self) -> list[Observation]:
        """Observations sorted by actual acquisition time; unknown times last.

        Ordering only - it implies nothing about comparability.
        """

        return sorted(
            self.observations,
            key=lambda o: (o.acquired_at is None, o.acquired_at or datetime.min),
        )


class QueryExecutionResult(BaseModel):
    """Structured, deterministic result of executing a :class:`SatQueryIntent`."""

    plan: ResolvedQueryPlan
    executed_modalities: list[Modality]
    skipped_modalities: list[SkippedModality]
    windows: list[ExecutedWindow]
    catalog: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observations(self) -> ObservationSet:
        """The actual acquisitions behind :attr:`windows`.

        Derived rather than stored, so it can never drift from ``windows``.
        Existing callers construct this model exactly as before; the field is
        additive on the wire and is recomputed on input rather than trusted.
        """

        return ObservationSet.from_windows(self.plan.bbox, self.windows)
