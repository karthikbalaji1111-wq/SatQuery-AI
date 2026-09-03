"""Analysis boundary contracts.

This layer interprets an *already-computed* :class:`QueryExecutionResult`. It
introduces no geometry, scene, imagery, plan, or intent shape of its own -
``QueryExecutionResult`` (and everything nested inside it) is reused verbatim.

The analysis task is **derived** from ``execution.plan.intent.task`` rather than
being passed separately, so there is exactly one source of truth for it.

Deliberate status convention: a task this phase does not implement is reported
as ``status="not_implemented"`` inside a normal HTTP 200 body, *not* as an
:class:`app.core.errors.NotImplementedFeatureError` (HTTP 501). The analysis did
run - it inspected the execution result and produced traceability
(``windows_considered``) and ``warnings`` - so returning an error body would
discard information the caller needs. Genuine failures (malformed request)
still surface through the existing error handlers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.services.query.compatibility import CompatibilityReport
from app.services.query.schemas import (
    Modality,
    QueryExecutionResult,
    QueryTask,
    TimeRange,
)

#: ``ok``              - the requested analysis ran and produced an answer.
#: ``not_implemented`` - the task is recognised but no engine exists for it yet.
#: Only values this phase can actually produce are listed; more are added when
#: an engine can emit them.
AnalysisStatus = Literal["ok", "not_implemented"]


class Measurement(BaseModel):
    """A single quantitative result produced by a deterministic engine.

    ``unit`` is free-form (e.g. ``"km^2"``, ``"count"``, ``"%"``) - the repo has
    no unit vocabulary and this phase does not invent one. No measurement is
    produced in this phase; the model exists so the contract is stable.
    """

    name: str
    value: float
    unit: str


class AnalysisWindowRef(BaseModel):
    """Slim, traceable reference to one executed (modality, window) pair.

    Deliberately *not* an :class:`ExecutedWindow`: the discovered ``scenes``
    list and the bounded ``imagery`` payload are never echoed back.
    """

    modality: Modality
    label: str
    time_range: TimeRange
    selected_scene_id: str | None


class ObservationIndexResult(BaseModel):
    """Index statistics computed for ONE observation, on its own.

    Each observation is read and indexed independently, at its own native
    resolution, over its own pixels. Two of these placed side by side are two
    separate summaries - they are not a spatial comparison and nothing here is
    resampled onto a shared grid.

    ``cloud_cover`` is carried straight from ``Scene`` as context: the index is
    NOT cloud-masked, so a reader needs it to judge the statistics.
    """

    window_label: str
    scene_id: str
    #: ``Observation.acquired_at`` - the real acquisition time, not the
    #: requested window. ``None`` when absent or unparseable.
    acquired_at: datetime | None
    cloud_cover: float | None
    measurements: list[Measurement] = Field(default_factory=list)
    #: Evidence from the ACTUAL quantitative read, not from STAC metadata. The
    #: Phase 13 compatibility report can only see metadata and will often say
    #: ``"unknown"`` for CRS and resolution; these fields say what the read
    #: really used. The two are different evidence sources, not a contradiction.
    crs: str | None = None
    resolution: float | None = None
    #: Pixels in the AOI window actually read for this observation (width x
    #: height after clamping to the scene). Together with
    #: ``ndwi_valid_pixel_count`` this is what makes AOI coverage inspectable:
    #: a scene footprint says nothing about how much of the AOI carried data.
    window_pixel_count: int | None = None
    #: Affine coefficients ``[a, b, c, d, e, f]`` of the band window this
    #: observation was indexed over, in ``crs``, carried verbatim from the
    #: raster read. Per observation and per grid: two observations are NOT
    #: co-registered and may sit on different transforms, so this is never
    #: shared between them, and it is never derived from the requested bbox.
    transform: list[float] | None = Field(
        default=None, min_length=6, max_length=6
    )


class NdwiOverlay(BaseModel):
    """The NDWI index rendered as a georeferenced picture.

    A *visualisation of the index*, not a classification: the colour ramp maps
    the NDWI value and nothing else, and a bright pixel means a high index, not
    detected water. The statistics in ``measurements`` remain the product; this
    is a way of looking at the same pixels they were computed from.

    Positioned exactly like Phase 16.2 RGB imagery - by ``transform`` and
    ``corners_wgs84``, both taken from the band window the index was computed
    on. The requested bbox is never involved, so the overlay cannot claim
    ground the analysis did not measure. When an honest footprint cannot be
    derived, the whole overlay is omitted rather than drawn in the wrong place.

    Pixels with no valid measurement are fully transparent, so the basemap
    stays readable where the index says nothing.
    """

    scene_id: str
    window_label: str
    media_type: Literal["image/png"] = "image/png"
    image_base64: str
    width: int
    height: int
    crs: str
    #: Affine of THIS raster, in ``crs`` - ``[a, b, c, d, e, f]``.
    transform: list[float] = Field(min_length=6, max_length=6)
    #: Four ``[lon, lat]`` pairs in EPSG:4326, ordered ``[NW, NE, SE, SW]``.
    corners_wgs84: list[list[float]] = Field(min_length=4, max_length=4)
    #: The index range actually rendered, so a legend can be honest about it.
    value_min: float
    value_max: float
    valid_pixel_count: int


class TemporalIndexComparison(BaseModel):
    """Two independently indexed observations, reported side by side.

    ``differences`` holds at most one measurement - the difference between the
    two aggregate means. It is a difference of *statistics*, computed from two
    separate sets of pixels; no pixel was ever compared against another pixel,
    and the value is suppressed entirely when that framing would mislead (no
    overlap, no valid pixels, or the same scene on both sides).

    ``compatibility`` is the Phase 13 report for this pair, carried verbatim so
    its ``limitations`` travel with the numbers rather than beside them.
    ``warnings`` are the comparison's own; orchestration warnings (no pair
    formed, a band read failed, further pairs not analysed) stay on
    :attr:`AnalysisResult.warnings`.
    """

    first: ObservationIndexResult
    second: ObservationIndexResult
    compatibility: CompatibilityReport
    differences: list[Measurement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    """Input to the analysis boundary.

    ``execution`` is the contract ``/query/execute`` produces, reused verbatim
    and re-validated. There is no separate ``task`` field - see the module
    docstring.

    ``include_ndwi`` opts in to single-scene Sentinel-2 NDWI statistics. It is a
    flag rather than a new :data:`QueryTask` value deliberately: a spectral
    index is a descriptive add-on, not one of the three user intents, and a new
    task value would have to propagate into ``SatQueryIntent``, the intent
    parser's instructions and the frontend task list. Defaulting to ``False``
    keeps every existing request byte-identical in behaviour.

    ``include_temporal_ndwi`` opts in to Temporal NDWI Statistics: ONE
    deterministic same-modality Sentinel-2 pair, each observation indexed
    independently. It is separate from ``include_ndwi`` (single-scene) and the
    two may be requested together. Defaulting to ``False`` means no temporal
    band read happens unless it is asked for.
    """

    execution: QueryExecutionResult
    include_ndwi: bool = False
    #: Additionally render the NDWI grid as a georeferenced PNG overlay.
    #: Requires ``include_ndwi``: the picture is a view of those same
    #: pixels, so it is never produced without the statistics.
    include_ndwi_overlay: bool = False
    include_temporal_ndwi: bool = False


class AnalysisResult(BaseModel):
    """Structured, deterministic interpretation of a query execution."""

    status: AnalysisStatus
    task: QueryTask
    answer: str
    windows_considered: list[AnalysisWindowRef]
    warnings: list[str] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    #: Temporal NDWI Statistics for one observation pair. ``None`` whenever the
    #: feature was not requested, or was requested but could not produce a valid
    #: comparison - the reason is then on :attr:`warnings`.
    #: The NDWI grid as a georeferenced picture, when it was requested and
    #: could be positioned honestly. ``None`` otherwise - a missing overlay
    #: never suppresses the measurements, which remain the product.
    ndwi_overlay: NdwiOverlay | None = None
    temporal_comparison: TemporalIndexComparison | None = None
