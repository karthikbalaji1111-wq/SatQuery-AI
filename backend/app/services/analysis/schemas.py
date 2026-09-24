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
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from app.services.query.compatibility import CompatibilityReport
from app.services.query.integrity import validate_execution_integrity
from app.services.query.schemas import (
    Modality,
    NdwiComparison,
    QueryExecutionResult,
    QueryTask,
    TimeRange,
)
from app.services.satellite.radiometry import RadiometricState

#: ``ok``              - the requested analysis ran and produced an answer.
#: ``not_implemented`` - the task is recognised but no engine exists for it yet.
#: Only values this phase can actually produce are listed; more are added when
#: an engine can emit them.
AnalysisStatus = Literal["ok", "not_implemented"]

#: What became of ONE requested analysis.
#:
#: ``completed``   - it ran and produced measurements.
#: ``unavailable`` - it was asked for and could not be produced; ``reason`` says
#:                   why, in the service's own words.
AnalysisOutcomeStatus = Literal["completed", "unavailable"]


class AnalysisOutcome(BaseModel):
    """Whether one requested analysis actually happened.

    **The defect this closes.** ``status`` answered a different question from
    the one readers were asking it. It is derived from the TASK - ``visualize``
    is implemented, so ``status`` is ``"ok"`` - and says nothing about whether
    the analysis the request asked for was produced. An NDWI request over an
    execution with no optical window therefore returned ``status: "ok"`` with an
    empty ``measurements`` list and a warning, and every consumer that keyed on
    the status read that as a successful analysis that happened to find nothing.

    Three different questions were collapsed into one field:

        transport success   - the HTTP request succeeded
        execution success   - the windows were retrieved (QueryExecutionResult)
        analysis completeness - the requested analysis was produced (HERE)

    ``status`` keeps its exact previous meaning and values, so existing clients
    are unaffected; this is the field that answers the third question.
    """

    #: The analysis asked for: an index key ("ndvi"/"ndwi"/"ndbi"),
    #: "temporal_ndwi", or "sar_backscatter".
    name: str
    status: AnalysisOutcomeStatus
    #: Present only when ``unavailable``. Carried verbatim from the warning the
    #: service produced, so the reason a reader sees is the reason the service
    #: gave - never a second, prettier explanation invented here.
    reason: str | None = None


class Measurement(BaseModel):
    """A single quantitative result produced by a deterministic engine.

    ``unit`` is free-form (e.g. ``"km^2"``, ``"count"``, ``"%"``) - the repo has
    no unit vocabulary and this phase does not invent one. No measurement is
    produced in this phase; the model exists so the contract is stable.
    """

    name: str
    value: float
    unit: str


QualityCategory = Literal[
    "nodata",
    "saturated_or_defective",
    "cloud",
    "cloud_shadow",
    "snow",
    "unknown_class",
    "other_masked",
]


class PixelQuality(BaseModel):
    """Which pixels of ONE optical index grid were usable, and why the rest were not.

    Stage 3 of the scientific pipeline. A catalog-valid scene (Stage 2) can
    still be clouded, shadowed or snow-covered pixel by pixel; this record
    MEASURES that, from the Sentinel-2 Scene Classification Layer placed on the
    final analysis grid, before any statistic is computed. It does not judge
    whether the result is acceptable - no threshold is applied here.

    Every pixel of the grid is counted in exactly one category, by the fixed
    precedence in :data:`QUALITY_PRECEDENCE` (``analysis.pixel_quality``):

        valid + nodata + saturated_or_defective + cloud + cloud_shadow
              + snow + unknown_class + other_masked == total

    ``contamination_fraction`` is ``masked / total`` - it includes nodata,
    because a pixel with no data is as absent from the statistic as a clouded
    one. Both fractions are ``None`` when the grid has no pixels: no
    denominator, no fraction. None of this is an uncertainty estimate.
    """

    model_config = ConfigDict(extra="forbid")

    #: The index this grid was assessed for ("ndvi", "ndwi", "ndbi"): each
    #: index reads its own bands, so its band-nodata can differ.
    index: str
    scene_id: str
    window_label: str
    mask_source: Literal["sentinel-2-scl"] = "sentinel-2-scl"
    #: SCL classes counted as usable surface observations.
    usable_scl_classes: list[int]
    #: ``known`` when the SCL asset's encoding was confirmed from the catalog
    #: item (Stage 2); ``unknown`` when the item published no raster metadata
    #: for it; ``not_validated`` when no catalog validation ran.
    scl_metadata_status: Literal["known", "unknown", "not_validated"]
    grid_width: int
    grid_height: int
    grid_crs: str | None = None
    grid_resolution: float | None = None
    total_pixels: int
    valid_pixels: int
    masked_pixels: int
    nodata_pixels: int
    saturated_or_defective_pixels: int
    cloud_pixels: int
    cloud_shadow_pixels: int
    snow_pixels: int
    unknown_class_pixels: int
    other_masked_pixels: int
    valid_fraction: float | None = None
    contamination_fraction: float | None = None
    #: SCL value -> pixels on the analysis grid carrying it (where the SCL has a
    #: value). String keys, so the record survives JSON unchanged.
    scl_class_counts: dict[str, int] = Field(default_factory=dict)
    #: SCL values present that are not documented classes. Always excluded.
    unknown_scl_values: list[int] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _accounts_for_every_pixel(self) -> Self:
        categories = (
            self.nodata_pixels + self.saturated_or_defective_pixels
            + self.cloud_pixels + self.cloud_shadow_pixels + self.snow_pixels
            + self.unknown_class_pixels + self.other_masked_pixels
        )
        if categories != self.masked_pixels:
            raise ValueError("masked_pixels must equal the sum of its categories")
        if self.valid_pixels + self.masked_pixels != self.total_pixels:
            raise ValueError("valid + masked must equal total")
        if self.total_pixels != self.grid_width * self.grid_height:
            raise ValueError("total_pixels must equal the grid's pixel count")
        return self


class GridInput(BaseModel):
    """One raster that took part in a computation, and how it met the grid."""

    model_config = ConfigDict(extra="forbid")

    role: str
    crs: str | None = None
    width: int
    height: int
    resolution_x: float
    resolution_y: float
    #: ``identical`` - the analysis grid itself or pixel-for-pixel equal to it;
    #: ``nested`` - a coarser grid whose cells cover whole blocks of it.
    relationship: Literal["analysis_grid", "identical", "nested"]
    ratio_x: int | None = None
    ratio_y: int | None = None
    #: Coarse-origin offset from the analysis origin, in analysis pixels.
    offset_x: int | None = None
    offset_y: int | None = None


class GridState(BaseModel):
    """The grid a computation ran on - or the reason it was refused one.

    Stage 5 of the scientific pipeline (``analysis.geometry``). ``stage`` says
    whether the decision was made from the catalog's published source grids
    (``pre_read``, before any pixel was read) or from the windows actually read
    (``post_read``). No confidence is expressed: the rules are exact up to a
    float-representation tolerance of ``geometry.TOLERANCE_PIXELS``.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["valid", "refused"]
    analysis: str
    scene_id: str | None = None
    stage: Literal["pre_read", "post_read"]
    crs: str | None = None
    width: int | None = None
    height: int | None = None
    resolution_x: float | None = None
    resolution_y: float | None = None
    transform: list[float] | None = None
    origin: list[float] | None = None
    #: ``[min_x, min_y, max_x, max_y]`` in ``crs``.
    bounds: list[float] | None = None
    grid_id: str | None = None
    inputs: list[GridInput] = Field(default_factory=list)
    refusal: str | None = None
    notes: list[str] = Field(default_factory=list)


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

    ``cloud_cover`` is carried straight from ``Scene`` as scene-level context.
    The index itself is computed only over pixels the Scene Classification
    Layer marks as clear surface; ``pixel_quality`` says how many that was.
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
    #: This observation's own pixel quality - never shared with the other side.
    pixel_quality: PixelQuality | None = None
    #: The radiometric representation its values were validated as (Stage 4).
    radiometry: RadiometricState | None = None
    #: The grid this observation was indexed on (Stage 5).
    grid: GridState | None = None


class SpatialMeasurement(BaseModel):
    """A threshold count over the analysed NDWI pixels, with its provenance.

    Deterministic evidence: ``matching_pixel_count`` and ``valid_pixel_count``
    are counted from the raster the statistics were computed on, and
    ``percentage`` is the ratio between them. No language model computes any of
    these; a model may only have chosen the threshold that was asked about.

    The denominator is VALID pixels, never the raster's pixel count. A nodata,
    non-finite or zero-denominator pixel was never measured, so it can neither
    match nor count against a match - including it would quietly understate
    every percentage over a partially covered window.

    Comparisons are exact. ``gt`` is ``>``; a pixel whose index equals the
    threshold matches ``gte`` and ``lte`` and nothing else.

    Deliberately carries no pixels. The picture lives in ``NdwiOverlay``; this
    is the number and where it came from.
    """

    #: The index this counts over. Only ``"ndwi"`` exists in this phase.
    metric: Literal["ndwi"] = "ndwi"
    operator: NdwiComparison
    threshold: float
    matching_pixel_count: int = Field(ge=0)
    valid_pixel_count: int = Field(gt=0)
    #: ``matching / valid * 100``.
    percentage: float = Field(ge=0.0, le=100.0)

    scene_id: str
    #: The real acquisition time, when the scene carried a usable one.
    acquired_at: datetime | None = None
    window_label: str
    #: The grid the count was performed on. ``None`` when the read carried no
    #: usable CRS - the count is still real, so it is reported without a
    #: footprint rather than withheld.
    crs: str | None = None
    #: Four ``[lon, lat]`` pairs, ``[NW, NE, SE, SW]``, from the read's own
    #: affine - never from the requested bbox.
    corners_wgs84: list[list[float]] | None = Field(
        default=None, min_length=4, max_length=4
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


class NdwiTemporalChange(BaseModel):
    """Paired-pixel NDWI change between a baseline and a target observation.

    The first result in SatQuery derived by comparing two observations PIXEL BY
    PIXEL, and therefore the first that has to refuse to. Two scenes sharing a
    requested bbox are not thereby on the same grid: each was reprojected and
    clamped on its own. This model exists only when the two grids were verified
    identical - same size, same CRS, same affine - so every difference in it is
    between two measurements of the same ground.

    ``change = second_NDWI - first_NDWI``, where *first* is the EARLIER
    acquisition and *second* the later one. Positive therefore means the index
    rose over time, negative that it fell.

    The sides are named by acquisition order, never by the requested roles:
    ``TemporalComparison`` does not require the baseline window to precede the
    target, so an inverted request would otherwise label the earlier scene
    "target". The requested roles stay readable in ``window_label``.

    It is an INDEX change, not water gained or lost:
    nothing here classifies water, and the statistics are computed only over
    pixels valid in BOTH observations.

    Nothing is resampled or co-registered. An incompatible pair produces no
    instance of this model at all, with the reason reported as a warning, which
    is the honest outcome while co-registration remains out of scope.

    The per-observation statistics stay in ``TemporalIndexComparison.first`` and
    ``.second`` and are not copied here.
    """

    first_scene_id: str
    second_scene_id: str
    first_acquired_at: datetime | None = None
    second_acquired_at: datetime | None = None
    window_label: str

    #: Pixels valid in BOTH observations - the denominator for every statistic
    #: below. A pixel measured only once is not evidence of change.
    paired_valid_pixel_count: int = Field(gt=0)
    change_mean: float
    change_min: float
    change_max: float

    #: The shared grid the comparison was performed on.
    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)
    corners_wgs84: list[list[float]] = Field(min_length=4, max_length=4)

    #: The difference rendered as a georeferenced PNG, when it could be drawn.
    overlay: NdwiOverlay | None = None


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
    #: Paired-pixel change, present ONLY when the two grids were verified
    #: identical. ``None`` when they were not - see ``warnings`` for why.
    change: NdwiTemporalChange | None = None
    warnings: list[str] = Field(default_factory=list)
    #: Stage 5: whether the two observations share ONE grid, which the paired
    #: change requires. ``refused`` withholds only the paired change - each
    #: side's own statistics never needed a common grid.
    pair_grid: GridState | None = None


# =========================================================================== #
# Sentinel-1 RTC backscatter
#
# The first QUANTITATIVE SAR result in SatQuery. Everything below is in
# DECIBELS of provider terrain-corrected gamma naught, and the unit string is
# always "dB" - never "index", which belongs to the normalised-difference
# engines and means something else entirely.
#
# The measurement names carry the polarization ("vv_mean_db", "vh_mean_db")
# rather than a shared name plus an attribute. That is deliberate: a claim is
# bound to evidence by metric identity, so distinct names are what makes
# "mean VV is -8 dB" fail structurally when only VH evidence supports -8 dB.
# =========================================================================== #

SarPolarization = Literal["vv", "vh"]


class SarPolarizationStatistics(BaseModel):
    """Gamma-naught statistics for ONE polarization, on its own pixels.

    ``measurements`` carry the decibel values; the pixel-count fields carry the
    provenance that makes them readable. ``valid_pixel_count`` counts the
    samples the statistics were actually computed from - finite, not nodata and
    strictly positive - out of ``window_pixel_count``, the pixels the AOI window
    covers. A sample the logarithm is undefined on is EXCLUDED and counted in
    ``nonpositive_pixel_count``; it is never clamped to a floor or replaced.
    """

    polarization: SarPolarization
    #: ``<pol>_valid_pixel_count`` plus, when that count is non-zero,
    #: ``<pol>_mean_db``, ``<pol>_min_db`` and ``<pol>_max_db``.
    measurements: list[Measurement] = Field(default_factory=list)
    valid_pixel_count: int = Field(ge=0)
    nonpositive_pixel_count: int = Field(ge=0)
    window_pixel_count: int = Field(ge=0)
    #: Evidence from the read itself, exactly as for an optical observation.
    crs: str | None = None
    resolution: float | None = None
    transform: list[float] | None = Field(default=None, min_length=6, max_length=6)


class SarPolarizationDifference(BaseModel):
    """VV minus VH, in decibels, over the pixels valid in BOTH polarizations.

    Both polarizations come from one acquisition and one grid - a requirement
    verified before this model is built, exactly as the temporal path verifies
    two observations - so this is a co-polarized/cross-polarized ratio over the
    same ground, expressed as a difference of decibels.

    It is a RATIO OF BACKSCATTER, not a classification. A large VV-VH is not
    "bare soil" and a small one is not "vegetation": no threshold is applied
    here and no land-cover label is derived from it.

    The two means are recomputed over the paired pixel set, so the difference
    is exactly ``vv_mean_db - vh_mean_db`` for the numbers reported in this
    model. They equal the per-polarization means whenever both polarizations
    were valid on the same pixels, which is the normal case.
    """

    vv_mean_db: float
    vh_mean_db: float
    vv_minus_vh_mean_db: float
    paired_valid_pixel_count: int = Field(gt=0)
    crs: str | None = None
    transform: list[float] | None = Field(default=None, min_length=6, max_length=6)


class SarBackscatterResult(BaseModel):
    """Sentinel-1 RTC gamma-naught statistics for ONE scene, in decibels.

    ``measurements`` is the flat list of every value in this result, so a
    caller citing a number never has to reach into the nested structure. Names
    are unique across polarizations, so nothing is overwritten.

    ``warnings`` carry this result's own qualifications - the averaging
    convention, the absence of any quality mask, excluded samples. Orchestration
    outcomes (no SAR window, a band that could not be read) stay on
    :attr:`AnalysisResult.warnings`.
    """

    scene_id: str
    window_label: str
    acquired_at: datetime | None = None
    collection: str | None = None
    polarizations: list[SarPolarizationStatistics] = Field(default_factory=list)
    #: Present only when both polarizations were read AND verified to sit on an
    #: identical grid. ``None`` otherwise, with the reason in ``warnings``.
    difference: SarPolarizationDifference | None = None
    measurements: list[Measurement] = Field(default_factory=list)
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

    ``include_sar_backscatter`` opts in to quantitative Sentinel-1 RTC
    gamma-naught statistics in decibels for one SAR window. It is independent of
    every optical flag - SAR and optical are different measurements of different
    physics - and defaults to ``False``, so no polarization is ever read unless
    it is asked for.

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
    #: Quantitative Sentinel-1 RTC backscatter (VV and VH) for one SAR window.
    include_sar_backscatter: bool = False
    #: Additional spectral indices to compute over the same optical window,
    #: by key ("ndvi", "ndwi", "ndbi").
    #:
    #: Additive and independent of ``include_ndwi``, which keeps its exact
    #: previous behaviour including the overlay and the threshold statistic.
    #: Asking for "ndwi" here as well is harmless - the measurement names are
    #: the same and the flat list is de-duplicated by name downstream - but the
    #: flag remains the path that produces an overlay.
    indices: list[str] = Field(default_factory=list)

    @field_validator("indices")
    @classmethod
    def _known_indices(cls, value: list[str]) -> list[str]:
        """Refuse an unknown index rather than silently computing a different one."""

        from app.services.analysis.indices import resolve_index

        seen: list[str] = []
        for key in value:
            resolved = resolve_index(key).key
            if resolved not in seen:
                seen.append(resolved)
        return seen

    @model_validator(mode="after")
    def _execution_is_internally_consistent(self) -> Self:
        """Structural validity is not provenance integrity.

        This is the boundary where a CLIENT supplies an execution result, so it
        is the boundary where the relations between its parts are checked -
        that the selected scene is among the scenes returned, that the windows
        belong to the intent they claim to answer, that counts match their
        lists. Parsing proves none of that, and everything downstream treats
        this object as established fact.
        """

        validate_execution_integrity(self.execution)
        return self


class AnalysisResult(BaseModel):
    """Structured, deterministic interpretation of a query execution."""

    status: AnalysisStatus
    task: QueryTask
    answer: str
    windows_considered: list[AnalysisWindowRef]
    warnings: list[str] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    #: One entry per analysis this request asked for. Empty when it asked for
    #: none, which is a different thing from asking and getting nothing.
    analysis_outcomes: list[AnalysisOutcome] = Field(default_factory=list)
    #: Temporal NDWI Statistics for one observation pair. ``None`` whenever the
    #: feature was not requested, or was requested but could not produce a valid
    #: comparison - the reason is then on :attr:`warnings`.
    #: The NDWI grid as a georeferenced picture, when it was requested and
    #: could be positioned honestly. ``None`` otherwise - a missing overlay
    #: never suppresses the measurements, which remain the product.
    ndwi_overlay: NdwiOverlay | None = None
    #: A deterministic threshold count over the analysed NDWI pixels, when the
    #: intent stated a threshold and there were valid pixels to count. ``None``
    #: otherwise - never a fabricated 0%.
    spatial_measurement: SpatialMeasurement | None = None
    temporal_comparison: TemporalIndexComparison | None = None
    #: Sentinel-1 RTC backscatter statistics in decibels, when requested and a
    #: SAR window with a selected scene was available. ``None`` otherwise - the
    #: reason is then on :attr:`warnings`.
    sar_backscatter: SarBackscatterResult | None = None
    #: Pixel quality for each single-scene optical index grid that was
    #: assessed, in the order computed. Temporal observations carry their own on
    #: :attr:`TemporalIndexComparison.first` / ``second``.
    pixel_quality: list[PixelQuality] = Field(default_factory=list)
    #: Stage 4: the radiometric state of every single-scene analysis assessed,
    #: REFUSED ones included, so a missing number can be traced to its reason.
    #: Temporal observations carry theirs on the comparison.
    radiometry: list[RadiometricState] = Field(default_factory=list)
    #: Stage 5: the grid of every single-scene analysis attempted, refused ones
    #: included. Temporal observations carry theirs on the comparison.
    grids: list[GridState] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def completeness(self) -> Literal["complete", "partial", "none", "not_requested"]:
        """Whether the analyses this request asked for were produced.

        Derived from :attr:`analysis_outcomes`, never stored, so it cannot
        disagree with them. ``not_requested`` is deliberately distinct from
        ``none``: asking for nothing and getting nothing is a complete answer to
        the question that was asked, while asking for NDWI and getting nothing
        is not.
        """

        if not self.analysis_outcomes:
            return "not_requested"
        produced = sum(
            1 for outcome in self.analysis_outcomes if outcome.status == "completed"
        )
        if produced == len(self.analysis_outcomes):
            return "complete"
        return "none" if produced == 0 else "partial"
