"""Structured query-intent contracts.

This layer is deliberately deterministic: no NLP, no LLM. It only defines the
shape of "what the user asks for" (:class:`SatQueryIntent`) and the shape of a
grounded plan (:class:`ResolvedQueryPlan`). Location grounding is delegated to
the existing Geospatial Service; the :class:`BoundingBox` type is reused verbatim.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from app.services.geospatial.schemas import BoundingBox
from app.services.satellite.schemas import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    SAR_IMAGERY_ASSET,
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


#: How a threshold compares an index value. Exact by construction: ``gt`` is
#: ``>``, never ``>=`` - a pixel exactly on the threshold is decided by the
#: operator the user asked for and by nothing else.
NdwiComparison = Literal["gt", "gte", "lt", "lte"]


class NdwiThreshold(BaseModel):
    """A numeric NDWI threshold the user stated explicitly.

    Carries the *question*, never the answer: the operator and the value come
    from the request, and the counting is done later from real pixels. A model
    may fill this in; a model never computes what it selects.

    ``value`` is bounded to NDWI's own range, and the bound rejects NaN and
    infinity, so an unusable threshold fails at the contract rather than
    silently matching nothing.
    """

    model_config = ConfigDict(extra="forbid")

    operator: NdwiComparison
    value: float = Field(ge=-1.0, le=1.0)


#: Upper bound on the number of temporal windows one query may span.
#:
#: This is a RESOURCE bound, not a scientific one. Execution runs one catalog
#: search per (modality x window) sequentially, and with imagery each of those
#: additionally performs a windowed COG read and a PNG encode whose base64 is
#: held in the response. The endpoint is unauthenticated, so an unbounded list
#: turns one request into an unbounded number of outbound requests against a
#: third-party catalog, and into unbounded memory here. Twenty-four windows
#: covers two years of monthly observations - well beyond anything the agent
#: plans - while keeping the worst case finite.
MAX_TIME_WINDOWS = 24


#: The earliest date any supported mission could have observed anything.
#: Sentinel-1A began routine acquisition in October 2014 and Sentinel-2A in
#: June 2015; this is the earlier of the two, so it bounds BOTH modalities
#: without preferring either. A window ending before this cannot be answered by
#: any catalog - not "no scenes matched", but "nothing was in orbit yet" - and
#: the two deserve different messages.
EARLIEST_OBSERVATION_DATE = date(2014, 10, 3)

#: How far past today a window may still START. A window that begins tomorrow
#: describes an observation that has not happened, and no catalog can return it.
#: One day of slack absorbs the difference between the caller's clock, its time
#: zone and UTC, so a legitimate "today" is never refused for being a few hours
#: ahead.
FUTURE_START_GRACE_DAYS = 1


def observation_period_problem(window: TimeRange, *, today: date) -> str | None:
    """Why no observation can exist in ``window``, or ``None`` when one can.

    The single statement of which periods are answerable at all. Both the
    intent contract and the analysis request gate apply it, so the two
    boundaries cannot drift into disagreeing about the same dates - which is
    what two copies of these comparisons would eventually do.
    """

    if window.end_date < EARLIEST_OBSERVATION_DATE:
        return (
            f"the window ending {window.end_date.isoformat()} precedes "
            f"the first available observation "
            f"({EARLIEST_OBSERVATION_DATE.isoformat()}); no satellite "
            "in this catalog was acquiring data yet"
        )
    if (window.start_date - today).days > FUTURE_START_GRACE_DAYS:
        return (
            f"the window starting {window.start_date.isoformat()} is in "
            "the future; only past observations can be retrieved"
        )
    return None


class SatQueryIntent(BaseModel):
    """What the user asks for, before any location grounding.

    ``extra="forbid"`` because a PLANNER fills this model. The agent contracts
    are closed for exactly this reason, and the intent was the one link in that
    chain that was not: a plan carrying ``bbox``, ``lat``, ``lon`` or ``center``
    was accepted and the field silently dropped. Nothing downstream read it - the
    location is resolved from ``location_query`` alone - so no substitution ever
    occurred; but "accepted and ignored" is the wrong answer to a model trying to
    supply its own coordinates. Refusing is how the closed contract is stated
    everywhere else, and it makes the intent say what it means: the geocoder
    grounds the place, the planner does not get a vote.
    """

    model_config = ConfigDict(extra="forbid")

    location_query: str = Field(min_length=1, max_length=300)
    temporal_mode: TemporalMode
    time_windows: TemporalComparison | list[TimeRange] = Field(
        union_mode="left_to_right",
    )
    modalities: list[Modality] = Field(min_length=1)
    task: QueryTask
    #: An explicit NDWI threshold, when the request stated one ("NDWI above
    #: 0.3"). Optional and additive: an intent without one behaves exactly as
    #: before. It travels with the intent so it survives plan -> execute ->
    #: analyse without a parallel request path.
    ndwi_threshold: NdwiThreshold | None = None

    @field_validator("location_query", mode="before")
    @classmethod
    def _strip_location(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        # Control characters are refused rather than stripped. No place name
        # contains a NUL, a CR or a LF, and this field is filled by a PLANNER -
        # untrusted output. httpx already percent-encodes them, so this closes
        # no live hole; it closes the CLASS, so the safety of a place name stops
        # depending on a third-party library's escaping behaviour staying the
        # same. It matches how `_require_readable_scheme` treats the other
        # untrusted string this system puts into a URL.
        if any(ch in value for ch in "\x00\r\n\t"):
            raise ValueError(
                "location_query must not contain control characters"
            )
        return value.strip()

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
        if count > MAX_TIME_WINDOWS:
            raise ValueError(
                f"a query may span at most {MAX_TIME_WINDOWS} time windows; "
                f"{count} were requested"
            )

        # Observability bounds, enforced HERE so they are provider-independent:
        # every path that builds an intent - Gemini, NVIDIA, the manual form or
        # a direct API call - inherits exactly the same rule, and no model can
        # invent a date the archive could never contain. Without this an
        # impossible window reached the catalog and came back empty, which the
        # UI then reported as "no scenes found" - indistinguishable from a real
        # gap in coverage, and pointing the reader at the wrong problem.
        ranges = (
            [windows.baseline, windows.target] if is_comparison else list(windows)
        )
        today = datetime.now(UTC).date()
        for window in ranges:
            problem = observation_period_problem(window, today=today)
            if problem is not None:
                raise ValueError(problem)
        return self


def expand_windows(intent: SatQueryIntent) -> list[tuple[str, TimeRange]]:
    """Expand a validated intent's ``time_windows`` into labelled windows.

    ``SatQueryIntent`` already guarantees the shape, so this is total:

    - ``single``     -> ``[("single", <the one range>)]``
    - ``timeseries`` -> ``[("series[0]", ...), ("series[1]", ...), ...]``
    - ``compare``    -> ``[("baseline", ...), ("target", ...)]``

    It lives on the CONTRACT rather than in the execution service because two
    parties need it and they must not disagree: the service produces windows
    from an intent, and the integrity validator checks a client-supplied
    result's windows AGAINST that intent. Two copies of this rule would drift,
    and the validator would then reject perfectly good results - or accept
    fabricated ones.
    """

    windows = intent.time_windows
    if isinstance(windows, TemporalComparison):
        return [("baseline", windows.baseline), ("target", windows.target)]
    if intent.temporal_mode == "single":
        return [("single", windows[0])]
    return [(f"series[{index}]", window) for index, window in enumerate(windows)]


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
    #: Which Sentinel-1 polarization to render when a SAR window retrieves
    #: imagery. Ignored for optical windows, which have their own asset. VV was
    #: previously hardcoded here, leaving VH unreachable through this route even
    #: though the imagery service already supported it; defaulting to VV keeps
    #: every existing caller byte-identical.
    sar_polarization: Literal["vv", "vh"] = SAR_IMAGERY_ASSET


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
    #: Which catalog answered THIS window.
    #:
    #: A mixed request reaches two different services - Sentinel-2 from Earth
    #: Search, Sentinel-1 RTC from the Planetary Computer - and the result
    #: carried ONE top-level ``catalog`` string, assigned inside the loop, so
    #: whichever window ran last silently spoke for all of them. Provenance has
    #: to travel with the observation it describes.
    #:
    #: Optional for backward compatibility: an older stored result, or a
    #: window whose discovery failed before any catalog answered, has ``None``.
    catalog: str | None = None
    #: How many scenes the catalog reported as MATCHING this window, when it
    #: said. ``scene_count`` is how many were returned and therefore how many
    #: deterministic selection actually chose between; a larger ``scenes_matched``
    #: means the chosen scene is the best of a bounded page, not of the archive.
    scenes_matched: int | None = None
    #: Why this window produced nothing, when DISCOVERY itself failed.
    #:
    #: Distinct from ``imagery_error``, which means the scene was found and its
    #: picture could not be read. This one means the catalog could not be asked,
    #: so the window contributes no observation at all - and says so, rather
    #: than looking like a window where nothing matched.
    error: str | None = None


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


#: ``completed`` - every requested window was executed.
#: ``partial``   - at least one window was executed and at least one failed.
#: ``failed``    - no window was executed.
#:
#: This describes EXECUTION, not scientific completeness: a window that ran and
#: matched no scene is ``completed``, because discovery worked and the honest
#: answer is that the archive holds nothing there. Whether the requested
#: ANALYSIS was actually produced is a separate question, answered by
#: :class:`~app.services.analysis.schemas.AnalysisResult`.
ExecutionStatus = Literal["completed", "partial", "failed"]


class QueryExecutionResult(BaseModel):
    """Structured, deterministic result of executing a :class:`SatQueryIntent`."""

    plan: ResolvedQueryPlan
    executed_modalities: list[Modality]
    skipped_modalities: list[SkippedModality]
    windows: list[ExecutedWindow]
    #: The catalog that answered FIRST, kept for backward compatibility.
    #:
    #: It was assigned inside the execution loop, so in a mixed Sentinel-1 +
    #: Sentinel-2 run it reported whichever service happened to answer last.
    #: :attr:`ExecutedWindow.catalog` is now the authoritative per-observation
    #: provenance and :attr:`catalogs` lists every service involved; this field
    #: remains so existing clients keep working, and is deliberately the FIRST
    #: rather than the last, which at least makes it deterministic.
    catalog: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def catalogs(self) -> list[str]:
        """Every catalog that answered, in the order first seen.

        Derived from the windows, so prose or a UI can name both services in a
        mixed run instead of implying that one answered for everything.
        """

        seen: list[str] = []
        for window in self.windows:
            if window.catalog and window.catalog not in seen:
                seen.append(window.catalog)
        return seen

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> ExecutionStatus:
        """Whether every requested window actually executed.

        DERIVED, never stored, for the same reason ``observations`` is: this
        model is accepted from a client at ``/query/analyze``, and a stored
        status would be a claim the client could make about its own payload.
        Computed here, "completed" can only mean that the windows in this very
        object carry no discovery error.
        """

        if not self.windows:
            return "failed"
        failed = sum(1 for window in self.windows if window.error is not None)
        if failed == 0:
            return "completed"
        return "failed" if failed == len(self.windows) else "partial"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observations(self) -> ObservationSet:
        """The actual acquisitions behind :attr:`windows`.

        Derived rather than stored, so it can never drift from ``windows``.
        Existing callers construct this model exactly as before; the field is
        additive on the wire and is recomputed on input rather than trusted.
        """

        return ObservationSet.from_windows(self.plan.bbox, self.windows)
