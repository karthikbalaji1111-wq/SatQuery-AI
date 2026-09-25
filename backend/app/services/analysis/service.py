"""Analysis service: interpretation of an already-computed query execution.

    QueryExecutionResult -> AnalysisService.analyze() -> AnalysisResult

This service performs no discovery, no STAC calls of its own, and no LLM/VLM
inference. It is the **dispatcher**: it reads the supplied execution result,
decides what to compute, obtains any pixels it needs through
:class:`~app.services.satellite.imagery.ImageryService`, and delegates all
arithmetic to the pure functions in :mod:`app.services.analysis.engines`. It
never performs pixel arithmetic itself.

(Amends the Phase 10 docstring, which stated this service holds no collaborators
and performs no imagery/raster I/O. Since Phase 11 it holds exactly one
collaborator - ``ImageryService`` - and reads bands through it. Discovery, scene
selection and STAC access remain outside this service.)

``visualize`` is answered with a deterministic, templated summary of what the
execution actually retrieved; every other task is reported as
``not_implemented`` (see ``analysis/schemas.py`` for why that is a 200 body
rather than a 501 error). Independently of the task, ``include_ndwi`` opts in to
single-scene Sentinel-2 NDWI statistics, and ``include_temporal_ndwi`` opts in
to Temporal NDWI Statistics for one deterministic Sentinel-2 pair - two
observations indexed independently, never compared pixel by pixel.
``include_sar_backscatter`` opts in to quantitative Sentinel-1 RTC gamma-naught
statistics in decibels for one SAR window. Future engines are dispatched from
here without changing this contract.
"""

from __future__ import annotations

from datetime import datetime

from fastapi.concurrency import run_in_threadpool

from app.core.errors import AppError
from app.core.logging import get_logger
from app.services.analysis import geometry
from app.services.analysis.engines import (
    _grids_are_comparable,
    compare_ndwi_observations,
    compute_index_measurements,
    compute_ndwi_measurements,
    compute_ndwi_temporal_change,
    compute_ndwi_threshold_measurement,
    coregister_to_finer_grid,
    render_ndwi_overlay,
)
from app.services.analysis.indices import bands_for, resolve_index
from app.services.analysis.pixel_quality import (
    align_scl_to_grid,
    mask_with_scl,
    quality_measurements,
    temporal_quality_note,
)
from app.services.analysis.sar import SAR_POLARIZATIONS, compute_sar_backscatter
from app.services.analysis.schemas import (
    AnalysisOutcome,
    AnalysisRequest,
    AnalysisResult,
    AnalysisWindowRef,
    GridState,
    Measurement,
    NdwiOverlay,
    ObservationIndexResult,
    PixelQuality,
    SarBackscatterResult,
    SpatialMeasurement,
    TemporalIndexComparison,
)
from app.services.analysis.validation import (
    DeclaresReadLimits,
    check_operations_area,
    operations_for_flags,
    requested_operations,
    validate_execution_analysis,
)
from app.services.base import DomainService
from app.services.geospatial.schemas import BoundingBox
from app.services.query.compatibility import compute_compatibility, pair_observations
from app.services.query.schemas import (
    ExecutedWindow,
    Modality,
    Observation,
    ObservationSet,
    QueryExecutionResult,
    QueryTask,
)
from app.services.satellite import ImageryService
from app.services.satellite.imagery import QuantitativeReadLimits
from app.services.satellite.radiometry import (
    RadiometricState,
    RadiometricValidationError,
    assess_radiometry,
    radiometric_pair_problem,
    require_usable,
)
from app.services.satellite.raster import BandWindow
from app.services.satellite.scene_validation import (
    SceneValidator,
    ValidatedScene,
    validate_scene_pair,
)

logger = get_logger("analysis")

_IMPLEMENTED_TASK: QueryTask = "visualize"

_OPTICAL_MODALITY: Modality = "sentinel-2-optical"
_SAR_MODALITY: Modality = "sentinel-1-sar"
#: Earth Search STAC asset keys (common names) for the two 10 m bands NDWI
#: needs: "green" is band B03, "nir" is band B08. Both come from the SAME scene,
#: so they share one grid and need no resampling or co-registration.
_NDWI_GREEN_ASSET = "green"
_NDWI_NIR_ASSET = "nir"
#: Sentinel-2's Scene Classification Layer: the pixel-quality source (Stage 3).
_SCL_ASSET = "scl"


def _window_ref(window: ExecutedWindow) -> AnalysisWindowRef:
    """Project an executed window onto its slim, traceable reference."""

    return AnalysisWindowRef(
        modality=window.modality,
        label=window.label,
        time_range=window.time_range,
        selected_scene_id=window.selected_scene_id,
    )


def _collect_warnings(execution: QueryExecutionResult) -> list[str]:
    """Surface execution defects that limit any interpretation of the result.

    Ordered by window so the output is deterministic.
    """

    warnings: list[str] = []

    if not execution.windows:
        warnings.append(
            "The execution produced no windows; there is nothing to interpret."
        )

    for window in execution.windows:
        if window.selected_scene_id is None:
            warnings.append(
                f"No scene was selected for the {window.modality} window "
                f"{window.label!r}; that window contributes no observation."
            )
        if window.imagery_error is not None:
            warnings.append(
                f"Imagery was unavailable for the {window.modality} window "
                f"{window.label!r}: {window.imagery_error}"
            )

    for skipped in execution.skipped_modalities:
        warnings.append(f"Modality {skipped.modality} was skipped: {skipped.reason}")

    return warnings


def _describe_window(ref: AnalysisWindowRef) -> str:
    scene = ref.selected_scene_id if ref.selected_scene_id is not None else "no scene"
    return (
        f"{ref.modality} {ref.label} "
        f"({ref.time_range.start_date.isoformat()} to "
        f"{ref.time_range.end_date.isoformat()}) -> {scene}"
    )


def _visualize_answer(
    execution: QueryExecutionResult, refs: list[AnalysisWindowRef]
) -> str:
    """Deterministic templated summary of what the execution retrieved.

    Derived only from the supplied execution result - no inference, no model.
    """

    location = execution.plan.intent.location_query
    if not refs:
        return f"No imagery windows were executed for {location!r}."

    parts = "; ".join(_describe_window(ref) for ref in refs)
    return (
        f"Retrieved {len(refs)} window(s) for {location!r} from "
        f"{execution.catalog}: {parts}."
    )


def _not_implemented_answer(task: QueryTask, window_count: int) -> str:
    return (
        f"The {task!r} analysis is not implemented in this phase. The query "
        f"executed and {window_count} window(s) were retrieved, but no "
        f"{task!r} was performed and no result is claimed."
    )


def _selected_acquisition(
    execution: QueryExecutionResult, window: ExecutedWindow
) -> datetime | None:
    """When the window's selected scene was actually acquired.

    The REAL acquisition time, not the requested window - the distinction
    Phase 12 exists to draw. Read through the derived observations, which
    already own that parsing, rather than re-parsing the timestamp here.
    ``None`` when the scene carries no usable timestamp, which is reported as
    unknown rather than filled in.
    """

    for observation in execution.observations.observations:
        if (
            observation.window_label == window.label
            and observation.scene.id == window.selected_scene_id
        ):
            return observation.acquired_at
    return None


def _scene_collection(window: ExecutedWindow) -> str | None:
    """STAC collection of the window's selected scene, from discovery itself.

    ``ExecutedWindow`` does not carry the collection, but each discovered
    :class:`Scene` does, so it is recovered by matching the selected id. ``None``
    lets ``ImageryService`` fall back to its configured default, which is exactly
    what ``QueryExecutionService`` passes for an optical window.
    """

    for scene in window.scenes:
        if scene.id == window.selected_scene_id:
            return scene.collection
    return None


def _outcome(name: str, *, produced: bool, reason: str | None) -> AnalysisOutcome:
    """What became of one requested analysis.

    The reason is the step's OWN warning that explains the absence, carried
    verbatim rather than rewritten here: a second explanation could disagree
    with the one the reader sees beside it, and then neither would be
    trustworthy. It is chosen by the step, never as "the first warning": a
    step's first warnings are records of checks that PASSED (radiometric state,
    AOI coverage), and a passed check is not why nothing was measured.
    """

    return AnalysisOutcome(
        name=name,
        status="completed" if produced else "unavailable",
        reason=None if produced else reason,
    )


def _quality_cause(label: str, quality: PixelQuality | None) -> str | None:
    """The pixel-quality note that explains an index with no usable pixel.

    ``_notes`` states the empty-grid or no-usable-pixel fact FIRST whenever no
    pixel is valid, so it is the first of the index's quality warnings.
    """

    if quality is None or quality.valid_pixels > 0:
        return None
    notes = _quality_warnings(label, quality)
    return notes[0] if notes else None


def _merge_outcomes(outcomes: list[AnalysisOutcome]) -> list[AnalysisOutcome]:
    """One entry per analysis, in the order first requested.

    ``include_ndwi`` and ``indices=["ndwi"]`` ask for the same measurements, so
    a request carrying both would otherwise report NDWI twice - and could
    report it as unavailable AND completed. A produced result wins: the
    measurements exist either way.
    """

    merged: dict[str, AnalysisOutcome] = {}
    for outcome in outcomes:
        existing = merged.get(outcome.name)
        if existing is None or (
            existing.status == "unavailable" and outcome.status == "completed"
        ):
            merged[outcome.name] = outcome
    return list(merged.values())


def _ndwi_candidates(execution: QueryExecutionResult) -> list[ExecutedWindow]:
    """Optical windows that actually have a scene to read."""

    return [
        window
        for window in execution.windows
        if window.modality == _OPTICAL_MODALITY and window.selected_scene_id is not None
    ]


def _sar_candidates(execution: QueryExecutionResult) -> list[ExecutedWindow]:
    """Sentinel-1 windows that actually have a scene to read."""

    return [
        window
        for window in execution.windows
        if window.modality == _SAR_MODALITY and window.selected_scene_id is not None
    ]


def _scl_metadata_status(scene: ValidatedScene | None) -> str:
    """Whether Stage 2 confirmed the SCL's encoding from its catalog item."""

    asset = scene.asset(_SCL_ASSET) if scene is not None else None
    return asset.metadata_status if asset is not None else "not_validated"


def _scl_unavailable(what: str, window: ExecutedWindow, reason: str) -> str:
    return (
        f"{what} was not computed for the {window.modality} window "
        f"{window.label!r}: the scene classification layer (SCL) is unavailable, "
        f"so no pixel can be established as clear. {reason}"
    )


def _radiometry_summary(label: str, state: RadiometricState) -> str:
    """One citable line: what representation was consumed, and on whose word."""

    scales = sorted({e.scale for e in state.encodings if e.scale is not None})
    units = sorted({e.unit for e in state.encodings if e.unit is not None})
    return (
        f"{label} radiometric state: {state.status} - {state.representation}. "
        f"Processing baseline {state.processing_baseline or 'not published'}; "
        f"reflectance offset {state.offset_state.replace('_', ' ')}; declared "
        f"scale {', '.join(map(str, scales)) or 'not declared'}; unit "
        f"{', '.join(units) or 'not declared'}; source: the catalog item."
    )


def _source_geometry(
    scene: ValidatedScene | None, scientific: tuple[str, ...], placed: tuple[str, ...]
) -> tuple[geometry.Problem, list[str]]:
    """Stage 5 before any read: the catalog's source grids for these assets."""

    if scene is None:
        return None, []
    assets = {
        key: asset
        for key in (*scientific, *placed)
        if (asset := scene.asset(key)) is not None
    }
    return geometry.source_problem(assets, None, scientific=scientific)


def _geometry_refusal(
    analysis: str,
    scene_id: str | None,
    problem: tuple[str, str],
    stage: str,
) -> tuple[GridState, str]:
    """The refused GridState and the warning text, for one geometric refusal."""

    code, reason = problem
    state = geometry.grid_state(
        analysis=analysis, scene_id=scene_id, grid=None, refusal=f"{code}: {reason}",
        stage=stage,  # type: ignore[arg-type]
    )
    return state, f"Geometric validation failed ({code}): {reason}."


def _quality_warnings(label: str, quality: PixelQuality) -> list[str]:
    return [f"{label} pixel quality: {note}" for note in quality.quality_notes]


def _coverage_warnings(scene: ValidatedScene | None, label: str) -> list[str]:
    """A partial footprint is reported, not hidden: the statistics shrink with it."""

    if scene is None or scene.aoi_coverage.status == "full":
        return []
    return [
        f"Scene {scene.scene_id} ({label!r}) covers about "
        f"{scene.aoi_coverage.fraction:.1%} of the requested area by its catalog "
        f"{scene.aoi_coverage.basis}; its statistics describe only the part it covers."
    ]


class AnalysisService(DomainService):
    """Interprets a :class:`QueryExecutionResult` into an :class:`AnalysisResult`.

    The generic :meth:`run` hook stays unimplemented; :meth:`analyze` is the
    typed entry point. ``ImageryService`` is injected with a real default, so
    zero-argument construction keeps working (the ``DomainService`` contract
    test relies on it) while tests can substitute a fake.
    """

    name = "analysis"

    def __init__(self, *, imagery_service: ImageryService | None = None) -> None:
        self._imagery = imagery_service or ImageryService()

    def _declared_read_limits(self) -> QuantitativeReadLimits | None:
        """The window the reader refuses beyond, if it declares one.

        The real :class:`ImageryService` does, and enforces the same numbers in
        ``read_band``. A reader that declares none gets no early size check -
        the rule is the reader's, and the gate only moves it earlier.
        """

        if isinstance(self._imagery, DeclaresReadLimits):
            return self._imagery.quantitative_read_limits()
        return None

    def precheck_plan(
        self,
        bbox: BoundingBox,
        *,
        indices: tuple[str, ...] | list[str] = (),
        include_ndwi: bool = False,
        include_temporal_ndwi: bool = False,
        include_sar_backscatter: bool = False,
    ) -> None:
        """The request-stage area rule, applied before discovery.

        The agent knows which analyses it will ask for before it searches the
        catalog, so an area this service would refuse is refused THEN - after
        geocoding, which is how the area is known, and before any STAC search.
        The same rule :meth:`analyze` applies, from the same reader limits.
        """

        operations = operations_for_flags(
            indices,
            include_ndwi=include_ndwi,
            include_temporal_ndwi=include_temporal_ndwi,
            include_sar_backscatter=include_sar_backscatter,
        )
        if operations:
            check_operations_area(
                bbox, operations, self._declared_read_limits(), "plan.bbox"
            )

    async def _validated_scene(
        self,
        *,
        scene_id: str,
        collection: str | None,
        modality: Modality,
        assets: tuple[str, ...],
        bbox: BoundingBox,
        require_all_assets: bool = True,
    ) -> ValidatedScene | None:
        """Stage 2: the catalog's own item for this scene, checked before any read.

        ``None`` when the reader does not validate scenes (a test double); the
        real ``ImageryService`` always does. Raises ``SceneValidationError``.
        """

        if not isinstance(self._imagery, SceneValidator):
            return None
        return await run_in_threadpool(
            self._imagery.validate_scene,
            scene_id=scene_id,
            collection=collection,
            modality=modality,
            assets=assets,
            bbox=bbox,
            require_all_assets=require_all_assets,
        )

    def describe(self) -> str:
        return (
            "Deterministic interpretation of an executed SatQuery result: "
            "status, templated answer, window traceability, warnings, "
            "opt-in single-scene Sentinel-2 NDWI statistics, opt-in "
            "Temporal NDWI Statistics for one Sentinel-2 observation pair, and "
            "opt-in Sentinel-1 RTC backscatter statistics in decibels."
        )

    async def _read_scl(
        self, *, scene_id: str, bbox: BoundingBox, collection: str | None
    ) -> BandWindow:
        return await run_in_threadpool(
            self._imagery.read_band,
            scene_id=scene_id,
            bbox=bbox,
            asset=_SCL_ASSET,
            collection=collection,
        )

    async def _index_measurements(
        self,
        execution: QueryExecutionResult,
        keys: tuple[str, ...],
        radiometry: list[RadiometricState],
        grids: list[GridState],
    ) -> tuple[list[Measurement], list[str], list[PixelQuality], dict[str, str]]:
        """Compute several spectral indices over one optical window.

        Each distinct band is read ONCE and shared: NIR appears in all three
        indices, so NDVI + NDWI + NDBI costs four reads rather than six. The
        arithmetic is delegated to the pure engine; this method performs none.

        A per-index failure degrades that index alone - the others still
        report - because a missing SWIR asset is no reason to withhold a
        perfectly good NDVI. The fourth element maps each index that was not
        computed to the warning that says why, so one index's absence is never
        explained by another's record.
        """

        warnings: list[str] = []
        absent: dict[str, str] = {}

        def unavailable(text: str, affected: tuple[str, ...]) -> None:
            warnings.append(text)
            for affected_key in affected:
                absent.setdefault(affected_key, text)

        candidates = _ndwi_candidates(execution)
        if not candidates:
            unavailable(
                "Spectral indices were requested but no Sentinel-2 optical "
                "window with a selected scene was available; nothing was "
                "computed.",
                keys,
            )
            return [], warnings, [], absent

        window = candidates[0]
        bbox = execution.plan.bbox
        collection = _scene_collection(window)
        if len(candidates) > 1:
            warnings.append(
                "Spectral indices are single-scene in this phase: they were "
                f"computed only for the {window.modality} window "
                f"{window.label!r}; {len(candidates) - 1} other optical "
                "window(s) were not analysed."
            )

        needed = bands_for(keys)
        try:
            # Partial availability is accepted here: a scene without SWIR still
            # has a perfectly good NDVI, exactly as a failed read already allowed.
            scene = await self._validated_scene(
                scene_id=window.selected_scene_id,
                collection=collection,
                modality=_OPTICAL_MODALITY,
                assets=(*needed, _SCL_ASSET),
                bbox=bbox,
                require_all_assets=False,
            )
        except AppError as exc:
            unavailable(
                "Spectral indices were not computed for the "
                f"{window.modality} window {window.label!r}: {exc.message}",
                keys,
            )
            return [], warnings, [], absent
        missing_assets = scene.unavailable_assets if scene is not None else {}
        if _SCL_ASSET in missing_assets:
            unavailable(
                _scl_unavailable("Spectral indices", window, missing_assets[_SCL_ASSET]),
                keys,
            )
            return [], warnings, [], absent
        warnings.extend(_coverage_warnings(scene, window.label))

        # Stage 4, before any read: an index whose bands are not on a
        # representation the engine can consume as-is is dropped here, so its
        # own bands are never read. Each index is judged on its own pair.
        runnable: list[str] = []
        for key in keys:
            index = resolve_index(key)
            if scene is not None:
                state = assess_radiometry(scene, (index.high_band, index.low_band))
                radiometry.append(state)
                try:
                    require_usable(state)
                except AppError as exc:
                    unavailable(
                        f"{index.label} was not computed: {exc.message}", (key,)
                    )
                    continue
                warnings.append(_radiometry_summary(index.label, state))
            # Stage 5, before any read: the catalog's source grids for this
            # index's bands and the SCL must be identical or exactly nested.
            problem, _ = _source_geometry(
                scene, (index.high_band, index.low_band), (_SCL_ASSET,)
            )
            if problem is not None:
                refused, text = _geometry_refusal(
                    index.key, window.selected_scene_id, problem, "pre_read"
                )
                grids.append(refused)
                unavailable(f"{index.label} was not computed: {text}", (key,))
                continue
            runnable.append(key)
        if not runnable:
            return [], warnings, [], absent
        keys = tuple(runnable)
        needed = bands_for(keys)

        # Pixel quality first: without it no pixel can be used, so a failure
        # here costs no spectral band read.
        try:
            scl = await self._read_scl(
                scene_id=window.selected_scene_id, bbox=bbox, collection=collection
            )
        except AppError as exc:
            unavailable(
                _scl_unavailable("Spectral indices", window, exc.message), keys
            )
            return [], warnings, [], absent

        def needing(asset: str) -> tuple[str, ...]:
            return tuple(
                k
                for k in keys
                if asset in (resolve_index(k).high_band, resolve_index(k).low_band)
            )

        # One read per distinct band, keyed by asset.
        bands: dict[str, BandWindow] = {}
        for asset in needed:
            if asset in missing_assets:
                unavailable(
                    f"The {asset} band could not be read, so any index needing "
                    f"it was not computed: {missing_assets[asset]}",
                    needing(asset),
                )
                continue
            try:
                bands[asset] = await run_in_threadpool(
                    self._imagery.read_band,
                    scene_id=window.selected_scene_id,
                    bbox=bbox,
                    asset=asset,
                    collection=collection,
                )
            except AppError as exc:
                unavailable(
                    f"The {asset} band could not be read, so any index needing "
                    f"it was not computed: {exc.message}",
                    needing(asset),
                )

        measurements: list[Measurement] = []
        qualities: list[PixelQuality] = []
        # Every index here lands on the same 10 m grid, so the SCL is aligned
        # once per distinct grid and reused, not re-aligned per index.
        scl_on_grid: dict[tuple[object, ...], BandWindow] = {}
        for key in keys:
            index = resolve_index(key)
            high = bands.get(index.high_band)
            low = bands.get(index.low_band)
            if high is None or low is None:
                continue
            high_source, low_source = high, low
            try:
                # Bands of different native resolution are placed on the finer
                # grid by explicit whole-cell assignment, never implicitly.
                if high.values.shape != low.values.shape:
                    if (high.resolution or 0) > (low.resolution or 0):
                        high = coregister_to_finer_grid(high, low)
                    else:
                        low = coregister_to_finer_grid(low, high)
                # Stage 3, on the FINAL grid: the mask is applied to the pair
                # before the engine sees it, so every statistic is masked.
                grid_key = (high.crs, high.values.shape, tuple(high.transform)[:6])
                if grid_key not in scl_on_grid:
                    scl_on_grid[grid_key] = align_scl_to_grid(scl, high)
                masked = mask_with_scl(
                    high,
                    low,
                    scl_on_grid[grid_key],
                    index=index.key,
                    label=index.label,
                    scene_id=window.selected_scene_id,
                    window_label=window.label,
                    scl_metadata_status=_scl_metadata_status(scene),
                )
                measurements.extend(
                    compute_index_measurements(index, masked.high, masked.low)
                )
                measurements.extend(quality_measurements(masked.quality))
                qualities.append(masked.quality)
                warnings.extend(_quality_warnings(index.label, masked.quality))
                if (cause := _quality_cause(index.label, masked.quality)) is not None:
                    absent.setdefault(key, cause)
                analysis_grid = geometry.Grid.of(high)
                state = geometry.grid_state(
                    analysis=index.key,
                    scene_id=window.selected_scene_id,
                    grid=analysis_grid,
                    inputs=[
                        geometry.grid_input(index.high_band, high_source, analysis_grid),
                        geometry.grid_input(index.low_band, low_source, analysis_grid),
                        geometry.grid_input(_SCL_ASSET, scl, analysis_grid),
                    ],
                )
                grids.append(state)
                warnings.append(geometry.summary(index.label, state))
            except geometry.GeometryError as exc:
                refused, text = _geometry_refusal(
                    index.key, window.selected_scene_id, (exc.code, exc.reason),
                    "post_read",
                )
                grids.append(refused)
                unavailable(f"{index.label} could not be computed: {text}", (key,))
                continue
            except AppError as exc:
                unavailable(
                    f"{index.label} could not be computed: {exc.message}", (key,)
                )
                continue
            if index.limiting_resolution_m > 10.0:
                warnings.append(
                    f"{index.label} uses a {index.limiting_resolution_m:.0f} m "
                    f"band, so it is sampled on the 10 m grid but resolves "
                    f"detail no finer than {index.limiting_resolution_m:.0f} m."
                )

        return measurements, warnings, qualities, absent

    async def _ndwi_measurements(
        self,
        execution: QueryExecutionResult,
        radiometry: list[RadiometricState],
        grids: list[GridState],
        *,
        with_overlay: bool = False,
    ) -> tuple[
        list[Measurement],
        list[str],
        NdwiOverlay | None,
        SpatialMeasurement | None,
        PixelQuality | None,
    ]:
        """Single-scene NDWI for one optical window. Returns (measurements, warnings).

        Pixels are read server-side through ``ImageryService`` and the arithmetic
        is delegated to the pure engine; this method performs none itself.
        """

        candidates = _ndwi_candidates(execution)
        if not candidates:
            return (
                [],
                [
                    "NDWI was requested but no Sentinel-2 optical window with a "
                    "selected scene was available; no index was computed."
                ],
                None,
                None,
                None,
            )

        window = candidates[0]
        warnings: list[str] = []
        if len(candidates) > 1:
            warnings.append(
                "NDWI is single-scene in this phase: it was computed only for "
                f"the {window.modality} window {window.label!r}; "
                f"{len(candidates) - 1} other optical window(s) were not analysed."
            )

        bbox = execution.plan.bbox
        collection = _scene_collection(window)
        try:
            scene = await self._validated_scene(
                scene_id=window.selected_scene_id,
                collection=collection,
                modality=_OPTICAL_MODALITY,
                assets=(_NDWI_GREEN_ASSET, _NDWI_NIR_ASSET, _SCL_ASSET),
                bbox=bbox,
            )
            # Stage 4, before any read.
            radiometric: RadiometricState | None = None
            if scene is not None:
                radiometric = assess_radiometry(
                    scene, (_NDWI_GREEN_ASSET, _NDWI_NIR_ASSET)
                )
                radiometry.append(radiometric)
                require_usable(radiometric)
                # Stage 5, before any read.
                problem, _ = _source_geometry(
                    scene, (_NDWI_GREEN_ASSET, _NDWI_NIR_ASSET), (_SCL_ASSET,)
                )
                if problem is not None:
                    raise geometry.GeometryError(
                        problem[0], _geometry_refusal("ndwi", None, problem, "pre_read")[1],
                        stage="pre_read",
                    )
            scl = await self._read_scl(
                scene_id=window.selected_scene_id, bbox=bbox, collection=collection
            )
            green = await run_in_threadpool(
                self._imagery.read_band,
                scene_id=window.selected_scene_id,
                bbox=bbox,
                asset=_NDWI_GREEN_ASSET,
                collection=collection,
            )
            nir = await run_in_threadpool(
                self._imagery.read_band,
                scene_id=window.selected_scene_id,
                bbox=bbox,
                asset=_NDWI_NIR_ASSET,
                collection=collection,
            )
            masked = mask_with_scl(
                green,
                nir,
                scl,
                index="ndwi",
                label="NDWI",
                scene_id=window.selected_scene_id,
                window_label=window.label,
                scl_metadata_status=_scl_metadata_status(scene),
            )
            # Everything below reads the MASKED pair: the statistics, the
            # overlay and the threshold count describe the same usable pixels.
            green_source, nir_source = green, nir
            green, nir = masked.high, masked.low
            measurements = compute_ndwi_measurements(green, nir)
            measurements.extend(quality_measurements(masked.quality))
            # Same two band windows, so the picture and the numbers describe
            # exactly the same pixels; the engine positions it from their own
            # affine and returns None rather than a misplaced overlay.
            overlay = (
                render_ndwi_overlay(
                    green,
                    nir,
                    scene_id=window.selected_scene_id,
                    window_label=window.label,
                )
                if with_overlay
                else None
            )
            # The threshold travels on the intent, so no separate request path
            # is needed. The engine does the counting; nothing here computes it.
            threshold = execution.plan.intent.ndwi_threshold
            spatial = (
                compute_ndwi_threshold_measurement(
                    green,
                    nir,
                    threshold=threshold,
                    scene_id=window.selected_scene_id,
                    window_label=window.label,
                    acquired_at=_selected_acquisition(execution, window),
                )
                if threshold is not None
                else None
            )
        except AppError as exc:
            if isinstance(exc, geometry.GeometryError):
                grids.append(
                    geometry.grid_state(
                        analysis="ndwi", scene_id=window.selected_scene_id, grid=None,
                        refusal=f"{exc.code}: {exc.reason}", stage=exc.stage,
                    )
                )
            logger.info(
                "NDWI unavailable for window %s [%s]: %s",
                window.label,
                exc.code,
                exc.message,
            )
            warnings.append(
                f"NDWI could not be computed for the {window.modality} window "
                f"{window.label!r}: {exc.message}"
            )
            return [], warnings, None, None, None

        warnings.extend(_coverage_warnings(scene, window.label))
        if radiometric is not None:
            warnings.append(_radiometry_summary("NDWI", radiometric))
        warnings.extend(_quality_warnings("NDWI", masked.quality))
        analysis_grid = geometry.Grid.of(green)
        ndwi_grid = geometry.grid_state(
            analysis="ndwi",
            scene_id=window.selected_scene_id,
            grid=analysis_grid,
            inputs=[
                geometry.grid_input(_NDWI_GREEN_ASSET, green_source, analysis_grid),
                geometry.grid_input(_NDWI_NIR_ASSET, nir_source, analysis_grid),
                geometry.grid_input(_SCL_ASSET, scl, analysis_grid),
            ],
        )
        grids.append(ndwi_grid)
        warnings.append(geometry.summary("NDWI", ndwi_grid))
        warnings.append(
            "NDWI values are a spectral index computed from raw Sentinel-2 "
            "digital numbers; they are not a validated water or flood "
            "classification."
        )
        if with_overlay and overlay is None:
            warnings.append(
                "The NDWI overlay was requested but could not be positioned "
                "from the read window's own georeferencing, so no overlay was "
                "produced; the statistics above are unaffected."
            )
        if threshold is not None and spatial is None:
            warnings.append(
                "An NDWI threshold was requested but no valid pixel was "
                "available to count, so no percentage was produced."
            )
        return measurements, warnings, overlay, spatial, masked.quality

    async def _observation_index(
        self,
        observation: Observation,
        bbox: BoundingBox,
        *,
        scl_metadata_status: str = "not_validated",
        radiometry: RadiometricState | None = None,
    ) -> tuple[ObservationIndexResult, BandWindow, BandWindow]:
        """Index ONE observation on its own pixels. Two reads, no comparison.

        Uses the unchanged ``ImageryService.read_band`` path - raw values at
        native resolution, never the display path. The collection comes straight
        off the acquired scene.

        The two band windows are returned alongside the result so a paired
        comparison can reuse them. They are already in memory; re-reading them
        would be a second retrieval path for the same pixels.
        """

        scl = await self._read_scl(
            scene_id=observation.scene_id, bbox=bbox, collection=observation.collection
        )
        green = await run_in_threadpool(
            self._imagery.read_band,
            scene_id=observation.scene_id,
            bbox=bbox,
            asset=_NDWI_GREEN_ASSET,
            collection=observation.collection,
        )
        nir = await run_in_threadpool(
            self._imagery.read_band,
            scene_id=observation.scene_id,
            bbox=bbox,
            asset=_NDWI_NIR_ASSET,
            collection=observation.collection,
        )
        # Each observation is masked by ITS OWN classification; the returned
        # bands carry that mask, so the paired change can only use pixels
        # usable on both dates.
        masked = mask_with_scl(
            green,
            nir,
            scl,
            index="ndwi",
            label="NDWI",
            scene_id=observation.scene_id,
            window_label=observation.window_label,
            scl_metadata_status=scl_metadata_status,
        )
        green_source, nir_source = green, nir
        green, nir = masked.high, masked.low
        observation_grid = geometry.Grid.of(green)
        result = ObservationIndexResult(
            window_label=observation.window_label,
            scene_id=observation.scene_id,
            acquired_at=observation.acquired_at,
            cloud_cover=observation.scene.cloud_cover,
            measurements=[
                *compute_ndwi_measurements(green, nir),
                *quality_measurements(masked.quality),
            ],
            pixel_quality=masked.quality,
            radiometry=radiometry,
            grid=geometry.grid_state(
                analysis="ndwi",
                scene_id=observation.scene_id,
                grid=observation_grid,
                inputs=[
                    geometry.grid_input(_NDWI_GREEN_ASSET, green_source, observation_grid),
                    geometry.grid_input(_NDWI_NIR_ASSET, nir_source, observation_grid),
                    geometry.grid_input(_SCL_ASSET, scl, observation_grid),
                ],
            ),
            # Evidence from the read itself. Both bands come from one scene and
            # share a grid, so ``green`` describes the read; carrying it lets
            # the engine state the AOI coverage and the grid actually used
            # instead of discarding what the read already established.
            crs=green.crs,
            resolution=green.resolution,
            window_pixel_count=green.width * green.height,
            # The affine the raster layer computed for this read, passed
            # through unchanged - not reconstructed from the plan bbox.
            transform=list(green.transform)[:6],
        )
        return result, green, nir

    async def _temporal_ndwi(
        self, execution: QueryExecutionResult
    ) -> tuple[TemporalIndexComparison | None, list[str]]:
        """Temporal NDWI Statistics for ONE deterministic Sentinel-2 pair.

        Orchestration only: select the pair (Phase 13, read-only), read the
        bands (Phase 11, unchanged), then hand both summaries to the pure engine.
        Returns ``(comparison, orchestration_warnings)``; the comparison's own
        warnings live on the returned model.
        """

        observations = execution.observations
        # SAR is filtered out before pairing: Sentinel-1 is not comparable to
        # Sentinel-2 without terrain correction, and NDWI is optical-only.
        # ``for_modality`` is the domain model's own query for exactly this -
        # reused rather than reimplemented.
        pairs, failures = pair_observations(
            ObservationSet(
                requested_bbox=observations.requested_bbox,
                observations=observations.for_modality(_OPTICAL_MODALITY),
            )
        )

        warnings: list[str] = []
        if not pairs:
            reasons = [failure.reason for failure in failures] or [
                "no Sentinel-2 observation pair was available."
            ]
            warnings.extend(
                "Temporal NDWI statistics were requested but no Sentinel-2 pair "
                f"could be formed: {reason}"
                for reason in reasons
            )
            return None, warnings

        pair = pairs[0]
        if len(pairs) > 1:
            warnings.append(
                "Temporal NDWI statistics cover one observation pair "
                f"({pair.first.window_label!r} and {pair.second.window_label!r}); "
                f"{len(pairs) - 1} further consecutive pair(s) were not analysed."
            )

        bbox = execution.plan.bbox
        scene_pair = None
        try:
            # Both scenes are validated against the catalog, independently, and
            # then as a pair - all before the first of the four band reads.
            scenes = [
                await self._validated_scene(
                    scene_id=observation.scene_id,
                    collection=observation.collection,
                    modality=_OPTICAL_MODALITY,
                    assets=(_NDWI_GREEN_ASSET, _NDWI_NIR_ASSET, _SCL_ASSET),
                    bbox=bbox,
                )
                for observation in (pair.first, pair.second)
            ]
            first_scene, second_scene = scenes
            states: list[RadiometricState | None] = [None, None]
            if first_scene is not None and second_scene is not None:
                scene_pair = validate_scene_pair(first_scene, second_scene)
                # Stage 4, before any of the six reads: each observation on
                # its own, then the pair. Nothing is corrected to make them agree.
                states = [
                    require_usable(
                        assess_radiometry(scene, (_NDWI_GREEN_ASSET, _NDWI_NIR_ASSET))
                    )
                    for scene in (first_scene, second_scene)
                ]
                problem = radiometric_pair_problem(states[0], states[1])  # type: ignore[arg-type]
                if problem is not None:
                    raise RadiometricValidationError("radiometric_incompatible", problem)
                # Stage 5, before any of the six reads: each scene's source
                # grids for green, NIR and the SCL.
                for validated in (first_scene, second_scene):
                    source, _ = _source_geometry(
                        validated, (_NDWI_GREEN_ASSET, _NDWI_NIR_ASSET), (_SCL_ASSET,)
                    )
                    if source is not None:
                        raise geometry.GeometryError(
                            source[0],
                            _geometry_refusal(
                                "ndwi", validated.scene_id, source, "pre_read"
                            )[1],
                            stage="pre_read",
                        )
            first, first_green, first_nir = await self._observation_index(
                pair.first,
                bbox,
                scl_metadata_status=_scl_metadata_status(first_scene),
                radiometry=states[0],
            )
            second, second_green, second_nir = await self._observation_index(
                pair.second,
                bbox,
                scl_metadata_status=_scl_metadata_status(second_scene),
                radiometry=states[1],
            )
        except AppError as exc:
            logger.info(
                "Temporal NDWI unavailable for pair %s/%s [%s]: %s",
                pair.first.window_label,
                pair.second.window_label,
                exc.code,
                exc.message,
            )
            warnings.append(
                "Temporal NDWI statistics could not be computed for the pair "
                f"{pair.first.window_label!r} / {pair.second.window_label!r}: "
                f"{exc.message}"
            )
            return None, warnings

        compatibility = compute_compatibility(pair.first, pair.second)
        differences, comparison_warnings = compare_ndwi_observations(
            first=first, second=second, compatibility=compatibility
        )
        if scene_pair is not None:
            for scene, label in (
                (scene_pair.earlier, first.window_label),
                (scene_pair.later, second.window_label),
            ):
                comparison_warnings.extend(_coverage_warnings(scene, label))
            comparison_warnings.extend(scene_pair.notes)

        # Paired-pixel change, on the bands already read. The engine verifies
        # the two grids are identical and returns None when they are not; this
        # method never resamples to force a comparison.
        incomparable = _grids_are_comparable(first_green, second_green)
        # Stage 5: the paired change needs ONE grid. Each side's own statistics
        # never did, so a refusal here withholds the paired change only.
        common = geometry.Grid.of(first_green)
        pair_grid = geometry.grid_state(
            analysis="temporal_ndwi_pair",
            scene_id=None,
            grid=common if incomparable is None else None,
            inputs=(
                [
                    geometry.grid_input("earlier", first_green, common),
                    geometry.grid_input("later", second_green, common),
                ]
                if incomparable is None
                else []
            ),
            refusal=(
                None if incomparable is None
                else f"paired change not computed: {incomparable}"
            ),
        )
        comparison_warnings.append(geometry.summary("Temporal NDWI pair", pair_grid))
        change = (
            compute_ndwi_temporal_change(
                first_green=first_green,
                first_nir=first_nir,
                second_green=second_green,
                second_nir=second_nir,
                first_scene_id=first.scene_id,
                second_scene_id=second.scene_id,
                first_acquired_at=first.acquired_at,
                second_acquired_at=second.acquired_at,
                window_label=f"{first.window_label}\u2192{second.window_label}",
            )
            if incomparable is None
            else None
        )
        if incomparable is not None:
            comparison_warnings.append(
                "Per-pixel NDWI change was not computed because "
                f"{incomparable}. The observations are NOT co-registered and "
                "nothing was resampled onto a shared grid, so the aggregate "
                "statistics above remain the only comparison available."
            )
        elif change is None:
            comparison_warnings.append(
                "Per-pixel NDWI change was not computed because no pixel was "
                "valid in both observations."
            )
        if first.pixel_quality is not None and second.pixel_quality is not None:
            comparison_warnings.append(
                temporal_quality_note(
                    first.pixel_quality,
                    second.pixel_quality,
                    change.paired_valid_pixel_count if change is not None else None,
                )
            )

        return (
            TemporalIndexComparison(
                first=first,
                second=second,
                compatibility=compatibility,
                differences=differences,
                change=change,
                warnings=comparison_warnings,
                pair_grid=pair_grid,
            ),
            warnings,
        )

    async def _sar_backscatter(
        self,
        execution: QueryExecutionResult,
        radiometry: list[RadiometricState],
        grids: list[GridState],
    ) -> tuple[SarBackscatterResult | None, list[str]]:
        """Sentinel-1 RTC gamma-naught statistics for ONE SAR window.

        Orchestration only: pick the window, read each polarization through the
        unchanged quantitative band path, and hand the arrays to the pure
        engine. Every decibel conversion, exclusion rule and grid check lives
        there; this method computes nothing.

        A polarization that cannot be read degrades to a warning rather than
        failing the result - a missing VH is no reason to withhold a perfectly
        good VV - and both failing yields no result at all.
        """

        candidates = _sar_candidates(execution)
        if not candidates:
            return (
                None,
                [
                    "Sentinel-1 backscatter was requested but no SAR window "
                    "with a selected scene was available; nothing was measured."
                ],
            )

        window = candidates[0]
        warnings: list[str] = []
        if len(candidates) > 1:
            warnings.append(
                "Sentinel-1 backscatter is single-scene in this phase: it was "
                f"computed only for the {window.modality} window "
                f"{window.label!r}; {len(candidates) - 1} other SAR window(s) "
                "were not analysed."
            )

        bbox = execution.plan.bbox
        collection = _scene_collection(window)
        try:
            # A missing VH is no reason to withhold VV, as a failed read allows.
            scene = await self._validated_scene(
                scene_id=window.selected_scene_id,
                collection=collection,
                modality=_SAR_MODALITY,
                assets=tuple(SAR_POLARIZATIONS),
                bbox=bbox,
                require_all_assets=False,
            )
        except AppError as exc:
            warnings.append(
                "Sentinel-1 backscatter was not measured for the "
                f"{window.modality} window {window.label!r}: {exc.message}"
            )
            return None, warnings
        unavailable = scene.unavailable_assets if scene is not None else {}
        warnings.extend(_coverage_warnings(scene, window.label))
        # Stage 4, before any read and before any asset is signed: the
        # polarizations must be declared as linear power, or not declared at
        # all - never decibels, never scaled or offset.
        if scene is not None:
            state = assess_radiometry(
                scene, tuple(p for p in SAR_POLARIZATIONS if p not in unavailable)
            )
            radiometry.append(state)
            try:
                require_usable(state)
            except AppError as exc:
                warnings.append(
                    "Sentinel-1 backscatter was not measured for the "
                    f"{window.modality} window {window.label!r}: {exc.message}"
                )
                return None, warnings
            warnings.append(_radiometry_summary("Sentinel-1 backscatter", state))
            # Stage 5, before any read or signing: the polarizations' source
            # grid as published (Planetary Computer publishes it per item).
            readable = tuple(p for p in SAR_POLARIZATIONS if p not in unavailable)
            problem, _ = _source_geometry(scene, readable, ())
            if problem is not None:
                refused, text = _geometry_refusal(
                    "sar_backscatter", window.selected_scene_id, problem, "pre_read"
                )
                grids.append(refused)
                warnings.append(
                    "Sentinel-1 backscatter was not measured for the "
                    f"{window.modality} window {window.label!r}: {text}"
                )
                return None, warnings
        bands: dict[str, BandWindow] = {}
        for polarization in SAR_POLARIZATIONS:
            if polarization in unavailable:
                warnings.append(
                    f"The {polarization} polarization could not be read for the "
                    f"{window.modality} window {window.label!r}, so no "
                    f"{polarization} statistics were computed: "
                    f"{unavailable[polarization]}"
                )
                continue
            try:
                bands[polarization] = await run_in_threadpool(
                    self._imagery.read_band,
                    scene_id=window.selected_scene_id,
                    bbox=bbox,
                    asset=polarization,
                    collection=collection,
                )
            except AppError as exc:
                logger.info(
                    "SAR %s unavailable for window %s [%s]: %s",
                    polarization,
                    window.label,
                    exc.code,
                    exc.message,
                )
                warnings.append(
                    f"The {polarization} polarization could not be read for the "
                    f"{window.modality} window {window.label!r}, so no "
                    f"{polarization} statistics were computed: {exc.message}"
                )

        if not bands:
            return None, warnings

        # Stage 5: each polarization's statistics use only its own pixels; the
        # VV-VH difference needs ONE grid (checked again by the engine, which
        # withholds the difference - never the per-polarization statistics).
        present = [p for p in SAR_POLARIZATIONS if p in bands]
        base = geometry.Grid.of(bands[present[0]])
        mismatch = (
            geometry.same_grid_problem(base, geometry.Grid.of(bands[present[1]]))
            if len(present) == 2
            else None
        )
        sar_grid = geometry.grid_state(
            analysis="sar_backscatter",
            scene_id=window.selected_scene_id,
            grid=base if mismatch is None else None,
            inputs=(
                [geometry.grid_input(p, bands[p], base) for p in present]
                if mismatch is None
                else []
            ),
            refusal=(
                None if mismatch is None
                else f"VV-VH difference not computed: {mismatch[0]}: {mismatch[1]}"
            ),
        )
        grids.append(sar_grid)
        warnings.append(geometry.summary("Sentinel-1 backscatter", sar_grid))

        return (
            compute_sar_backscatter(
                vv=bands.get("vv"),
                vh=bands.get("vh"),
                scene_id=window.selected_scene_id,
                window_label=window.label,
                acquired_at=_selected_acquisition(execution, window),
                collection=collection,
            ),
            warnings,
        )

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Interpret ``request.execution``; the task comes from its intent."""

        # Stage 1 of the scientific pipeline, before the first read: a request
        # that cannot be measured is refused here, with a specific code, rather
        # than discovered after a catalog fetch and a remote open. The reader is
        # consulted only when something will be read: an interpretation with no
        # quantitative operation never touched it, and still does not.
        validate_execution_analysis(
            request,
            limits=self._declared_read_limits() if requested_operations(request) else None,
        )

        execution = request.execution
        task = execution.plan.intent.task
        refs = [_window_ref(window) for window in execution.windows]
        warnings = _collect_warnings(execution)

        if task == _IMPLEMENTED_TASK:
            status = "ok"
            answer = _visualize_answer(execution, refs)
        else:
            status = "not_implemented"
            answer = _not_implemented_answer(task, len(refs))

        measurements: list[Measurement] = []
        ndwi_overlay: NdwiOverlay | None = None
        spatial_measurement: SpatialMeasurement | None = None
        # What was ASKED FOR, and what came of it. `status` answers a different
        # question - whether the task has an engine - so on its own it reported
        # "ok" for a request whose analysis produced nothing at all.
        outcomes: list[AnalysisOutcome] = []
        # Multi-index runs share their band reads; NDWI keeps its own path
        # because it alone also produces the overlay and threshold statistic.
        pixel_quality: list[PixelQuality] = []
        radiometry: list[RadiometricState] = []
        grids: list[GridState] = []
        if request.indices:
            (
                index_measurements,
                index_warnings,
                index_qualities,
                index_absent,
            ) = await self._index_measurements(
                execution, tuple(request.indices), radiometry, grids
            )
            pixel_quality.extend(index_qualities)
            warnings.extend(index_warnings)
            measurements.extend(index_measurements)
            # Each index is judged on its own: a missing SWIR band costs NDBI
            # and leaves a perfectly good NDVI standing.
            names = {m.name for m in index_measurements}
            outcomes.extend(
                _outcome(
                    key, produced=f"{key}_mean" in names, reason=index_absent.get(key)
                )
                for key in request.indices
            )

        if request.include_ndwi:
            (
                ndwi_measurements,
                ndwi_warnings,
                ndwi_overlay,
                spatial_measurement,
                ndwi_quality,
            ) = await self._ndwi_measurements(
                execution, radiometry, grids, with_overlay=request.include_ndwi_overlay
            )
            # ``indices=["ndwi"]`` and ``include_ndwi`` assess the same grid.
            if ndwi_quality is not None and not any(
                q.index == ndwi_quality.index and q.scene_id == ndwi_quality.scene_id
                for q in pixel_quality
            ):
                pixel_quality.append(ndwi_quality)
            warnings.extend(ndwi_warnings)
            # Pixel counts alone are not an NDWI: a grid masked to the last
            # pixel produced counts and no statistic, exactly as for any index.
            ndwi_produced = any(m.name == "ndwi_mean" for m in ndwi_measurements)
            outcomes.append(
                _outcome(
                    "ndwi",
                    produced=ndwi_produced,
                    # A path that gave up said why last; a masked grid says it
                    # in its quality notes.
                    reason=(
                        _quality_cause("NDWI", ndwi_quality)
                        if ndwi_quality is not None
                        else (ndwi_warnings[-1] if ndwi_warnings else None)
                    ),
                )
            )
            # Names are unique per index, so a combined request reports both
            # rather than one replacing the other.
            existing = {m.name for m in measurements}
            measurements.extend(
                m for m in ndwi_measurements if m.name not in existing
            )
            if ndwi_measurements:
                answer = (
                    f"{answer} NDWI index statistics were computed for one "
                    "Sentinel-2 scene at native 10 m resolution."
                )

        sar_backscatter: SarBackscatterResult | None = None
        if request.include_sar_backscatter:
            sar_backscatter, sar_warnings = await self._sar_backscatter(
                execution, radiometry, grids
            )
            warnings.extend(sar_warnings)
            outcomes.append(
                _outcome(
                    "sar_backscatter",
                    # A result with no valid positive sample measured nothing,
                    # whatever object came back.
                    produced=sar_backscatter is not None
                    and any(
                        p.valid_pixel_count for p in sar_backscatter.polarizations
                    ),
                    # No result: the path said why last. A result with no
                    # sample: its engine said why first.
                    reason=(
                        (sar_warnings[-1] if sar_warnings else None)
                        if sar_backscatter is None
                        else next(iter(sar_backscatter.warnings), None)
                    ),
                )
            )
            if sar_backscatter is not None:
                existing = {m.name for m in measurements}
                measurements.extend(
                    m for m in sar_backscatter.measurements if m.name not in existing
                )
                if any(p.valid_pixel_count for p in sar_backscatter.polarizations):
                    answer = (
                        f"{answer} Sentinel-1 RTC gamma naught backscatter "
                        "statistics were computed in decibels for one SAR scene at "
                        "native resolution."
                    )
                else:
                    answer = (
                        f"{answer} No valid positive SAR samples were available, "
                        "so no backscatter statistics in decibels were computed."
                    )

        temporal_comparison: TemporalIndexComparison | None = None
        if request.include_temporal_ndwi:
            temporal_comparison, temporal_warnings = await self._temporal_ndwi(
                execution
            )
            warnings.extend(temporal_warnings)
            outcomes.append(
                _outcome(
                    "temporal_ndwi",
                    produced=temporal_comparison is not None,
                    # Every path that returns no comparison says why last.
                    reason=temporal_warnings[-1] if temporal_warnings else None,
                )
            )
            if temporal_comparison is not None:
                if task == "change_detection":
                    status = "ok"
                    answer = (
                        "Temporal NDWI comparison completed."
                        if temporal_comparison.change is not None
                        else "Temporal NDWI aggregate comparison completed; "
                        "a paired-pixel change map is unavailable."
                    )
                answer = (
                    f"{answer} Temporal NDWI statistics were computed for two "
                    "Sentinel-2 observations, each indexed independently at "
                    "native 10 m resolution."
                )

        logger.info(
            "Analysed execution (task=%s, status=%s, windows=%d, warnings=%d, "
            "measurements=%d, temporal=%s)",
            task,
            status,
            len(refs),
            len(warnings),
            len(measurements),
            temporal_comparison is not None,
        )

        return AnalysisResult(
            status=status,
            task=task,
            answer=answer,
            windows_considered=refs,
            warnings=warnings,
            measurements=measurements,
            analysis_outcomes=_merge_outcomes(outcomes),
            ndwi_overlay=ndwi_overlay,
            spatial_measurement=spatial_measurement,
            temporal_comparison=temporal_comparison,
            pixel_quality=pixel_quality,
            radiometry=radiometry,
            grids=grids,
            sar_backscatter=sar_backscatter,
        )
