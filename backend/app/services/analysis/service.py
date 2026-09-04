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
observations indexed independently, never compared pixel by pixel. Future
engines are dispatched from here without changing this contract.
"""

from __future__ import annotations

from datetime import datetime

from fastapi.concurrency import run_in_threadpool

from app.core.errors import AppError
from app.core.logging import get_logger
from app.services.analysis.engines import (
    compare_ndwi_observations,
    compute_ndwi_measurements,
    compute_ndwi_threshold_measurement,
    render_ndwi_overlay,
)
from app.services.analysis.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisWindowRef,
    Measurement,
    NdwiOverlay,
    ObservationIndexResult,
    SpatialMeasurement,
    TemporalIndexComparison,
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

logger = get_logger("analysis")

_IMPLEMENTED_TASK: QueryTask = "visualize"

_OPTICAL_MODALITY: Modality = "sentinel-2-optical"
#: Earth Search STAC asset keys (common names) for the two 10 m bands NDWI
#: needs: "green" is band B03, "nir" is band B08. Both come from the SAME scene,
#: so they share one grid and need no resampling or co-registration.
_NDWI_GREEN_ASSET = "green"
_NDWI_NIR_ASSET = "nir"


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


def _ndwi_candidates(execution: QueryExecutionResult) -> list[ExecutedWindow]:
    """Optical windows that actually have a scene to read."""

    return [
        window
        for window in execution.windows
        if window.modality == _OPTICAL_MODALITY and window.selected_scene_id is not None
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

    def describe(self) -> str:
        return (
            "Deterministic interpretation of an executed SatQuery result: "
            "status, templated answer, window traceability, warnings, "
            "opt-in single-scene Sentinel-2 NDWI statistics, and opt-in "
            "Temporal NDWI Statistics for one Sentinel-2 observation pair."
        )

    async def _ndwi_measurements(
        self, execution: QueryExecutionResult, *, with_overlay: bool = False
    ) -> tuple[
        list[Measurement], list[str], NdwiOverlay | None, SpatialMeasurement | None
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
            measurements = compute_ndwi_measurements(green, nir)
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
            return [], warnings, None, None

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
        return measurements, warnings, overlay, spatial

    async def _observation_index(
        self, observation: Observation, bbox: BoundingBox
    ) -> ObservationIndexResult:
        """Index ONE observation on its own pixels. Two reads, no comparison.

        Uses the unchanged ``ImageryService.read_band`` path - raw values at
        native resolution, never the display path. The collection comes straight
        off the acquired scene.
        """

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
        return ObservationIndexResult(
            window_label=observation.window_label,
            scene_id=observation.scene_id,
            acquired_at=observation.acquired_at,
            cloud_cover=observation.scene.cloud_cover,
            measurements=compute_ndwi_measurements(green, nir),
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
        try:
            first = await self._observation_index(pair.first, bbox)
            second = await self._observation_index(pair.second, bbox)
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
        return (
            TemporalIndexComparison(
                first=first,
                second=second,
                compatibility=compatibility,
                differences=differences,
                warnings=comparison_warnings,
            ),
            warnings,
        )

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Interpret ``request.execution``; the task comes from its intent."""

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
        if request.include_ndwi:
            (
                measurements,
                ndwi_warnings,
                ndwi_overlay,
                spatial_measurement,
            ) = await self._ndwi_measurements(
                execution, with_overlay=request.include_ndwi_overlay
            )
            warnings.extend(ndwi_warnings)
            if measurements:
                answer = (
                    f"{answer} NDWI index statistics were computed for one "
                    "Sentinel-2 scene at native 10 m resolution."
                )

        temporal_comparison: TemporalIndexComparison | None = None
        if request.include_temporal_ndwi:
            temporal_comparison, temporal_warnings = await self._temporal_ndwi(
                execution
            )
            warnings.extend(temporal_warnings)
            if temporal_comparison is not None:
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
            ndwi_overlay=ndwi_overlay,
            spatial_measurement=spatial_measurement,
            temporal_comparison=temporal_comparison,
        )
