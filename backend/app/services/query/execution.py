"""Query execution orchestration.

A thin composition layer over services that already exist:

    SatQueryIntent
      -> QueryService.build_plan()      (existing grounding, unchanged)
      -> SatelliteService.search()      (existing STAC discovery, single entry point)
      -> deterministic scene selection  (per-modality, pure)
      -> ImageryService.retrieve()      (existing bounded imagery, Sentinel-2 only, optional)

Every requested modality is executed independently against every expanded
temporal window. Sentinel-1 is discovered and selected here; Sentinel-1
*imagery* retrieval is out of scope and no ``ImageryRequest`` is ever built for
it. This module owns *composition only*: it performs no HTTP, no STAC, no raster
I/O and no language-model calls, and must not import provider SDKs or the
low-level transport/raster helpers - only the public service entry points.
"""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.services.base import DomainService
from app.services.query.schemas import (
    ExecutedWindow,
    Modality,
    QueryExecutionRequest,
    QueryExecutionResult,
    SatQueryIntent,
    TemporalComparison,
    TimeRange,
)
from app.services.query.service import QueryService
from app.services.satellite import (
    ImageryRequest,
    ImageryService,
    SatelliteService,
    Scene,
    SceneSearchRequest,
)
from app.services.satellite.schemas import DEFAULT_IMAGERY_ASSET

logger = get_logger("query.execution")

_OPTICAL: Modality = "sentinel-2-optical"
_SAR: Modality = "sentinel-1-sar"


def _collection_for(modality: Modality, settings: Settings) -> str | None:
    """STAC collection override for a modality, for ``SatelliteService.search``.

    ``None`` means "use SatelliteService's configured default" (the Sentinel-2
    collection). Sentinel-1 targets the configured S1 collection. Deliberately
    tiny - not a registry.
    """

    return settings.stac_s1_collection if modality == _SAR else None


def _expand_windows(intent: SatQueryIntent) -> list[tuple[str, TimeRange]]:
    """Expand a validated intent's ``time_windows`` into labelled windows.

    ``SatQueryIntent`` already guarantees the shape, so this is total:

    - ``single``     -> ``[("single", <the one range>)]``
    - ``timeseries`` -> ``[("series[0]", ...), ("series[1]", ...), ...]``
    - ``compare``    -> ``[("baseline", ...), ("target", ...)]``
    """

    windows = intent.time_windows
    if isinstance(windows, TemporalComparison):
        return [("baseline", windows.baseline), ("target", windows.target)]
    if intent.temporal_mode == "single":
        return [("single", windows[0])]
    return [(f"series[{index}]", window) for index, window in enumerate(windows)]


def _select_scene(scenes: list[Scene]) -> Scene | None:
    """Deterministically pick one scene from a discovery result.

    Ordering: ``cloud_cover`` ascending, then ``datetime`` ascending, then
    ``id`` lexicographically ascending. A ``None`` ``cloud_cover`` sorts after
    any numeric value; a ``None`` ``datetime`` sorts after any string. An empty
    list yields ``None`` and is not treated as a failure.
    """

    if not scenes:
        return None

    def sort_key(scene: Scene) -> tuple[tuple[int, float], tuple[int, str], str]:
        cloud = scene.cloud_cover
        cloud_key = (1, 0.0) if cloud is None else (0, float(cloud))
        moment = scene.datetime
        datetime_key = (1, "") if moment is None else (0, moment)
        return (cloud_key, datetime_key, scene.id)

    return min(scenes, key=sort_key)


def _select_scene_sar(scenes: list[Scene]) -> Scene | None:
    """Deterministically pick one Sentinel-1 (SAR) scene.

    Ordering: ``datetime`` ascending, then ``id`` lexicographically ascending.
    A ``None`` ``datetime`` sorts after any string. Cloud cover is not a SAR
    concept and is never consulted; no polarization or orbit-direction
    preference is applied. An empty list yields ``None`` and is not a failure.
    """

    if not scenes:
        return None

    def sort_key(scene: Scene) -> tuple[tuple[int, str], str]:
        moment = scene.datetime
        datetime_key = (1, "") if moment is None else (0, moment)
        return (datetime_key, scene.id)

    return min(scenes, key=sort_key)


class QueryExecutionService(DomainService):
    """Composes existing grounding, discovery, selection and bounded imagery.

    The generic :meth:`run` hook stays unimplemented; :meth:`execute` is the
    typed entry point. Collaborators are injected (defaulting to the real
    services) so tests can substitute fakes. Every requested modality is
    executed independently per temporal window; this service adds no capability
    of its own beyond temporal-window expansion and deterministic per-modality
    scene selection.
    """

    name = "query.execution"

    def __init__(
        self,
        *,
        query_service: QueryService | None = None,
        satellite_service: SatelliteService | None = None,
        imagery_service: ImageryService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._query = query_service or QueryService()
        self._satellite = satellite_service or SatelliteService()
        self._imagery = imagery_service or ImageryService()
        self._settings = settings or get_settings()

    def describe(self) -> str:
        return (
            "End-to-end execution of a SatQueryIntent: location grounding, "
            "Sentinel-1 and Sentinel-2 discovery, deterministic per-modality "
            "scene selection, and optional bounded Sentinel-2 imagery retrieval."
        )

    async def execute(self, request: QueryExecutionRequest) -> QueryExecutionResult:
        """Ground, then discover + select per (modality, temporal window).

        Geospatial and STAC failures propagate unchanged. A bounded-imagery
        failure is confined to its Sentinel-2 window via ``imagery_error`` and
        never aborts the whole execution. Sentinel-1 never triggers imagery
        retrieval.
        """

        intent = request.intent
        plan = await self._query.build_plan(intent)

        executed_modalities: list[Modality] = list(intent.modalities)
        windows: list[ExecutedWindow] = []
        catalog = self._settings.stac_base_url

        for modality in executed_modalities:
            is_optical = modality == _OPTICAL
            collection = _collection_for(modality, self._settings)
            select = _select_scene if is_optical else _select_scene_sar

            for label, time_range in _expand_windows(intent):
                search_response = await self._satellite.search(
                    SceneSearchRequest(
                        bbox=plan.bbox,
                        start_date=time_range.start_date,
                        end_date=time_range.end_date,
                        collection=collection,
                        max_cloud_cover=(
                            request.max_cloud_cover if is_optical else None
                        ),
                        limit=request.limit,
                    )
                )
                catalog = search_response.catalog
                selected = select(search_response.scenes)

                imagery = None
                imagery_error = None
                if is_optical and request.include_imagery and selected is not None:
                    try:
                        imagery = await run_in_threadpool(
                            self._imagery.retrieve,
                            ImageryRequest(
                                scene_id=selected.id,
                                bbox=plan.bbox,
                                asset=DEFAULT_IMAGERY_ASSET,
                            ),
                        )
                    except AppError as exc:
                        imagery_error = exc.message
                        logger.info(
                            "Imagery retrieval failed for %s window %s [%s]: %s",
                            modality,
                            label,
                            exc.code,
                            exc.message,
                        )

                windows.append(
                    ExecutedWindow(
                        modality=modality,
                        label=label,
                        time_range=time_range,
                        scene_count=search_response.scene_count,
                        scenes=search_response.scenes,
                        selected_scene_id=(
                            selected.id if selected is not None else None
                        ),
                        imagery=imagery,
                        imagery_error=imagery_error,
                    )
                )

        logger.info(
            "Executed query for %r: %d window(s) across modalities %s",
            intent.location_query,
            len(windows),
            executed_modalities,
        )

        return QueryExecutionResult(
            plan=plan,
            executed_modalities=executed_modalities,
            skipped_modalities=[],
            windows=windows,
            catalog=catalog,
        )
