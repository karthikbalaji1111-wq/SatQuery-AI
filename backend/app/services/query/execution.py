"""Query execution orchestration.

A thin composition layer over services that already exist:

    SatQueryIntent
      -> QueryService.build_plan()      (existing grounding, unchanged)
      -> SatelliteService.search()      (existing STAC discovery, single entry point)
      -> deterministic scene selection  (per-modality, pure)
      -> ImageryService.retrieve()      (existing bounded imagery, optional)

Every requested modality is executed independently against every expanded
temporal window. Optical windows retrieve the ``visual`` asset; SAR windows
retrieve the ``vv`` asset (display-normalized to grayscale by the raster
layer). This module owns *composition only*: it performs no HTTP, no STAC, no
raster I/O and no language-model calls, and must not import provider SDKs or the
low-level transport/raster helpers - only the public service entry points.
"""

from __future__ import annotations

from collections.abc import Callable

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
    ResolvedQueryPlan,
    expand_windows,
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


#: The window expansion now lives on the contract, so the integrity validator
#: can check a client-supplied result against the same rule this service used
#: to produce one. Kept as a module-level name because it is this module's
#: vocabulary and several tests address it here.
_expand_windows = expand_windows


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

    async def execute(
        self,
        request: QueryExecutionRequest,
        *,
        before_discovery: Callable[[ResolvedQueryPlan], None] | None = None,
    ) -> QueryExecutionResult:
        """Ground, then discover + select per (modality, temporal window).

        Geospatial and STAC failures propagate unchanged. A bounded-imagery
        failure is confined to its window via ``imagery_error`` and never aborts
        the whole execution.

        ``before_discovery`` runs on the grounded plan before the first catalog
        search, and whatever it raises propagates: it lets a caller that already
        knows what it will analyse refuse an area without paying for discovery.
        This service does not know what that check is - the query layer stays
        free of any analysis import.
        """

        intent = request.intent
        plan = await self._query.build_plan(intent)
        if before_discovery is not None:
            before_discovery(plan)

        executed_modalities: list[Modality] = list(intent.modalities)
        windows: list[ExecutedWindow] = []
        catalog: str | None = None
        first_failure: AppError | None = None

        for modality in executed_modalities:
            is_optical = modality == _OPTICAL
            collection = _collection_for(modality, self._settings)
            select = _select_scene if is_optical else _select_scene_sar

            for label, time_range in _expand_windows(intent):
                try:
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
                except AppError as exc:
                    # One window's catalog failure used to abort the whole
                    # execution, discarding every window that had already
                    # succeeded - a request for three months lost two good
                    # months because the third could not be reached. The
                    # failure is recorded against the window it belongs to and
                    # the rest of the run continues.
                    first_failure = first_failure or exc
                    logger.info(
                        "Discovery failed for %s window %s [%s]: %s",
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
                            scene_count=0,
                            scenes=[],
                            selected_scene_id=None,
                            error=exc.message,
                        )
                    )
                    continue

                # The FIRST catalog to answer, not the last: the top-level
                # field is deterministic, and per-window provenance below is
                # what actually describes each observation.
                catalog = catalog or search_response.catalog
                selected = select(search_response.scenes)

                imagery = None
                imagery_error = None
                if request.include_imagery and selected is not None:
                    asset = (
                        DEFAULT_IMAGERY_ASSET
                        if is_optical
                        else request.sar_polarization
                    )
                    try:
                        imagery = await run_in_threadpool(
                            self._imagery.retrieve,
                            ImageryRequest(
                                scene_id=selected.id,
                                bbox=plan.bbox,
                                asset=asset,
                                collection=collection,
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
                        # Provenance travels with the observation, so a mixed
                        # run stays attributable window by window.
                        catalog=search_response.catalog,
                        # And the scope of the choice travels with it too.
                        scenes_matched=search_response.scenes_matched,
                    )
                )

        if first_failure is not None and all(w.error is not None for w in windows):
            # NOTHING survived. Reporting a 200 carrying only failures would
            # dress a total outage as a result; the original error propagates
            # exactly as it did before partial results existed.
            raise first_failure

        result = QueryExecutionResult(
            plan=plan,
            executed_modalities=executed_modalities,
            skipped_modalities=[],
            windows=windows,
            catalog=catalog or self._settings.stac_base_url,
        )
        logger.info(
            "Executed query for %r: %d window(s) across modalities %s (status=%s)",
            intent.location_query,
            len(windows),
            executed_modalities,
            result.status,
        )
        return result
