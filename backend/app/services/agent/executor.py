"""Deterministic execution of a validated agent plan.

    AgentPlan -> AgentExecutor -> {QueryExecutionService, AnalysisService}
                              -> AgentEvidence + [AgentToolStep]

**There is no model call anywhere in this module.** The executor turns an
already-validated plan into real work and records what happened. It holds
exactly two collaborators - the query-execution service and the analysis
service - and deliberately no imagery service, raster handle, HTTP client or
provider, so there is no reachable path from here to the network or the
filesystem except through those two services.

Dispatch is driven by the registry's operation kind, never by resolving a
model-supplied name against Python objects. A name that is not on the allowlist
is refused during classification - before any service call - so a plan carrying
one executes nothing at all.

Two invariants are architectural rather than incidental:

**Analysis calls are coalesced.** However many analysis tools a plan names, the
executor makes exactly ONE ``AnalysisService.analyze`` call carrying the union
of their flags. Calling ``analyze`` once per tool would re-interpret the same
execution result and double the band reads for no benefit.

**The resource budget is server-controlled.** ``limit`` was removed from the
model-facing parameters during the Commit 1 hardening pass, so:

    MODEL CONTROLLED   location, temporal mode, time windows, modalities, task,
                       include_imagery, max_cloud_cover
    SERVER CONTROLLED  the result limit / resource budget

The executor injects :data:`SERVER_QUERY_LIMIT` when it builds the real
``QueryExecutionRequest``. A planner cannot influence it, and it cannot be
expressed in the contract it emits.

The executor also does not second-guess the analysis service. A plan may be
structurally valid yet semantically inert - a temporal request against a
single-window intent, say. ``AnalysisService`` remains authoritative: it
answers with its own warnings, and the executor preserves them verbatim rather
than substituting an interpretation of its own.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.core.errors import AppError, GeocodingUnavailableError, InvalidInputError
from app.core.logging import get_logger
from app.services.agent.registry import AnalysisFlag, resolve_tool
from app.services.agent.schemas import (
    AgentEvidence,
    AgentPlan,
    AgentToolStep,
    AgentVisual,
    EvidenceItem,
    ExecuteQueryParams,
    RsModelParams,
    SpectralIndicesParams,
    ToolCall,
    VisualObservation,
)
from app.services.agent.visual import VisualAnalyst
from app.services.analysis.schemas import AnalysisRequest, AnalysisResult, Measurement
from app.services.analysis.service import AnalysisService
from app.services.geospatial.schemas import BoundingBox
from app.services.query.execution import QueryExecutionService
from app.services.query.schemas import (
    QueryExecutionRequest,
    QueryExecutionResult,
    ResolvedQueryPlan,
)
from app.services.satellite.schemas import DEFAULT_LIMIT, ImageryResponse


@runtime_checkable
class PrechecksPlan(Protocol):
    """An analysis service that can refuse an area before discovery.

    The real :class:`AnalysisService` does. A test double that does not is
    simply not consulted, and discovery runs as it always has.
    """

    def precheck_plan(
        self,
        bbox: BoundingBox,
        *,
        indices: tuple[str, ...] = (),
        include_ndwi: bool = False,
        include_temporal_ndwi: bool = False,
        include_sar_backscatter: bool = False,
    ) -> None: ...

logger = get_logger("agent.executor")

#: The server-side discovery budget, injected by the executor. Reuses the
#: repository's existing default rather than inventing a second number, so the
#: agent path and the manual endpoints share one budget.
SERVER_QUERY_LIMIT: int = DEFAULT_LIMIT

_NDWI_PRODUCER = "analysis.engines.compute_ndwi_measurements"
_TEMPORAL_PRODUCER = "analysis.engines.compare_ndwi_observations"
_COMPATIBILITY_PRODUCER = "query.compatibility.compute_compatibility"
_EXECUTION_PRODUCER = "query.execution.QueryExecutionService"
_ANALYSIS_PRODUCER = "analysis.service.AnalysisService"


@dataclass(frozen=True)
class ExecutionOutcome:
    """What the executor did, and what it found.

    Deliberately not an agent *contract*: it is the executor's return value,
    consumed later by the synthesis and grounding layers. The two pieces it
    carries - trace steps and evidence - are both Commit 1 contracts.
    """

    steps: list[AgentToolStep] = field(default_factory=list)
    evidence: AgentEvidence = field(default_factory=AgentEvidence)
    #: The ``AppError`` code discovery failed with, when it failed. Lets the
    #: orchestrator tell "no such place" (a question to put back to the user)
    #: from an outage, without matching on message text.
    discovery_failure_code: str | None = None
    #: The wait that failure named, when it named one (a geocoder cooldown or
    #: Retry-After). ``None`` means unknown - never zero.
    discovery_failure_retry_after_seconds: float | None = None
    #: What became of a requested visual description; ``None`` when none was.
    visual: AgentVisual | None = None


def _execution_request(params: ExecuteQueryParams) -> QueryExecutionRequest:
    """Build the real request, injecting the server-controlled budget.

    Every field the model may influence comes from ``params``; ``limit`` does
    not exist there and is supplied here. ``QueryExecutionRequest`` validates
    the result, so its bounds are enforced once, in the place that owns them.
    """

    return QueryExecutionRequest(
        intent=params.intent,
        include_imagery=params.include_imagery,
        sar_polarization=params.sar_polarization,
        max_cloud_cover=params.max_cloud_cover,
        limit=SERVER_QUERY_LIMIT,
    )


def _measurement_items(
    measurements: list[Measurement], *, prefix: str, source: str, produced_by: str
) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id=f"{prefix}.{measurement.name}",
            source=source,  # type: ignore[arg-type]
            measurement=measurement,
            produced_by=produced_by,
        )
        for measurement in measurements
    ]


def _execution_items(execution: QueryExecutionResult) -> list[EvidenceItem]:
    """Citable facts about the retrieval itself, one per executed window.

    The id carries the modality as well as the label. ``QueryExecutionService``
    runs every requested modality against every temporal window, so a
    two-modality query yields two windows sharing one label ("single",
    "baseline", ...). Keying on the label alone collided, and ``AgentEvidence``
    correctly refused the duplicate - which failed the whole request. The
    modality is what actually distinguishes them, and the pair stays
    deterministic.
    """

    items = [
        EvidenceItem(
            id=f"execution.{window.modality}.{window.label}.scene_count",
            source="execution",
            measurement=Measurement(
                name=f"{window.modality}_{window.label}_scene_count",
                value=float(window.scene_count),
                unit="count",
            ),
            produced_by=_EXECUTION_PRODUCER,
        )
        for window in execution.windows
    ]
    for window in execution.windows:
        if window.modality != "sentinel-1-sar" or window.imagery is None:
            continue
        image = window.imagery
        selected = next((s for s in window.scenes if s.id == image.scene_id), None)
        if selected is None or selected.collection != "sentinel-1-rtc":
            continue
        polarization = image.asset.upper()
        items.append(EvidenceItem(
            id=f"execution.{window.modality}.{window.label}.imagery",
            source="execution",
            text=f"Sentinel-1 RTC {polarization} imagery was retrieved. "
                 "The provider supplies terrain-corrected gamma naught. "
                 "SatQuery renders a grayscale display and does not perform SAR "
                 "calibration; quantitative statistics are reported only when computed.",
            produced_by=_EXECUTION_PRODUCER,
        ))
    return items


#: How each index describes its own provenance in the evidence trail.
_INDEX_PRODUCERS = {
    "ndvi": "deterministic NDVI over Sentinel-2 red and NIR",
    "ndwi": _NDWI_PRODUCER,
    "ndbi": "deterministic NDBI over Sentinel-2 SWIR and NIR",
}


def _analysis_items(analysis: AnalysisResult) -> list[EvidenceItem]:
    """Flatten an analysis result into citable evidence.

    Ids are namespaced by source so they stay unique across every producer -
    ``AgentEvidence`` rejects duplicates, which makes an id collision a test
    failure rather than a silently ambiguous reference.
    """

    # Attribute each measurement to the index that produced it. A single
    # "ndwi" prefix over the whole list would label an NDVI value as NDWI -
    # wrong in exactly the trail the product exists to make trustworthy.
    items: list[EvidenceItem] = []
    for key in ("ndvi", "ndwi", "ndbi"):
        owned = [
            m for m in analysis.measurements if m.name.startswith(f"{key}_")
        ]
        if owned:
            items.extend(
                _measurement_items(
                    owned,
                    prefix=key,
                    source=key,
                    produced_by=_INDEX_PRODUCERS[key],
                )
            )
    sar = analysis.sar_backscatter
    if sar is not None:
        items.extend(_measurement_items(
            sar.measurements, prefix="sar_backscatter", source="sar_backscatter",
            produced_by="analysis.sar.compute_sar_backscatter",
        ))
        for index, warning in enumerate(sar.warnings):
            items.append(EvidenceItem(
                id=f"sar_backscatter.warning.{index}", source="sar_backscatter",
                text=warning, produced_by="analysis.sar.compute_sar_backscatter",
            ))
    sar_names = {m.name for m in sar.measurements} if sar is not None else set()
    # Anything that named no index still reports, under the analysis source it
    # already had, rather than being dropped.
    unclaimed = [
        m
        for m in analysis.measurements
        if m.name not in sar_names
        and not any(m.name.startswith(f"{k}_") for k in ("ndvi", "ndwi", "ndbi"))
    ]
    if unclaimed:
        items.extend(
            _measurement_items(
                unclaimed,
                prefix="ndwi",
                source="ndwi",
                produced_by=_NDWI_PRODUCER,
            )
        )

    items.extend(
        EvidenceItem(
            id=f"execution.warning.{index}",
            source="execution",
            text=warning,
            produced_by=_ANALYSIS_PRODUCER,
        )
        for index, warning in enumerate(analysis.warnings)
    )

    comparison = analysis.temporal_comparison
    if comparison is None:
        return items

    items.extend(
        _measurement_items(
            comparison.first.measurements,
            prefix="temporal_ndwi.first",
            source="temporal_ndwi",
            produced_by=_TEMPORAL_PRODUCER,
        )
    )
    items.extend(
        _measurement_items(
            comparison.second.measurements,
            prefix="temporal_ndwi.second",
            source="temporal_ndwi",
            produced_by=_TEMPORAL_PRODUCER,
        )
    )
    items.extend(
        _measurement_items(
            comparison.differences,
            prefix="temporal_ndwi.difference",
            source="temporal_ndwi",
            produced_by=_TEMPORAL_PRODUCER,
        )
    )
    # The paired-pixel change, when the two grids were verified identical.
    #
    # These are the numbers the interface actually shows for a temporal query,
    # and without them here an answer quoting the mean change would be judged
    # ungrounded and withheld - a real, pixel-derived measurement rejected
    # because nothing had offered it as evidence.
    change = comparison.change
    if change is not None:
        items.extend(
            _measurement_items(
                [
                    Measurement(
                        name="ndwi_change_mean",
                        value=change.change_mean,
                        unit="index",
                    ),
                    Measurement(
                        name="ndwi_change_min",
                        value=change.change_min,
                        unit="index",
                    ),
                    Measurement(
                        name="ndwi_change_max",
                        value=change.change_max,
                        unit="index",
                    ),
                    Measurement(
                        name="paired_valid_pixel_count",
                        value=float(change.paired_valid_pixel_count),
                        unit="pixels",
                    ),
                ],
                prefix="temporal_ndwi.change",
                source="temporal_ndwi",
                produced_by=_TEMPORAL_PRODUCER,
            )
        )
        items.append(
            EvidenceItem(
                id="temporal_ndwi.change.pair",
                source="temporal_ndwi",
                text=(
                    f"Paired-pixel NDWI change from {change.first_scene_id} "
                    f"(earlier) to {change.second_scene_id} (later), computed "
                    f"on one shared {change.crs} grid over "
                    f"{change.paired_valid_pixel_count} pixels valid in both."
                ),
                produced_by=_TEMPORAL_PRODUCER,
            )
        )

    items.extend(
        EvidenceItem(
            id=f"temporal_ndwi.warning.{index}",
            source="temporal_ndwi",
            text=warning,
            produced_by=_TEMPORAL_PRODUCER,
        )
        for index, warning in enumerate(comparison.warnings)
    )
    items.extend(
        EvidenceItem(
            id=f"compatibility.limitation.{index}",
            source="compatibility",
            text=limitation,
            produced_by=_COMPATIBILITY_PRODUCER,
        )
        for index, limitation in enumerate(comparison.compatibility.limitations)
    )
    return items


#: The single imagery artifact Phase 18.1 will show a model: the Sentinel-2
#: true-colour product. Identified by the whole triple - optical modality, the
#: ``visual`` asset, and RGB bands - not by any one of them, because "it is a
#: PNG" or "it has three bands" would also admit the Sentinel-1 VV rendering.
_VISUAL_MODALITY = "sentinel-2-optical"
_VISUAL_ASSET = "visual"
_VISUAL_BANDS = ("red", "green", "blue")


def _visual_image(
    execution: QueryExecutionResult | None,
) -> tuple[ImageryResponse, bytes] | str:
    """The S2 true-colour image to observe, or a sentence saying why not.

    Server-authoritative and deliberately narrow. Sentinel-1 VV is a *display*
    rendering - a per-scene 2nd-98th percentile stretch - so its brightness
    carries no fixed meaning and a model asked about it would be inventing.
    Only the optical true-colour product is admitted, and only when it actually
    carries bytes.

    The plan cannot influence this. Nothing here reads a tool parameter: the
    image comes from the execution result the discovery step already produced.
    """

    if execution is None:
        return (
            "no execution result was produced, so there was no image to observe"
        )

    for window in execution.windows:
        image = window.imagery
        if image is None:
            continue
        if window.modality != _VISUAL_MODALITY:
            continue
        if image.asset != _VISUAL_ASSET or tuple(image.bands) != _VISUAL_BANDS:
            continue
        try:
            raw = base64.b64decode(image.image_base64, validate=True)
        except (ValueError, binascii.Error):
            return "the retrieved image could not be decoded"
        if not raw:
            return "the retrieved image carried no bytes"
        return image, raw

    return (
        "no Sentinel-2 true-colour image was available; this phase observes "
        "only the optical 'visual' product, and Sentinel-1 SAR is a display "
        "rendering rather than a picture a model may be asked to interpret"
    )


_THIS_AREA = "this area"


def _without_place(question: str, place: str | None) -> str:
    """The question as the model sees it: the place's name replaced by "this area".

    A model told "Cubbon Park" can describe a park it has read about instead of
    the picture it was handed. The image is the only evidence it may use, so
    the place is removed from what it is asked - the full name first, then each
    comma-separated part of it. Nothing else in the question changes.
    """

    if not place:
        return question
    names = {place.strip(), *(part.strip() for part in place.split(","))}
    out = question
    for name in sorted((name for name in names if len(name) >= 3), key=len, reverse=True):
        out = re.sub(re.escape(name), _THIS_AREA, out, flags=re.IGNORECASE)
    out = re.sub(rf"{_THIS_AREA}(?:\s*,\s*{_THIS_AREA})+", _THIS_AREA, out)
    return " ".join(out.split())


def _acquired(execution: QueryExecutionResult | None, scene_id: str) -> str | None:
    """The catalog's acquisition date for ``scene_id``, YYYY-MM-DD."""

    if execution is None:
        return None
    for window in execution.windows:
        for scene in window.scenes:
            if scene.id == scene_id and scene.datetime:
                return str(scene.datetime)[:10]
    return None


_VISUAL_UNAVAILABLE = (
    "The satellite image was retrieved, but describing it needs an AI visual "
    "model, which is not available here."
)
_VISUAL_FAILED = (
    "The satellite image was retrieved, but the AI visual model did not answer."
)
_VISUAL_NO_IMAGE = "No normal-colour satellite image was available to describe."


class AgentExecutor:
    """Runs a validated :class:`AgentPlan` against the deterministic services.

    Both collaborators are required and injected, so a test substitutes fakes
    and production wires the real services. Nothing else is reachable from here.
    """

    def __init__(
        self,
        *,
        query_execution_service: QueryExecutionService,
        analysis_service: AnalysisService,
        visual_analyst: VisualAnalyst | None = None,
    ) -> None:
        self._query = query_execution_service
        self._analysis = analysis_service
        # Optional so every existing construction site keeps working; the
        # visual tool simply cannot run without one, which the step records.
        self._visual = visual_analyst

    async def execute(self, plan: AgentPlan) -> ExecutionOutcome:
        """Execute ``plan`` and report every step.

        Ordering comes from the plan, whose shape ``AgentPlan`` already
        validated; this method does not re-implement those validators.

        The primary defence against an unknown tool is the Pydantic
        discriminator, which rejects it while the plan is being parsed - long
        before this method sees it. The registry check in :meth:`_classify` is
        defence-in-depth for calls that are already validated, and it raises
        rather than recording a step: a trace step naming an unregistered tool
        cannot be constructed, because ``AgentToolStep.parameters`` is itself
        the closed ``ToolCall`` union.

        The one status this method assigns for a non-executed step is
        ``skipped``: an analysis step with no execution result to interpret.
        Neither path dispatches anything.
        """

        discovery, analysis_steps, visual_steps = self._classify(plan)

        # What the analysis step will ask for, known before discovery runs.
        # Explicit type check rather than a dynamic attribute lookup: the
        # executor is deliberately free of reflection primitives, and the one
        # tool that carries indices is known by name.
        requested_indices: list[str] = []
        for _, params, _ in analysis_steps:
            if isinstance(params, SpectralIndicesParams):
                for key in params.indices:
                    if key not in requested_indices:
                        requested_indices.append(key)
        analysis_flags = [flag for _, _, flag in analysis_steps if flag is not None]

        steps: dict[int, AgentToolStep] = {}
        execution: QueryExecutionResult | None = None
        discovery_failure: str | None = None
        discovery_failure_code: str | None = None
        discovery_failure_retry_after: float | None = None

        if discovery is not None:
            index, params = discovery
            try:
                execution = await self._execute_discovery(
                    params, analysis_flags, requested_indices
                )
            except AppError as exc:
                logger.info(
                    "Agent discovery failed [%s]: %s", exc.code, exc.message
                )
                discovery_failure = exc.message
                discovery_failure_code = exc.code
                # Only a location-service outage names a wait worth keeping.
                if isinstance(exc, GeocodingUnavailableError):
                    discovery_failure_retry_after = exc.retry_after_seconds
                steps[index] = AgentToolStep(
                    status="failed", parameters=params, error_message=exc.message
                )
            else:
                steps[index] = AgentToolStep(status="ok", parameters=params)

        analysis = None
        analysis_failure: str | None = None
        if analysis_steps:
            analysis, analysis_status, message = await self._run_analysis(
                execution, analysis_flags, requested_indices
            )
            if analysis_status == "failed":
                analysis_failure = message
            for index, params, _ in analysis_steps:
                steps[index] = AgentToolStep(
                    status=analysis_status,
                    parameters=params,
                    error_message=message,
                    rejection_reason=(
                        "no execution result was produced, so there was nothing "
                        "to analyse"
                        if analysis_status == "skipped"
                        else None
                    ),
                )

        visual: EvidenceItem | None = None
        visual_failure: str | None = None
        visual_state: AgentVisual | None = None
        if visual_steps:
            visual, visual_step_state, visual_state = await self._run_visual(
                execution, visual_steps[0][1]
            )
            status, message, rejection = visual_step_state
            if status == "failed":
                visual_failure = message
            for index, params in visual_steps:
                steps[index] = AgentToolStep(
                    status=status,
                    parameters=params,
                    error_message=message,
                    rejection_reason=rejection,
                )

        evidence = self._assemble_evidence(
            execution,
            analysis,
            visual,
            discovery_failure=discovery_failure,
            analysis_failure=analysis_failure,
            visual_failure=visual_failure,
        )
        ordered = [steps[index] for index in sorted(steps)]

        logger.info(
            "Agent executed %d step(s): %s",
            len(ordered),
            ", ".join(f"{step.tool}={step.status}" for step in ordered),
        )
        return ExecutionOutcome(
            steps=ordered,
            evidence=evidence,
            discovery_failure_code=discovery_failure_code,
            discovery_failure_retry_after_seconds=discovery_failure_retry_after,
            visual=visual_state,
        )

    # -- discovery -------------------------------------------------------- #

    async def _execute_discovery(
        self,
        params: ExecuteQueryParams,
        flags: list[AnalysisFlag],
        indices: list[str],
    ) -> QueryExecutionResult:
        """Run discovery, refusing an unmeasurable area BEFORE the catalog search.

        When the plan will analyse, the analysis service's own request-stage
        area rule runs on the grounded plan through ``before_discovery``: an
        area the analysis would refuse costs a geocode, not a set of catalog
        searches. Without an analysis step - or with an analysis service that
        declares no precheck - discovery runs exactly as before.
        """

        request = _execution_request(params)
        if not (flags or indices) or not isinstance(self._analysis, PrechecksPlan):
            return await self._query.execute(request)
        analysis = self._analysis

        def precheck(plan: ResolvedQueryPlan) -> None:
            analysis.precheck_plan(
                plan.bbox,
                indices=tuple(indices),
                include_ndwi="include_ndwi" in flags,
                include_temporal_ndwi="include_temporal_ndwi" in flags,
                include_sar_backscatter="include_sar_backscatter" in flags,
            )

        return await self._query.execute(request, before_discovery=precheck)

    # -- planning-time classification ------------------------------------- #

    def _classify(
        self, plan: AgentPlan
    ) -> tuple[
        tuple[int, ExecuteQueryParams] | None,
        list[tuple[int, ToolCall, AnalysisFlag]],
        list[tuple[int, RsModelParams]],
    ]:
        """Split the plan into its discovery step and its analysis steps.

        Dispatch is driven by the registry's ``operation`` kind, never by a
        model-supplied name resolved against Python objects. Anything not on the
        allowlist is refused here, before any service call.
        """

        discovery: tuple[int, ExecuteQueryParams] | None = None
        analysis: list[tuple[int, ToolCall, AnalysisFlag]] = []
        visual: list[tuple[int, RsModelParams]] = []

        for index, params in enumerate(plan.steps):
            # ``resolve_tool`` raises for anything off the allowlist. Because
            # classification completes BEFORE any service call, refusing here
            # means a plan carrying an unpermitted tool executes nothing at all
            # - not even its valid steps. That is the safest failure available,
            # and it is why the refusal is raised rather than recorded: a step
            # naming an unregistered tool is unrepresentable in the trace, since
            # ``AgentToolStep.parameters`` is itself the closed ``ToolCall``
            # union.
            spec = resolve_tool(params.tool)

            if spec.operation == "discovery" and isinstance(
                params, ExecuteQueryParams
            ):
                discovery = (index, params)
            elif spec.operation == "analysis":
                # `spectral_indices` carries its own parameter instead of a
                # flag, so it is collected the same way and read out below.
                analysis.append((index, params, spec.analysis_flag))
            elif spec.operation == "visual" and isinstance(params, RsModelParams):
                visual.append((index, params))
            else:  # pragma: no cover - unreachable while the registry is closed
                raise InvalidInputError(
                    f"Tool {params.tool!r} has no executable operation."
                )

        return discovery, analysis, visual

    # -- analysis --------------------------------------------------------- #

    async def _run_analysis(
        self,
        execution: QueryExecutionResult | None,
        flags: list[AnalysisFlag],
        indices: list[str] | None = None,
    ) -> tuple[AnalysisResult | None, str, str | None]:
        """ONE analyze call carrying the union of the requested flags.

        Coalescing is the architectural invariant: the analysis service
        interprets a single execution result, so running it once per tool would
        repeat that interpretation and duplicate the band reads.
        """

        if execution is None:
            return None, "skipped", None

        indices = indices or []
        request = AnalysisRequest(
            execution=execution,
            include_ndwi="include_ndwi" in flags,
            include_temporal_ndwi="include_temporal_ndwi" in flags,
            include_sar_backscatter="include_sar_backscatter" in flags,
            indices=indices,
        )
        try:
            result = await self._analysis.analyze(request)
        except AppError as exc:
            logger.info("Agent analysis failed [%s]: %s", exc.code, exc.message)
            return None, "failed", exc.message
        return result, "ok", None

    # -- visual observation ----------------------------------------------- #

    async def _run_visual(
        self, execution: QueryExecutionResult | None, params: RsModelParams
    ) -> tuple[
        EvidenceItem | None, tuple[str, str | None, str | None], AgentVisual
    ]:
        """Ask the analyst one question about the image the server chose.

        Every precondition is re-checked here rather than trusted to the
        planner: a plan is model output, so it is an input to be validated, not
        a security boundary. The image is selected from the execution result -
        the plan cannot name one - and only the Sentinel-2 true-colour product
        is admitted.

        Returns the evidence item (or ``None``), the step's observed state and
        the reader-facing :class:`AgentVisual`. A refusal or a provider failure
        yields no OBSERVATION rather than a placeholder: an observation nobody
        made must not appear as one. (A failure is separately explained by a
        plain execution text item - see ``_assemble_evidence`` - which carries
        no ``visual`` field.)

        The image is established first, so "no model here" is only ever said
        about an image that was actually retrieved - the reader is then shown
        that image even though nothing described it.
        """

        selected = _visual_image(execution)
        if isinstance(selected, str):
            return (
                None,
                ("rejected", None, selected),
                AgentVisual(status="no_image", message=_VISUAL_NO_IMAGE),
            )
        image, raw = selected
        where = {
            "scene_id": image.scene_id,
            "acquired": _acquired(execution, image.scene_id),
        }

        if self._visual is None:
            return (
                None,
                (
                    "rejected",
                    None,
                    "no visual analyst is configured, so no model could observe the image",
                ),
                AgentVisual(status="unavailable", message=_VISUAL_UNAVAILABLE, **where),
            )

        place = execution.plan.intent.location_query if execution is not None else None
        try:
            answer = await self._visual.observe(
                question=_without_place(params.question, place),
                image=raw,
                media_type=image.media_type,
            )
        except AppError as exc:
            logger.info("Agent visual analysis failed [%s]: %s", exc.code, exc.message)
            return (
                None,
                ("failed", exc.message, None),
                AgentVisual(status="failed", message=_VISUAL_FAILED, **where),
            )

        item = EvidenceItem(
            # Namespaced by source and scene, so it cannot collide with
            # execution / ndwi / temporal evidence and stays deterministic.
            id=f"model.visual.{image.scene_id}",
            source="model",
            visual=VisualObservation(
                statement=answer.answer,
                provider=self._visual.provider_name,
                model=self._visual.model_name,
                scene_id=image.scene_id,
            ),
            produced_by=self._visual.model_name,
        )
        logger.info(
            "Agent visual observation recorded (model=%s, scene=%s)",
            self._visual.model_name,
            image.scene_id,
        )
        return (
            item,
            ("ok", None, None),
            AgentVisual(
                status="observed",
                message=f"Described by an AI visual model ({self._visual.model_name}).",
                **where,
            ),
        )

    # -- evidence --------------------------------------------------------- #

    def _assemble_evidence(
        self,
        execution: QueryExecutionResult | None,
        analysis: AnalysisResult | None,
        visual: EvidenceItem | None = None,
        *,
        discovery_failure: str | None = None,
        analysis_failure: str | None = None,
        visual_failure: str | None = None,
    ) -> AgentEvidence:
        """Collect the deterministic outputs into the Commit 1 evidence shape.

        Nothing is interpreted, summarised or rounded here - the results are
        carried verbatim and the flattened ``items`` view exists only so a later
        grounding step can resolve a reference by id.

        A failed step becomes one citable text item explaining the absence.
        Without it the synthesiser received evidence with a hole in it and -
        correctly, given that - wrote "Insufficient evidence", so a catalog
        outage or a busy vision endpoint was presented to the user as a
        successful abstention (both observed live). The message is the
        system-authored ``AppError`` text already shown in the trace, never
        upstream content. It adds no measurement, and it is never shaped as an
        observation: it has no ``visual`` field, so a failed look cannot read as
        something a model saw.
        """

        items: list[EvidenceItem] = []
        failures = (
            ("execution.discovery_failure", "Scene discovery", discovery_failure),
            ("execution.analysis_failure", "The analysis", analysis_failure),
            ("execution.visual_failure", "The visual observation", visual_failure),
        )
        explanations = [
            EvidenceItem(
                id=item_id,
                source="execution",
                text=f"{stage} did not complete: {message}",
                produced_by="agent.executor",
            )
            for item_id, stage, message in failures
            if message is not None
        ]
        if execution is not None:
            items.extend(_execution_items(execution))
        if analysis is not None:
            items.extend(_analysis_items(analysis))
        if visual is not None:
            # Last among results, and structurally distinct: it is the only item
            # here that a model authored rather than an engine computed.
            items.append(visual)
        items.extend(explanations)

        return AgentEvidence(items=items, execution=execution, analysis=analysis)
