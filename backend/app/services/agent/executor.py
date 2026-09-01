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

from dataclasses import dataclass, field

from app.core.errors import AppError, InvalidInputError
from app.core.logging import get_logger
from app.services.agent.registry import AnalysisFlag, resolve_tool
from app.services.agent.schemas import (
    AgentEvidence,
    AgentPlan,
    AgentToolStep,
    EvidenceItem,
    ExecuteQueryParams,
    ToolCall,
)
from app.services.analysis.schemas import AnalysisRequest, AnalysisResult, Measurement
from app.services.analysis.service import AnalysisService
from app.services.query.execution import QueryExecutionService
from app.services.query.schemas import QueryExecutionRequest, QueryExecutionResult
from app.services.satellite.schemas import DEFAULT_LIMIT

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


def _execution_request(params: ExecuteQueryParams) -> QueryExecutionRequest:
    """Build the real request, injecting the server-controlled budget.

    Every field the model may influence comes from ``params``; ``limit`` does
    not exist there and is supplied here. ``QueryExecutionRequest`` validates
    the result, so its bounds are enforced once, in the place that owns them.
    """

    return QueryExecutionRequest(
        intent=params.intent,
        include_imagery=params.include_imagery,
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

    return [
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


def _analysis_items(analysis: AnalysisResult) -> list[EvidenceItem]:
    """Flatten an analysis result into citable evidence.

    Ids are namespaced by source so they stay unique across every producer -
    ``AgentEvidence`` rejects duplicates, which makes an id collision a test
    failure rather than a silently ambiguous reference.
    """

    items = _measurement_items(
        analysis.measurements,
        prefix="ndwi",
        source="ndwi",
        produced_by=_NDWI_PRODUCER,
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
    ) -> None:
        self._query = query_execution_service
        self._analysis = analysis_service

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

        discovery, analysis_steps = self._classify(plan)

        steps: dict[int, AgentToolStep] = {}
        execution: QueryExecutionResult | None = None

        if discovery is not None:
            index, params = discovery
            try:
                execution = await self._query.execute(_execution_request(params))
            except AppError as exc:
                logger.info(
                    "Agent discovery failed [%s]: %s", exc.code, exc.message
                )
                steps[index] = AgentToolStep(
                    status="failed", parameters=params, error_message=exc.message
                )
            else:
                steps[index] = AgentToolStep(status="ok", parameters=params)

        analysis = None
        if analysis_steps:
            analysis, analysis_status, message = await self._run_analysis(
                execution, [flag for _, _, flag in analysis_steps]
            )
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

        evidence = self._assemble_evidence(execution, analysis)
        ordered = [steps[index] for index in sorted(steps)]

        logger.info(
            "Agent executed %d step(s): %s",
            len(ordered),
            ", ".join(f"{step.tool}={step.status}" for step in ordered),
        )
        return ExecutionOutcome(steps=ordered, evidence=evidence)

    # -- planning-time classification ------------------------------------- #

    def _classify(
        self, plan: AgentPlan
    ) -> tuple[
        tuple[int, ExecuteQueryParams] | None,
        list[tuple[int, ToolCall, AnalysisFlag]],
    ]:
        """Split the plan into its discovery step and its analysis steps.

        Dispatch is driven by the registry's ``operation`` kind, never by a
        model-supplied name resolved against Python objects. Anything not on the
        allowlist is refused here, before any service call.
        """

        discovery: tuple[int, ExecuteQueryParams] | None = None
        analysis: list[tuple[int, ToolCall, AnalysisFlag]] = []

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
            elif spec.operation == "analysis" and spec.analysis_flag is not None:
                analysis.append((index, params, spec.analysis_flag))
            else:  # pragma: no cover - unreachable while the registry is closed
                raise InvalidInputError(
                    f"Tool {params.tool!r} has no executable operation."
                )

        return discovery, analysis

    # -- analysis --------------------------------------------------------- #

    async def _run_analysis(
        self, execution: QueryExecutionResult | None, flags: list[AnalysisFlag]
    ) -> tuple[AnalysisResult | None, str, str | None]:
        """ONE analyze call carrying the union of the requested flags.

        Coalescing is the architectural invariant: the analysis service
        interprets a single execution result, so running it once per tool would
        repeat that interpretation and duplicate the band reads.
        """

        if execution is None:
            return None, "skipped", None

        request = AnalysisRequest(
            execution=execution,
            include_ndwi="include_ndwi" in flags,
            include_temporal_ndwi="include_temporal_ndwi" in flags,
        )
        try:
            result = await self._analysis.analyze(request)
        except AppError as exc:
            logger.info("Agent analysis failed [%s]: %s", exc.code, exc.message)
            return None, "failed", exc.message
        return result, "ok", None

    # -- evidence --------------------------------------------------------- #

    def _assemble_evidence(
        self,
        execution: QueryExecutionResult | None,
        analysis: AnalysisResult | None,
    ) -> AgentEvidence:
        """Collect the deterministic outputs into the Commit 1 evidence shape.

        Nothing is interpreted, summarised or rounded here - the results are
        carried verbatim and the flattened ``items`` view exists only so a later
        grounding step can resolve a reference by id.
        """

        items: list[EvidenceItem] = []
        if execution is not None:
            items.extend(_execution_items(execution))
        if analysis is not None:
            items.extend(_analysis_items(analysis))

        return AgentEvidence(items=items, execution=execution, analysis=analysis)
