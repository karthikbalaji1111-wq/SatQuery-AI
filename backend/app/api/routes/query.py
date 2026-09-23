"""Structured query endpoints: NL intent parsing and plan resolution."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.errors import AppError
from app.core.limits import rate_limited, workflow_slot
from app.core.observability import workflow
from app.services.agent.executor import AgentExecutor
from app.services.agent.providers.factory import (
    get_agent_providers,
    get_intent_parser,
)
from app.services.agent.schemas import AgentQuestionRequest, AgentResult
from app.services.agent.service import AgentService
from app.services.ai import AiService, ParsePromptRequest
from app.services.analysis import AnalysisRequest, AnalysisResult, AnalysisService
from app.services.query import (
    QueryExecutionRequest,
    QueryExecutionResult,
    QueryExecutionService,
    QueryService,
    ResolvedQueryPlan,
    SatQueryIntent,
)

router = APIRouter()


def get_query_service() -> QueryService:
    """Provider for :class:`QueryService`; overridden in tests."""

    return QueryService()


def get_ai_service() -> AiService:
    """Production provider for :class:`AiService` - a provider-resolved parser.

    Overridden in tests to inject ``MockIntentParser`` or a fake parser.

    The provider comes from the same factory the agent path uses, so
    ``AI_PROVIDER`` governs this endpoint too - there is one selection
    mechanism, not a second one here. A missing credential for the selected
    provider raises at request time and is reported as an upstream failure; it
    is never answered by the other provider.
    """

    return AiService(parser=get_intent_parser())


def get_query_execution_service() -> QueryExecutionService:
    """Provider for :class:`QueryExecutionService`; overridden in tests.

    Composes the real ``QueryService``, ``SatelliteService`` and
    ``ImageryService``. Cheap to construct; performs no network at
    import/startup time.
    """

    return QueryExecutionService()


def get_analysis_service() -> AnalysisService:
    """Provider for :class:`AnalysisService`; overridden in tests.

    The service has no collaborators and no external dependencies, so this is
    a plain construction.
    """

    return AnalysisService()


def build_agent_service(
    provider: str | None = None, model: str | None = None
) -> AgentService:
    """Provider for :class:`AgentService`; overridden in tests.

    Composition lives here rather than inside the service: this is the only
    place that names concrete providers, which is what keeps ``AgentService``
    itself free of any SDK.

    All three AI roles come from ONE provider bundle, so selecting a provider
    swaps the entire reasoning path - planning, seeing and describing - not one
    step of it. The deterministic services below are constructed identically
    whichever provider is chosen: discovery, retrieval, the raster path and the
    analyses are shared, and provider selection cannot reach them.
    """

    providers = get_agent_providers(provider=provider, model=model)
    return AgentService(
        planner=providers.planner,
        executor=AgentExecutor(
            query_execution_service=QueryExecutionService(),
            analysis_service=AnalysisService(),
            # The visual analyst is the WITNESS - it observes the retrieved
            # image. Injected here, beside the other roles, so the executor
            # stays free of any SDK and never learns which provider answered.
            visual_analyst=providers.visual_analyst,
        ),
        synthesizer=providers.synthesizer,
    )


def get_agent_service() -> AgentService | None:
    """Provider for :class:`AgentService`; overridden in tests.

    Returns ``None`` instead of raising when the CONFIGURED DEFAULT provider
    cannot be built. FastAPI resolves a dependency before the handler body
    runs, so raising here decided the run before the request's own
    ``provider`` had been read: a deployment holding only an NVIDIA key could
    not use NVIDIA, because constructing the unconfigured Gemini default
    failed first - and the error then named Gemini, a provider the caller had
    not asked for.

    The failure is not swallowed. The handler re-raises it, unchanged, when the
    request does not name a provider of its own, so an unconfigured deployment
    still gets the same actionable 502 naming the variable to set.
    """

    try:
        return build_agent_service()
    except AppError:
        return None


@router.post(
    "/parse",
    response_model=SatQueryIntent,
    # One provider call, which may be metered.
    dependencies=[Depends(rate_limited), Depends(workflow_slot)],
)
async def parse_intent(
    request: ParsePromptRequest,
    service: AiService = Depends(get_ai_service),
) -> SatQueryIntent:
    """Convert a natural-language request into a structured ``SatQueryIntent``.

    This endpoint ONLY parses text into an intent. It does not geocode, build a
    plan, call STAC, retrieve imagery, or perform external AI inference."""

    return await service.parse_intent(request.prompt)


@router.post(
    "/build-plan",
    response_model=ResolvedQueryPlan,
    # Geocoding only; the geocoder has its own application-wide budget.
    dependencies=[Depends(rate_limited)],
)
async def build_plan(
    intent: SatQueryIntent,
    service: QueryService = Depends(get_query_service),
) -> ResolvedQueryPlan:
    """Validate a query intent and ground its location to a bounding box via
    the Geospatial Service.

    This endpoint performs no STAC discovery, no imagery retrieval, and no
    LLM/AI inference - only validation and location resolution."""

    return await service.build_plan(intent)


@router.post(
    "/execute",
    response_model=QueryExecutionResult,
    dependencies=[Depends(rate_limited), Depends(workflow_slot)],
)
async def execute_query(
    request: QueryExecutionRequest,
    service: QueryExecutionService = Depends(get_query_execution_service),
) -> QueryExecutionResult:
    """Execute a validated ``SatQueryIntent`` end to end.

    Grounds the location via the Geospatial Service, runs Sentinel-2 discovery
    once per temporal window, deterministically selects one scene per window,
    and - when ``include_imagery`` is set - retrieves one bounded RGB window for
    each selected scene. Sentinel-1 SAR is reported under ``skipped_modalities``
    and is not executed in this phase. No LLM/AI inference happens here."""

    return await service.execute(request)


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    dependencies=[Depends(rate_limited), Depends(workflow_slot)],
)
async def analyze_query(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResult:
    """Interpret an already-computed ``QueryExecutionResult``.

    Returns the analysis status, the task derived from
    ``execution.plan.intent.task``, a deterministic answer, slim per-window
    traceability, and warnings. A task with no engine yet is reported as
    ``status="not_implemented"`` in a 200 body rather than as an error.

    This endpoint performs no scene discovery, no STAC search, no imagery
    retrieval, no raster I/O, and no LLM/VLM inference - it only reads the
    execution result it is given."""

    return await service.analyze(request)


@router.post(
    "/agent",
    response_model=AgentResult,
    # The most expensive route in the system: planning, discovery, raster
    # reads, a visual call and synthesis, all inside one HTTP request.
    dependencies=[Depends(rate_limited), Depends(workflow_slot)],
)
async def answer_question(
    request: AgentQuestionRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentResult:
    """Answer a free-form question by planning, executing and describing.

    A language model chooses which of the existing deterministic analyses to
    run; the server validates that choice against a closed tool set, executes
    it through the same services the manual endpoints use, and validates the
    generated answer against the evidence before returning it.

    Every agent outcome is a 200, including the failures. ``planner_unavailable``,
    ``synthesis_unavailable`` and ``answer_withheld`` all carry whatever
    deterministic evidence was established, because that evidence is the
    product and the prose is only a presentation of it - converting those into
    a 5xx would discard the useful half of the response. Genuine faults still
    surface through the existing error handlers.

    This endpoint is an HTTP adapter only. It performs no orchestration, no tool
    execution, no grounding and no model call of its own - ``AgentService`` owns
    all of that.

    A request may name a ``provider`` and a ``model`` to select the inference
    backend for that run. They change nothing else: the same plan, the same
    deterministic tools, the same grounding and the same evidence shape. A
    model that cannot accept an image is refused before the request is made
    rather than being asked to describe a picture it never received."""

    # One correlated scope for the whole run. Every line any layer emits while
    # it is open carries the same id, which is what makes an interleaved log
    # readable - and the provider is recorded here because this is the only
    # layer that knows which one was selected.
    with workflow(
        "agent",
        provider=request.provider or "configured-default",
        model=request.model or "configured-default",
    ):
        if request.provider is not None or request.model is not None:
            # The request names its own backend, so the configured default is
            # irrelevant to this run - including whether it could be built at all.
            service = build_agent_service(request.provider, request.model)
        elif service is None:
            # No override, and the default could not be built: surface that now,
            # with the message naming the provider actually selected.
            service = build_agent_service()
        return await service.answer(request)
