"""Structured query endpoints: NL intent parsing and plan resolution."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.agent.executor import AgentExecutor
from app.services.agent.providers.gemini import (
    GeminiAgentPlanner,
    GeminiAnswerSynthesizer,
)
from app.services.agent.schemas import AgentQuestionRequest, AgentResult
from app.services.agent.service import AgentService
from app.services.ai import AiService, GeminiIntentParser, ParsePromptRequest
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
    """Production provider for :class:`AiService` - a Gemini-backed parser.

    Overridden in tests to inject ``MockIntentParser`` or a fake parser. The
    parser is created lazily, so this stays cheap and does not require
    ``GEMINI_API_KEY`` to be set at import/startup time.
    """

    return AiService(parser=GeminiIntentParser())


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


def get_agent_service() -> AgentService:
    """Provider for :class:`AgentService`; overridden in tests.

    Composition lives here rather than inside the service: this is the only
    place that names a concrete provider, which is what keeps ``AgentService``
    itself free of any SDK. The Gemini planner and synthesizer create their
    clients lazily, so this stays cheap and needs no ``GEMINI_API_KEY`` at
    import/startup time.
    """

    return AgentService(
        planner=GeminiAgentPlanner(),
        executor=AgentExecutor(
            query_execution_service=QueryExecutionService(),
            analysis_service=AnalysisService(),
        ),
        synthesizer=GeminiAnswerSynthesizer(),
    )


@router.post("/parse", response_model=SatQueryIntent)
async def parse_intent(
    request: ParsePromptRequest,
    service: AiService = Depends(get_ai_service),
) -> SatQueryIntent:
    """Convert a natural-language request into a structured ``SatQueryIntent``.

    This endpoint ONLY parses text into an intent. It does not geocode, build a
    plan, call STAC, retrieve imagery, or perform external AI inference."""

    return await service.parse_intent(request.prompt)


@router.post("/build-plan", response_model=ResolvedQueryPlan)
async def build_plan(
    intent: SatQueryIntent,
    service: QueryService = Depends(get_query_service),
) -> ResolvedQueryPlan:
    """Validate a query intent and ground its location to a bounding box via
    the Geospatial Service.

    This endpoint performs no STAC discovery, no imagery retrieval, and no
    LLM/AI inference - only validation and location resolution."""

    return await service.build_plan(intent)


@router.post("/execute", response_model=QueryExecutionResult)
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


@router.post("/analyze", response_model=AnalysisResult)
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


@router.post("/agent", response_model=AgentResult)
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
    all of that."""

    return await service.answer(request)
