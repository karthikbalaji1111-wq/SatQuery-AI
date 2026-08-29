"""Structured query endpoints: NL intent parsing and plan resolution."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.ai import AiService, GeminiIntentParser, ParsePromptRequest
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
