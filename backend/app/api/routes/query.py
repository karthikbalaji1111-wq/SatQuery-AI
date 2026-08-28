"""Structured query endpoints: NL intent parsing and plan resolution."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.ai import AiService, ParsePromptRequest
from app.services.query import QueryService, ResolvedQueryPlan, SatQueryIntent

router = APIRouter()


def get_query_service() -> QueryService:
    """Provider for :class:`QueryService`; overridden in tests."""

    return QueryService()


def get_ai_service() -> AiService:
    """Provider for :class:`AiService`; overridden in tests.

    Defaults to the ``MockIntentParser``. Swap this for a provider-backed
    parser later without changing the route or the response contract.
    """

    return AiService()


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
