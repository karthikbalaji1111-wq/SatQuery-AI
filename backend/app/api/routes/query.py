"""Structured query-plan endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.query import QueryService, ResolvedQueryPlan, SatQueryIntent

router = APIRouter()


def get_query_service() -> QueryService:
    """Provider for :class:`QueryService`; overridden in tests."""

    return QueryService()


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
