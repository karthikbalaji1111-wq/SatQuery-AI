"""Query-plan resolution.

SatQueryIntent -> existing Geospatial Service -> BoundingBox -> ResolvedQueryPlan.
No geocoding is performed here; :class:`GeospatialService` owns that.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.services.base import DomainService
from app.services.geospatial import GeospatialService, ResolveRequest
from app.services.query.schemas import ResolvedQueryPlan, SatQueryIntent

logger = get_logger("query")


class QueryService(DomainService):
    """Turns a validated :class:`SatQueryIntent` into a grounded plan.

    The generic :meth:`run` hook stays unimplemented; :meth:`build_plan` is the
    typed entry point for this phase.
    """

    name = "query"

    def __init__(self, geospatial_service: GeospatialService | None = None) -> None:
        self._geospatial = geospatial_service or GeospatialService()

    def describe(self) -> str:
        return "Structured query-intent grounding and plan resolution."

    async def build_plan(self, intent: SatQueryIntent) -> ResolvedQueryPlan:
        """Resolve ``intent.location_query`` via the Geospatial Service and
        attach the resulting bounding box to the intent."""

        resolved = await self._geospatial.resolve(
            ResolveRequest(place=intent.location_query)
        )
        logger.info(
            "Built query plan for %r (task=%s, mode=%s, modalities=%s)",
            intent.location_query,
            intent.task,
            intent.temporal_mode,
            intent.modalities,
        )
        return ResolvedQueryPlan(intent=intent, bbox=resolved.bbox)
