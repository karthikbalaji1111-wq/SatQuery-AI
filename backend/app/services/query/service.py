"""Query-plan resolution.

SatQueryIntent -> existing Geospatial Service -> BoundingBox -> ResolvedQueryPlan.
No geocoding is performed here; :class:`GeospatialService` owns that.
"""

from __future__ import annotations

from app.core.errors import PointLocationError
from app.core.logging import get_logger
from app.services.base import DomainService
from app.services.geospatial import GeospatialService, ResolveRequest
from app.services.geospatial.schemas import ResolveResponse
from app.services.query.schemas import ResolvedQueryPlan, SatQueryIntent

logger = get_logger("query")


def _point_location_message(query: str, resolved: ResolveResponse) -> str:
    """Name what the geocoder matched, in its own words, and why it stops here."""

    parts = [part.strip() for part in (resolved.display_name or "").split(",")]
    matched = ", ".join(part for part in parts[:2] if part) or query
    kind = " · ".join(
        value.replace("_", " ")
        for value in (resolved.place_class, resolved.place_type)
        if value
    )
    described = f"{matched} ({kind})" if kind else matched
    return (
        f"'{query}' resolved only to a single point on the map: {described}. "
        "A point has no area to measure, and SatQuery does not draw one around "
        "it, so nothing was searched."
    )


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
        # An analysis area has to BE an area. When the geocoder has only a
        # point for the place (and none of its other results is the same place
        # with a real extent), the box around that point is a display box -
        # measuring it would report on ~2 x 2 pixels as if they were the place.
        # Refused here, before any catalog search; never widened.
        if resolved.point_like:
            raise PointLocationError(
                _point_location_message(intent.location_query, resolved)
            )
        logger.info(
            "Built query plan for %r (task=%s, mode=%s, modalities=%s)",
            intent.location_query,
            intent.task,
            intent.temporal_mode,
            intent.modalities,
        )
        return ResolvedQueryPlan(
            intent=intent,
            bbox=resolved.bbox,
            matched_name=resolved.display_name,
            matched_class=resolved.place_class,
            matched_type=resolved.place_type,
        )
