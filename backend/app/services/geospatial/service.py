"""Geospatial grounding service."""

from __future__ import annotations

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import InvalidInputError
from app.core.logging import get_logger
from app.services.base import DomainService
from app.services.geospatial.nominatim import geocode
from app.services.geospatial.schemas import ResolveRequest, ResolveResponse

logger = get_logger("geospatial")


class GeospatialService(DomainService):
    """Geocoding, area-of-interest handling, and spatial grounding.

    The generic :meth:`run` hook stays unimplemented; :meth:`resolve` is the
    typed entry point for this phase.
    """

    name = "geospatial"

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    def describe(self) -> str:
        return "Geocoding, area-of-interest handling, and spatial grounding."

    async def resolve(self, request: ResolveRequest) -> ResolveResponse:
        """Resolve a place name or bounding box into a validated representation."""

        if request.bbox is not None:
            return ResolveResponse(
                query_type="bbox",
                display_name=None,
                center=request.bbox.center,
                bbox=request.bbox,
                source="input",
            )

        if request.place is None:  # defensive; ResolveRequest guarantees one input
            raise InvalidInputError("provide exactly one of 'place' or 'bbox'")

        place = await geocode(
            request.place,
            settings=self._settings,
            transport=self._transport,
        )
        logger.info("Resolved place %r -> %s", request.place, place.display_name)
        return ResolveResponse(
            query_type="place",
            display_name=place.display_name,
            center=place.center,
            bbox=place.bbox,
            source="nominatim",
        )
