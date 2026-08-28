"""Geocoding, area-of-interest handling, and spatial grounding."""

from __future__ import annotations

from app.services.base import DomainService


class GeospatialService(DomainService):
    """Geocoding, area-of-interest handling, and spatial grounding.

    Foundation stub - see :class:`DomainService`. No logic implemented yet.
    """

    name = "geospatial"

    def describe(self) -> str:
        return "Geocoding, area-of-interest handling, and spatial grounding."


__all__ = ["GeospatialService"]
