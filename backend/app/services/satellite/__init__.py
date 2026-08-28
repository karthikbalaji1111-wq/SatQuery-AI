"""Sentinel-1 SAR and Sentinel-2 optical imagery retrieval."""

from __future__ import annotations

from app.services.base import DomainService


class SatelliteService(DomainService):
    """Sentinel-1 SAR and Sentinel-2 optical imagery retrieval.

    Foundation stub - see :class:`DomainService`. No logic implemented yet.
    """

    name = "satellite"

    def describe(self) -> str:
        return "Sentinel-1 SAR and Sentinel-2 optical imagery retrieval."


__all__ = ["SatelliteService"]
