"""Map layer and tile preparation for the interactive frontend."""

from __future__ import annotations

from app.services.base import DomainService


class MapService(DomainService):
    """Map layer and tile preparation for the interactive frontend.

    Foundation stub - see :class:`DomainService`. No logic implemented yet.
    """

    name = "map"

    def describe(self) -> str:
        return "Map layer and tile preparation for the interactive frontend."


__all__ = ["MapService"]
