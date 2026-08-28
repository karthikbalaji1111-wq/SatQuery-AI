"""Model inference and reasoning over geospatial inputs."""

from __future__ import annotations

from app.services.base import DomainService


class AiService(DomainService):
    """Model inference and reasoning over geospatial inputs.

    Foundation stub - see :class:`DomainService`. No logic implemented yet.
    """

    name = "ai"

    def describe(self) -> str:
        return "Model inference and reasoning over geospatial inputs."


__all__ = ["AiService"]
