"""Natural-language query parsing and orchestration."""

from __future__ import annotations

from app.services.base import DomainService


class QueryService(DomainService):
    """Natural-language query parsing and orchestration.

    Foundation stub - see :class:`DomainService`. No logic implemented yet.
    """

    name = "query"

    def describe(self) -> str:
        return "Natural-language query parsing and orchestration."


__all__ = ["QueryService"]
