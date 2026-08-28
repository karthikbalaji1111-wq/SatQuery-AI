"""Multitemporal change detection over image stacks."""

from __future__ import annotations

from app.services.base import DomainService


class TemporalService(DomainService):
    """Multitemporal change detection over image stacks.

    Foundation stub - see :class:`DomainService`. No logic implemented yet.
    """

    name = "temporal"

    def describe(self) -> str:
        return "Multitemporal change detection over image stacks."


__all__ = ["TemporalService"]
