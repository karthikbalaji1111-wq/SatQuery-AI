"""Multimodal fusion and analysis across SAR, optical, and text."""

from __future__ import annotations

from app.services.base import DomainService


class MultimodalService(DomainService):
    """Multimodal fusion and analysis across SAR, optical, and text.

    Foundation stub - see :class:`DomainService`. No logic implemented yet.
    """

    name = "multimodal"

    def describe(self) -> str:
        return "Multimodal fusion and analysis across SAR, optical, and text."


__all__ = ["MultimodalService"]
