"""The provider-neutral intent-parsing port.

Separated from :mod:`app.services.ai.parser` for one reason: that module
imports the Gemini SDK at module scope, so implementing this ABC from another
provider's adapter would drag ``google-genai`` in behind it. The port itself
depends on nothing but the domain schema.

:mod:`app.services.ai.parser` re-exports :class:`IntentParser`, so existing
imports are unaffected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.query.schemas import SatQueryIntent


class IntentParser(ABC):
    """Abstract translator from raw user text to :class:`SatQueryIntent`.

    Implementations must not leak provider-specific concepts through this
    interface - swapping in a real parser must not change the API contract.
    """

    @abstractmethod
    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        """Return a validated :class:`SatQueryIntent` derived from ``prompt``."""
