"""Provider-agnostic natural-language -> :class:`SatQueryIntent` extraction.

This module defines the *boundary* only. No external AI provider (Anthropic,
Gemini, NVIDIA, Ollama, local weights, ...) is imported here or anywhere in this
package. A real parser is added in a later phase; for now the only concrete
implementation is :class:`MockIntentParser`, used for tests and local dev.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from app.services.query.schemas import SatQueryIntent, TimeRange

# A fixed, schema-valid intent the mock always returns. Deterministic on purpose
# so tests can assert exact equality.
_MOCK_INTENT = SatQueryIntent(
    location_query="Chennai",
    temporal_mode="single",
    time_windows=[TimeRange(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))],
    modalities=["sentinel-2-optical"],
    task="visualize",
)


class IntentParser(ABC):
    """Abstract translator from raw user text to :class:`SatQueryIntent`.

    Implementations must not leak provider-specific concepts through this
    interface - swapping in a real parser must not change the API contract.
    """

    @abstractmethod
    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        """Return a validated :class:`SatQueryIntent` derived from ``prompt``."""


class MockIntentParser(IntentParser):
    """TEST / DEVELOPMENT ONLY - performs no language understanding.

    Returns a fixed, valid :class:`SatQueryIntent` regardless of the prompt so
    the rest of the pipeline can be exercised without an AI provider. Replace
    with a real :class:`IntentParser` (e.g. a provider-backed one) in a later
    phase.
    """

    is_mock = True

    def __init__(self, intent: SatQueryIntent | None = None) -> None:
        self._intent = intent if intent is not None else _MOCK_INTENT.model_copy(deep=True)

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        # The prompt is intentionally ignored - this mock does no NLP.
        del prompt
        return self._intent.model_copy(deep=True)
