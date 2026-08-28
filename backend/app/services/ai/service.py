"""AI service: natural language -> existing SatQueryIntent.

This service owns *only* intent extraction and holds no model/provider logic -
it delegates to an injected :class:`IntentParser`. It does not resolve
locations, build plans, discover scenes, or retrieve imagery.
"""

from __future__ import annotations

from app.core.errors import InvalidInputError
from app.core.logging import get_logger
from app.services.ai.parser import IntentParser, MockIntentParser
from app.services.base import DomainService
from app.services.query.schemas import SatQueryIntent

logger = get_logger("ai")


class AiService(DomainService):
    """Wraps an :class:`IntentParser` to expose :meth:`parse_intent`.

    The generic :meth:`run` hook stays unimplemented; :meth:`parse_intent` is
    the typed entry point. This class holds no provider logic - the parser is
    injected. Production wires a ``GeminiIntentParser`` in via the route's
    dependency; the constructor default is the credential-free
    :class:`MockIntentParser` so tests and offline use never need a key.
    """

    name = "ai"

    def __init__(self, parser: IntentParser | None = None) -> None:
        self._parser: IntentParser = parser if parser is not None else MockIntentParser()

    def describe(self) -> str:
        return "Natural-language intent extraction into SatQueryIntent."

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        """Translate ``prompt`` into a validated :class:`SatQueryIntent`."""

        cleaned = prompt.strip()
        if not cleaned:
            raise InvalidInputError("prompt must not be empty")

        intent = await self._parser.parse_intent(cleaned)
        logger.info(
            "Parsed prompt into intent (parser=%s, mode=%s, task=%s)",
            type(self._parser).__name__,
            intent.temporal_mode,
            intent.task,
        )
        return intent
