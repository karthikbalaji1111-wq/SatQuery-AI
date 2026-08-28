"""Model inference and reasoning over geospatial inputs.

Implemented so far: a provider-agnostic natural-language -> ``SatQueryIntent``
boundary (:class:`IntentParser`). Concrete parsers:

- :class:`GeminiIntentParser` - production, Google Gemini API.
- :class:`MockIntentParser` - deterministic, no provider (tests / local dev).
"""

from app.services.ai.parser import (
    GeminiIntentParser,
    IntentParser,
    MockIntentParser,
)
from app.services.ai.schemas import ParsePromptRequest
from app.services.ai.service import AiService

__all__ = [
    "AiService",
    "GeminiIntentParser",
    "IntentParser",
    "MockIntentParser",
    "ParsePromptRequest",
]
