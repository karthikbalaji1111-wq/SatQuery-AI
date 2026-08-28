"""Model inference and reasoning over geospatial inputs.

Implemented so far: a provider-agnostic natural-language -> ``SatQueryIntent``
boundary. The only concrete parser is :class:`MockIntentParser` (test/dev);
no external AI provider is wired in yet.
"""

from app.services.ai.parser import IntentParser, MockIntentParser
from app.services.ai.schemas import ParsePromptRequest
from app.services.ai.service import AiService

__all__ = [
    "AiService",
    "IntentParser",
    "MockIntentParser",
    "ParsePromptRequest",
]
