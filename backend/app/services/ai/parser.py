"""Natural-language -> :class:`SatQueryIntent` extraction.

:class:`IntentParser` is the provider-neutral abstract boundary. Provider
coupling is confined to this module:

- :class:`MockIntentParser` - deterministic, no provider (tests / local dev).
- :class:`GeminiIntentParser` - real parser backed by the Google Gemini API
  via the official ``google-genai`` SDK.

Everything outside this module (``AiService``, the route) depends only on
:class:`IntentParser`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.core.logging import get_logger
from app.services.query.schemas import SatQueryIntent, TimeRange

logger = get_logger("ai.parser")

# A fixed, schema-valid intent the mock always returns. Deterministic on purpose
# so tests can assert exact equality.
_MOCK_INTENT = SatQueryIntent(
    location_query="Chennai",
    temporal_mode="single",
    time_windows=[TimeRange(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))],
    modalities=["sentinel-2-optical"],
    task="visualize",
)

_SYSTEM_INSTRUCTION = """\
You extract a structured satellite-imagery query intent from the user's request
and return ONLY a JSON object matching the provided response schema. No prose.

FIELDS

location_query (string, required)
- The geographic place or region the user asked about, kept verbatim as a
  human-readable place name suitable for a downstream geocoder
  (e.g. "Chennai", "Port of Rotterdam", "Sundarbans").
- NEVER invent or output coordinates. Preserve the name the user gave.

temporal_mode (string, required) - exactly one of:
- "single"     : the user wants one observation window.
- "compare"    : the user contrasts two windows (before/after, baseline/target).
- "timeseries" : the user wants a sequence of three or more windows over time.

time_windows (required) - the shape depends on temporal_mode:
- "single":     a list with EXACTLY ONE object
                {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}.
- "timeseries": a list with TWO OR MORE such objects, in chronological order.
- "compare":    a single object
                {"baseline": {"start_date": "...", "end_date": "..."},
                 "target":   {"start_date": "...", "end_date": "..."}}.
- All dates are ISO calendar dates. end_date must be >= start_date. Resolve
  unambiguous relative expressions ("last summer", "June 2024", "2023") into
  explicit ISO ranges: a single named day sets start_date == end_date; a named
  month or year expands to that period's first and last day.

modalities (required) - a non-empty list; allowed values ONLY:
- "sentinel-2-optical" : optical / true-colour / cloud-free / visual imagery.
- "sentinel-1-sar"     : radar / SAR / all-weather / night / see-through-cloud.
- Default to ["sentinel-2-optical"] when the user names no sensor.
- No duplicate values.

task (string, required) - exactly one of:
- "visualize"             : "show / view / get imagery of".
- "change_detection"      : "what changed", "before vs after", growth / loss.
- "object_identification" : "find / count / locate <objects>".

RULES
- Extract only information the user's request actually supports. Use the allowed
  enum values only.
- NEVER invent coordinates, satellite scenes, or analysis results.
- Do NOT geocode, do NOT retrieve imagery, and do NOT claim any image was
  analysed - you only produce the structured intent.
- Prefer the safe defaults above over guessing. Do not fabricate a location or a
  date range that the request does not imply.
- Output only the structured JSON response.
"""


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
    the pipeline can be exercised without any AI provider. Retained for tests
    and offline development after the real parser landed.
    """

    is_mock = True

    def __init__(self, intent: SatQueryIntent | None = None) -> None:
        self._intent = intent if intent is not None else _MOCK_INTENT.model_copy(deep=True)

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        # The prompt is intentionally ignored - this mock does no NLP.
        del prompt
        return self._intent.model_copy(deep=True)


class GeminiIntentParser(IntentParser):
    """Real :class:`IntentParser` backed by the Google Gemini API.

    Uses the official ``google-genai`` async client with structured output: the
    *existing* :class:`SatQueryIntent` model is passed as the response schema and
    the model's JSON is re-validated with Pydantic - the LLM is never
    authoritative. The client is created lazily so the service can be
    constructed without credentials; tests inject a stub ``client``.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: object | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        if not self._settings.gemini_api_key:
            raise UpstreamServiceError(
                "GEMINI_API_KEY is not configured; natural-language parsing is "
                "unavailable."
            )
        self._client = genai.Client(
            api_key=self._settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(self._settings.gemini_timeout_seconds * 1000)
            ),
        )
        return self._client

    def _build_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            # Pass the EXISTING Pydantic model as the schema - no duplication.
            response_schema=SatQueryIntent,
            temperature=0.0,
            candidate_count=1,
        )

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        client = self._get_client()

        try:
            response = await client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=self._build_config(),
            )
        except genai_errors.APIError as exc:
            # Log the status code only - never the message (may carry request data)
            # and never the key.
            logger.warning("Gemini API error (status=%s)", getattr(exc, "code", "?"))
            raise UpstreamServiceError(
                "The language-model service is unavailable."
            ) from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            logger.warning("Gemini transport error: %s", type(exc).__name__)
            raise UpstreamServiceError(
                "The language-model service timed out."
            ) from exc
        except Exception as exc:  # unknown SDK/transport failure - never leak details
            logger.warning("Unexpected Gemini failure: %s", type(exc).__name__)
            raise UpstreamServiceError(
                "The language-model service failed."
            ) from exc

        raw = getattr(response, "text", None)
        if not raw or not raw.strip():
            raise IntentParsingError(
                "The language model returned an empty response."
            )

        try:
            intent = SatQueryIntent.model_validate_json(raw)
        except ValidationError as exc:
            logger.info(
                "Gemini output failed SatQueryIntent validation (%d error(s))",
                exc.error_count(),
            )
            raise IntentParsingError(
                "Could not extract a reliable structured intent from the request."
            ) from exc

        logger.info(
            "Gemini parsed intent (model=%s, mode=%s, task=%s)",
            self._settings.gemini_model,
            intent.temporal_mode,
            intent.task,
        )
        return intent
