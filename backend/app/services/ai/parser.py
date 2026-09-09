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

from datetime import date

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.core.logging import get_logger
from app.services.ai.ports import IntentParser
from app.services.ai.prompts import _SYSTEM_INSTRUCTION
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


def _intent_response_schema() -> types.Schema:
    """SDK- and API-compatible schema for a :class:`SatQueryIntent`.

    Derived from the existing contract - not hand-written - so it cannot drift
    from what the parser will actually accept back. Built with the SDK's public
    ``Schema.from_json_schema``, the same conversion the Phase 15 agent provider
    uses, rather than handing the Pydantic model straight to the SDK.

    That distinction is the whole point. ``SatQueryIntent.ndwi_threshold`` is an
    ``NdwiThreshold``, which sets ``extra="forbid"``, so Pydantic emits
    ``additionalProperties: false``. ``t_schema`` accepts that field and carries
    it onto the wire, where ``generateContent`` rejects the entire request:

        400 INVALID_ARGUMENT - Unknown name "additional_properties" at
        'generation_config.response_schema.properties[5].value'

    A schema that translates is therefore not necessarily a schema the API will
    take, which is why the regression test asserts the SERIALIZED form.

    This schema is deliberately only structurally representative - it describes
    the shape the model should return and nothing more. It does NOT relax the
    application contract: the response is still parsed through
    ``SatQueryIntent``, so ``extra="forbid"`` still rejects an unexpected field
    even though the request no longer advertises that rule.
    """

    return types.Schema.from_json_schema(
        json_schema=types.JSONSchema(**SatQueryIntent.model_json_schema())
    )


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
            response_schema=_intent_response_schema(),
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
