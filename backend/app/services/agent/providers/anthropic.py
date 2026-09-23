"""The Anthropic (Claude) provider.

A third implementation of the four provider-neutral roles - planner, answer
synthesizer, visual analyst and intent parser - interchangeable with the Gemini
and NVIDIA ones. Nothing above this module changes when the provider changes:
the executor, the grounding rules, the evidence shape and the API contract are
all provider-neutral, and an answer from any provider reaches them as the same
validated SatQuery model.

The official ``anthropic`` SDK is used rather than raw HTTP. The NVIDIA adapter
speaks plain ``httpx`` because NVIDIA publishes an OpenAI-compatible dialect
with no first-party Python client worth the dependency; Anthropic publishes one,
and hand-rolling its authentication, retry and error taxonomy would be a
reimplementation with no upside. The SDK is confined to THIS FILE, exactly as
``google-genai`` is confined to ``gemini.py``, and an AST test enforces it.

Three request-shape decisions, each of which is a fact about the current models
rather than a preference:

* **No ``temperature``.** The Gemini and NVIDIA adapters pin it to 0.0 for
  reproducibility. Sampling parameters were REMOVED from the current Claude
  models - ``temperature``, ``top_p`` and ``top_k`` return HTTP 400 - so
  sending one would not make this provider more deterministic, it would make
  every request fail.
* **No ``thinking`` or ``output_config``.** Claude thinks adaptively by default
  on the current models, and the depth control (``output_config.effort``) is
  rejected by some older ones. Sending neither keeps this adapter compatible
  with any model id an operator configures, catalogued or not. Reasoning is
  never *read*: ``_message_text`` takes text blocks only, so no thinking block
  is parsed, stored, returned or rendered - the Phase 15 rule holds unchanged.
* **No server-side refusal fallbacks.** Anthropic can reroute a refused request
  to another model automatically. That is deliberately not enabled: this
  repository's rule is that an unattributable answer is worse than a missing
  one, and a silently-substituted model would break the attribution that
  travels with every observation. A refusal is reported as a refusal.

**The model is never authoritative.** Whatever it returns is parsed through the
existing contracts - ``AgentPlan``'s closed discriminated union, ``DraftAnswer``,
``SatQueryIntent``, ``VisualAnswer`` - so an unrecognised tool, a smuggled field
or a malformed intent fails validation here rather than reaching the executor.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import anthropic
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.core.logging import get_logger
from app.services.agent.grounding import DraftAnswer
from app.services.agent.planner import AgentPlanner
from app.services.agent.prompts import (
    _SYNTHESIS_INSTRUCTION,
    _VISUAL_INSTRUCTION,
    _render_evidence,
    _system_instruction,
)
from app.services.agent.schemas import AgentEvidence, AgentPlan
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.agent.visual import VisualAnalyst, VisualAnswer
from app.services.ai.ports import IntentParser
from app.services.ai.prompts import _SYSTEM_INSTRUCTION
from app.services.query.schemas import SatQueryIntent

logger = get_logger("agent.anthropic")

#: Image formats the Messages API accepts as a base64 content block.
_SUPPORTED_MEDIA_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)

#: Upstream statuses worth one more attempt: rate limiting (429), Anthropic's
#: own overload signal (529) and server faults. A 4xx that is not 429 means the
#: request itself is wrong, and repeating it would only repeat the mistake.
_RETRYABLE_STATUSES: frozenset[int] = frozenset(
    {429, 500, 502, 503, 504, 529}
)

#: Attempts for one planning call, mirroring the NVIDIA planner. A retry asks
#: for another sample; it never repairs or invents one. Bounded deliberately -
#: after this the genuine failure is raised unchanged.
_PLAN_ATTEMPTS = 3

#: Short, linear backoff between planning attempts.
_PLAN_BACKOFF_SECONDS = 1.5

#: Output budget for a JSON response. Generous because thinking tokens are
#: drawn from the same budget on the current models, and a plan truncated
#: mid-JSON is indistinguishable from a malformed one at the parser.
_JSON_MAX_TOKENS = 4096

#: Output budget for one visual observation - one or two plain sentences,
#: capped at 4000 characters by ``VisualAnswer`` itself.
_VISUAL_MAX_TOKENS = 2048


def _extract_json(text: str) -> str:
    """Recover the JSON object from an assistant message.

    The instruction asks for bare JSON, and this tolerates the one thing a
    model still does anyway: wrap it in a Markdown fence or surround it with a
    sentence. It never repairs malformed JSON - it only locates it, and what it
    finds is still validated by the caller's own contract.

    Deliberately not shared with the NVIDIA adapter's namesake. That one also
    unwraps a bare top-level array, because NIM models were OBSERVED returning
    plans that way; encoding another provider's observed quirks here would
    claim evidence this file does not have.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```", 2)
        if len(parts) >= 2:
            fenced = parts[1]
            if fenced.startswith("json"):
                fenced = fenced[4:]
            stripped = fenced.strip()

    if stripped.startswith("{"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _message_text(message: Any) -> str:
    """The assistant's text from one Messages response.

    SDK-shaped objects stop here: nothing above this module sees a content
    block, a stop reason or a usage record. Two non-answers are refused rather
    than returned as text a caller might parse:

    * ``refusal`` - the model declined. Reported as an upstream failure, not
      as an empty answer.
    * ``max_tokens`` - the response was cut off. Truncated JSON would surface
      as "the model returned a malformed plan", blaming the model for a budget
      this adapter set.

    Thinking blocks are skipped rather than concatenated. That is what keeps
    "no reasoning is requested, stored or displayed" true of this provider.
    """

    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "refusal":
        raise UpstreamServiceError(
            "The language model declined to answer this request."
        )
    if stop_reason == "max_tokens":
        raise IntentParsingError(
            "The language model's response was cut off before it was complete."
        )

    text = "".join(
        block.text
        for block in getattr(message, "content", None) or []
        if getattr(block, "type", None) == "text"
        and isinstance(getattr(block, "text", None), str)
    )
    if not text.strip():
        raise IntentParsingError(
            "The language model returned an empty or unusable response."
        )
    return text.strip()


class _AnthropicMessagesClient:
    """The one place an Anthropic request is made.

    Shared by all four roles so transport, authentication, error mapping and
    the never-log-the-body rule exist once rather than four times. It returns
    text; turning that text into an ``AgentPlan``, a ``DraftAnswer``, a
    ``SatQueryIntent`` or a ``VisualAnswer`` is each role's own business, and
    SDK-shaped objects never travel further than this file.

    The client is created lazily, so every role constructs without a credential
    and a deployment that never selects Anthropic never pays for one.
    """

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._model = model or settings.anthropic_model

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = self._settings.anthropic_api_key
        if not key:
            raise UpstreamServiceError(
                "ANTHROPIC_API_KEY is not configured; the anthropic provider "
                "is unavailable."
            )
        options: dict[str, Any] = {
            "api_key": key,
            "timeout": self._settings.anthropic_timeout_seconds,
        }
        if self._settings.anthropic_base_url:
            options["base_url"] = self._settings.anthropic_base_url
        self._client = anthropic.AsyncAnthropic(**options)
        return self._client

    async def complete(
        self,
        *,
        system: str,
        content: str | list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """Send one message and return the assistant's text.

        Every SDK exception is mapped to this repository's own error types, and
        no provider payload is logged: an upstream error body can echo the
        request, and a visual request carries image bytes.
        """

        client = self._get_client()
        try:
            message = await client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APITimeoutError as exc:
            logger.warning("Anthropic transport timeout")
            raise UpstreamServiceError(
                "The language-model service timed out."
            ) from exc
        except anthropic.APIConnectionError as exc:
            logger.warning(
                "Anthropic transport error: %s", type(exc).__name__
            )
            raise UpstreamServiceError(
                "The language-model service is unavailable."
            ) from exc
        except anthropic.APIStatusError as exc:
            # Status only - never the body.
            logger.warning(
                "Anthropic API error (status=%s)", exc.status_code
            )
            error = UpstreamServiceError(
                "The language-model service is unavailable."
            )
            # Marked locally rather than by widening the shared error type,
            # matching the NVIDIA adapter: only the planner's retry loop reads
            # it, and changing `UpstreamServiceError` would touch every raiser.
            error.retryable = exc.status_code in _RETRYABLE_STATUSES
            raise error from exc
        except anthropic.AnthropicError as exc:
            # The SDK's own base class, so this stays a narrow catch: an
            # ordinary bug in this module still surfaces as a 500 rather than
            # being recoded as a provider outage.
            logger.warning("Anthropic SDK error: %s", type(exc).__name__)
            raise UpstreamServiceError(
                "The language-model service is unavailable."
            ) from exc

        return _message_text(message)


class AnthropicAgentPlanner(AgentPlanner):
    """Real :class:`AgentPlanner` backed by a Claude model.

    Issues the **same** planning instruction as the other providers - imported
    from ``prompts``, not re-written here - so all three are asked for the same
    thing and can be compared. What comes back is validated through
    ``AgentPlan``: the closed tool union is the authority, so an unrecognised
    tool is a failure here too and never something the executor has to defend
    against.

    Planning needs only text generation, so any catalogued Claude model can
    serve this role.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _AnthropicMessagesClient(
            settings or get_settings(), client, model
        )

    async def plan(self, question: str) -> AgentPlan:
        """Ask for a plan, re-asking a bounded number of times.

        Two transient conditions are retried and neither is papered over: the
        endpoint can rate-limit or report overload, and a model can return an
        unparseable shape. In both cases another sample is a reasonable
        request, and in neither case is anything invented. After
        ``_PLAN_ATTEMPTS`` the genuine failure is raised unchanged.
        """

        last: Exception | None = None
        for attempt in range(_PLAN_ATTEMPTS):
            try:
                return await self._plan_once(question)
            except IntentParsingError as exc:
                last = exc
            except UpstreamServiceError as exc:
                if not getattr(exc, "retryable", False):
                    raise
                last = exc
            if attempt + 1 < _PLAN_ATTEMPTS:
                await asyncio.sleep(_PLAN_BACKOFF_SECONDS * (attempt + 1))
        assert last is not None
        raise last

    async def _plan_once(self, question: str) -> AgentPlan:
        raw = await self._chat.complete(
            system=_system_instruction(),
            content=f"QUESTION\n{question}",
            max_tokens=_JSON_MAX_TOKENS,
        )
        try:
            return AgentPlan.model_validate_json(_extract_json(raw))
        except ValidationError as exc:
            raise IntentParsingError(
                "The language model returned a plan that did not match the "
                "expected shape."
            ) from exc
        except ValueError as exc:
            raise IntentParsingError(
                "The language model returned a plan that was not valid JSON."
            ) from exc


class AnthropicAnswerSynthesizer(AnswerSynthesizer):
    """Real :class:`AnswerSynthesizer` backed by a Claude model.

    Uses the same synthesis instruction and the same evidence rendering as the
    other synthesizers. As there, the prompt is not the control: the response is
    parsed through ``DraftAnswer``, and the grounding validator that runs
    outside this class is what establishes that every stated number is
    traceable. A number this model invents is refused by exactly the same check.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _AnthropicMessagesClient(
            settings or get_settings(), client, model
        )

    async def synthesize(
        self, question: str, evidence: AgentEvidence
    ) -> DraftAnswer:
        raw = await self._chat.complete(
            system=_SYNTHESIS_INSTRUCTION,
            content=(
                f"QUESTION\n{question}\n\n"
                f"EVIDENCE (cite by id)\n{_render_evidence(evidence)}\n"
            ),
            max_tokens=_JSON_MAX_TOKENS,
        )
        try:
            return DraftAnswer.model_validate_json(_extract_json(raw))
        except ValidationError as exc:
            raise IntentParsingError(
                "The language model returned an answer that did not match the "
                "expected shape."
            ) from exc
        except ValueError as exc:
            raise IntentParsingError(
                "The language model returned an answer that was not valid JSON."
            ) from exc


class AnthropicIntentParser(IntentParser):
    """Real :class:`IntentParser` backed by a Claude model.

    Issues the **same** extraction instruction as the other parsers - imported
    from ``ai.prompts``, not restated here - so their output is comparable.
    What comes back is validated through :class:`SatQueryIntent`, so a
    malformed or invented field is a failure rather than something the query
    pipeline has to defend against.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _AnthropicMessagesClient(
            settings or get_settings(), client, model
        )

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        raw = await self._chat.complete(
            system=_SYSTEM_INSTRUCTION,
            content=prompt,
            max_tokens=_JSON_MAX_TOKENS,
        )
        try:
            return SatQueryIntent.model_validate_json(_extract_json(raw))
        except ValidationError as exc:
            raise IntentParsingError(
                "The language model returned an intent that did not match the "
                "expected shape."
            ) from exc
        except ValueError as exc:
            raise IntentParsingError(
                "The language model returned an intent that was not valid JSON."
            ) from exc


class AnthropicVisualAnalyst(VisualAnalyst):
    """Real :class:`VisualAnalyst` backed by a Claude model's image input.

    Sends exactly two content blocks: the PNG bytes the query pipeline already
    retrieved, and the question. It sends **no** georeferencing and **no**
    measurements - no CRS, no affine, no corners, no scene id, no NDWI, no
    bbox - for the same reason the other providers do not: an observation that
    has been told the answer is not independent of it, and the point of this
    tool is a reading that can be set against the deterministic evidence rather
    than derived from it.

    It cannot fetch anything. The bytes arrive as an argument; there is no URL,
    path or catalog handle in this class.
    """

    provider_name = "anthropic"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _AnthropicMessagesClient(
            settings or get_settings(), client, model
        )

    @property
    def model_name(self) -> str:
        """The configured model, so a claim is attributable to a version."""

        return self._chat.model

    async def observe(
        self, *, question: str, image: bytes, media_type: str
    ) -> VisualAnswer:
        """Send the image and the question; parse what comes back."""

        if media_type not in _SUPPORTED_MEDIA_TYPES:
            raise UpstreamServiceError(
                f"The anthropic provider cannot accept {media_type!r}; Claude "
                "models accept PNG, JPEG, GIF or WebP."
            )

        raw = await self._chat.complete(
            system=_VISUAL_INSTRUCTION,
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.standard_b64encode(image).decode(
                            "ascii"
                        ),
                    },
                },
                {"type": "text", "text": f"QUESTION\n{question}"},
            ],
            max_tokens=_VISUAL_MAX_TOKENS,
        )
        # `VisualAnswer` caps its field; truncating would silently alter what
        # the model said, so an over-long answer is refused instead.
        if len(raw) > 4000:
            raise IntentParsingError(
                "The language model returned an observation longer than the "
                "contract allows."
            )
        return VisualAnswer(answer=raw)
