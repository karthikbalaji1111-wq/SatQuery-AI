"""The NVIDIA NIM vision-language provider.

A second implementation of :class:`~app.services.agent.visual.VisualAnalyst`,
interchangeable with the Gemini one. Nothing above this module changes when the
provider changes: the executor, the grounding rules, the evidence shape and the
API contract are all provider-neutral, and an observation from either provider
reaches them as the same :class:`VisualAnswer`.

**Image input is a hard requirement here.** The visual path exists to look at a
satellite PNG, so a text-only model - including most Nemotron variants - cannot
serve it. The configured default is a Nemotron *VL* model, and the request is
built in the OpenAI-compatible shape NVIDIA documents for NIM VLMs:
``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``.

No SDK is used. NVIDIA's hosted endpoints speak the OpenAI chat-completions
dialect over plain HTTP, and the project already depends on ``httpx``; adding a
vendor SDK to send one JSON body would buy nothing and widen the dependency
surface.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.errors import IntentParsingError, UpstreamServiceError
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

logger = logging.getLogger("satquery.agent.visual.nvidia")

#: Formats NVIDIA documents for NIM VLM image input.
_SUPPORTED_MEDIA_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/jpg"}
)

class NvidiaVisualAnalyst(VisualAnalyst):
    """Real :class:`VisualAnalyst` backed by an NVIDIA-hosted VLM.

    Sends exactly two content parts: the question, and the PNG bytes the query
    pipeline already retrieved. It sends **no** georeferencing and **no**
    measurements - no CRS, no affine, no corners, no scene id, no NDWI, no
    bbox - for the same reason the Gemini provider does not: an observation
    that has been told the answer is not independent of it, and the whole point
    of this tool is a reading that can be set against the deterministic
    evidence rather than derived from it.

    It cannot fetch anything. The bytes arrive as an argument; there is no URL,
    path or catalog handle in this class.
    """

    provider_name = "nvidia"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _NvidiaChatClient(settings or get_settings(), client, model)

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
                f"The nvidia provider cannot accept {media_type!r}; NIM vision "
                "models accept PNG or JPEG."
            )

        raw = await self._chat.complete(
            [
                {"role": "system", "content": _VISUAL_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"QUESTION\n{question}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{media_type};base64,"
                                    f"{base64.b64encode(image).decode('ascii')}"
                                )
                            },
                        },
                    ],
                },
            ],
            max_tokens=512,
        )
        # `VisualAnswer` caps its field; truncating would silently alter what
        # the model said, so an over-long answer is refused instead.
        if len(raw) > 4000:
            raise IntentParsingError(
                "The language model returned an observation longer than the "
                "contract allows."
            )
        return VisualAnswer(answer=raw)


def _completion_text(response: httpx.Response) -> str:
    """The assistant's text from one chat-completions body.

    Provider-shaped JSON stops here: nothing above this module sees a `choices`
    array or a `message` object. A body that carries no usable text raises
    rather than yielding an empty result that a caller might treat as an answer.
    """

    try:
        body = response.json()
    except ValueError as exc:
        raise IntentParsingError(
            "The language model returned a response that was not JSON."
        ) from exc

    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise IntentParsingError(
            "The language model returned no completion for the image."
        )

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None

    # Some NIM models return the assistant turn as a list of content parts.
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )

    if not isinstance(content, str) or not content.strip():
        raise IntentParsingError(
            "The language model returned an empty or unusable observation."
        )

    return content.strip()


class _NvidiaChatClient:
    """The one place an NVIDIA HTTP request is made.

    Shared by all three roles so transport, authentication, error mapping and
    the never-log-the-body rule exist once rather than three times. It returns
    text; turning that text into an `AgentPlan`, a `DraftAnswer` or a
    `VisualAnswer` is each role's own business, and provider-shaped JSON never
    travels further than this file.
    """

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._model = model or settings.nvidia_model

    @property
    def model(self) -> str:
        return self._model

    def _require_key(self) -> str:
        key = self._settings.nvidia_api_key
        if not key:
            raise UpstreamServiceError(
                "NVIDIA_API_KEY is not configured; the nvidia provider is "
                "unavailable."
            )
        return key

    def _endpoint(self) -> str:
        return f"{self._settings.nvidia_base_url.rstrip('/')}/chat/completions"

    async def complete(
        self, messages: list[dict[str, Any]], *, max_tokens: int = 1024
    ) -> str:
        """POST one chat completion and return the assistant's text."""

        key = self._require_key()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            # Deterministic decoding, matching the Gemini providers, so a rerun
            # is as reproducible as the provider allows.
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

        try:
            if self._client is not None:
                response = await self._client.post(
                    self._endpoint(), json=payload, headers=headers
                )
            else:
                async with httpx.AsyncClient(
                    timeout=self._settings.nvidia_timeout_seconds
                ) as client:
                    response = await client.post(
                        self._endpoint(), json=payload, headers=headers
                    )
        except httpx.TimeoutException as exc:
            logger.warning("NVIDIA transport timeout")
            raise UpstreamServiceError(
                "The language-model service timed out."
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("NVIDIA transport error: %s", type(exc).__name__)
            raise UpstreamServiceError(
                "The language-model service is unavailable."
            ) from exc

        if response.status_code >= 400:
            # Status only. A provider error body can echo the request, and a
            # visual request carries image bytes.
            logger.warning("NVIDIA API error (status=%s)", response.status_code)
            raise UpstreamServiceError(
                "The language-model service is unavailable."
            )

        return _completion_text(response)


def _extract_json(text: str) -> str:
    """Recover the JSON object from a chat completion.

    Gemini is asked for JSON through a response schema the SDK enforces. The
    OpenAI-compatible surface has no equivalent that every NIM model honours,
    so the instruction asks for JSON and this tolerates the two things models
    still do: wrap it in a Markdown fence, or add a sentence around it. It
    never repairs malformed JSON - only locates it.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        body = stripped.split("```", 2)
        if len(body) >= 2:
            fenced = body[1]
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


class NvidiaAgentPlanner(AgentPlanner):
    """Real :class:`AgentPlanner` backed by an NVIDIA-hosted model.

    Issues the **same** planning instruction as the Gemini planner - imported
    from `prompts`, not re-written here - so the two providers are asked for
    the same thing and can be compared. What comes back is validated through
    `AgentPlan`, exactly as the Gemini path validates it: the closed tool union
    is the authority, so an unrecognised tool is a failure here too and never
    something the executor has to defend against.

    Planning needs only text generation, so a text-only NVIDIA model can serve
    this role.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _NvidiaChatClient(settings or get_settings(), client, model)

    async def plan(self, question: str) -> AgentPlan:
        raw = await self._chat.complete(
            [
                {"role": "system", "content": _system_instruction()},
                {"role": "user", "content": f"QUESTION\n{question}"},
            ],
            max_tokens=1024,
        )
        try:
            body = json.loads(_extract_json(raw))
            
            # Llama often wraps the plan in a "plan" key and echoes the intent.
            if isinstance(body, dict):
                if "plan" in body and isinstance(body["plan"], dict) and "steps" in body["plan"]:
                    body = body["plan"]
                if "intent" in body:
                    del body["intent"]

            if isinstance(body, dict) and isinstance(body.get("steps"), list):
                for index, step in enumerate(body["steps"]):
                    # Only NVIDIA's exact tool/parameters envelope is adapted.
                    # Never discard extra fields or allow an inner tool override.
                    if (
                        isinstance(step, dict)
                        and set(step) == {"tool", "parameters"}
                        and isinstance(step["parameters"], dict)
                        and "tool" not in step["parameters"]
                    ):
                        body["steps"][index] = {"tool": step["tool"], **step["parameters"]}
            return AgentPlan.model_validate_json(json.dumps(body))
        except ValidationError as exc:
            raise IntentParsingError(
                "The language model returned a plan that did not match the "
                "expected shape."
            ) from exc
        except ValueError as exc:
            raise IntentParsingError(
                "The language model returned a plan that was not valid JSON."
            ) from exc


class NvidiaAnswerSynthesizer(AnswerSynthesizer):
    """Real :class:`AnswerSynthesizer` backed by an NVIDIA-hosted model.

    Uses the same synthesis instruction and the same evidence rendering as the
    Gemini synthesizer. As there, the prompt is not the control: the response is
    parsed through `DraftAnswer`, and the grounding validator that runs outside
    this class is what establishes that every stated number is traceable. A
    number this model invents is refused by exactly the same check.

    Synthesis needs only text generation, so a text-only NVIDIA model can serve
    this role.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _NvidiaChatClient(settings or get_settings(), client, model)

    async def synthesize(
        self, question: str, evidence: AgentEvidence
    ) -> DraftAnswer:
        raw = await self._chat.complete(
            [
                {"role": "system", "content": _SYNTHESIS_INSTRUCTION},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION\n{question}\n\n"
                        f"EVIDENCE (cite by id)\n{_render_evidence(evidence)}\n"
                    ),
                },
            ],
            max_tokens=1024,
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


class NvidiaIntentParser(IntentParser):
    """Real :class:`IntentParser` backed by an NVIDIA-hosted model.

    Issues the **same** extraction instruction as the Gemini parser - imported
    from ``ai.prompts``, not restated here - so the two providers are asked for
    the same thing and their output is comparable. What comes back is validated
    through :class:`SatQueryIntent`, exactly as the Gemini path validates it:
    the schema is the authority, so a malformed or invented field is a failure
    here too rather than something the query pipeline has to defend against.

    Parsing needs only text generation, so a text-only NVIDIA model can serve
    this role.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _NvidiaChatClient(settings or get_settings(), client, model)

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        raw = await self._chat.complete(
            [
                {"role": "system", "content": _SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
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
