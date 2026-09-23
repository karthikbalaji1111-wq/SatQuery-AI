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

import asyncio
import base64
import json
import logging
import re
from collections.abc import Awaitable, Callable
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

        messages = [
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
        ]
        raw = await _retry_transient(
            lambda: self._chat.complete(messages, max_tokens=512)
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


def _tool_call_arguments(response: httpx.Response) -> str | None:
    """The ``arguments`` JSON string of the first tool call, or ``None``.

    ``None`` means "this model did not answer with a tool call" - a shape
    question, not a failure. The caller falls back to the plain-JSON path, so a
    NIM model that ignores ``tools`` behaves exactly as it did before tool
    calling was offered.

    Provider-shaped JSON stops here, as it does in :func:`_completion_text`:
    nothing above this module sees ``choices`` or ``tool_calls``.
    """

    try:
        body = response.json()
    except ValueError:
        return None
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or not calls:
        return None
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if isinstance(arguments, str) and arguments.strip():
        return arguments.strip()
    return None


def _finish_reason(response: httpx.Response) -> str:
    """The completion's ``finish_reason`` if it is a plain token, else a label."""

    try:
        choices = response.json().get("choices")
        reason = choices[0].get("finish_reason")
    except (ValueError, AttributeError, IndexError, TypeError):
        return "unavailable"
    if isinstance(reason, str) and _FINISH_REASON.fullmatch(reason):
        return reason
    return "unrecognised"


#: Upstream statuses worth one more attempt. 503 is what the NIM endpoint
#: returns as "Worker local total request limit reached (16/16)" - shared
#: capacity, not a fault in the request - and 429 is ordinary rate limiting.
#: A 4xx that is not 429 means the request itself is wrong and retrying it
#: would only repeat the mistake, so it is deliberately excluded.
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Attempts for one planning call. The endpoint is non-deterministic about the
#: JSON envelope it returns, so re-asking is a legitimate response to an
#: unparseable shape - it requests another sample, it does not repair or invent
#: one. Bounded deliberately: three attempts, then the real failure is raised.
_PLAN_ATTEMPTS = 3

#: Short, linear backoff between planning attempts.
_PLAN_BACKOFF_SECONDS = 1.5

#: Token budget for one planning call. Measured live: the configured reasoning
#: model spent 1,430-2,179 completion tokens on a single plan (its reasoning
#: counts), well above the 1,024 previously requested. A budget below what the
#: model needs ends the turn before the tool call is emitted, which then reads
#: downstream as an unusable plan. A model that needs fewer tokens stops early,
#: so the headroom costs nothing when it is not used.
_PLAN_MAX_TOKENS = 4096

#: A provider ``finish_reason`` is logged only if it looks like one.
_FINISH_REASON = re.compile(r"[a-z_]{1,32}")

#: Attempts for one synthesis or visual call that fails with a TRANSIENT status
#: (the retryable set above). Observed live: one shared-capacity 503 ("Worker
#: local total request limit reached") during synthesis discarded an otherwise
#: complete run as ``synthesis_unavailable`` - seconds after the planner, which
#: already retried, had succeeded against the same endpoint.
_TRANSIENT_ATTEMPTS = 3

#: Short, linear backoff between those attempts.
_TRANSIENT_BACKOFF_SECONDS = 1.5


async def _retry_transient(call: Callable[[], Awaitable[str]]) -> str:
    """Run ``call``, re-issuing it only after a retryable upstream failure.

    Narrower than the planner's loop on purpose. Only a transient HTTP status
    is retried - the endpoint was busy, and asking again asks the same question.
    A malformed or empty RESPONSE is not: for synthesis and observation that is
    a real outcome of the request, and the caller reports it as one. A timeout
    is not retried either; each already cost the full client timeout.

    Bounded; after the last attempt the genuine failure is raised unchanged, so
    a persistent outage still surfaces as one.
    """

    for attempt in range(_TRANSIENT_ATTEMPTS):
        try:
            return await call()
        except UpstreamServiceError as exc:
            if not getattr(exc, "retryable", False) or attempt + 1 == _TRANSIENT_ATTEMPTS:
                raise
        await asyncio.sleep(_TRANSIENT_BACKOFF_SECONDS * (attempt + 1))
    raise AssertionError("unreachable: the final attempt returns or raises")


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

    async def complete_tool_call(
        self,
        messages: list[dict[str, Any]],
        *,
        tool: dict[str, Any],
        max_tokens: int = 1024,
    ) -> str | None:
        """Ask for ONE named tool call and return its ``arguments`` JSON string.

        This is the structured-output path that actually holds. ``response_format``
        only asks for "some JSON object", so a NIM model is free to invent the
        envelope and the field types, and observed live it does: the configured
        default returned ``{"steps": ["execute_query", "ndwi_statistics"]}`` -
        tool names as bare strings, carrying none of the parameters a plan needs
        - while another catalogued model stringified every nested value
        (``"modalities": "['sentinel-2-optical']"``, a Python repr rather than
        JSON). Neither is repairable without inventing the missing arguments.

        Constrained decoding against the real JSON Schema is what fixes that:
        with ``tools`` + a forced ``tool_choice``, the same default model returns
        properly typed nested arrays and a plan that validates unchanged.

        Returns ``None`` - never raises - when the model answers without a tool
        call, so the caller can fall back to the older path. Transport and HTTP
        failures are raised exactly as :meth:`complete` raises them.
        """

        key = self._require_key()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": [tool],
            "tool_choice": {
                "type": "function",
                "function": {"name": tool["function"]["name"]},
            },
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = await self._post(payload, key)
        arguments = _tool_call_arguments(response)
        if arguments is None:
            # Structure only - never content. Without this a planning failure
            # was undiagnosable: bodies are deliberately never logged, so the
            # log could not say whether the model ran out of budget ("length")
            # or chose to answer in prose ("stop").
            logger.warning(
                "NVIDIA planning returned no tool call (finish_reason=%s)",
                _finish_reason(response),
            )
        return arguments

    async def _post(self, payload: dict[str, Any], key: str) -> httpx.Response:
        """Issue the POST and map transport/HTTP failures. Body never logged."""

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
            error = UpstreamServiceError(
                "The language-model service is unavailable."
            )
            # Marked locally rather than by widening the shared error type:
            # only this provider's retry loop consults it, and changing
            # `UpstreamServiceError`'s signature would touch every raiser.
            error.retryable = response.status_code in _RETRYABLE_STATUSES
            raise error
        return response

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """POST one chat completion and return the assistant's text.

        ``response_format`` is a GENERATION HINT ONLY. It constrains what the
        endpoint emits so the caller does not have to guess between the several
        equivalent envelopes a NIM model will otherwise pick from. It is never
        the authority on correctness: the response is still parsed and validated
        by the caller's own contract, exactly as it is when the hint is absent
        or silently ignored by a model that does not honour it.
        """

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
        if response_format is not None:
            payload["response_format"] = response_format

        return _completion_text(await self._post(payload, key))


def _plan_response_format() -> dict[str, Any]:
    """Ask the endpoint for JSON rather than prose. A hint, never authority.

    ``json_object`` is used rather than a full ``json_schema``. Both were tried
    against the live endpoint: plain requests and ``{"type": "json_object"}``
    return 200, while ``{"type": "json_schema", ...}`` returned
    ``503 ResourceExhausted: Worker local total request limit reached (16/16)``
    - a capacity limit on the constrained-decoding path, not a rejection of the
    schema. Depending on a route that answers 503 under load would trade a
    parsing problem for an availability problem, so the schema is expressed in
    the prompt and enforced afterwards by `AgentPlan` instead.

    What this DOES buy is the elimination of prose around the JSON and of the
    "two concatenated objects" responses seen live, which `_extract_json` can
    only partially recover from. Shape variance that remains is handled by
    `_plan_envelope`, and correctness is `AgentPlan`'s alone.
    """

    return {"type": "json_object"}


#: The function name the planning tool call is forced to. Arbitrary but fixed:
#: it is echoed back by the endpoint and never reaches a prompt or a response.
_PLAN_TOOL_NAME = "submit_plan"


def _plan_tool() -> dict[str, Any]:
    """The planning step expressed as an OpenAI-style function definition.

    The parameter schema is ``AgentPlan``'s OWN JSON Schema, generated rather
    than written out here, so the shape the endpoint is constrained to and the
    shape the contract validates can never drift apart. `AgentPlan` remains the
    sole authority on correctness: this only improves the odds that what comes
    back is already the right shape, and a response that validates against the
    schema is still validated again on arrival.
    """

    return {
        "type": "function",
        "function": {
            "name": _PLAN_TOOL_NAME,
            "description": "Submit the remote-sensing analysis plan.",
            "parameters": AgentPlan.model_json_schema(),
        },
    }


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

    # A bare top-level array is one of the shapes NIM models return for a plan
    # ("[{...}, {...}]" rather than '{"steps": [...]}'). It must be recognised
    # HERE rather than left to the brace scan below, which would start at the
    # first "{" and end at the last "}" - spanning both elements and producing
    # "Extra data" from an otherwise perfectly well-formed response.
    if stripped.startswith("["):
        end = stripped.rfind("]")
        if end > 0:
            return stripped[: end + 1]
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _plan_envelope(body: Any) -> Any:
    """Reduce NIM's several equivalent plan envelopes to the contract's one.

    Observed live against the NVIDIA endpoint, the SAME model returns the SAME
    semantic plan wrapped four different ways from call to call, because nothing
    in an OpenAI-compatible chat completion pins the envelope the way Gemini's
    response schema does. Each shape below was seen in a real response:

    =====================================  =================================
    Returned                               Meaning
    =====================================  =================================
    ``{"steps": [...]}``                   already the contract's shape
    ``{"plan": [...]}``                    step list under a "plan" key
    ``{"plan": {"steps": [...]}}``         contract shape nested under "plan"
    ``{"tool": ..., <args>}``              a single bare step, unwrapped
    ``[{...}, {...}]``                     a bare step list, no wrapper
    =====================================  =================================

    plus, per step, OpenAI's function-calling envelope
    ``{"tool": ..., "parameters": {<args>}}``.

    This is ENVELOPE normalisation, deliberately not content normalisation. It
    only relocates an already-complete step list; it never renames a field,
    supplies a default, coerces a type, drops an unrecognised key, or repairs a
    step's arguments. Anything it cannot place is returned untouched so that
    `AgentPlan` - which remains the sole validation authority, with its closed
    tool union and ``extra="forbid"`` - rejects it. Widening this set is a
    deliberate act requiring a newly observed shape, not a general fallback.
    """

    # A bare step list, with no enclosing object at all.
    if isinstance(body, list):
        body = {"steps": body}

    if not isinstance(body, dict):
        return body

    plan = body.get("plan")
    if "steps" not in body:
        if isinstance(plan, list):
            body = {"steps": plan}
        elif isinstance(plan, dict) and "steps" in plan:
            body = plan
        elif "tool" in body:
            # A single step returned bare, without the surrounding list.
            body = {"steps": [body]}

    if isinstance(body, dict):
        # The intent is sometimes echoed beside the plan; it is not a plan field.
        body.pop("intent", None)
        steps = body.get("steps")
        if isinstance(steps, list):
            for index, step in enumerate(steps):
                if (
                    isinstance(step, dict)
                    and set(step) == {"tool", "parameters"}
                    and isinstance(step["parameters"], dict)
                    and "tool" not in step["parameters"]
                ):
                    steps[index] = {"tool": step["tool"], **step["parameters"]}
    return body


#: A provider-local restatement of two rules the shared instruction already
#: gives. It is repeated here, in the user turn, because the NIM models comply
#: with them far less reliably than Gemini does: observed live, this model
#: returned a plan of `execute_query` ALONE for a question explicitly asking for
#: an NDWI value, which discovers a scene, measures nothing, and leaves the
#: synthesizer to answer "insufficient evidence" from one scene-count item - a
#: correct sentence about an incomplete plan, which reads to a user as a broken
#: product.
#:
#: This adds NO rule and grants NO capability. It restates the existing
#: contract more emphatically for a weaker instruction-follower; the tool
#: allowlist, `AgentPlan` validation and grounding are all unchanged, and a plan
#: that ignores this is refused exactly as before.
_PLAN_REMINDER = """
REMINDER - the two rules most often missed:
1. If the question asks for a MEASUREMENT or a VALUE (NDWI, NDVI, NDBI, water,
   vegetation, built-up, change over time), the plan MUST contain an analysis
   step AFTER execute_query. execute_query alone only finds a scene; it
   measures nothing, so a plan with only that step cannot answer the question.
2. Return ONLY the JSON object, shaped {"steps": [ ... ]}. Do not wrap it in a
   "plan" key, do not nest arguments under "parameters", and do not add prose.
"""


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
        """Ask for a plan, re-asking a bounded number of times.

        Two different transient conditions are retried, and neither is papered
        over. The endpoint returns 503 under shared load, and the model returns
        the plan in a differently-shaped envelope from one call to the next; in
        both cases another sample is a reasonable request, and in neither case
        is anything invented - a retry asks the model again, it never edits an
        answer. After `_PLAN_ATTEMPTS` the genuine failure is raised unchanged,
        so a persistent problem still surfaces as a planner failure.
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
        messages = [
            {"role": "system", "content": _system_instruction()},
            {
                "role": "user",
                "content": f"QUESTION\n{question}\n{_PLAN_REMINDER}",
            },
        ]

        # Preferred path: constrained decoding against `AgentPlan`'s schema.
        # Falls through to the plain-JSON path below when the model answers
        # without a tool call, so a model that ignores `tools` is unaffected.
        raw = await self._chat.complete_tool_call(
            messages, tool=_plan_tool(), max_tokens=_PLAN_MAX_TOKENS
        )
        if raw is None:
            raw = await self._chat.complete(
                messages,
                max_tokens=_PLAN_MAX_TOKENS,
                response_format=_plan_response_format(),
            )
        try:
            body = _plan_envelope(json.loads(_extract_json(raw)))
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
        messages = [
            {"role": "system", "content": _SYNTHESIS_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"QUESTION\n{question}\n\n"
                    f"EVIDENCE (cite by id)\n{_render_evidence(evidence)}\n"
                ),
            },
        ]
        raw = await _retry_transient(
            lambda: self._chat.complete(messages, max_tokens=1024)
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
