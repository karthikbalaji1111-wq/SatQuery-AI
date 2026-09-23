"""The local open-weight provider: a Qwen3-VL model served by Ollama.

All four provider-neutral roles - planning, visual observation, synthesis and
intent parsing - over Ollama's native chat endpoint (``POST /api/chat``), using
plain ``httpx`` exactly as the NVIDIA provider does. No SDK is added.

**Why the native endpoint.** Its ``format`` field takes a JSON Schema, which
Ollama compiles into a decoding grammar: the model cannot emit a token that
leaves the schema. The synthesiser is constrained to ``DraftAnswer``'s schema,
the intent parser to ``SatQueryIntent``'s, and the planner to a grammar
derived from ``AgentPlan``'s (see :func:`_plan_grammar`) - all generated from
the contracts, so they cannot drift. The
grammar is a generation constraint only. Every response is still validated
through the same Pydantic contract the cloud providers use, and the closed tool
allowlist, the executor's own checks and grounding are all unchanged: a model
running on this machine is trusted exactly as little as one running elsewhere.

**Offline by construction.** Requests go to ``LOCAL_AI_BASE_URL`` and nowhere
else, and no credential exists to leak. There is no fallback: when Ollama is
not running or the model is not installed, the run fails with a message saying
so, and is never answered by a cloud provider in its place.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
from enum import Enum
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

logger = logging.getLogger("satquery.agent.local")

#: What a user is told when nothing answers at the local endpoint.
UNAVAILABLE_MESSAGE = (
    "Local AI provider is unavailable. Start Ollama and ensure the selected "
    "Qwen3-VL model is installed."
)

#: What a user is told when Ollama answers but cannot read its installed models.
MODELS_UNREADABLE_MESSAGE = (
    "Ollama is running but cannot read its installed models. If they are "
    "stored on an external drive, check that it is connected, then try again."
)

#: What a user is told when an error answer names no cause this module knows.
_FAILED_MESSAGE = "The local AI provider failed to answer."

#: Formats Ollama accepts as image input.
_SUPPORTED_MEDIA_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/jpg"}
)

#: A connection that cannot be opened in this long means nothing is listening;
#: the generous overall timeout is for the model thinking, not for connecting.
_CONNECT_TIMEOUT_SECONDS = 5.0

#: Planning attempts. The grammar fixes the SHAPE, but ``AgentPlan`` also
#: enforces order (discovery first, no repeated tool) that a grammar cannot
#: express, so one re-ask is a legitimate second sample - never a repair.
_PLAN_ATTEMPTS = 2

#: ``VisualAnswer`` caps its field; truncating would alter what the model said.
_MAX_OBSERVATION_CHARS = 4000

#: How long the model catalog waits to CONNECT to Ollama. Nothing listening on
#: this machine is refused at once; this only bounds a socket that never opens.
_PROBE_CONNECT_SECONDS = 2.0

#: How long it then waits for the model list. Ollama reads its model folder to
#: answer, and a folder on an external hard disk that has gone to sleep waits
#: for the disk to spin up. Measured on the development machine (a USB hard
#: disk; macOS spins disks down after 10 idle minutes): about 4 s. The single
#: 2 s budget this replaces reported a running Ollama as "not running".
_PROBE_READ_SECONDS = 10.0

#: The one planning step every plan starts with.
_DISCOVERY_REF = "#/$defs/ExecuteQueryParams"

#: A restatement of the shape the grammar below enforces, in the user turn. It
#: adds no rule: the shared instruction already says discovery comes first.
_PLAN_SHAPE_NOTE = (
    '\nReturn {"discovery": <the execute_query step>, "analysis": [<up to two '
    'analysis steps>]}. "discovery" is always present; "analysis" lists what to '
    "measure or observe."
)

#: The only keys a plan may arrive in. The grammar forbids any other, but it is
#: a generation constraint, not validation.
_PLAN_SLOTS = frozenset({"discovery", "analysis"})


class ProbeFailure(Enum):
    """Why a REACHABLE Ollama gave no model list.

    Kept apart from ``None`` - nothing listening - because the fix differs:
    starting a service that is already running helps no one. Observed live:
    with the external drive holding the models disconnected, Ollama kept
    running and answered ``/api/tags`` with 500 and every chat with 400.
    """

    MODELS_UNREADABLE = "models_unreadable"


def _plan_grammar() -> dict[str, Any]:
    """The decoding grammar for a plan: discovery first, by construction.

    Derived from ``AgentPlan``'s own JSON Schema, never written by hand. Two
    changes, both expressing rules ``AgentPlan`` already enforces:

    * The step list becomes two slots - ``discovery`` (an ``execute_query``
      step) and ``analysis`` (up to two of the other tools). Observed live, a
      4B model decoding under the plain schema returned
      ``{"steps": [{"tool": "ndwi_statistics"}]}``: every step well-formed,
      discovery missing, so ``AgentPlan`` refused it - identically on every
      re-ask at temperature 0. The rule "exactly one execute_query, first"
      lives in a validator, which a schema-derived grammar cannot see, and
      Ollama ignores ``prefixItems``, the construct that would express it. A
      required ``discovery`` property uses only constructs it does honour.
    * ``tool`` is required in every step, so a step always names itself.

    The provider only ORDERS the two slots into ``steps``; it supplies no value
    of its own, and ``AgentPlan`` validates the result exactly as before.
    """

    schema = AgentPlan.model_json_schema()
    defs = copy.deepcopy(schema["$defs"])
    for definition in defs.values():
        if "tool" in definition.get("properties", {}):
            definition["required"] = sorted({*definition.get("required", []), "tool"})
    steps = schema["properties"]["steps"]
    others = [ref for ref in steps["items"]["oneOf"] if ref["$ref"] != _DISCOVERY_REF]
    return {
        "$defs": defs,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "discovery": {"$ref": _DISCOVERY_REF},
            "analysis": {
                "type": "array",
                "items": {"oneOf": others},
                "maxItems": steps["maxItems"] - 1,
            },
        },
        "required": ["discovery", "analysis"],
    }


def _plan_from_slots(raw: str) -> AgentPlan:
    """Order the two slots into ``steps`` and let ``AgentPlan`` decide.

    Anything outside the two slots is refused, not ignored. ``AgentPlan`` sets
    ``extra="forbid"`` so that a smuggled field is visible rather than absorbed;
    an adapter that silently dropped a top-level ``"steps"`` or ``"evidence"``
    key would undo that one level up.
    """

    try:
        slots = json.loads(raw)
        if not isinstance(slots, dict) or set(slots) != _PLAN_SLOTS:
            raise ValueError("a plan carries exactly a discovery and an analysis slot")
        if not isinstance(slots["discovery"], dict) or not isinstance(
            slots["analysis"], list
        ):
            raise ValueError("discovery is one step and analysis a list of steps")
        steps = [slots["discovery"], *slots["analysis"]]
        return AgentPlan.model_validate({"steps": steps})
    except (ValueError, KeyError, TypeError, ValidationError) as exc:
        raise IntentParsingError(
            "The local model returned a plan that did not match the expected shape."
        ) from exc


class _OllamaChatClient:
    """The one place a request to the local model is made.

    Shared by all four roles so transport, error mapping and the
    never-log-the-body rule exist once. It returns the assistant's text; turning
    that into a plan, an answer or an observation is each role's own business.
    """

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._model = model or settings.local_ai_model

    @property
    def model(self) -> str:
        return self._model

    def _endpoint(self) -> str:
        return f"{self._settings.local_ai_base_url.rstrip('/')}/api/chat"

    async def chat(
        self, messages: list[dict[str, Any]], *, schema: dict[str, Any] | None = None
    ) -> str:
        """POST one chat turn and return the assistant's text.

        ``schema`` constrains decoding to that JSON Schema. Deterministic
        decoding (temperature 0), matching the cloud providers, so a rerun is as
        reproducible as the model allows.
        """

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            # Qwen3-VL is a thinking-capable model. Left to think, it emits
            # hundreds of hidden reasoning tokens per call - tens of seconds on
            # a small machine - before a plan whose shape the grammar already
            # fixes. Off, for stable latency; the answer is still validated.
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": self._settings.local_ai_num_ctx,
            },
        }
        if schema is not None:
            payload["format"] = schema

        try:
            if self._client is not None:
                response = await self._client.post(self._endpoint(), json=payload)
            else:
                timeout = httpx.Timeout(
                    self._settings.local_ai_timeout_seconds,
                    connect=_CONNECT_TIMEOUT_SECONDS,
                )
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self._endpoint(), json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.warning("Local AI provider unreachable (%s)", type(exc).__name__)
            raise UpstreamServiceError(UNAVAILABLE_MESSAGE) from exc
        except httpx.TimeoutException as exc:
            logger.warning("Local model timed out (model=%s)", self._model)
            raise UpstreamServiceError(
                f"The local model {self._model!r} did not answer within "
                f"{self._settings.local_ai_timeout_seconds:.0f} seconds."
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("Local AI transport error (%s)", type(exc).__name__)
            raise UpstreamServiceError(UNAVAILABLE_MESSAGE) from exc

        if response.status_code >= 400:
            raise (
                _http_failure(response, self._model)
                or await self._unexplained_failure()
            )
        _log_timings(response, self._model, self._settings.local_ai_num_ctx)
        return _message_text(response)

    async def _unexplained_failure(self) -> UpstreamServiceError:
        """An error answer that names no cause, told apart by one more question.

        Ollama's own text is never repeated, so it cannot say what went wrong.
        Observed live: while the external drive holding the models was
        disconnected, Ollama refused every chat with 400 in milliseconds, and
        its model list answered 500. That list is what tells "cannot read its
        models" apart from any other failure.
        """

        listed = await installed_models(self._settings, self._client)
        if listed is ProbeFailure.MODELS_UNREADABLE:
            return UpstreamServiceError(MODELS_UNREADABLE_MESSAGE)
        return UpstreamServiceError(_FAILED_MESSAGE)


def _http_failure(
    response: httpx.Response, model: str
) -> UpstreamServiceError | None:
    """Classify an error answer. The body is read to classify, never repeated.

    The message is written here: Ollama's own text is third-party content and,
    like every provider payload, does not reach a response or a log. ``None``
    when the answer names no cause recognised here.
    """

    logger.warning(
        "Local AI provider error (status=%s, model=%s)", response.status_code, model
    )
    try:
        detail = response.text.lower()
    except UnicodeDecodeError:  # pragma: no cover - Ollama answers JSON
        detail = ""
    if response.status_code == 404 or "not found" in detail:
        return UpstreamServiceError(
            f"The local model {model!r} is not installed. Install it with: "
            f"ollama pull {model}"
        )
    if "memory" in detail:
        return UpstreamServiceError(
            f"The local model {model!r} could not be loaded: this machine does "
            "not have enough free memory for it. Close other applications or "
            "select a smaller model."
        )
    return None


def _log_timings(response: httpx.Response, model: str, num_ctx: int) -> None:
    """Record how long the model took - numbers only, never content.

    Ollama reports durations in nanoseconds. A prompt that fills the context
    window is logged as a warning, because Ollama truncates it from the start.
    """

    try:
        body = response.json()
    except ValueError:
        return
    if not isinstance(body, dict):
        return

    def seconds(key: str) -> float:
        value = body.get(key)
        return value / 1e9 if isinstance(value, (int, float)) else 0.0

    prompt_tokens = body.get("prompt_eval_count")
    logger.info(
        "Local model answered (model=%s, load=%.1fs, prompt_tokens=%s, "
        "output_tokens=%s, total=%.1fs)",
        model,
        seconds("load_duration"),
        prompt_tokens,
        body.get("eval_count"),
        seconds("total_duration"),
    )
    if isinstance(prompt_tokens, int) and prompt_tokens >= num_ctx:
        logger.warning(
            "Local prompt filled the %d-token context window; the start may "
            "have been truncated. Raise SATQUERY_LOCAL_AI_NUM_CTX.",
            num_ctx,
        )


def _message_text(response: httpx.Response) -> str:
    """The assistant's text from one chat response, or an explicit failure."""

    try:
        body = response.json()
    except ValueError as exc:
        raise IntentParsingError(
            "The local model returned a response that was not JSON."
        ) from exc
    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise IntentParsingError("The local model returned an empty answer.")
    return content.strip()


# --------------------------------------------------------------------------- #
# The four roles
# --------------------------------------------------------------------------- #


class LocalAgentPlanner(AgentPlanner):
    """Plans with the shared instruction, constrained to ``AgentPlan``'s schema."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _OllamaChatClient(settings or get_settings(), client, model)

    async def plan(self, question: str) -> AgentPlan:
        last: IntentParsingError | None = None
        for _ in range(_PLAN_ATTEMPTS):
            try:
                return await self._plan_once(question)
            except IntentParsingError as exc:
                last = exc
        assert last is not None
        raise last

    async def _plan_once(self, question: str) -> AgentPlan:
        raw = await self._chat.chat(
            [
                {"role": "system", "content": _system_instruction()},
                {"role": "user", "content": f"QUESTION\n{question}{_PLAN_SHAPE_NOTE}"},
            ],
            schema=_plan_grammar(),
        )
        return _plan_from_slots(raw)


class LocalAnswerSynthesizer(AnswerSynthesizer):
    """Describes the evidence, constrained to ``DraftAnswer``'s schema.

    Grounding still runs afterwards, outside this class: a number the local
    model invents is refused by exactly the same check as one a cloud model
    invents.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _OllamaChatClient(settings or get_settings(), client, model)

    async def synthesize(
        self, question: str, evidence: AgentEvidence
    ) -> DraftAnswer:
        raw = await self._chat.chat(
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
            schema=DraftAnswer.model_json_schema(),
        )
        try:
            return DraftAnswer.model_validate_json(raw)
        except ValidationError as exc:
            raise IntentParsingError(
                "The local model returned an answer that did not match the "
                "expected shape."
            ) from exc


class LocalVisualAnalyst(VisualAnalyst):
    """Looks at the exact PNG the query pipeline retrieved.

    The bytes it is handed are sent as they are - base64 in Ollama's ``images``
    field - and nothing is re-fetched, re-encoded or described in their place.
    """

    provider_name = "local"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _OllamaChatClient(settings or get_settings(), client, model)

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self._chat.model

    async def observe(
        self, *, question: str, image: bytes, media_type: str
    ) -> VisualAnswer:
        if media_type not in _SUPPORTED_MEDIA_TYPES:
            raise UpstreamServiceError(
                f"The local provider cannot accept {media_type!r}; it accepts "
                "PNG or JPEG."
            )
        raw = await self._chat.chat(
            [
                {"role": "system", "content": _VISUAL_INSTRUCTION},
                {
                    "role": "user",
                    "content": f"QUESTION\n{question}",
                    "images": [base64.b64encode(image).decode("ascii")],
                },
            ]
        )
        if len(raw) > _MAX_OBSERVATION_CHARS:
            raise IntentParsingError(
                "The local model returned an observation longer than the "
                "contract allows."
            )
        return VisualAnswer(answer=raw)


class LocalIntentParser(IntentParser):
    """Extracts a ``SatQueryIntent``, constrained to its schema."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        self._chat = _OllamaChatClient(settings or get_settings(), client, model)

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        raw = await self._chat.chat(
            [
                {"role": "system", "content": _SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            schema=SatQueryIntent.model_json_schema(),
        )
        try:
            return SatQueryIntent.model_validate_json(raw)
        except ValidationError as exc:
            raise IntentParsingError(
                "The local model returned an intent that did not match the "
                "expected shape."
            ) from exc


# --------------------------------------------------------------------------- #
# Discovery, for the model catalog
# --------------------------------------------------------------------------- #


async def installed_models(
    settings: Settings, client: httpx.AsyncClient | None = None
) -> frozenset[str] | ProbeFailure | None:
    """Model names Ollama reports as installed.

    ``None`` when nothing answers: Ollama is not running.
    :attr:`ProbeFailure.MODELS_UNREADABLE` when it answers with no usable list,
    or with none in time. One read-only ``GET /api/tags``: the local provider
    is the one case where reachability is cheap to ask and actionable to
    report - the service runs on this machine, and each outcome has a
    one-line fix.
    """

    url = f"{settings.local_ai_base_url.rstrip('/')}/api/tags"
    try:
        if client is not None:
            response = await client.get(url)
        else:
            timeout = httpx.Timeout(
                _PROBE_READ_SECONDS, connect=_PROBE_CONNECT_SECONDS
            )
            async with httpx.AsyncClient(timeout=timeout) as probe:
                response = await probe.get(url)
    except httpx.ConnectTimeout:
        return None
    except httpx.TimeoutException:
        # Connected, then no answer in time: running, but not listing.
        return ProbeFailure.MODELS_UNREADABLE
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return ProbeFailure.MODELS_UNREADABLE
    try:
        listed = response.json().get("models", [])
    except (ValueError, AttributeError):
        return ProbeFailure.MODELS_UNREADABLE
    names: set[str] = set()
    for entry in listed if isinstance(listed, list) else []:
        if isinstance(entry, dict):
            names.update(
                value
                for value in (entry.get("name"), entry.get("model"))
                if isinstance(value, str)
            )
    return frozenset(names)
