"""Gemini-backed planner and answer synthesizer.

The only SDK-aware module in the agent package.

Mirrors the provider pattern already established by
``app.services.ai.parser.GeminiIntentParser``: a lazily-created client so the
service constructs without credentials, an injectable stub for tests,
structured output against an existing Pydantic model, deterministic decoding,
and provider errors normalised into the repository's own error types.

That parser is **not** reused or subclassed. It is hardcoded to produce a
``SatQueryIntent``, so bending it to produce an ``AgentPlan`` would mean either
editing it for this layer's convenience or making the agent depend on the AI
package's internals. A thin local adapter keeps both boundaries intact; the
cost is a few lines of similar-looking plumbing, which is the cheaper trade.

**The model is never authoritative.** Whatever it returns is parsed through
``AgentPlan`` - the closed discriminated union the rest of the system uses - so
an unrecognised tool, a smuggled field, a planner-supplied ``limit`` or a
malformed intent all fail validation here rather than reaching the executor.

Why a provider-local generation schema exists
---------------------------------------------
google-genai 2.20.0 **cannot** translate ``AgentPlan`` directly: Pydantic emits
``discriminator`` and ``oneOf`` for a discriminated union, and the SDK's own
``Schema`` model forbids both, so a live request would fail while
``GenerateContentConfig(response_schema=AgentPlan)`` constructs happily - it
stores the model and translates later.

So the request carries an SDK-compatible ``types.Schema`` built here, using
``any_of`` to express the same union without those keywords. It is a
*generation hint only*. It is derived from the existing contracts rather than
hand-written - tool names come from :data:`TOOL_REGISTRY`, the intent shape from
``SatQueryIntent``, the step bounds from ``AgentPlan`` - so it cannot drift from
what the executor will accept. **The discriminated union remains the sole
validation authority**, applied to the response immediately below.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
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
from app.services.agent.registry import TOOL_REGISTRY, ToolOperation
from app.services.agent.schemas import AgentEvidence, AgentPlan
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.agent.visual import VisualAnalyst, VisualAnswer
from app.services.analysis.indices import SPECTRAL_INDICES
from app.services.query.schemas import SatQueryIntent

logger = get_logger("agent.gemini")


# --------------------------------------------------------------------------- #
# Transient-failure handling
#
# Measured against the live endpoint (2026-09): the free tier allows 5 requests
# per minute per model per project, and one agent run costs two or three calls
# (plan, answer, and optionally a visual observation). Two runs inside a minute
# therefore exhaust the quota, and the endpoint also returns 503/504 under load
# - a trivial two-word prompt was observed taking 27 s and then failing.
#
# So a failure here is usually TEMPORARY, and the two temporary kinds need
# different handling:
#
#   * 500/502/503/504 - overload. A second attempt moments later often works,
#     so a short bounded retry is worth its latency.
#   * 429 - quota. The server states exactly how long to wait, and it is
#     typically tens of seconds. Sleeping that long inside a request would turn
#     a fast honest failure into a hung one, so the delay is REPORTED rather
#     than waited out.
#
# Nothing here fabricates a success, and nothing retries indefinitely: when the
# budget is spent the real failure is raised, classified so the caller can say
# which of the two it was.
# --------------------------------------------------------------------------- #

#: HTTP statuses worth attempting again. A 4xx other than 429 is the request's
#: own fault and repeating it would only waste quota.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Total attempts, retries included. Three is the point where a second
#: independent overload becomes unlikely without materially extending latency.
_MAX_ATTEMPTS = 3

#: The most this provider will ever spend sleeping between attempts, across the
#: whole call. A request that cannot be served inside this budget is reported,
#: not waited on - the caller has its own deadline and a user is watching.
_MAX_RETRY_WAIT_SECONDS = 8.0

#: Backoff used when the server names no delay of its own.
_BACKOFF_SECONDS = (0.5, 1.5)

#: ``retryDelay`` as google.rpc.RetryInfo writes it, e.g. ``"36.36s"``.
_RETRY_DELAY = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)s\s*$")


def _retry_delay_seconds(exc: genai_errors.APIError) -> float | None:
    """The delay the SERVER asked for, or ``None`` if it named none.

    Only the number is taken. The upstream error body is never surfaced or
    logged as text: it is third-party content echoing an unknown amount of the
    request, and this repository's rule is that provider payloads do not reach
    responses or logs. A duration is safe; the prose around it is not.
    """

    details: Any = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    error = details.get("error")
    entries = error.get("details") if isinstance(error, dict) else None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not str(entry.get("@type", "")).endswith("RetryInfo"):
            continue
        match = _RETRY_DELAY.match(str(entry.get("retryDelay", "")))
        if match is not None:
            return float(match.group(1))
    return None


def _upstream_failure(exc: genai_errors.APIError) -> UpstreamServiceError:
    """Translate a provider error into one this system is willing to show.

    A rate limit is reported under its own ``rate_limited`` code rather than
    the generic ``upstream_error``, because the two call for different actions:
    a quota exhaustion clears on its own in a stated number of seconds, while
    an outage does not. Collapsing them told every caller the same unhelpful
    thing.

    The message is written HERE, from the status code alone. No upstream text
    is ever passed through.
    """

    code = getattr(exc, "code", None)
    if code == 429:
        delay = _retry_delay_seconds(exc)
        # A delay under a second is real but not worth quoting: rendering it
        # with ":.0f" produced "Please retry in about 0 seconds", which reads as
        # a bug rather than as guidance. Observed live - Gemini does return
        # "0s". Anything below a second is therefore reported as "shortly",
        # which is what a sub-second wait actually means to a reader.
        wait = (
            f" Please retry in about {delay:.0f} seconds."
            if delay is not None and delay >= 1.0
            else " Please retry shortly."
        )
        error = UpstreamServiceError(
            "The language-model service is rate limited (its request quota is "
            "temporarily exhausted)." + wait,
            code="rate_limited",
        )
        # An optional hint, read reflectively by the agent service so that no
        # provider is REQUIRED to supply one. Absent means "unknown", never
        # "zero".
        error.retry_after_seconds = delay  # type: ignore[attr-defined]
        return error
    return UpstreamServiceError("The language-model service is unavailable.")


async def _generate(
    client: Any,
    *,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig,
    role: str,
) -> Any:
    """Issue one generation request, retrying only what is worth retrying.

    Shared by all three roles so the retry policy, the failure classification
    and the logging exist once. Three copies of this block drifted apart
    before: a synthesis failure was logged under the planner's logger name,
    which made a live incident read as the wrong stage failing.

    ``role`` names the stage for the log only; it never reaches the request.
    """

    budget = _MAX_RETRY_WAIT_SECONDS

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            # Status code only - the message may echo request data, and the key
            # must never reach a log.
            logger.warning(
                "Gemini API error (role=%s, status=%s, attempt=%d/%d)",
                role,
                code,
                attempt,
                _MAX_ATTEMPTS,
            )
            if attempt == _MAX_ATTEMPTS or code not in _RETRYABLE_STATUS:
                raise _upstream_failure(exc) from exc
            delay = _retry_delay_seconds(exc)
            if delay is None:
                if code == 429:
                    # Quota, and the server named no delay. The short backoff
                    # below is calibrated for OVERLOAD (500/502/503/504), where
                    # a moment later often works. A quota window is seconds to
                    # minutes, so retrying inside two seconds cannot clear it -
                    # it only spends two more of the very units that ran out.
                    # Measured: three attempts in 2.01s, tripling the cost of a
                    # request that was always going to fail. Report it instead,
                    # with the same rate-limited classification the caller
                    # already distinguishes from an outage.
                    raise _upstream_failure(exc) from exc
                delay = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)]
            if delay > budget:
                # The server wants longer than this request can honestly wait.
                # Report the real failure now, with the delay it named, rather
                # than holding the connection open or pretending to succeed.
                raise _upstream_failure(exc) from exc
            budget -= delay
            await asyncio.sleep(delay)
        except (TimeoutError, ConnectionError, OSError) as exc:
            logger.warning(
                "Gemini transport error (role=%s, kind=%s, attempt=%d/%d)",
                role,
                type(exc).__name__,
                attempt,
                _MAX_ATTEMPTS,
            )
            if attempt == _MAX_ATTEMPTS:
                raise UpstreamServiceError(
                    "The language-model service timed out."
                ) from exc
            delay = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)]
            if delay > budget:
                raise UpstreamServiceError(
                    "The language-model service timed out."
                ) from exc
            budget -= delay
            await asyncio.sleep(delay)
        except Exception as exc:  # unknown SDK/transport failure - never leak details
            # Deliberately NOT retried. An unrecognised failure is as likely to
            # be a bug here as a blip there, and repeating it would multiply a
            # side effect nobody has characterised.
            logger.warning(
                "Unexpected Gemini failure (role=%s, kind=%s)",
                role,
                type(exc).__name__,
            )
            raise UpstreamServiceError("The language-model service failed.") from exc

    # Unreachable: the loop either returns or raises on its final attempt.
    raise UpstreamServiceError(  # pragma: no cover
        "The language-model service is unavailable."
    )


def _tool_names(operation: ToolOperation) -> list[str]:
    """Permitted tool names for one operation kind, straight from the allowlist."""

    return [
        spec.name for spec in TOOL_REGISTRY.values() if spec.operation == operation
    ]


def _step_bounds() -> tuple[int, int]:
    """Plan length bounds, read off ``AgentPlan`` so the two cannot disagree."""

    metadata = AgentPlan.model_fields["steps"].metadata
    minimum = next(
        (m.min_length for m in metadata if hasattr(m, "min_length")), 1
    )
    maximum = next(
        (m.max_length for m in metadata if hasattr(m, "max_length")), 3
    )
    return minimum, maximum


def _intent_schema() -> types.Schema:
    """The intent shape, converted from the existing ``SatQueryIntent``.

    Uses the SDK's public ``Schema.from_json_schema`` rather than a private
    transformer, and derives from the contract rather than restating it, so the
    model is never offered an intent shape the validator would reject.
    """

    return types.Schema.from_json_schema(
        json_schema=types.JSONSchema(**SatQueryIntent.model_json_schema())
    )


#: Derived from the contracts so the generation hint cannot drift from what
#: the executor will accept.
_INDEX_TOOL = "spectral_indices"
_INDEX_KEYS = tuple(sorted(SPECTRAL_INDICES))


def _plan_response_schema() -> types.Schema:
    """An SDK-compatible schema for the plan - generation hint only.

    Expresses the tool union with ``any_of`` over three concrete branches
    instead of Pydantic's ``discriminator``/``oneOf``, which this SDK rejects.
    Giving the analysis branch no ``intent`` property is deliberate: the model
    is not even offered a field that ``extra="forbid"`` would later reject.

    The visual branch offers ``question`` and nothing else - no scene, no asset,
    no URL. The model is never shown a way to choose which image it looks at,
    because it does not get to: the executor selects the image from the
    validated execution result.
    """

    minimum, maximum = _step_bounds()

    execute_branch = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "tool": types.Schema(
                type=types.Type.STRING, enum=_tool_names("discovery")
            ),
            "intent": _intent_schema(),
            "include_imagery": types.Schema(type=types.Type.BOOLEAN),
            "max_cloud_cover": types.Schema(type=types.Type.NUMBER),
        },
        required=["tool", "intent"],
    )
    # The parameterless analysis tools. Offering no property but the name is
    # deliberate: the model is not shown a field the contract would reject.
    analysis_branch = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "tool": types.Schema(
                type=types.Type.STRING,
                enum=[
                    name
                    for name in _tool_names("analysis")
                    if name != _INDEX_TOOL
                ],
            )
        },
        required=["tool"],
    )
    # The one analysis tool that takes a parameter. WHICH index answers a
    # question is a planning decision; how each index is computed is not, so
    # the branch offers the index names and nothing else - no band, no
    # threshold, no scene.
    index_branch = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "tool": types.Schema(type=types.Type.STRING, enum=[_INDEX_TOOL]),
            "indices": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.STRING, enum=list(_INDEX_KEYS)
                ),
                min_items=1,
                max_items=len(_INDEX_KEYS),
            ),
        },
        required=["tool", "indices"],
    )
    visual_branch = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "tool": types.Schema(type=types.Type.STRING, enum=_tool_names("visual")),
            "question": types.Schema(type=types.Type.STRING),
        },
        required=["tool", "question"],
    )

    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "steps": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    any_of=[
                        execute_branch,
                        index_branch,
                        analysis_branch,
                        visual_branch,
                    ]
                ),
                min_items=minimum,
                max_items=maximum,
            )
        },
        required=["steps"],
    )


class GeminiAgentPlanner(AgentPlanner):
    """Real :class:`AgentPlanner` backed by the Google Gemini API.

    The client is created lazily so the planner can be constructed without
    credentials; tests inject a stub client and no live call is ever made.
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
                "GEMINI_API_KEY is not configured; agent planning is unavailable."
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
            system_instruction=_system_instruction(),
            response_mime_type="application/json",
            # SDK-compatible generation hint, derived from the contracts.
            # AgentPlan itself cannot be sent - see the module docstring - but
            # it remains the validation authority for whatever comes back.
            response_schema=_plan_response_schema(),
            temperature=0.0,
            candidate_count=1,
        )

    async def plan(self, question: str) -> AgentPlan:
        """Ask the model for a plan, then validate it before returning."""

        client = self._get_client()

        response = await _generate(
            client,
            model=self._settings.gemini_model,
            contents=question,
            config=self._build_config(),
            role="planning",
        )

        raw = getattr(response, "text", None)
        # ``.text`` is documented as ``str | None``, but a malformed or proxied
        # response can carry anything. Checking the type keeps a surprising
        # payload a handled failure rather than an AttributeError crash.
        if not isinstance(raw, str) or not raw.strip():
            raise IntentParsingError(
                "The language model returned an empty or unusable plan."
            )

        try:
            plan = AgentPlan.model_validate_json(raw)
        except ValidationError as exc:
            # Includes an unknown tool, a smuggled field, a planner-supplied
            # limit, and any malformed intent - all rejected by the contract.
            logger.info(
                "Gemini output failed AgentPlan validation (%d error(s))",
                exc.error_count(),
            )
            raise IntentParsingError(
                "Could not extract a valid analysis plan from the request."
            ) from exc

        logger.info(
            "Gemini planned %d step(s) (model=%s): %s",
            len(plan.steps),
            self._settings.gemini_model,
            ", ".join(step.tool for step in plan.steps),
        )
        return plan


# =========================================================================== #
# Answer synthesis
#
# Same provider pattern as the planner above: lazy client, injectable stub,
# structured output through the SDK's public schema conversion, deterministic
# decoding, and provider errors normalised into the repository's own types.
#
# The schema here needs no union - a DraftAnswer is one flat object - so it
# converts cleanly and the Commit 4 discriminator/oneOf problem cannot recur.
# A test asserts that anyway.
# =========================================================================== #

def _answer_response_schema() -> types.Schema:
    """SDK-compatible schema for a :class:`DraftAnswer`.

    Converted from the existing contract with the SDK's public
    ``Schema.from_json_schema``, so the model is offered exactly the shape the
    parser will accept and nothing else.
    """

    return types.Schema.from_json_schema(
        json_schema=types.JSONSchema(**DraftAnswer.model_json_schema())
    )


class GeminiAnswerSynthesizer(AnswerSynthesizer):
    """Real :class:`AnswerSynthesizer` backed by the Google Gemini API.

    The prompt asks the model to stay inside the evidence, but the prompt is not
    the control: the response is parsed through ``DraftAnswer``, and the Commit 3
    grounding validator - which runs outside this class - is what actually
    establishes that every stated number is traceable.
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
                "GEMINI_API_KEY is not configured; answer synthesis is "
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
            system_instruction=_SYNTHESIS_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_answer_response_schema(),
            temperature=0.0,
            candidate_count=1,
        )

    async def synthesize(
        self, question: str, evidence: AgentEvidence
    ) -> DraftAnswer:
        """Ask the model for an answer, then validate it before returning."""

        client = self._get_client()
        prompt = (
            f"QUESTION\n{question}\n\n"
            f"EVIDENCE (cite by id)\n{_render_evidence(evidence)}\n"
        )

        response = await _generate(
            client,
            model=self._settings.gemini_model,
            contents=prompt,
            config=self._build_config(),
            role="synthesis",
        )

        raw = getattr(response, "text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise IntentParsingError(
                "The language model returned an empty or unusable answer."
            )

        try:
            answer = DraftAnswer.model_validate_json(raw)
        except ValidationError as exc:
            # Includes a missing summary, a missing or malformed evidence_refs,
            # and any smuggled reasoning/confidence/tool-call field.
            logger.info(
                "Gemini output failed DraftAnswer validation (%d error(s))",
                exc.error_count(),
            )
            raise IntentParsingError(
                "Could not extract a valid answer from the language model."
            ) from exc

        logger.info(
            "Gemini synthesised an answer (model=%s, %d citation(s))",
            self._settings.gemini_model,
            len(answer.evidence_refs),
        )
        return answer


# --------------------------------------------------------------------------- #
# Phase 18.1 - visual analyst
# --------------------------------------------------------------------------- #

#: What the model is asked to do with the picture, and what it must not do.
#:
#: The prompt is not the control - nothing downstream trusts it. The real
#: guarantees are structural: the answer is parsed through ``VisualAnswer``, it
#: becomes ``source="model"`` evidence, and ``grounding._allowed_values``
#: refuses to let that evidence authorise any number. The instruction exists to
#: make the honest answer the easy one, not to enforce anything.
def _visual_response_schema() -> types.Schema:
    """SDK-compatible schema for a :class:`VisualAnswer`.

    Converted from the contract with the SDK's public ``Schema.from_json_schema``
    - the same conversion the answer path uses - so the model is offered exactly
    the shape the parser accepts, and no field that could masquerade as a
    measurement.
    """

    return types.Schema.from_json_schema(
        json_schema=types.JSONSchema(**VisualAnswer.model_json_schema())
    )


class GeminiVisualAnalyst(VisualAnalyst):
    """Real :class:`VisualAnalyst` backed by Gemini's multimodal input.

    Sends exactly two parts: the PNG bytes the query pipeline already retrieved,
    and the question. It sends **no** georeferencing and **no** measurements -
    no CRS, no affine, no corners, no scene id, no NDWI statistics, no bbox. A
    model told "the mean NDWI is 0.146" would answer from that number rather
    than from the picture, and the point of this tool is an observation that is
    independent of the deterministic evidence, so the two can be read against
    each other.

    It cannot fetch anything. The bytes arrive as an argument; there is no URL,
    path or catalog handle in this class.
    """

    provider_name = "gemini"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: object | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

    @property
    def model_name(self) -> str:
        """The configured model, so a claim is attributable to a version."""

        return self._settings.gemini_model

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        if not self._settings.gemini_api_key:
            raise UpstreamServiceError(
                "GEMINI_API_KEY is not configured; visual analysis is unavailable."
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
            system_instruction=_VISUAL_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_visual_response_schema(),
            temperature=0.0,
            candidate_count=1,
        )

    async def observe(
        self, *, question: str, image: bytes, media_type: str
    ) -> VisualAnswer:
        """Send the image and the question; parse what comes back."""

        client = self._get_client()
        # Exactly one image part, from the bytes handed in. Never re-encoded,
        # never re-fetched, and never logged.
        parts = [
            types.Part.from_bytes(data=image, mime_type=media_type),
            types.Part.from_text(text=f"QUESTION\n{question}"),
        ]

        response = await _generate(
            client,
            model=self._settings.gemini_model,
            contents=[types.Content(role="user", parts=parts)],
            config=self._build_config(),
            role="visual",
        )

        raw = getattr(response, "text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise IntentParsingError(
                "The language model returned an empty or unusable observation."
            )
        try:
            answer = VisualAnswer.model_validate_json(raw)
        except ValidationError as exc:
            raise IntentParsingError(
                "The language model returned an observation that did not match "
                "the expected shape."
            ) from exc

        logger.info(
            "Gemini visual observation (model=%s, %d image byte(s) sent)",
            self._settings.gemini_model,
            len(image),
        )
        return answer
