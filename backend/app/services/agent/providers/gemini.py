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

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.core.logging import get_logger
from app.services.agent.grounding import DraftAnswer
from app.services.agent.planner import AgentPlanner
from app.services.agent.registry import TOOL_REGISTRY, ToolOperation
from app.services.agent.schemas import AgentEvidence, AgentPlan
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.query.schemas import SatQueryIntent

logger = get_logger("agent.planner.gemini")


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


def _plan_response_schema() -> types.Schema:
    """An SDK-compatible schema for the plan - generation hint only.

    Expresses the tool union with ``any_of`` over two concrete branches instead
    of Pydantic's ``discriminator``/``oneOf``, which this SDK rejects. Giving
    the analysis branch no ``intent`` property is deliberate: the model is not
    even offered a field that ``extra="forbid"`` would later reject.
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
    analysis_branch = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "tool": types.Schema(
                type=types.Type.STRING, enum=_tool_names("analysis")
            )
        },
        required=["tool"],
    )

    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "steps": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(any_of=[execute_branch, analysis_branch]),
                min_items=minimum,
                max_items=maximum,
            )
        },
        required=["steps"],
    )


def _tool_catalogue() -> str:
    """Describe the permitted tools, straight from the allowlist.

    Generated from :data:`TOOL_REGISTRY` rather than written out by hand, so the
    model can never be told about a capability the executor would refuse - or
    left ignorant of one it would accept.
    """

    return "\n".join(
        f"- {spec.name}: {spec.description}" for spec in TOOL_REGISTRY.values()
    )


def _system_instruction() -> str:
    """The planning instruction. Deliberately asks for a plan and nothing else.

    It requests no explanation, no justification and no account of how the plan
    was arrived at, because none of that has anywhere to go: ``AgentPlan`` has
    no field for it, and the trace shown to a user records decisions and
    outcomes only.

    It also describes only what a planner needs - the tools and the shape of a
    plan - not the repository's architecture.
    """

    return f"""\
You select which remote-sensing analyses to run for a user's question, and
return ONLY a JSON object matching the provided response schema. No prose, no
explanation, no commentary.

AVAILABLE TOOLS

{_tool_catalogue()}

PLAN RULES

- A plan has 1 to 3 steps.
- The first step MUST be "execute_query"; it appears exactly once. The analysis
  tools interpret its result, so nothing can run before it.
- Do not repeat a tool.
- Choose an analysis tool only when the question calls for it. A request merely
  to find or view imagery needs "execute_query" alone.
- "temporal_ndwi_statistics" compares two Sentinel-2 acquisitions, so use it
  only with a "compare" temporal mode carrying a baseline and a target window.
- "ndwi_statistics" is optical-only; it needs "sentinel-2-optical" among the
  modalities.

EXECUTE_QUERY PARAMETERS

- intent.location_query: the place named by the user, verbatim. NEVER invent or
  output coordinates.
- intent.temporal_mode: "single" (one window), "compare" (a baseline and a
  target window), or "timeseries" (three or more windows).
- intent.time_windows: for "single" a list of exactly one
  {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}; for "timeseries" a
  list of two or more; for "compare" an object with "baseline" and "target".
  Resolve unambiguous relative expressions into explicit ISO ranges.
- intent.modalities: a non-empty list from "sentinel-2-optical" and
  "sentinel-1-sar", with no duplicates. Default to ["sentinel-2-optical"] when
  the user names no sensor.
- intent.task: "visualize", "change_detection" or "object_identification".
- include_imagery: true only when the user asks to see the imagery.
- max_cloud_cover: 0-100, only when the user asks for cloud-free imagery.

RULES

- Use only the tools listed above. Never invent a tool name.
- Emit only the fields described here; extra fields are rejected.
- Extract only what the user's question supports. Never invent a location, a
  date range, a scene, or a measurement.
- You are planning only. You do not run the tools and you never report results.
"""


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

        try:
            response = await client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=question,
                config=self._build_config(),
            )
        except genai_errors.APIError as exc:
            # Status code only - the message may echo request data, and the key
            # must never reach a log.
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

_SYNTHESIS_INSTRUCTION = """\
You write one short, factual answer to the user's question using ONLY the
evidence supplied below, and return it as JSON matching the response schema.
No prose outside the schema, no commentary.

RULES

- Use only the supplied evidence. Never introduce an observation, a place, a
  date, a scene or a measurement that does not appear in it.
- State no number that is not present in the evidence. Quote a value as given,
  or round it; never estimate, extrapolate or infer one.
- Cite the evidence you used in "evidence_refs", using the exact ids shown.
  Cite nothing by returning an empty list - never omit the field.
- When the evidence carries limitations or warnings, say so plainly rather than
  presenting a result as more certain than it is.
- Describe what was measured. Do not claim detection, classification,
  co-registration, alignment, or comparison between individual pixels.
- If the evidence does not answer the question, say that instead of guessing.
- Return only "summary" and "evidence_refs". No other field is accepted.
"""


def _answer_response_schema() -> types.Schema:
    """SDK-compatible schema for a :class:`DraftAnswer`.

    Converted from the existing contract with the SDK's public
    ``Schema.from_json_schema``, so the model is offered exactly the shape the
    parser will accept and nothing else.
    """

    return types.Schema.from_json_schema(
        json_schema=types.JSONSchema(**DraftAnswer.model_json_schema())
    )


def _render_evidence(evidence: AgentEvidence) -> str:
    """Present the evidence as citable lines: ``id | source | content``.

    Only the flattened, citable ``items`` are shown - each with the id the
    answer must cite. The full execution and analysis results are deliberately
    not dumped in: everything a sentence may legitimately quote is already an
    item, and a smaller prompt is a smaller surface for the model to wander off
    into.
    """

    if not evidence.items:
        return "(no evidence was collected)"

    lines = []
    for item in evidence.items:
        if item.measurement is not None:
            content = (
                f"{item.measurement.name} = {item.measurement.value} "
                f"{item.measurement.unit}"
            )
        else:
            content = item.text or ""
        lines.append(f"- {item.id} | {item.source} | {content}")
    return "\n".join(lines)


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

        try:
            response = await client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=self._build_config(),
            )
        except genai_errors.APIError as exc:
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
