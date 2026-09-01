"""Agent orchestration contracts.

    question -> AgentPlan -> (deterministic tools) -> AgentEvidence -> AgentResult

This module defines the *shapes* only. There is no planner, no executor, no
provider and no route here, and nothing in this file performs I/O.

Several properties are enforced structurally rather than by convention, because
a later executor and grounding validator will rely on them:

1. **The tool set is closed.** ``ToolCall`` is a discriminated union over three
   literal tool names, so an unknown tool fails validation before any code can
   dispatch on it.
2. **Unexpected fields are refused, not dropped.** Every contract model sets
   ``extra="forbid"``. A planner that smuggles ``code``, ``limit`` or an asset
   href gets a validation error rather than a silently sanitised object - this
   is a safety boundary, so an attempt should be visible, not absorbed.
3. **A plan carries no executable content.** Tool parameters are typed domain
   models - never code, a command, a path or a URL.
4. **A trace step cannot lie about which tool ran.** :class:`AgentToolStep`
   holds no ``tool`` field at all; the tool name is *derived* from the
   validated parameters, so a step naming one tool while carrying another's
   parameters is unrepresentable rather than merely rejected.
5. **The trace carries no reasoning.** Chain-of-thought is not modelled here
   under that name or any synonym, so it cannot be stored, returned or rendered.
6. **An ``ok`` result cannot exist without its answer.**

The model-facing surface is deliberately narrower than the server's own:
``ExecuteQueryParams`` exposes only genuine analytical decisions and omits
``limit``, which is a server resource budget rather than something a question
implies. The executor injects the configured limit when it builds the real
``QueryExecutionRequest``.

``SatQueryIntent``, ``QueryExecutionResult``, ``AnalysisResult`` and
``Measurement`` are reused verbatim; no validator is duplicated.

Dependency direction is ``api -> agent -> {analysis, query} -> satellite``.
Nothing in ``analysis``, ``query``, ``satellite`` or ``core`` may import this
package.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.analysis.schemas import AnalysisResult, Measurement
from app.services.query.schemas import QueryExecutionResult, SatQueryIntent

#: The closed set of tools a planner may select. Adding a name here is the
#: deliberate act of granting a model access to a capability - which is why a
#: future remote-sensing model tool is NOT listed yet.
ToolName = Literal["execute_query", "ndwi_statistics", "temporal_ndwi_statistics"]

#: Outcome of one planned step, as observed by the executor. ``rejected`` is a
#: step the executor declined (e.g. its precondition was not met); ``failed`` is
#: one that ran and raised; ``skipped`` is one never reached.
ToolStepStatus = Literal["ok", "rejected", "failed", "skipped"]

#: ``ok``                     - a grounded answer was produced.
#: ``planner_unavailable``    - no plan; nothing executed.
#: ``synthesis_unavailable``  - tools ran, the answer could not be generated.
#: ``answer_withheld``        - an answer was generated but failed validation.
#: In the last three the evidence is still returned: the deterministic result
#: never depends on a language-model provider being reachable.
AgentStatus = Literal[
    "ok", "planner_unavailable", "synthesis_unavailable", "answer_withheld"
]

#: Where one piece of evidence came from. ``model`` is RESERVED for a future
#: remote-sensing model tool and is unused today; it exists so that adding such
#: a tool later is a registration, not a contract change.
EvidenceSource = Literal[
    "execution", "ndwi", "temporal_ndwi", "compatibility", "model"
]

#: Result of each mechanical check applied to a generated answer.
#: ``not_run`` distinguishes "checked and passed" from "never checked", so a
#: missing answer can never read as a validated one.
ValidationOutcome = Literal["pass", "fail", "not_run"]

_MAX_PLAN_STEPS = 3


class _StrictModel(BaseModel):
    """Base for every agent contract: unexpected input is an error.

    ``extra="forbid"`` is the point. These models sit on the boundary where
    untrusted planner output is parsed, and silently discarding an unrecognised
    field would turn an attempt to smuggle one into a no-op the system never
    sees. Refusing makes the attempt observable.
    """

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Tool parameters
#
# One model per tool. The discriminator field is the tool name itself, so the
# parameter type and the tool identity can never disagree.
# --------------------------------------------------------------------------- #


class ExecuteQueryParams(_StrictModel):
    """Discovery + selection + optional bounded imagery.

    Exposes only decisions a question can actually imply:

    * ``intent``          - where, when, which sensor, what kind of answer.
      The existing :class:`SatQueryIntent` is reused whole, so its temporal-mode,
      window-shape and modality validators all apply unchanged.
    * ``include_imagery`` - whether the user also wants to *see* the scene. The
      model never receives the image; this is a UI-directed request.
    * ``max_cloud_cover`` - a real filter ("cloud-free imagery of ...").

    ``limit`` is deliberately absent. Nothing in a natural-language question
    maps to "return 10 versus 100 candidate scenes", deterministic selection
    picks exactly one scene regardless, and it is a server resource budget. The
    executor supplies it.
    """

    tool: Literal["execute_query"] = "execute_query"
    intent: SatQueryIntent
    include_imagery: bool = False
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)


class NdwiParams(_StrictModel):
    """Single-scene Sentinel-2 NDWI statistics.

    Deliberately parameterless. The index threshold, band choice and raw-DN
    decision are scientific constants owned by the engine, not knobs a language
    model may turn.
    """

    tool: Literal["ndwi_statistics"] = "ndwi_statistics"


class TemporalNdwiParams(_StrictModel):
    """Temporal NDWI Statistics for one deterministic Sentinel-2 pair.

    Also parameterless: pair selection, suppression rules and the compatibility
    report are deterministic and are not open to negotiation.
    """

    tool: Literal["temporal_ndwi_statistics"] = "temporal_ndwi_statistics"


#: A single validated tool call. Discriminated on ``tool``, so an unrecognised
#: name is a validation error rather than a runtime dispatch problem.
ToolCall = Annotated[
    ExecuteQueryParams | NdwiParams | TemporalNdwiParams,
    Field(discriminator="tool"),
]


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


class AgentPlan(_StrictModel):
    """What the planner proposes to run, after validation.

    The shape rules encode real preconditions rather than taste: the analysis
    tools interpret a ``QueryExecutionResult``, so exactly one ``execute_query``
    must come first; repeating an analysis tool would duplicate work the
    executor coalesces anyway; and the step cap means a planner cannot loop.

    Note the deliberate limit of this validation: it enforces *structural*
    preconditions, not *semantic* ones. A plan pairing
    ``temporal_ndwi_statistics`` with a single-window intent is well-formed and
    will validate; the executor reports it as a warning rather than the plan
    layer duplicating domain logic that already lives in the analysis service.
    """

    steps: list[ToolCall] = Field(min_length=1, max_length=_MAX_PLAN_STEPS)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        tools = [step.tool for step in self.steps]

        if tools.count("execute_query") != 1:
            raise ValueError(
                "a plan must contain exactly one 'execute_query' step"
            )
        if tools[0] != "execute_query":
            raise ValueError(
                "'execute_query' must be the first step; the analysis tools "
                "interpret its result"
            )
        if len(set(tools)) != len(tools):
            raise ValueError("a plan must not repeat a tool")
        return self


# --------------------------------------------------------------------------- #
# Trace - observable decisions and outcomes ONLY
# --------------------------------------------------------------------------- #


class AgentToolStep(_StrictModel):
    """One executed (or refused) step.

    There is no ``tool`` field. The tool name is derived from ``parameters``,
    which is the POST-validation tool call, so the trace can only ever name the
    tool whose parameters it actually carries. A step claiming ``execute_query``
    while holding NDWI parameters is not rejected - it cannot be expressed.

    ``parameters`` shows what actually ran, including any server-supplied
    defaults, so it records the *effective* call rather than only the fields a
    planner chose to send.
    """

    status: ToolStepStatus
    parameters: ToolCall
    #: Why the executor declined a validated step.
    rejection_reason: str | None = None
    #: Message from a handled ``AppError``; never a stack trace.
    error_message: str | None = None

    @property
    def tool(self) -> ToolName:
        """The authoritative tool name, from the validated parameters."""

        return self.parameters.tool


class AnswerValidation(_StrictModel):
    """Outcome of the mechanical checks applied to a generated answer.

    These are containment, not proof: they catch ungrounded numbers, forbidden
    vocabulary and dangling evidence references. They cannot establish that a
    qualitative statement follows from the evidence.
    """

    numeric_grounding: ValidationOutcome = "not_run"
    forbidden_terms: ValidationOutcome = "not_run"
    evidence_refs: ValidationOutcome = "not_run"


class AgentTrace(_StrictModel):
    """What the system decided and what happened - never why it thought so.

    There is deliberately no field for chain-of-thought, rationale or any
    synonym. Everything here is externally observable: the validated plan, the
    per-step outcome, the evidence referenced, and the answer checks.
    """

    plan: AgentPlan | None = None
    steps: list[AgentToolStep] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    answer_validation: AnswerValidation | None = None


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class EvidenceItem(_StrictModel):
    """One citable fact: a measurement, or a piece of qualifying text.

    ``id`` is a stable reference (e.g. ``"ndwi.ndwi_mean"``) so a generated
    answer can point at what it used and the reference can be checked. Ids are
    unique within an :class:`AgentEvidence` - grounding resolves them as keys.

    ``produced_by`` names what computed the item (a deterministic engine
    function today; a model identifier and version when a remote-sensing model
    is added later). It is optional so existing deterministic evidence stays
    valid, and it is the reason ``source="model"`` alone is not the whole
    provenance story.

    **Known limitation.** ``Measurement.value`` is a ``float``, so this shape
    carries *scalar numeric* evidence only. It cannot represent a categorical
    model output such as a land-cover class, nor an attached confidence. A
    future remote-sensing classifier will therefore need a contract extension,
    not merely a registration - that is recorded here rather than glossed over.
    """

    id: str = Field(min_length=1, max_length=200)
    source: EvidenceSource
    measurement: Measurement | None = None
    #: Warning or limitation text, for evidence that is not a number.
    text: str | None = None
    #: What computed this item - engine function, or model id/version.
    produced_by: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _require_content(self) -> Self:
        if self.measurement is None and self.text is None:
            raise ValueError(
                "an evidence item must carry a measurement or text; an empty "
                "item cites nothing"
            )
        return self


class AgentEvidence(_StrictModel):
    """Everything the deterministic layer established, in one place.

    ``execution`` and ``analysis`` are the existing results carried verbatim;
    the compatibility report already travels inside
    ``analysis.temporal_comparison``, so it needs no separate field. ``items``
    is the flattened, citable view over them.

    An empty instance is valid: a failed plan still returns a well-formed shape.
    """

    items: list[EvidenceItem] = Field(default_factory=list)
    execution: QueryExecutionResult | None = None
    analysis: AnalysisResult | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"evidence ids must be unique; duplicated: {', '.join(duplicates)}"
            )
        return self

    def ids(self) -> set[str]:
        """The set of citable evidence ids."""

        return {item.id for item in self.items}


# --------------------------------------------------------------------------- #
# Request / result
# --------------------------------------------------------------------------- #


class AgentQuestionRequest(_StrictModel):
    """A free-form question for the agent to plan against."""

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class AgentResult(_StrictModel):
    """The agent's response: what ran, what was found, and - maybe - an answer.

    ``answer`` is optional by design. When synthesis is unavailable or the
    generated answer fails validation, the answer is withheld and the evidence
    is still returned. The deterministic result is the product; the prose is a
    presentation layer over it.

    Two integrity rules are enforced here because this is the only place both
    halves are visible: an ``ok`` result must carry the answer it claims, and
    the trace may not cite evidence the result does not contain.
    """

    status: AgentStatus
    answer: str | None = None
    trace: AgentTrace
    evidence: AgentEvidence

    @model_validator(mode="after")
    def _check_integrity(self) -> Self:
        if self.status == "ok" and self.answer is None:
            raise ValueError(
                "status 'ok' requires an answer; use 'synthesis_unavailable' or "
                "'answer_withheld' when there is none"
            )

        unknown = sorted(set(self.trace.evidence_refs) - self.evidence.ids())
        if unknown:
            raise ValueError(
                "trace.evidence_refs name evidence that is not present: "
                f"{', '.join(unknown)}"
            )
        return self
