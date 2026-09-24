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
from app.services.query.schemas import QueryExecutionResult, SatQueryIntent, TimeRange

#: The closed set of tools a planner may select. Adding a name here is the
#: deliberate act of granting a model access to a capability - which is why a
#: future remote-sensing model tool is NOT listed yet.
ToolName = Literal[
    "execute_query",
    "spectral_indices",
    "sar_backscatter_statistics",
    "ndwi_statistics",
    "temporal_ndwi_statistics",
    "rs_model_analysis",
]

#: Outcome of one planned step, as observed by the executor. ``rejected`` is a
#: step the executor declined (e.g. its precondition was not met); ``failed`` is
#: one that ran and raised; ``skipped`` is one never reached.
ToolStepStatus = Literal["ok", "rejected", "failed", "skipped"]

#: ``ok``                     - a grounded answer was produced.
#: ``planner_unavailable``    - no plan; nothing executed.
#: ``synthesis_unavailable``  - tools ran, the answer could not be generated.
#: ``answer_withheld``        - an answer was generated but failed validation.
#: ``needs_clarification``    - the question does not state something the
#:                              workflow needs (or asks for something it does
#:                              not do); nothing was measured and
#:                              ``clarification`` says what to add.
#: ``location_unavailable``   - the question was usable, but the location
#:                              service (the geocoder) could not be used right
#:                              now, so nothing was searched or measured.
#:                              NOT a clarification - the user has nothing to
#:                              add - and NOT an evidence verdict: no scene was
#:                              ever looked at. ``failure`` says why and, when
#:                              known, how long to wait.
#: In the middle three the evidence is still returned: the deterministic result
#: never depends on a language-model provider being reachable.
AgentStatus = Literal[
    "ok",
    "planner_unavailable",
    "synthesis_unavailable",
    "answer_withheld",
    "needs_clarification",
    "location_unavailable",
]

#: Why a question could not be executed as asked. Each names ONE missing or
#: unsupported fact, so a caller can act on it without parsing prose.
ClarificationReason = Literal[
    "analysis_missing",
    "analysis_unsupported",
    "analysis_ambiguous",
    "location_missing",
    "location_not_found",
    "area_too_large",
    "date_missing",
    "date_ambiguous",
    "date_invalid",
    "comparison_incomplete",
    "conflicting_request",
    "requires_ai_model",
]

#: Where one piece of evidence came from. ``model`` is RESERVED for a future
#: remote-sensing model tool and is unused today; it exists so that adding such
#: a tool later is a registration, not a contract change.
EvidenceSource = Literal[
    "execution",
    "ndvi",
    "ndwi",
    "ndbi",
    "temporal_ndwi",
    "sar_backscatter",
    "compatibility",
    "model",
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
    sar_polarization: Literal["vv", "vh"] = "vv"
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)


class NdwiParams(_StrictModel):
    """Single-scene Sentinel-2 NDWI statistics.

    Deliberately parameterless. The index threshold, band choice and raw-DN
    decision are scientific constants owned by the engine, not knobs a language
    model may turn.
    """

    tool: Literal["ndwi_statistics"] = "ndwi_statistics"


class SpectralIndicesParams(_StrictModel):
    """Which spectral indices to compute over the discovered optical scene.

    This is the ONE analysis tool that takes a parameter, and the distinction
    matters: choosing *which* index answers a question is a planning decision,
    exactly what the model is for. How each index is computed - the bands, the
    raw-DN decision, the co-registration rule - stays a scientific constant
    owned by the engine. The model selects; it still never computes.

    One tool rather than three also keeps a plan affordable: the plan budget is
    three steps, so a tool per index would leave no room for discovery and a
    visual observation in the same run.
    """

    tool: Literal["spectral_indices"] = "spectral_indices"
    #: Closed set, validated here so an unrecognised index fails before
    #: dispatch rather than being quietly swapped for a different one.
    indices: list[Literal["ndvi", "ndwi", "ndbi"]] = Field(
        min_length=1, max_length=3
    )

    @field_validator("indices")
    @classmethod
    def _no_duplicates(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("indices must not contain duplicates")
        return value


class SarBackscatterParams(_StrictModel):
    """Quantitative provider RTC VV/VH statistics; no model-controlled calibration."""

    tool: Literal["sar_backscatter_statistics"] = "sar_backscatter_statistics"


class TemporalNdwiParams(_StrictModel):
    """Temporal NDWI Statistics for one deterministic Sentinel-2 pair.

    Also parameterless: pair selection, suppression rules and the compatibility
    report are deterministic and are not open to negotiation.
    """

    tool: Literal["temporal_ndwi_statistics"] = "temporal_ndwi_statistics"


class RsModelParams(_StrictModel):
    """A visual question about the image the server already retrieved.

    Carries the QUESTION and nothing else. There is deliberately no field for a
    scene id, an asset, a URL, a path or image bytes: the server decides which
    image is looked at, and a plan cannot redirect that. ``extra="forbid"``
    means an attempt to add one is a validation error, not a silently dropped
    key.

    The image itself comes from the preceding validated ``execute_query`` step,
    and the executor re-checks that independently - the planner is not trusted
    to enforce it.
    """

    tool: Literal["rs_model_analysis"] = "rs_model_analysis"
    question: str = Field(min_length=1, max_length=500)


#: A single validated tool call. Discriminated on ``tool``, so an unrecognised
#: name is a validation error rather than a runtime dispatch problem.
ToolCall = Annotated[
    ExecuteQueryParams
    | SpectralIndicesParams
    | NdwiParams
    | TemporalNdwiParams
    | SarBackscatterParams
    | RsModelParams,
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

        # Looking at an image requires retrieving one. Observed live: for a
        # plainly visual question the planner sometimes asked for no imagery and
        # then asked to observe it, so the executor refused and the run produced
        # nothing. That is a precondition, not a preference, so it is settled
        # here rather than requested in a prompt - a validated plan simply
        # cannot express the contradiction.
        #
        # Enabled rather than rejected, because the model's INTENT was coherent;
        # only the flag was wrong, and failing the whole request over a field
        # the server owns would be worse for no gain. `include_imagery` is a
        # retrieval switch, not a claim, so setting it invents nothing - and the
        # trace shows the plan that actually ran.
        #
        # This does NOT guarantee an image exists: discovery may still return a
        # window without one. The executor's own check remains the thing that
        # decides whether the visual tool may run.
        # Spectral results are displayed beside the same selected scene's RGB.
        # Request it even when the planner omitted the display-only switch.
        if set(tools) & {"rs_model_analysis", "ndwi_statistics", "spectral_indices",
                         "temporal_ndwi_statistics", "sar_backscatter_statistics"}:
            for index, step in enumerate(self.steps):
                if isinstance(step, ExecuteQueryParams) and not step.include_imagery:
                    self.steps[index] = step.model_copy(
                        update={"include_imagery": True}
                    )
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
    #: Whether the answer rests on a model observation. ``attributed`` is NOT a
    #: pass: it records that a visual claim is present and is being presented as
    #: a named model's observation, because no mechanical check can validate
    #: one. ``not_run`` means no model evidence was involved at all.
    visual_claims: Literal["attributed", "not_run"] = "not_run"


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


class VisualObservation(_StrictModel):
    """What a vision-language model said about one retrieved image.

    An ATTRIBUTED observation, never a verified fact. Nothing mechanical can
    check "water is visible": there is no evidence such a sentence could be
    contained by, which is exactly why this shape carries the model's identity
    beside the statement rather than presenting the statement alone.

    Deliberately not a :class:`Measurement`. Even when the model says a number,
    that number is part of ``statement`` and never becomes a measurement value -
    see the source filter in ``grounding._allowed_values``, which is what stops
    a model from authorising its own figure.
    """

    #: Exactly what the model said, verbatim.
    statement: str = Field(min_length=1, max_length=4000)
    #: The provider family (e.g. "gemini"), for attribution in the UI.
    provider: str = Field(min_length=1, max_length=100)
    #: The specific model that produced it, so a claim is traceable to a version.
    model: str = Field(min_length=1, max_length=200)
    #: The scene whose image was actually shown to the model.
    scene_id: str = Field(min_length=1, max_length=200)


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
    #: An attributed model observation about an image. Mutually informative
    #: with, and never a substitute for, ``measurement``: a visual statement is
    #: not a measurement and must not be rendered or validated as one.
    visual: VisualObservation | None = None
    #: What computed this item - engine function, or model id/version.
    produced_by: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _require_content(self) -> Self:
        if self.measurement is None and self.text is None and self.visual is None:
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
    #: Which inference backend answers the visual step for THIS run. ``None``
    #: uses the configured default. Present so two providers can be compared
    #: over the same scene without restarting the service; it selects an
    #: inference backend and nothing else - the deterministic pipeline, the
    #: grounding rules and the evidence shape are identical either way.
    provider: str | None = Field(default=None, max_length=32)
    #: Which model that provider should use for THIS run. ``None`` uses the
    #: provider's configured default. Validated against the catalog, so a
    #: model that cannot see an image is refused before any request is made.
    model: str | None = Field(default=None, max_length=200)

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip().lower()
            return stripped or None
        return value

    @field_validator("model", mode="before")
    @classmethod
    def _normalise_model(cls, value: object) -> object:
        # Model ids are case-sensitive and vendor-namespaced, so only
        # surrounding whitespace is removed.
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("question", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class AgentFailure(_StrictModel):
    """Why a provider stage could not complete - stated, not merely implied.

    Without this, every provider failure collapsed into a bare status, and the
    three situations it hides call for three different actions:

    ============================  ===============================================
    ``rate_limited``              temporary; the quota clears on its own
    ``upstream_error``            the service is down, or a key is not configured
    ``intent_parse_error``        the model's output did not match the contract
    ============================  ===============================================

    It also keeps an ACTIONABLE failure actionable. "GEMINI_API_KEY is not
    configured" is a message someone can act on, and it was being logged and
    then discarded, leaving the caller an opaque status and no next step.

    Note what this is NOT. It is never a substitute for an answer, and it never
    appears beside one: a present ``failure`` means a stage did not run, so
    ``status`` is already one of the ``*_unavailable`` values. In particular it
    must not be confused with a synthesizer that ran fine and honestly reported
    that the evidence did not answer the question - that is a successful run
    carrying an abstention as its ``answer``, and it has no ``failure`` at all.

    ``message`` is always written by this system from a status code, never
    passed through from a provider: an upstream error body is third-party text
    that may echo the request, and provider payloads do not reach responses.
    """

    #: Which stage failed. The executor's own tool failures are reported
    #: per-step in the trace and never here - with ONE exception, ``location``:
    #: the geocoder is a dependency every run needs before any tool can do
    #: anything, and its outage decides the whole result.
    stage: Literal["planning", "synthesis", "location"]
    #: The originating :class:`~app.core.errors.AppError` code, so a caller can
    #: branch on the KIND of failure without matching on prose.
    code: str = Field(min_length=1, max_length=100)
    #: A safe, human-readable statement of what went wrong.
    message: str = Field(min_length=1, max_length=1000)
    #: How long the SERVICE asked us to wait, when it said so. Only a rate
    #: limit carries one. ``None`` means the wait is unknown - never zero.
    retry_after_seconds: float | None = Field(default=None, ge=0)
    #: The external dependency that failed, when the failure is one. Set for
    #: ``location`` (``"geocoder"``); provider stages leave it unset.
    dependency: Literal["geocoder"] | None = None


class AgentClarification(_StrictModel):
    """What a question has to add before it can be executed.

    A question, not a guess. When the interpreter cannot establish an analysis,
    a place or a period from what was written, it says which one and what the
    supported choices are - rather than running a default nobody asked for and
    presenting the result as an answer.

    ``understood_*`` repeat back what WAS established, so a reader can see that
    only the missing part needs adding. They are facts read from the question,
    never inferred values.
    """

    reason: ClarificationReason
    #: The question put to the user. System-authored, never model output.
    message: str = Field(min_length=1, max_length=1000)
    #: The supported choices that would resolve it, when there is a closed set.
    options: list[str] = Field(default_factory=list, max_length=10)
    #: For each option, a complete question that asks for it - built only from
    #: what the question already established, so choosing one never adds a
    #: place or a date the user did not give. Empty, or one per option.
    option_questions: list[str] = Field(default_factory=list, max_length=10)
    understood_analyses: list[str] = Field(default_factory=list, max_length=10)
    understood_location: str | None = Field(default=None, max_length=300)
    understood_periods: list[TimeRange] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _questions_match_options(self) -> Self:
        if self.option_questions and len(self.option_questions) != len(self.options):
            raise ValueError("option_questions must be empty or one per option")
        return self


class AgentResult(_StrictModel):
    """The agent's response: what ran, what was found, and - maybe - an answer.

    ``answer`` is optional by design. When synthesis is unavailable or the
    generated answer fails validation, the answer is withheld and the evidence
    is still returned. The deterministic result is the product; the prose is a
    presentation layer over it.

    Three integrity rules are enforced here because this is the only place all
    the halves are visible: an ``ok`` result must carry the answer it claims,
    the trace may not cite evidence the result does not contain, and a
    ``failure`` may appear only where a provider stage actually failed.
    """

    status: AgentStatus
    answer: str | None = None
    #: Present exactly when a stage failed, i.e. for ``planner_unavailable``,
    #: ``synthesis_unavailable`` and ``location_unavailable``. Optional rather
    #: than required for the first two so an existing caller constructing a
    #: bare failure result stays valid; the service always supplies it, and
    #: ``location_unavailable`` requires it.
    failure: AgentFailure | None = None
    #: Present exactly when ``status`` is ``needs_clarification``.
    clarification: AgentClarification | None = None
    trace: AgentTrace
    evidence: AgentEvidence

    @model_validator(mode="after")
    def _check_integrity(self) -> Self:
        if self.status == "ok" and self.answer is None:
            raise ValueError(
                "status 'ok' requires an answer; use 'synthesis_unavailable' or "
                "'answer_withheld' when there is none"
            )

        # A clarification is a question back to the user. It never sits beside
        # an answer, and a result that asks for one must say what it asks.
        if (self.status == "needs_clarification") != (self.clarification is not None):
            raise ValueError(
                "'clarification' is present exactly when status is "
                "'needs_clarification'"
            )
        if self.status == "needs_clarification" and self.answer is not None:
            raise ValueError("a result that needs clarification carries no answer")

        # A failure beside a delivered or withheld answer would misdescribe the
        # run: in both of those the providers did their work, and what happened
        # afterwards is reported through ``answer_validation``.
        if self.failure is not None and self.status not in (
            "planner_unavailable",
            "synthesis_unavailable",
            "location_unavailable",
        ):
            raise ValueError(
                "a 'failure' may only accompany 'planner_unavailable', "
                f"'synthesis_unavailable' or 'location_unavailable'; got {self.status!r}"
            )

        # A location outage says WHY nothing ran, and says nothing more: no
        # answer (not even an abstention - no evidence was ever weighed).
        if self.status == "location_unavailable" and (
            self.failure is None or self.answer is not None
        ):
            raise ValueError(
                "'location_unavailable' requires a 'failure' and carries no answer"
            )

        # The stage and the status must tell the same story. They are derived
        # from one another in practice, so disagreement is a bug, not a case.
        expected = {
            "planner_unavailable": "planning",
            "synthesis_unavailable": "synthesis",
            "location_unavailable": "location",
        }.get(self.status)
        if self.failure is not None and self.failure.stage != expected:
            raise ValueError(
                f"status {self.status!r} implies stage {expected!r}, but the "
                f"failure names {self.failure.stage!r}"
            )

        unknown = sorted(set(self.trace.evidence_refs) - self.evidence.ids())
        if unknown:
            raise ValueError(
                "trace.evidence_refs name evidence that is not present: "
                f"{', '.join(unknown)}"
            )
        return self
