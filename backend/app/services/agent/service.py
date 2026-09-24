"""Agent orchestration: coordinate four components, own none of their logic.

    AgentQuestionRequest
      -> AgentPlanner.plan()          propose      (provider, may fail)
      -> AgentExecutor.execute()      execute      (deterministic)
      -> AnswerSynthesizer.synthesize() describe   (provider, may fail)
      -> validate_answer()            check        (pure, Commit 3)
      -> AgentResult

This service calls those four in order and translates their outcomes into the
existing statuses. It does **not** parse language, choose tools, execute
anything, compute an index, query a catalog, touch a raster, call a provider,
or ground an answer. Every one of those already has an owner, and duplicating
any of them here would create a second source of truth.

Failure is the interesting part, so it is stated plainly:

===========================  ==========================  ====================
Where it broke               Status                      What survives
===========================  ==========================  ====================
planner                      ``planner_unavailable``     nothing ran, so
                                                         nothing is claimed
synthesizer                  ``synthesis_unavailable``   the evidence
grounding rejected the       ``answer_withheld``         the evidence, the
answer                                                   trace, the checks
everything passed            ``ok``                      the answer too
===========================  ==========================  ====================

No failure is ever recoded as ``ok``, and no gap is ever filled with invented
prose or invented evidence. When the answer is withheld the deterministic
result is still returned, because the measurements are the product and the
sentence is only a presentation of them.

The two provider failures also carry an :class:`AgentFailure` saying WHICH
stage broke and why, so a caller can tell a temporary rate limit from an
outage from a misconfigured key. That distinction matters most in the case
that looks least like a failure: ``synthesis_unavailable`` returns a complete
set of valid measurements, and describing it as a lack of evidence would be
false. A synthesizer that genuinely found the evidence wanting says so in a
successful answer instead, and carries no failure at all.

An executed step is reported because the **executor** said it ran, never
because it appeared in the requested plan. ``trace.plan`` records what was
asked for; ``trace.steps`` records what happened; they are allowed to differ.

Collaborators are injected. This service constructs no client, no provider and
no downstream service - composition belongs to the layer above, which is what
keeps this class free of any SDK.
"""

from __future__ import annotations

from app.core.errors import AppError
from app.core.logging import get_logger
from app.core.observability import stage
from app.services.agent.executor import AgentExecutor
from app.services.agent.grounding import validate_answer
from app.services.agent.interpretation import ClarificationRequiredError
from app.services.agent.plan_completion import complete_plan
from app.services.agent.planner import AgentPlanner
from app.services.agent.schemas import (
    AgentClarification,
    AgentEvidence,
    AgentFailure,
    AgentPlan,
    AgentQuestionRequest,
    AgentResult,
    AgentToolStep,
    AgentTrace,
    AnswerValidation,
    ExecuteQueryParams,
)
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.base import DomainService

logger = get_logger("agent.service")


def _passed(validation: AnswerValidation) -> bool:
    """Whether every mechanical check cleared.

    All three must pass. ``not_run`` is deliberately not treated as success -
    an unchecked answer is not a validated one.
    """

    return (
        validation.numeric_grounding == "pass"
        and validation.forbidden_terms == "pass"
        and validation.evidence_refs == "pass"
    )


def _failure(stage: str, exc: AppError) -> AgentFailure:
    """Record why a provider stage failed, in a shape a caller can act on.

    The message and code come from the :class:`AppError` the provider raised.
    Those messages are written by this system - never passed through from an
    upstream response - so they are safe to return, and some of them are the
    only actionable thing a misconfigured deployment gets ("GEMINI_API_KEY is
    not configured..."). Discarding them left the caller a bare status.

    ``retry_after_seconds`` is read reflectively because it is an OPTIONAL hint:
    a provider that knows how long the service asked us to wait may attach it,
    and one that does not is not obliged to invent a number. Reading it here
    rather than importing a provider type keeps this service free of any
    provider import, which is the property the whole package is arranged
    around.
    """

    hint = getattr(exc, "retry_after_seconds", None)
    return AgentFailure(
        stage=stage,  # type: ignore[arg-type]
        code=exc.code,
        message=exc.message,
        retry_after_seconds=hint if isinstance(hint, (int, float)) else None,
    )


def _place(plan: AgentPlan) -> str | None:
    discovery = plan.steps[0]
    return (
        discovery.intent.location_query
        if isinstance(discovery, ExecuteQueryParams)
        else None
    )


def _oversized_place(plan: AgentPlan, steps: list[AgentToolStep]) -> AgentClarification:
    """The clarification for a place larger than one native-resolution read.

    The measured extent and the limit are the analysis gate's own words, taken
    from the failed step verbatim - nothing is re-derived here.
    """

    place = _place(plan)
    reason = next(
        (step.error_message for step in steps if step.error_message), None
    ) or "The area is too large to measure at native resolution."
    reason = reason.split(": ", 1)[-1]  # drop the "plan.bbox: " field prefix
    named = f"'{place}' is too large to analyse: {reason}" if place else reason
    return AgentClarification(
        reason="area_too_large",
        message=(
            f"{named} Name a neighbourhood, landmark or smaller district - for "
            "example '<landmark>, <city>' - or give coordinates as 'lat, lon'."
        )[:1000],
        understood_location=place,
    )


def _unresolved_place(plan: AgentPlan) -> AgentClarification:
    """The clarification for a place the geospatial service could not find."""

    place = _place(plan)
    named = f"No place matching '{place}' was found. " if place else "The place was not found. "
    return AgentClarification(
        reason="location_not_found",
        message=named
        + "Name a city, district or landmark - adding the city or state helps, "
        "for example '<landmark>, <city>' - or give coordinates as 'lat, lon'.",
        understood_location=place,
    )


class AgentService(DomainService):
    """Coordinates planning, execution, synthesis and validation.

    All three collaborators are required and injected - there is no default,
    because a default would mean constructing a provider or a downstream
    service in here, and this layer must be able to run with fakes and no
    network at all.

    Note the deliberate departure from the repository's other services: this
    one is not zero-argument constructible, so it is absent from the
    ``test_services`` contract list. Injecting a planner and a synthesizer is
    the whole point of the class.
    """

    name = "agent"

    def __init__(
        self,
        *,
        planner: AgentPlanner,
        executor: AgentExecutor,
        synthesizer: AnswerSynthesizer,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._synthesizer = synthesizer

    def describe(self) -> str:
        return (
            "Agentic orchestration over the deterministic remote-sensing "
            "tools: plan, execute, synthesise, validate."
        )

    async def answer(self, request: AgentQuestionRequest) -> AgentResult:
        """Answer ``request`` by coordinating the four stages."""

        # --- 1. Plan. A failure here means nothing has run, so nothing is
        # claimed: no plan, no steps, no evidence, no validation.
        try:
            with stage("planning"):
                plan = await self._planner.plan(request.question)
        except ClarificationRequiredError as exc:
            # Not a failure: the question does not yet say what to run. It is
            # put back to the user rather than answered with a default.
            logger.info("Agent needs clarification [%s]", exc.clarification.reason)
            return AgentResult(
                status="needs_clarification",
                clarification=exc.clarification,
                trace=AgentTrace(),
                evidence=AgentEvidence(),
            )
        except AppError as exc:
            logger.info("Agent planning failed [%s]: %s", exc.code, exc.message)
            return AgentResult(
                status="planner_unavailable",
                answer=None,
                failure=_failure("planning", exc),
                trace=AgentTrace(),
                evidence=AgentEvidence(),
            )

        # --- 2. Execute. The executor reports per-step outcomes and handles
        # its own tool failures; whatever it returns is what actually happened.
        #
        # An index the question names outright is always computed, and a
        # question asking what is visible always gets the observation - even
        # when the planner left that step out (observed live). The trace keeps
        # the plan as the planner returned it; an added step appears in
        # ``trace.steps`` because it ran.
        executed_plan = complete_plan(request.question, plan)
        if executed_plan is not plan:
            logger.info("Plan completed with an explicitly requested step")
        with stage("execution", steps=len(executed_plan.steps)):
            outcome = await self._executor.execute(executed_plan)

        # A place the geospatial service cannot find is a question for the
        # user, not an outage and not an absence of scenes. Nothing was
        # searched, so nothing is described; the failed step stays in the trace.
        #
        # So is a place too large to measure: the analysis gate refused it after
        # geocoding and before any catalog search, and the remedy is a smaller
        # place - the user's to choose, not ours to crop.
        clarification = None
        if outcome.discovery_failure_code == "not_found":
            clarification = _unresolved_place(plan)
        elif outcome.discovery_failure_code == "aoi_too_large":
            clarification = _oversized_place(plan, outcome.steps)
        if clarification is not None:
            return AgentResult(
                status="needs_clarification",
                clarification=clarification,
                trace=AgentTrace(plan=plan, steps=outcome.steps),
                evidence=outcome.evidence,
            )

        # --- 3. Synthesise. A failure here loses the prose, never the
        # evidence that was already established.
        try:
            with stage("synthesis", evidence=len(outcome.evidence.items)):
                draft = await self._synthesizer.synthesize(
                    request.question, outcome.evidence
                )
        except AppError as exc:
            logger.info("Agent synthesis failed [%s]: %s", exc.code, exc.message)
            return AgentResult(
                status="synthesis_unavailable",
                answer=None,
                # The evidence below is intact and was computed deterministically.
                # This records that the PROSE stage failed - not that the
                # question went unanswered for want of evidence.
                failure=_failure("synthesis", exc),
                trace=AgentTrace(plan=plan, steps=outcome.steps),
                evidence=outcome.evidence,
            )

        # --- 4. Validate, using the Commit 3 validator unchanged. This service
        # performs no check of its own and knows nothing about how any of them
        # work; it only reads the three outcomes.
        with stage("grounding"):
            validation = validate_answer(draft, outcome.evidence)
        accepted = _passed(validation)

        # Only references the evidence can actually resolve are recorded. A
        # dangling citation is reported through ``validation.evidence_refs``;
        # repeating it here would make the trace assert evidence that is not
        # present, which ``AgentResult`` rightly refuses.
        resolvable = outcome.evidence.ids()
        cited = [ref for ref in draft.evidence_refs if ref in resolvable]

        logger.info(
            "Agent answered (steps=%d, accepted=%s, numeric=%s, terms=%s, refs=%s)",
            len(outcome.steps),
            accepted,
            validation.numeric_grounding,
            validation.forbidden_terms,
            validation.evidence_refs,
        )

        return AgentResult(
            status="ok" if accepted else "answer_withheld",
            answer=draft.summary if accepted else None,
            trace=AgentTrace(
                plan=plan,
                steps=outcome.steps,
                evidence_refs=cited,
                answer_validation=validation,
            ),
            evidence=outcome.evidence,
        )
