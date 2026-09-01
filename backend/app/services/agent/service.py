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
from app.services.agent.executor import AgentExecutor
from app.services.agent.grounding import validate_answer
from app.services.agent.planner import AgentPlanner
from app.services.agent.schemas import (
    AgentEvidence,
    AgentQuestionRequest,
    AgentResult,
    AgentTrace,
    AnswerValidation,
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
            plan = await self._planner.plan(request.question)
        except AppError as exc:
            logger.info("Agent planning failed [%s]: %s", exc.code, exc.message)
            return AgentResult(
                status="planner_unavailable",
                answer=None,
                trace=AgentTrace(),
                evidence=AgentEvidence(),
            )

        # --- 2. Execute. The executor reports per-step outcomes and handles
        # its own tool failures; whatever it returns is what actually happened.
        outcome = await self._executor.execute(plan)

        # --- 3. Synthesise. A failure here loses the prose, never the
        # evidence that was already established.
        try:
            draft = await self._synthesizer.synthesize(
                request.question, outcome.evidence
            )
        except AppError as exc:
            logger.info("Agent synthesis failed [%s]: %s", exc.code, exc.message)
            return AgentResult(
                status="synthesis_unavailable",
                answer=None,
                trace=AgentTrace(plan=plan, steps=outcome.steps),
                evidence=outcome.evidence,
            )

        # --- 4. Validate, using the Commit 3 validator unchanged. This service
        # performs no check of its own and knows nothing about how any of them
        # work; it only reads the three outcomes.
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
