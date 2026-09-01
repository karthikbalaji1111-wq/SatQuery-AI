"""The synthesis boundary: evidence in, prose out.

    question + AgentEvidence -> AnswerSynthesizer.synthesize() -> DraftAnswer

:class:`AnswerSynthesizer` is the provider-neutral abstraction, mirroring
:class:`~app.services.agent.planner.AgentPlanner`. **This module contains no
provider SDK import** - concrete providers live in
:mod:`app.services.agent.providers`, and an AST test enforces that the SDK
appears in exactly one file of the agent package.

A synthesizer *describes*; it does not act. It receives evidence that has
already been collected and turns it into a sentence. It executes no tool,
retrieves no imagery, computes no statistic, holds no service handle, and
creates no ``Measurement`` or ``EvidenceItem`` - the ids it emits are
*references* to evidence that already exists, never new evidence.

**It does not validate its own output.** Grounding stays where Commit 3 put it,
as a separate pure function over ``DraftAnswer`` and ``AgentEvidence``. A
generator that marked its own homework would establish nothing, so the check
that a stated number is traceable deliberately lives outside this boundary.
The prompt asks a model to stay within the evidence; ``validate_answer`` is
what actually enforces it.

``DraftAnswer`` is reused verbatim from :mod:`app.services.agent.grounding`,
where the validator that consumes it lives. It carries a summary and evidence
references and nothing else - no reasoning, no confidence, no tool calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.agent.grounding import DraftAnswer
from app.services.agent.schemas import AgentEvidence


class AnswerSynthesizer(ABC):
    """Turns collected evidence into a :class:`DraftAnswer`.

    Implementations must not leak provider concepts through this interface, and
    must return a parsed model rather than raw provider output. The contract is
    deliberately tiny: a question and the evidence in, a draft answer out, and
    no other capability.
    """

    @abstractmethod
    async def synthesize(
        self, question: str, evidence: AgentEvidence
    ) -> DraftAnswer:
        """Describe ``evidence`` in answer to ``question``.

        Implementations raise rather than inventing prose when they cannot
        produce an answer: a synthesis failure must stay a failure, so the
        caller can withhold the answer and still return the evidence.

        ``evidence`` must not be mutated.
        """


#: What the mock returns when nothing more specific is configured. Cites
#: nothing, which is explicit rather than accidental: an empty list is a claim
#: that no evidence was used, and grounding treats it as such.
_DEFAULT_ANSWER = DraftAnswer(
    summary="The requested observations were retrieved.",
    evidence_refs=[],
)


class MockAnswerSynthesizer(AnswerSynthesizer):
    """TEST / DEVELOPMENT ONLY - performs no language generation.

    Returns a fixed, already-validated answer regardless of the question or the
    evidence, so the agent pipeline can be exercised with no provider and no
    network. It does not summarise the evidence and does not simulate a model;
    pretending otherwise would make tests pass for reasons unrelated to how a
    real synthesizer behaves.

    Each call returns an independent copy, so a caller that mutates an answer
    cannot corrupt the fixture for the next one.
    """

    is_mock = True

    def __init__(self, answer: DraftAnswer | None = None) -> None:
        self._answer = answer if answer is not None else _DEFAULT_ANSWER

    async def synthesize(
        self, question: str, evidence: AgentEvidence
    ) -> DraftAnswer:
        # Both inputs are intentionally ignored - this mock does no synthesis,
        # and it must never touch the evidence it is handed.
        del question, evidence
        return self._answer.model_copy(deep=True)
