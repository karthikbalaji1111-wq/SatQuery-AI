"""Provider selection must not touch the deterministic pipeline.

The scientific claim this system makes is that its measurements come from the
raster and the catalog, not from a model. That claim only holds if swapping the
AI provider leaves the deterministic half byte-for-byte identical.

So: run the SAME agent execution twice with mocked providers whose only
difference is which provider they represent, and assert that everything
deterministic matches exactly - the scene selected, the imagery retrieved, the
evidence produced - while only the model-authored prose differs.

Mocked deliberately. A live comparison would confound provider differences with
quota, latency and catalog drift, and could not assert equality of anything.
"""

from __future__ import annotations

import asyncio

from app.services.agent.executor import AgentExecutor
from app.services.agent.grounding import DraftAnswer
from app.services.agent.planner import MockAgentPlanner
from app.services.agent.schemas import AgentEvidence, AgentQuestionRequest
from app.services.agent.service import AgentService
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.agent.visual import VisualAnalyst, VisualAnswer

from tests.test_agent_executor import (  # reuse the established fakes
    FakeAnalysisService,
    FakeQueryExecutionService,
)


class ProviderTaggedSynthesizer(AnswerSynthesizer):
    """A synthesizer whose prose names its provider - and nothing else does."""

    def __init__(self, provider: str) -> None:
        self.provider = provider

    async def synthesize(
        self, question: str, evidence: AgentEvidence
    ) -> DraftAnswer:
        del question
        item = evidence.items[0]
        assert item.measurement is not None
        count = item.measurement.value
        # Vary the wording, not the supported fact. Uncited provider-tagged
        # prose no longer bypasses grounding merely because it has no numbers.
        summary = (f"{count:g} scenes were reported." if self.provider == "gemini"
                   else f"Reported scene count: {count:g}.")
        return DraftAnswer(
            summary=summary, evidence_refs=[item.id]
        )


class ProviderTaggedVisualAnalyst(VisualAnalyst):
    def __init__(self, provider: str) -> None:
        self.provider_name = provider
        self.model_name = f"{provider}-model"

    async def observe(
        self, *, question: str, image: bytes, media_type: str
    ) -> VisualAnswer:
        del question, image, media_type
        return VisualAnswer(answer=f"Observed by {self.provider_name}.")


def _run(provider: str):
    """One full agent run whose only provider-dependent parts are the models."""

    service = AgentService(
        planner=MockAgentPlanner(),
        executor=AgentExecutor(
            query_execution_service=FakeQueryExecutionService(),
            analysis_service=FakeAnalysisService(),
            visual_analyst=ProviderTaggedVisualAnalyst(provider),
        ),
        synthesizer=ProviderTaggedSynthesizer(provider),
    )
    return asyncio.run(
        service.answer(AgentQuestionRequest(question="Any water at the coast?"))
    )


def test_scene_selection_is_identical_across_providers() -> None:
    gemini = _run("gemini")
    nvidia = _run("nvidia")

    def scenes(result: object) -> list[str | None]:
        execution = result.evidence.execution  # type: ignore[attr-defined]
        return [] if execution is None else [
            window.selected_scene_id for window in execution.windows
        ]

    assert scenes(gemini) == scenes(nvidia)


def test_retrieved_imagery_is_identical_across_providers() -> None:
    gemini = _run("gemini")
    nvidia = _run("nvidia")

    def imagery(result: object) -> list[object]:
        execution = result.evidence.execution  # type: ignore[attr-defined]
        return [] if execution is None else [
            None if window.imagery is None else window.imagery.model_dump()
            for window in execution.windows
        ]

    assert imagery(gemini) == imagery(nvidia)


def test_deterministic_evidence_is_identical_across_providers() -> None:
    """The measurements are the product; they must not move with the provider."""

    gemini = _run("gemini")
    nvidia = _run("nvidia")

    def deterministic(result: object) -> list[dict[str, object]]:
        return [
            item.model_dump()
            for item in result.evidence.items  # type: ignore[attr-defined]
            if item.source != "model"
        ]

    assert deterministic(gemini) == deterministic(nvidia)


def test_the_executed_plan_is_identical_across_providers() -> None:
    gemini = _run("gemini")
    nvidia = _run("nvidia")

    assert [step.parameters.tool for step in gemini.trace.steps] == [
        step.parameters.tool for step in nvidia.trace.steps
    ]
    assert [step.status for step in gemini.trace.steps] == [
        step.status for step in nvidia.trace.steps
    ]


def test_only_the_model_authored_parts_differ() -> None:
    """The prose and the observation may differ - and are attributed."""

    gemini = _run("gemini")
    nvidia = _run("nvidia")

    assert gemini.answer != nvidia.answer
    assert gemini.status == nvidia.status == "ok"
    assert "scenes were reported" in (gemini.answer or "")
    assert "scene count" in (nvidia.answer or "")

    def observations(result: object) -> list[tuple[str, str]]:
        return [
            (item.visual.provider, item.visual.statement)
            for item in result.evidence.items  # type: ignore[attr-defined]
            if item.visual is not None
        ]

    gemini_visual = observations(gemini)
    nvidia_visual = observations(nvidia)
    if gemini_visual or nvidia_visual:
        assert gemini_visual != nvidia_visual
        for provider, _ in gemini_visual:
            assert provider == "gemini"
        for provider, _ in nvidia_visual:
            assert provider == "nvidia"
