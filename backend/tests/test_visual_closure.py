"""Final product closure: a question about what an image SHOWS is answered.

Before this pass the standard workflow refused every visual question up front
("Describing what is visible in an image needs an AI model") - nothing was
retrieved, so there was no image to show either. Now:

* a visual question retrieves the normal-colour Sentinel-2 image and plans ONE
  visual step; a measurement joins it only when the question asks to measure;
* the executor chooses that image and hands its exact bytes to the analyst -
  with the place's name removed from the question, so a model describes the
  picture rather than what it knows about "Cubbon Park";
* the result states what became of the description (``AgentResult.visual``):
  observed, unavailable (no model here - the image is still shown), no image,
  or failed. Never "Insufficient evidence", never a fabricated observation;
* the standard workflow uses the deployment's configured visual analyst for
  that one step, and runs without one.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import pytest
from app.api.routes import query as query_routes
from app.core.errors import UpstreamServiceError
from app.services.agent.executor import AgentExecutor, _without_place
from app.services.agent.intent_router import route_operation
from app.services.agent.interpretation import ClarificationRequiredError, interpret
from app.services.agent.prompts import _VISUAL_INSTRUCTION
from app.services.agent.schemas import AgentQuestionRequest
from app.services.agent.service import AgentService
from app.services.agent.standard import ABSTENTION, StandardPlanner, StandardReport
from app.services.agent.visual import VisualAnalyst

from tests.test_agent_visual import (
    PNG_BYTES,
    FakeAnalysis,
    FakeQuery,
    RecordingAnalyst,
    build_executor,
    execution,
    run,
    visual_plan,
    window,
)
from tests.test_intent_model import FakeClassifier

TODAY = date(2026, 9, 25)


def tools(question: str) -> list[str]:
    return [step.tool for step in interpret(question, today=TODAY).plan().steps]


# =========================================================================== #
# 1. Which questions look, which measure, which do both
# =========================================================================== #


@pytest.mark.parametrize(
    "question",
    [
        "What is happening in the image around Hussain Sagar, Hyderabad in March 2024?",
        "What do you see in the satellite image around Cubbon Park, Bengaluru in December 2024?",
        "Describe the satellite image of Bandra West, Mumbai in January 2025",
        "What does the image show around Marina Beach, Chennai in January 2025?",
        "What is visible around Hussain Sagar, Hyderabad in March 2024?",
        "Is there visible water in the image of Hussain Sagar, Hyderabad in March 2024?",
        "Describe the landscape around Lalbagh Botanical Gardens, Bengaluru in December 2024",
        "Can you describe what is happening at India Gate, New Delhi in December 2024?",
        "Does the image show water around Dal Lake, Srinagar in January 2025?",
    ],
)
def test_a_visual_question_retrieves_the_image_and_plans_one_look(question: str) -> None:
    interpretation = interpret(question, today=TODAY)
    assert interpretation.analyses == ("imagery",)
    assert tools(question) == ["execute_query", "rs_model_analysis"]
    discovery = interpretation.plan().steps[0]
    assert discovery.include_imagery is True  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("question", "tool_list"),
    [
        ("What is the NDVI around Cubbon Park, Bengaluru in December 2024?",
         ["execute_query", "spectral_indices"]),
        ("Calculate NDWI around Marina Beach, Chennai in January 2025",
         ["execute_query", "spectral_indices"]),
        ("Compare water around Marina Beach, Chennai between January 2024 and January 2025",
         ["execute_query", "temporal_ndwi_statistics"]),
        ("Show vegetation around Cubbon Park, Bengaluru in December 2024",
         ["execute_query", "spectral_indices"]),
    ],
)
def test_a_measurement_question_never_reaches_a_visual_model(
    question: str, tool_list: list[str]
) -> None:
    assert tools(question) == tool_list


@pytest.mark.parametrize(
    ("question", "index"),
    [
        ("What do you see around Hussain Sagar, Hyderabad in March 2024, and what does "
         "the water measurement show?", "ndwi"),
        ("What do you see in the image of Cubbon Park, Bengaluru in December 2024 and "
         "what is the NDVI?", "ndvi"),
    ],
)
def test_a_question_that_asks_to_look_and_to_measure_gets_both(question: str, index: str) -> None:
    plan = interpret(question, today=TODAY).plan()
    assert [step.tool for step in plan.steps] == [
        "execute_query", "spectral_indices", "rs_model_analysis",
    ]
    assert plan.steps[1].indices == [index]  # type: ignore[union-attr]


def test_a_term_asked_about_a_place_is_a_request_a_bare_term_is_a_definition() -> None:
    from app.services.agent.interpretation import _VISUAL
    from app.services.agent.plan_completion import requested_matches

    assert requested_matches("What is visible around Hussain Sagar?", _VISUAL)
    assert requested_matches("What is visible in the image?", _VISUAL)
    assert not requested_matches("What is visible light?", _VISUAL)


def test_a_confident_label_cannot_add_a_measurement_to_a_look() -> None:
    chosen, _, decision = route_operation(
        "What do you see around Cubbon Park, Bengaluru in December 2024?",
        FakeClassifier("NDBI"),
    )
    assert chosen == ["imagery"]
    assert decision.reason == "visual_question"


@pytest.mark.parametrize(
    "question",
    [
        "Describe the radar backscatter around Marina Beach, Chennai in January 2025",
        "Compare water around Marina Beach, Chennai between January 2024 and January "
        "2025 and describe what you see",
    ],
)
def test_a_description_of_radar_or_of_two_dates_is_refused_plainly(question: str) -> None:
    with pytest.raises(ClarificationRequiredError) as refused:
        interpret(question, today=TODAY)
    assert refused.value.clarification.reason == "analysis_unsupported"
    assert "one satellite image" in refused.value.clarification.message or (
        "normal-colour satellite image" in refused.value.clarification.message
    )


# =========================================================================== #
# 2. The actual image, and only the image, reaches the model
# =========================================================================== #


def test_the_model_gets_the_retrieved_bytes_and_a_question_without_the_place() -> None:
    ex, _, analyst = build_executor()
    outcome = run(
        ex, visual_plan("What do you see around Marina Beach, Chennai in January 2025?")
    )

    [call] = analyst.calls
    assert call["image"] == PNG_BYTES
    assert call["media_type"] == "image/png"
    assert call["question"] == "What do you see around this area in January 2025?"
    assert "Marina" not in call["question"] and "Chennai" not in call["question"]

    assert outcome.visual.status == "observed"
    assert outcome.visual.scene_id == "S2B_44PMV_20250104_0_L2A"
    assert outcome.visual.acquired == "2025-01-04"
    [observation] = [item for item in outcome.evidence.items if item.visual is not None]
    assert observation.visual.scene_id == "S2B_44PMV_20250104_0_L2A"
    assert observation.measurement is None  # an observation is never a number


@pytest.mark.parametrize(
    ("question", "place", "expected"),
    [
        ("What do you see in the satellite image around Cubbon Park, Bengaluru in "
         "December 2024?", "Cubbon Park, Bengaluru",
         "What do you see in the satellite image around this area in December 2024?"),
        ("What is visible around Hussain Sagar, Hyderabad in March 2024?",
         "Hussain Sagar, Hyderabad", "What is visible around this area in March 2024?"),
        ("Describe the image of Bandra West, Mumbai near bandra west",
         "Bandra West, Mumbai", "Describe the image of this area near this area"),
        ("Describe what is happening at India Gate, New Delhi", "India Gate, New Delhi",
         "Describe what is happening at this area"),
        ("Describe the landscape", None, "Describe the landscape"),
    ],
)
def test_the_place_name_never_reaches_the_model(
    question: str, place: str | None, expected: str
) -> None:
    assert _without_place(question, place) == expected


def test_the_instruction_asks_for_plain_words_and_no_place_guessing() -> None:
    assert "plain everyday words" in _VISUAL_INSTRUCTION
    assert "Do not name or guess the place" in _VISUAL_INSTRUCTION
    assert "Do NOT state precise quantities" in _VISUAL_INSTRUCTION


# =========================================================================== #
# 3. What became of the description - stated, never invented
# =========================================================================== #


def test_no_model_here_the_image_is_still_retrieved_and_the_result_says_so() -> None:
    query = FakeQuery()
    ex = AgentExecutor(
        query_execution_service=query,  # type: ignore[arg-type]
        analysis_service=FakeAnalysis(),  # type: ignore[arg-type]
    )
    outcome = run(ex, visual_plan())

    assert outcome.visual.status == "unavailable"
    assert outcome.visual.message == (
        "The satellite image was retrieved, but describing it needs an AI visual "
        "model, which is not available here."
    )
    assert outcome.visual.scene_id == "S2B_44PMV_20250104_0_L2A"
    assert outcome.visual.acquired == "2025-01-04"
    # No observation is fabricated; the retrieved image is in the evidence.
    assert not any(item.visual is not None for item in outcome.evidence.items)
    assert outcome.evidence.execution.windows[0].imagery is not None


def test_no_image_is_its_own_state() -> None:
    ex, _, analyst = build_executor(query=FakeQuery(execution([window(img=None)])))
    outcome = run(ex, visual_plan())
    assert outcome.visual.status == "no_image"
    assert analyst.calls == []


def test_a_model_that_does_not_answer_is_a_failure_not_an_observation() -> None:
    ex, _, _ = build_executor(
        analyst=RecordingAnalyst(error=UpstreamServiceError("provider down"))
    )
    outcome = run(ex, visual_plan())
    assert outcome.visual.status == "failed"
    assert outcome.visual.message == (
        "The satellite image was retrieved, but the AI visual model did not answer."
    )
    assert not any(item.visual is not None for item in outcome.evidence.items)


# =========================================================================== #
# 4. End to end through the standard workflow
# =========================================================================== #


def standard(analyst: VisualAnalyst | None) -> AgentService:
    return AgentService(
        planner=StandardPlanner(),
        executor=AgentExecutor(
            query_execution_service=FakeQuery(),  # type: ignore[arg-type]
            analysis_service=FakeAnalysis(),  # type: ignore[arg-type]
            visual_analyst=analyst,
        ),
        synthesizer=StandardReport(),
    )


QUESTION = "What do you see in the satellite image around Marina Beach, Chennai in January 2025?"


def ask(service: AgentService) -> Any:
    return asyncio.run(service.answer(AgentQuestionRequest(question=QUESTION)))


def test_standard_without_a_model_reports_the_image_not_insufficient_evidence() -> None:
    result = ask(standard(None))

    assert result.status == "ok"
    assert result.answer != ABSTENTION
    assert "S2B_44PMV_20250104_0_L2A" in result.answer
    assert result.visual is not None and result.visual.status == "unavailable"
    assert result.evidence.execution.windows[0].imagery is not None


def test_standard_with_a_model_attributes_its_observation_and_keeps_numbers_out() -> None:
    analyst = RecordingAnalyst(answer="About 40% of the image is water beside a sandy shore.")
    result = ask(standard(analyst))

    assert result.status == "ok"
    assert result.visual.status == "observed"
    [observation] = [item for item in result.evidence.items if item.visual is not None]
    assert observation.source == "model"
    assert observation.visual.model == "fake-vlm/1"
    # The model's "40%" is its words, never a measurement or part of the answer.
    assert observation.measurement is None
    assert "40" not in (result.answer or "")
    assert not any(
        item.measurement is not None and item.source == "model" for item in result.evidence.items
    )


def test_a_measurement_question_through_standard_has_no_visual_state() -> None:
    service = standard(RecordingAnalyst())
    question = "Show the satellite image of Marina Beach, Chennai in January 2025"
    result = asyncio.run(service.answer(AgentQuestionRequest(question=question)))
    assert result.visual is None


# =========================================================================== #
# 5. The standard workflow runs with or without a configured model
# =========================================================================== #


def test_standard_mode_has_no_model_when_none_is_configured() -> None:
    assert query_routes.configured_visual_analyst() is None
    service = query_routes.build_standard_agent_service()
    assert service._executor._visual is None  # type: ignore[attr-defined]


def test_standard_mode_uses_the_configured_model_for_the_look_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = RecordingAnalyst()
    monkeypatch.setattr(query_routes, "get_visual_analyst", lambda: configured)
    service = query_routes.build_standard_agent_service()
    assert service._executor._visual is configured  # type: ignore[attr-defined]
    # Interpretation and wording stay model-free.
    assert isinstance(service._planner, StandardPlanner)  # type: ignore[attr-defined]
    assert isinstance(service._synthesizer, StandardReport)  # type: ignore[attr-defined]
