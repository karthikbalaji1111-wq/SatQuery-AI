"""A question that names an index by acronym gets that index computed.

Observed live with both providers: "What is the NDWI of Marina Beach, Chennai in
January 2025?" was planned as ``execute_query`` ALONE, so nothing was measured
and the run ended "Insufficient evidence" beside a perfectly good scene. These
tests pin the narrow completion that closes that gap - and, as importantly,
every case where it must leave the planner's plan alone.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.services.agent.plan_completion import ensure_requested_indices, named_indices
from app.services.agent.schemas import (
    AgentPlan,
    ExecuteQueryParams,
    SpectralIndicesParams,
)

from tests.test_agent_service import RecordingExecutor, RecordingPlanner, ask, build


def intent(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "location_query": "Marina Beach, Chennai",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    body.update(overrides)
    return body


def plan(*extra: dict[str, Any], **intent_overrides: Any) -> AgentPlan:
    return AgentPlan.model_validate(
        {"steps": [{"tool": "execute_query", "intent": intent(**intent_overrides)}, *extra]}
    )


def tools(p: AgentPlan) -> list[str]:
    return [step.tool for step in p.steps]


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the NDWI of Marina Beach?", ["ndwi"]),
        ("ndvi and NDBI for Delhi", ["ndvi", "ndbi"]),
        ("NDWI, then ndwi again", ["ndwi"]),
        ("How green is Chennai?", []),
        ("Is there water at Marina Beach?", []),
        ("ndwis are not a thing", []),
        ("scene S2B_NDWI_TILE", []),
    ],
)
def test_only_whole_acronyms_are_recognised(question: str, expected: list[str]) -> None:
    assert named_indices(question) == expected


# --------------------------------------------------------------------------- #
# Completion
# --------------------------------------------------------------------------- #


def test_a_discovery_only_plan_gains_the_named_index() -> None:
    original = plan()
    completed = ensure_requested_indices("What is the NDWI of Marina Beach?", original)

    assert tools(completed) == ["execute_query", "spectral_indices"]
    step = completed.steps[1]
    assert isinstance(step, SpectralIndicesParams)
    assert step.indices == ["ndwi"]


def test_completion_enables_the_imagery_the_analysis_is_displayed_beside() -> None:
    completed = ensure_requested_indices("NDVI of Chennai", plan())
    discovery = completed.steps[0]
    assert isinstance(discovery, ExecuteQueryParams)
    assert discovery.include_imagery is True


def test_completion_changes_nothing_about_where_or_when() -> None:
    original = plan()
    completed = ensure_requested_indices("NDBI of Marina Beach", original)
    assert completed.steps[0].intent == original.steps[0].intent  # type: ignore[union-attr]


def test_an_existing_spectral_step_is_extended_not_repeated() -> None:
    original = plan({"tool": "spectral_indices", "indices": ["ndvi"]})
    completed = ensure_requested_indices("Compare NDVI and NDBI", original)

    assert tools(completed) == ["execute_query", "spectral_indices"]
    step = completed.steps[1]
    assert isinstance(step, SpectralIndicesParams)
    assert step.indices == ["ndvi", "ndbi"]


def test_the_input_plan_is_not_mutated() -> None:
    original = plan({"tool": "spectral_indices", "indices": ["ndvi"]})
    ensure_requested_indices("NDVI and NDWI", original)
    step = original.steps[1]
    assert isinstance(step, SpectralIndicesParams)
    assert step.indices == ["ndvi"]


# --------------------------------------------------------------------------- #
# Every case that must be left alone - returned as the SAME object
# --------------------------------------------------------------------------- #


def test_a_plan_that_already_computes_the_index_is_returned_unchanged() -> None:
    original = plan({"tool": "spectral_indices", "indices": ["ndwi"]})
    assert ensure_requested_indices("NDWI please", original) is original


def test_the_dedicated_ndwi_tool_counts_as_computing_ndwi() -> None:
    original = plan({"tool": "ndwi_statistics"})
    assert ensure_requested_indices("NDWI please", original) is original


def test_a_question_naming_no_acronym_is_the_planners_judgement() -> None:
    original = plan()
    assert ensure_requested_indices("Is there water at Marina Beach?", original) is original


def test_a_comparison_is_left_to_its_own_tool() -> None:
    original = plan(
        temporal_mode="compare",
        time_windows={
            "baseline": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
            "target": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
        },
    )
    assert ensure_requested_indices("How did NDWI change?", original) is original


def test_an_index_is_never_requested_from_sar() -> None:
    original = plan(modalities=["sentinel-1-sar"])
    assert ensure_requested_indices("NDWI from Sentinel-1", original) is original


def test_a_full_step_budget_keeps_the_planners_steps() -> None:
    """Three steps is the cap. Appending would break it, and dropping a step the
    planner chose to make room would be worse than not completing."""

    original = plan(
        {"tool": "ndwi_statistics"},
        {"tool": "rs_model_analysis", "question": "What is visible?"},
    )
    assert ensure_requested_indices("NDVI of Chennai", original) is original


# --------------------------------------------------------------------------- #
# Through the service
# --------------------------------------------------------------------------- #


def test_the_service_executes_the_completed_plan_and_traces_the_requested_one() -> None:
    requested = plan()
    service, _, executor, _ = build(
        planner=RecordingPlanner(plan=requested), executor=RecordingExecutor()
    )
    result = ask(service, "What is the NDWI of Marina Beach, Chennai in January 2025?")

    [executed] = executor.calls
    assert tools(executed) == ["execute_query", "spectral_indices"]
    # The trace keeps what the planner actually asked for.
    assert result.trace.plan is not None
    assert tools(result.trace.plan) == ["execute_query"]


def test_the_service_leaves_a_complete_plan_alone() -> None:
    requested = plan({"tool": "ndwi_statistics"})
    service, _, executor, _ = build(
        planner=RecordingPlanner(plan=requested), executor=RecordingExecutor()
    )
    ask(service, "What is the NDWI of Chennai?")
    assert executor.calls == [requested]


def test_the_planning_instruction_says_when_each_task_applies() -> None:
    """Observed live: an NDVI question planned as object_identification, so the
    run was reported not_implemented beside four real NDVI measurements and the
    UI highlighted an unbuilt capability as the analysis that ran."""

    from app.services.agent.prompts import _system_instruction

    text = _system_instruction()
    assert '"visualize" for viewing imagery or for measuring an index' in text
    assert '"change_detection"\n  only with a "compare" temporal mode' in text
    assert "never choose it for a measurement" in text


# --------------------------------------------------------------------------- #
# Visual observation
# --------------------------------------------------------------------------- #

from app.services.agent.plan_completion import (  # noqa: E402
    asks_what_is_visible,
    complete_plan,
    ensure_requested_observation,
)
from app.services.agent.schemas import RsModelParams  # noqa: E402

VISIBLE = "Is there visible water in the Sentinel-2 image of Marina Beach, Chennai?"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (VISIBLE, True),
        ("What does the coastline look like?", True),
        ("Can you see ships near the port?", True),
        ("Is anything visually unusual here?", True),
        ("Show me Sentinel-2 imagery of Chennai", False),
        ("Get the image of Marina Beach", False),
        ("What is the NDWI of Marina Beach?", False),
    ],
)
def test_only_an_explicit_request_to_look_is_recognised(question: str, expected: bool) -> None:
    assert asks_what_is_visible(question) is expected


def test_a_visual_question_gains_the_observation_with_its_own_words() -> None:
    completed = ensure_requested_observation(VISIBLE, plan())

    assert tools(completed) == ["execute_query", "rs_model_analysis"]
    step = completed.steps[1]
    assert isinstance(step, RsModelParams)
    assert step.question == VISIBLE
    # The image the model is shown must be retrieved.
    discovery = completed.steps[0]
    assert isinstance(discovery, ExecuteQueryParams)
    assert discovery.include_imagery is True


def test_an_existing_observation_is_left_alone() -> None:
    original = plan({"tool": "rs_model_analysis", "question": "Is water visible?"})
    assert ensure_requested_observation(VISIBLE, original) is original


def test_a_request_merely_to_view_imagery_stays_discovery_only() -> None:
    original = plan()
    assert ensure_requested_observation("Show me imagery of Chennai", original) is original


def test_no_observation_is_requested_for_sar() -> None:
    original = plan(modalities=["sentinel-1-sar"])
    assert ensure_requested_observation("Is water visible in the SAR image?", original) is original


def test_no_observation_is_requested_for_a_comparison() -> None:
    original = plan(
        temporal_mode="compare",
        time_windows={
            "baseline": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
            "target": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
        },
    )
    assert ensure_requested_observation("Is more water visible now?", original) is original


def test_an_over_long_question_is_not_truncated_into_a_different_one() -> None:
    long_question = "Is water visible here? " + "x" * 600
    original = plan()
    assert ensure_requested_observation(long_question, original) is original


def test_indices_and_observation_complete_together_within_the_budget() -> None:
    question = "What is the NDWI, and is water visible in the image?"
    completed = complete_plan(question, plan())
    assert tools(completed) == ["execute_query", "spectral_indices", "rs_model_analysis"]


def test_a_full_budget_keeps_the_planners_steps_for_the_observation_too() -> None:
    original = plan({"tool": "ndwi_statistics"}, {"tool": "spectral_indices", "indices": ["ndvi"]})
    assert ensure_requested_observation(VISIBLE, original) is original


def test_the_service_runs_the_observation_the_question_asked_for() -> None:
    service, _, executor, _ = build(
        planner=RecordingPlanner(plan=plan()), executor=RecordingExecutor()
    )
    result = ask(service, VISIBLE)

    [executed] = executor.calls
    assert tools(executed) == ["execute_query", "rs_model_analysis"]
    assert result.trace.plan is not None
    assert tools(result.trace.plan) == ["execute_query"]


# --------------------------------------------------------------------------- #
# Temporal NDWI
# --------------------------------------------------------------------------- #

from app.services.agent.plan_completion import ensure_requested_comparison  # noqa: E402

COMPARE = {
    "temporal_mode": "compare",
    "time_windows": {
        "baseline": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        "target": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
    },
}
WATER_CHANGE = "How has water changed at Marina Beach between January 2024 and January 2025?"


def test_a_water_comparison_gains_temporal_ndwi() -> None:
    completed = ensure_requested_comparison(WATER_CHANGE, plan(**COMPARE))
    assert tools(completed) == ["execute_query", "temporal_ndwi_statistics"]


def test_naming_ndwi_in_a_comparison_counts_too() -> None:
    completed = ensure_requested_comparison("Compare NDWI across the two dates", plan(**COMPARE))
    assert tools(completed) == ["execute_query", "temporal_ndwi_statistics"]


def test_a_comparison_about_something_else_is_left_alone() -> None:
    original = plan(**COMPARE)
    assert ensure_requested_comparison("How has vegetation changed?", original) is original


def test_a_single_window_water_question_is_not_made_temporal() -> None:
    original = plan()
    assert ensure_requested_comparison("Is there water at Marina Beach?", original) is original


def test_a_sar_comparison_is_left_alone() -> None:
    original = plan(modalities=["sentinel-1-sar"], **COMPARE)
    assert ensure_requested_comparison(WATER_CHANGE, original) is original


def test_an_existing_temporal_step_is_left_alone() -> None:
    original = plan({"tool": "temporal_ndwi_statistics"}, **COMPARE)
    assert ensure_requested_comparison(WATER_CHANGE, original) is original


def test_the_service_runs_the_comparison_the_question_asked_for() -> None:
    service, _, executor, _ = build(
        planner=RecordingPlanner(plan=plan(**COMPARE)), executor=RecordingExecutor()
    )
    ask(service, WATER_CHANGE)
    [executed] = executor.calls
    assert tools(executed) == ["execute_query", "temporal_ndwi_statistics"]
