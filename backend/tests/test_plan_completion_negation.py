"""Plan completion must read the request, not scan it for acronyms.

``ensure_requested_indices`` exists because a planner can return a structurally
valid plan that omits the analysis the user named outright. The mechanism that
closes that gap - "the question contains NDVI, so compute NDVI" - is substring
presence, and substring presence is not evidence of intent:

    "Show imagery only. Do not calculate NDVI."

contains ``NDVI`` and asks for the opposite. Adding the step there is not a
harmless extra: it spends two more band reads, puts an unrequested measurement
in the evidence, and lets a synthesized answer discuss an index the user
explicitly refused.

These tests pin the four ways an acronym can appear without being a request -
negation, quotation, hypothetical and definition - and, as importantly, pin that
a real request still gets its analysis. Completion may only ever ADD work, so
the safe direction is to add nothing when the mention is not a request.

Every expectation here is about the PUBLIC completion functions, not about how
they classify internally, so the implementation stays free to change.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.services.agent.plan_completion import (
    complete_plan,
    ensure_requested_comparison,
    ensure_requested_indices,
    ensure_requested_observation,
)
from app.services.agent.schemas import AgentPlan, SpectralIndicesParams

SINGLE_WINDOW = [{"start_date": "2025-01-01", "end_date": "2025-01-31"}]
COMPARE_WINDOWS = {
    "baseline": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
    "target": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
}


def plan(*extra: dict[str, Any], **intent_overrides: Any) -> AgentPlan:
    intent: dict[str, Any] = {
        "location_query": "Marina Beach, Chennai",
        "temporal_mode": "single",
        "time_windows": SINGLE_WINDOW,
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    intent.update(intent_overrides)
    return AgentPlan.model_validate(
        {"steps": [{"tool": "execute_query", "intent": intent}, *extra]}
    )


def compare_plan(*extra: dict[str, Any]) -> AgentPlan:
    return plan(*extra, temporal_mode="compare", time_windows=COMPARE_WINDOWS)


def tools(p: AgentPlan) -> list[str]:
    return [step.tool for step in p.steps]


def added_indices(p: AgentPlan) -> list[str]:
    """Every index the plan computes, in order. Empty when it computes none."""

    return [
        key
        for step in p.steps
        if isinstance(step, SpectralIndicesParams)
        for key in step.indices
    ]


# --------------------------------------------------------------------------- #
# Explicit negation
#
# The user named the index in order to refuse it. Each phrasing below was
# written as a separate case rather than a shared regex, because the point is
# the BEHAVIOUR under ordinary English, not a particular cue list.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question",
    [
        "Show imagery only. Do not calculate NDVI.",
        "Do not calculate NDWI.",
        "Don't compute NDVI.",
        "Show me Chennai without NDVI.",
        "Skip the NDBI computation.",
        "No NDWI - just the picture, please.",
        "Exclude NDBI from the analysis.",
        "Never compute NDWI for this scene.",
        "Show the true-colour image rather than NDVI.",
        "I want imagery instead of NDBI.",
    ],
)
def test_a_refused_index_is_never_added(question: str) -> None:
    original = plan()
    completed = ensure_requested_indices(question, original)

    assert added_indices(completed) == []
    # Nothing was added at all, so the planner's own plan runs untouched.
    assert completed is original


def test_negation_scopes_over_every_index_that_follows_it() -> None:
    """"Do not compute NDVI and NDBI" refuses BOTH, not just the first."""

    completed = ensure_requested_indices(
        "Do not compute NDVI and NDBI.", plan()
    )
    assert added_indices(completed) == []


def test_a_request_and_a_refusal_in_one_sentence_are_separated() -> None:
    """The refusal scopes forward; what precedes it is still a request."""

    completed = ensure_requested_indices("Show NDWI, not NDVI.", plan())
    assert added_indices(completed) == ["ndwi"]


def test_a_contrastive_clause_restores_the_request() -> None:
    completed = ensure_requested_indices(
        "Do not bother with the imagery, but do compute NDWI.", plan()
    )
    assert added_indices(completed) == ["ndwi"]


# --------------------------------------------------------------------------- #
# Mentioned, not requested
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question",
    [
        # Quotation: the acronym is the subject of the sentence, not an order.
        'Explain what "NDBI" stands for.',
        # Definition.
        "NDVI is a normalized difference vegetation index.",
        "The textbook defines NDWI as a normalized difference of green and NIR.",
        "What is NDWI?",
        # Hypothetical.
        "If I wanted NDVI, would this scene work? Just show the imagery for now.",
        "Suppose NDBI were available - would it help here?",
    ],
)
def test_a_mention_is_not_a_request(question: str) -> None:
    completed = ensure_requested_indices(question, plan())
    assert added_indices(completed) == []


# --------------------------------------------------------------------------- #
# Non-vacuity: a real request still gets its analysis
#
# Without these, every test above would pass on a function that simply never
# added anything.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the NDWI of Marina Beach, Chennai in January 2025?", ["ndwi"]),
        ("Compute NDVI and NDBI for Chennai.", ["ndvi", "ndbi"]),
        ("NDWI of Marina Beach", ["ndwi"]),
        ("What is the NDVI of this area?", ["ndvi"]),
        # "what is ... of <place>" asks for a measurement, not a definition.
        ("What is the NDBI over Chennai?", ["ndbi"]),
        # A measurement request with no place after it at all. An earlier
        # version of the definition rule refused this one.
        ("What is the NDWI, and is water visible in the image?", ["ndwi"]),
    ],
)
def test_an_explicit_request_still_gets_its_index(
    question: str, expected: list[str]
) -> None:
    completed = ensure_requested_indices(question, plan())
    assert added_indices(completed) == expected


# --------------------------------------------------------------------------- #
# The same rule for the other two completions
# --------------------------------------------------------------------------- #


def test_a_refused_visual_observation_is_never_added() -> None:
    completed = ensure_requested_observation(
        "Do not describe what is visible; compute NDWI instead.", plan()
    )
    assert "rs_model_analysis" not in tools(completed)


def test_a_requested_visual_observation_is_still_added() -> None:
    completed = ensure_requested_observation(
        "Is there visible water at Marina Beach?", plan()
    )
    assert "rs_model_analysis" in tools(completed)


def test_a_refused_water_comparison_is_never_added() -> None:
    completed = ensure_requested_comparison(
        "Compare January and March, but do not compute water indices.",
        compare_plan(),
    )
    assert "temporal_ndwi_statistics" not in tools(completed)


def test_a_requested_water_comparison_is_still_added() -> None:
    completed = ensure_requested_comparison(
        "How has water changed between January 2024 and January 2025?",
        compare_plan(),
    )
    assert "temporal_ndwi_statistics" in tools(completed)


# --------------------------------------------------------------------------- #
# End to end, through the function the service actually calls
# --------------------------------------------------------------------------- #


def test_the_full_completion_respects_a_refusal() -> None:
    question = "Show the Sentinel-2 image of Marina Beach. Do not calculate NDVI."
    completed = complete_plan(question, plan())

    assert added_indices(completed) == []
    assert tools(completed) == ["execute_query"]


def test_the_full_completion_still_serves_a_request() -> None:
    question = "What is the NDWI of Marina Beach in January 2025?"
    completed = complete_plan(question, plan())

    assert added_indices(completed) == ["ndwi"]
