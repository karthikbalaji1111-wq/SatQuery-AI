"""Measurements reach the synthesiser rounded for reading, never re-estimated.

Observed live: the answer read "The mean NDWI was 0.1463908465206975 index." -
the raw float, quoted faithfully because that is what the evidence line showed.
The evidence itself must keep full precision; only the displayed line changes.
The property that matters most is the last one here: every value shown is still
accepted by grounding, because display rounding that grounding refused would
turn a presentation fix into withheld answers.
"""

from __future__ import annotations

import pytest
from app.services.agent.grounding import DraftAnswer, validate_answer
from app.services.agent.prompts import _display_value, _render_evidence
from app.services.agent.schemas import AgentEvidence


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (0.1463908465206975, "0.1464"),
        (-0.06136076172332139, "-0.06136"),
        (0.9729119638826185, "0.9729"),
        (-0.7808971620384498, "-0.7809"),
        (0.011835606151453555, "0.01184"),
        (-12.345678, "-12.35"),
        (0.9, "0.9"),
        (0.99996, "1"),
        (33600.0, "33600"),
        (5.0, "5"),
        (0.0, "0"),
        (123456.7, "123457"),
    ],
)
def test_a_value_is_shown_to_four_significant_digits(value: float, shown: str) -> None:
    assert _display_value(value) == shown


def _evidence(index: str, value: float) -> AgentEvidence:
    return AgentEvidence.model_validate(
        {
            "items": [
                {
                    "id": f"{index}.{index}_mean",
                    "source": index,
                    "measurement": {"name": f"{index}_mean", "value": value, "unit": "index"},
                    "produced_by": "analysis.engines.test",
                }
            ]
        }
    )


def test_the_rendered_line_is_rounded_and_the_evidence_is_not() -> None:
    evidence = _evidence("ndwi", 0.1463908465206975)
    rendered = _render_evidence(evidence)

    assert "ndwi_mean = 0.1464 index" in rendered
    assert "0.1463908465206975" not in rendered
    # The authoritative value is untouched.
    assert evidence.items[0].measurement is not None
    assert evidence.items[0].measurement.value == 0.1463908465206975


@pytest.mark.parametrize("index", ["ndvi", "ndwi", "ndbi"])
@pytest.mark.parametrize(
    "value",
    [
        0.1463908465206975,
        -0.06136076172332139,
        0.9729119638826185,
        -0.7808971620384498,
        0.011835606151453555,
    ],
)
def test_every_displayed_mean_is_accepted_by_grounding(index: str, value: float) -> None:
    evidence = _evidence(index, value)
    shown = _display_value(value)
    draft = DraftAnswer(
        summary=f"The mean {index.upper()} was {shown} index.",
        evidence_refs=[f"{index}.{index}_mean"],
    )

    validation = validate_answer(draft, evidence)

    assert validation.numeric_grounding == "pass"
    assert validation.evidence_refs == "pass"


# --------------------------------------------------------------------------- #
# Provenance context
#
# Observed live: without it the synthesiser abstained on about half of direct
# "What is the NDWI of <place> in <month>?" questions, because the citable
# lines named a metric and a value but never where or when. With it, the same
# evidence was answered and grounded 6/6.
# --------------------------------------------------------------------------- #

from tests.test_agent_executor import make_execution_result  # noqa: E402


def _evidence_with_execution() -> AgentEvidence:
    return AgentEvidence.model_validate(
        {
            "items": [
                {
                    "id": "ndwi.ndwi_mean",
                    "source": "ndwi",
                    "measurement": {"name": "ndwi_mean", "value": 0.2777, "unit": "index"},
                    "produced_by": "analysis.engines.test",
                }
            ],
            "execution": make_execution_result(),
        }
    )


def _selected(evidence: AgentEvidence):  # type: ignore[no-untyped-def]
    assert evidence.execution is not None
    window = evidence.execution.windows[0]
    # Guards against a vacuous pass: the fixture must actually select a scene.
    assert window.selected_scene_id is not None
    scene = next(s for s in window.scenes if s.id == window.selected_scene_id)
    assert scene.datetime
    return window, scene


def test_the_context_names_what_was_queried_and_what_was_measured() -> None:
    evidence = _evidence_with_execution()
    window, scene = _selected(evidence)
    rendered = _render_evidence(evidence)

    assert evidence.execution is not None
    assert f"location queried: {evidence.execution.plan.intent.location_query}" in rendered
    assert window.time_range.start_date.isoformat() in rendered
    assert window.time_range.end_date.isoformat() in rendered
    assert f"measured scene {scene.id}, acquired {scene.datetime[:10]}" in rendered
    assert rendered.index("CONTEXT") < rendered.index("CITABLE ITEMS") < rendered.index(
        "ndwi.ndwi_mean"
    )


def test_the_context_carries_no_citable_id() -> None:
    rendered = _render_evidence(_evidence_with_execution())
    context = rendered.split("CITABLE ITEMS", 1)[0]
    assert "|" not in context


def test_without_an_execution_there_is_no_context() -> None:
    rendered = _render_evidence(_evidence("ndwi", 0.2777))
    assert "CONTEXT" not in rendered
    assert "ndwi_mean = 0.2777 index" in rendered


def test_empty_evidence_is_still_reported_as_empty() -> None:
    assert _render_evidence(AgentEvidence()) == "(no evidence was collected)"


def test_repeating_the_context_is_still_grounded() -> None:
    """Showing a model the scene and date must not make its answer withheld."""

    evidence = _evidence_with_execution()
    _, scene = _selected(evidence)
    draft = DraftAnswer(
        summary=(
            f"The mean NDWI was 0.2777 index. Scene {scene.id} was selected. "
            f"The scene was acquired on {scene.datetime[:10]}."
        ),
        evidence_refs=["ndwi.ndwi_mean"],
    )

    validation = validate_answer(draft, evidence)

    assert validation.numeric_grounding == "pass"
    assert validation.evidence_refs == "pass"
    assert validation.forbidden_terms == "pass"


def test_a_discovery_failure_can_be_reported_verbatim() -> None:
    """The explanation is only useful if the synthesiser may repeat it."""

    sentence = "Scene discovery did not complete: The satellite catalog is unavailable."
    evidence = AgentEvidence.model_validate(
        {
            "items": [
                {
                    "id": "execution.discovery_failure",
                    "source": "execution",
                    "text": sentence,
                    "produced_by": "agent.executor",
                }
            ]
        }
    )
    draft = DraftAnswer(summary=sentence, evidence_refs=["execution.discovery_failure"])

    validation = validate_answer(draft, evidence)

    assert validation.numeric_grounding == "pass"
    assert validation.evidence_refs == "pass"
    assert validation.forbidden_terms == "pass"
