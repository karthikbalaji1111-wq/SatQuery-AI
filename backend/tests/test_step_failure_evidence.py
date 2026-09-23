"""A failed step is explained in the evidence, never left as a silent hole.

Observed live twice: a catalog outage (discovery) and a busy vision endpoint
(observation) each left the synthesiser evidence with a gap and no word about
why, so the run ended "Insufficient evidence" - a failure presented to the user
as a successful abstention. Each failure is now one plain execution text item
carrying the service's own message: no measurement, and never shaped as an
observation.
"""

from __future__ import annotations

from app.core.errors import UpstreamServiceError

from tests.test_agent_executor import FakeAnalysisService, execute_params, make_plan, run


def _failures(outcome):  # type: ignore[no-untyped-def]
    return [item for item in outcome.evidence.items if item.id.endswith("_failure")]


def test_an_analysis_failure_is_explained_and_discovery_evidence_survives() -> None:
    outcome, _, _ = run(
        make_plan(execute_params(), {"tool": "ndwi_statistics"}),
        analysis=FakeAnalysisService(error=UpstreamServiceError("provider down")),
    )

    [item] = _failures(outcome)
    assert item.id == "execution.analysis_failure"
    assert item.source == "execution"
    assert item.text == "The analysis did not complete: provider down"
    assert item.measurement is None
    assert item.visual is None
    # What discovery established is still there beside the explanation.
    assert any(
        i.id.startswith("execution.") and i.measurement is not None
        for i in outcome.evidence.items
    )


def test_a_successful_run_carries_no_failure_explanation() -> None:
    outcome, _, _ = run(make_plan(execute_params(), {"tool": "ndwi_statistics"}))
    assert _failures(outcome) == []


def test_a_failed_observation_is_explained_but_never_shaped_as_one() -> None:
    """The case observed live: the vision endpoint answered 503 three times."""

    from tests.test_agent_visual import RecordingAnalyst, build_executor, visual_plan
    from tests.test_agent_visual import run as run_visual

    message = "The language-model service is unavailable."
    executor, _, analyst = build_executor(
        analyst=RecordingAnalyst(error=UpstreamServiceError(message))
    )
    outcome = run_visual(executor, visual_plan())

    assert len(analyst.calls) == 1  # the image really was offered
    [item] = _failures(outcome)
    assert item.id == "execution.visual_failure"
    assert item.source == "execution"
    assert item.text == f"The visual observation did not complete: {message}"
    assert item.visual is None
    # No observation appears where none was made.
    assert not [i for i in outcome.evidence.items if i.source == "model"]
