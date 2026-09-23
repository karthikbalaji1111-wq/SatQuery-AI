"""A client-supplied execution result must agree with itself and its intent.

``POST /query/analyze`` takes a whole ``QueryExecutionResult`` in the body.
Pydantic proves the SHAPE; it cannot prove the parts agree. Each case below
parses perfectly and is nonetheless impossible:

    structural validity  !=  provenance integrity

The consequence is not an error somewhere later - it is a measurement reported
under a scene nobody discovered, or a window nobody requested, carried onward
into the evidence panel, the export and any grounded prose as though the server
had established it.

What is NOT claimed: that the scene exists, or that its pixels are what the body
says. Verifying that would mean re-running discovery. These tests pin the
narrower property - self-consistency, and consistency with the stated intent -
and the last test pins that a genuine result still passes, without which every
assertion here could be satisfied by refusing everything.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from app.main import create_app
from app.services.analysis.schemas import AnalysisRequest
from app.services.query.schemas import (
    ExecutedWindow,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    TimeRange,
)
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.test_query_execution import (
    DEFAULT_BBOX,
    make_imagery_response,
    make_scene,
)

CATALOG = "https://earth-search.aws.element84.com/v1"
ANALYZE_URL = "/api/v1/query/analyze"

JANUARY = TimeRange(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31))
FEBRUARY = TimeRange(start_date=date(2025, 2, 1), end_date=date(2025, 2, 28))


def single_intent(**overrides: Any) -> SatQueryIntent:
    body: dict[str, Any] = {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [JANUARY],
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    body.update(overrides)
    return SatQueryIntent(**body)


def compare_intent() -> SatQueryIntent:
    return SatQueryIntent(
        location_query="Chennai",
        temporal_mode="compare",
        time_windows={"baseline": JANUARY, "target": FEBRUARY},  # type: ignore[arg-type]
        modalities=["sentinel-2-optical"],
        task="visualize",
    )


def window(**overrides: Any) -> ExecutedWindow:
    body: dict[str, Any] = {
        "modality": "sentinel-2-optical",
        "label": "single",
        "time_range": JANUARY,
        "scene_count": 1,
        "scenes": [make_scene("scene-a", cloud_cover=4.0)],
        "selected_scene_id": "scene-a",
        "catalog": CATALOG,
    }
    body.update(overrides)
    return ExecutedWindow(**body)


def execution(
    intent: SatQueryIntent | None = None, *windows: ExecutedWindow
) -> QueryExecutionResult:
    resolved = intent or single_intent()
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=resolved, bbox=DEFAULT_BBOX),
        executed_modalities=list(resolved.modalities),
        skipped_modalities=[],
        windows=list(windows) or [window()],
        catalog=CATALOG,
    )


def refuses(result: QueryExecutionResult) -> str:
    """The reason this result was refused. Fails the test if it was accepted."""

    with pytest.raises(ValidationError) as raised:
        AnalysisRequest(execution=result)
    return str(raised.value)


# --------------------------------------------------------------------------- #
# The result must agree with itself
# --------------------------------------------------------------------------- #


def test_a_selected_scene_must_be_among_the_scenes_returned() -> None:
    """Every later read is keyed on this id; nothing else checks it exists."""

    reason = refuses(execution(None, window(selected_scene_id="never-discovered")))
    assert "not among the scenes it returned" in reason


def test_a_window_cannot_carry_more_scenes_than_it_found() -> None:
    reason = refuses(
        execution(
            None,
            window(
                scene_count=1,
                scenes=[make_scene("scene-a"), make_scene("scene-b")],
            ),
        )
    )
    assert "reports 1 scene(s) while carrying 2" in reason


def test_a_bounded_subset_of_the_matches_is_accepted() -> None:
    """The other direction is normal, not a defect.

    A catalog page is a bounded subset of what matched, so a window may report
    more scenes than it carries. Refusing that would make honest pagination
    unrepresentable - see the selection-scope policy.
    """

    request = AnalysisRequest(execution=execution(None, window(scene_count=12)))
    assert request.execution.windows[0].scene_count == 12


def test_imagery_must_belong_to_the_selected_scene() -> None:
    reason = refuses(
        execution(None, window(imagery=make_imagery_response("a-different-scene")))
    )
    assert "imagery for a different scene" in reason


def test_imagery_requires_a_selected_scene() -> None:
    reason = refuses(
        execution(
            None,
            window(
                selected_scene_id=None,
                scenes=[],
                scene_count=0,
                imagery=make_imagery_response("scene-a"),
            ),
        )
    )
    assert "imagery without having selected a scene" in reason


def test_a_failed_window_cannot_also_carry_results() -> None:
    """An outage and a successful discovery are different outcomes."""

    reason = refuses(execution(None, window(error="catalog down")))
    assert "discovery failure while also carrying results" in reason


def test_a_window_cannot_appear_twice() -> None:
    reason = refuses(execution(None, window(), window()))
    assert "appears twice" in reason


# --------------------------------------------------------------------------- #
# The result must agree with the intent it claims to answer
# --------------------------------------------------------------------------- #


def test_a_window_of_an_unrequested_modality_is_refused() -> None:
    reason = refuses(
        execution(None, window(modality="sentinel-1-sar", label="single"))
    )
    assert "the intent requested" in reason


def test_a_label_outside_the_systems_vocabulary_is_refused() -> None:
    """Labels are names for requested windows, not free text."""

    reason = refuses(execution(None, window(label="whatever-i-like")))
    assert "not one this system produces" in reason


def test_a_period_the_intent_never_requested_is_refused() -> None:
    """The binding check.

    A measurement attributed to dates nobody asked about answers a different
    question, and the evidence, the export and any grounded prose would carry
    that attribution onward as established fact.
    """

    reason = refuses(execution(None, window(time_range=FEBRUARY)))
    assert "covers a period the intent never requested" in reason


def test_a_loose_label_over_a_requested_period_is_accepted() -> None:
    """The other side of the line drawn above.

    A window labelled "baseline" over the one period the intent requested
    fabricates nothing; refusing it would enforce a naming convention rather
    than integrity.
    """

    request = AnalysisRequest(
        execution=execution(None, window(label="baseline", time_range=JANUARY))
    )
    assert request.execution.status == "completed"


def test_a_label_carrying_another_labels_period_is_refused() -> None:
    """The label is the KEY to a requested period.

    A window labelled "baseline" carrying the target's dates moves an answer to
    a period the user did not ask about, while every field still parses.
    """

    intent = compare_intent()
    reason = refuses(
        execution(
            intent,
            window(label="baseline", time_range=FEBRUARY),
            window(label="target", time_range=FEBRUARY),
        )
    )
    assert "a period the intent does not assign to that label" in reason


def test_executed_modalities_cannot_exceed_what_was_requested() -> None:
    intent = single_intent()
    result = QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=intent, bbox=DEFAULT_BBOX),
        executed_modalities=["sentinel-2-optical", "sentinel-1-sar"],
        skipped_modalities=[],
        windows=[window()],
        catalog=CATALOG,
    )

    assert "did not request" in refuses(result)


# --------------------------------------------------------------------------- #
# Non-vacuity, and the boundary itself
# --------------------------------------------------------------------------- #


def test_a_genuine_result_is_accepted() -> None:
    """Without this, refusing everything would satisfy every test above."""

    request = AnalysisRequest(execution=execution())
    assert request.execution.status == "completed"


def test_a_partial_result_is_accepted() -> None:
    """A failed window is legitimate - as long as it carries no results."""

    intent = compare_intent()
    request = AnalysisRequest(
        execution=execution(
            intent,
            window(label="baseline", time_range=JANUARY),
            window(
                label="target",
                time_range=FEBRUARY,
                scenes=[],
                scene_count=0,
                selected_scene_id=None,
                error="The satellite catalog is unavailable.",
                catalog=None,
            ),
        )
    )

    assert request.execution.status == "partial"


def test_the_endpoint_refuses_an_inconsistent_body_in_the_one_envelope() -> None:
    client = TestClient(create_app(), raise_server_exceptions=False)
    body = {
        "execution": execution(
            None, window(selected_scene_id="never-discovered")
        ).model_dump(mode="json")
    }

    response = client.post(ANALYZE_URL, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    # The reason names the RELATION that failed, never the offending value: a
    # body may carry anything, and echoing it back is how a request's own
    # contents return to its sender.
    assert "never-discovered" not in response.text
