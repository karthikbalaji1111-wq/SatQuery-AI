"""Completed windows survive a later window's failure.

Execution runs one catalog search per (modality x window), sequentially, and a
single ``await`` was unguarded: the first catalog failure raised straight out of
``execute()``. A three-month request whose third month could not be reached
therefore returned nothing at all, discarding two months that had already been
searched - and the caller could not distinguish "the catalog was down" from
"the archive holds nothing there".

What is asserted here:

* completed windows keep their scenes and their selected scene;
* a failed window says WHY it is empty, distinctly from an empty archive;
* the result's ``status`` reports ``partial`` rather than claiming completeness;
* a run in which NOTHING succeeded still raises, so a total outage is never
  dressed up as a result;
* ``status`` is derived from the windows, so a client cannot assert it.

The mixed-modality case lives in ``test_query_execution.py`` beside the
contract it changed.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from app.core.errors import UpstreamServiceError
from app.services.query import QueryExecutionRequest, QueryExecutionResult
from app.services.query.schemas import (
    ExecutedWindow,
    ResolvedQueryPlan,
    SatQueryIntent,
    TimeRange,
)
from app.services.satellite.schemas import SceneSearchResponse

from tests.test_query_execution import (
    DEFAULT_BBOX,
    build_service,
    make_scene,
    make_search_response,
    run,
)

PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"


class SequencedSatellite:
    """Answers each search in order; an ``Exception`` entry is raised instead.

    The shared fake routes by collection, which cannot express "the second of
    three windows failed" - the case this module is about.
    """

    def __init__(self, *outcomes: SceneSearchResponse | Exception) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def search(self, request: Any) -> SceneSearchResponse:
        index = min(self.calls, len(self._outcomes) - 1)
        self.calls += 1
        outcome = self._outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def series_intent(*months: tuple[str, str]) -> SatQueryIntent:
    """A time-series intent over the given (start, end) date pairs."""

    return SatQueryIntent(
        location_query="Chennai",
        temporal_mode="timeseries",
        time_windows=[
            TimeRange(
                start_date=date.fromisoformat(start), end_date=date.fromisoformat(end)
            )
            for start, end in months
        ],
        modalities=["sentinel-2-optical"],
        task="visualize",
    )


JANUARY = ("2025-01-01", "2025-01-31")
FEBRUARY = ("2025-02-01", "2025-02-28")
MARCH = ("2025-03-01", "2025-03-31")


def execute(satellite: SequencedSatellite, intent: SatQueryIntent):
    service = build_service(satellite=satellite)  # type: ignore[arg-type]
    return run(service.execute(QueryExecutionRequest(intent=intent)))


# --------------------------------------------------------------------------- #
# The four orders the brief names
# --------------------------------------------------------------------------- #


def test_the_first_window_survives_a_second_window_failure() -> None:
    result = execute(
        SequencedSatellite(
            make_search_response(make_scene("january", cloud_cover=4.0)),
            UpstreamServiceError("The satellite catalog is unavailable."),
        ),
        series_intent(JANUARY, FEBRUARY),
    )

    assert result.status == "partial"
    first, second = result.windows
    assert first.selected_scene_id == "january"
    assert [scene.id for scene in first.scenes] == ["january"]
    assert first.error is None
    assert second.error == "The satellite catalog is unavailable."
    assert second.selected_scene_id is None


def test_a_failure_in_the_middle_does_not_stop_the_windows_after_it() -> None:
    """Execution continues rather than abandoning the rest of the request."""

    result = execute(
        SequencedSatellite(
            make_search_response(make_scene("january", cloud_cover=4.0)),
            UpstreamServiceError("The satellite catalog is unavailable."),
            make_search_response(make_scene("march", cloud_cover=9.0)),
        ),
        series_intent(JANUARY, FEBRUARY, MARCH),
    )

    assert result.status == "partial"
    assert [window.selected_scene_id for window in result.windows] == [
        "january",
        None,
        "march",
    ]
    assert [window.error is None for window in result.windows] == [True, False, True]


def test_a_failed_window_is_distinguishable_from_an_empty_archive() -> None:
    """Both have no scenes; only one of them is a failure.

    Without ``error`` the two are identical on the wire, and the UI reported an
    outage as "no scenes found" - which points the reader at the wrong problem.
    """

    result = execute(
        SequencedSatellite(
            make_search_response(),  # searched fine, matched nothing
            UpstreamServiceError("The satellite catalog is unavailable."),
        ),
        series_intent(JANUARY, FEBRUARY),
    )

    empty, failed = result.windows
    assert empty.scene_count == 0 and empty.error is None
    assert failed.scene_count == 0 and failed.error is not None


def test_every_window_failing_raises_rather_than_returning_emptiness() -> None:
    with pytest.raises(UpstreamServiceError):
        execute(
            SequencedSatellite(
                UpstreamServiceError("The satellite catalog is unavailable.")
            ),
            series_intent(JANUARY, FEBRUARY),
        )


def test_a_fully_successful_run_is_completed() -> None:
    """Non-vacuity: the status must not read ``partial`` for everything."""

    result = execute(
        SequencedSatellite(
            make_search_response(make_scene("january", cloud_cover=4.0)),
            make_search_response(make_scene("february", cloud_cover=6.0)),
        ),
        series_intent(JANUARY, FEBRUARY),
    )

    assert result.status == "completed"
    assert all(window.error is None for window in result.windows)


# --------------------------------------------------------------------------- #
# The status is derived, not asserted
# --------------------------------------------------------------------------- #


def plan_for(intent: SatQueryIntent) -> ResolvedQueryPlan:
    return ResolvedQueryPlan(intent=intent, bbox=DEFAULT_BBOX)


def window(label: str, *, error: str | None = None) -> ExecutedWindow:
    return ExecutedWindow(
        modality="sentinel-2-optical",
        label=label,
        time_range=TimeRange(
            start_date=date(2025, 1, 1), end_date=date(2025, 1, 31)
        ),
        scene_count=0,
        scenes=[],
        selected_scene_id=None,
        error=error,
    )


def test_a_client_cannot_claim_a_run_completed() -> None:
    """``/query/analyze`` accepts this model FROM a client.

    A stored status would be the client's own claim about its own payload.
    Derived from the windows, "completed" can only mean that these windows
    carry no failure.
    """

    result = QueryExecutionResult.model_validate(
        {
            "plan": plan_for(series_intent(JANUARY, FEBRUARY)).model_dump(mode="json"),
            "executed_modalities": ["sentinel-2-optical"],
            "skipped_modalities": [],
            "windows": [
                window("series[0]").model_dump(mode="json"),
                window("series[1]", error="catalog down").model_dump(mode="json"),
            ],
            "catalog": "https://earth-search.aws.element84.com/v1",
            "status": "completed",  # the lie
        }
    )

    assert result.status == "partial"


def test_no_windows_at_all_is_not_a_completed_run() -> None:
    result = QueryExecutionResult(
        plan=plan_for(series_intent(JANUARY, FEBRUARY)),
        executed_modalities=["sentinel-2-optical"],
        skipped_modalities=[],
        windows=[],
        catalog="https://earth-search.aws.element84.com/v1",
    )

    assert result.status == "failed"


# --------------------------------------------------------------------------- #
# Per-window provenance (mixed catalogs)
# --------------------------------------------------------------------------- #


def test_each_window_records_the_catalog_that_answered_it() -> None:
    result = execute(
        SequencedSatellite(
            make_search_response(make_scene("january", cloud_cover=4.0)),
            make_search_response(make_scene("february", cloud_cover=6.0), catalog=PLANETARY),
        ),
        series_intent(JANUARY, FEBRUARY),
    )

    first, second = result.windows
    assert first.catalog != second.catalog
    assert second.catalog == PLANETARY
    # Every service that answered, in the order first seen - so prose and the
    # UI can name both instead of implying one answered for everything.
    assert result.catalogs == [first.catalog, PLANETARY]
    # The top-level field stays, and is now deterministic: the FIRST answer.
    assert result.catalog == first.catalog
