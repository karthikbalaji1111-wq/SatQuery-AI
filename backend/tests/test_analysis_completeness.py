"""A successful STATUS is not a produced ANALYSIS.

``AnalysisResult.status`` is derived from the TASK: ``visualize`` has an engine,
so the status is ``"ok"``. It says nothing about whether the analysis the
request asked for was produced. So an NDWI request over an execution with no
optical window returned:

    status: "ok"          measurements: []          warnings: ["...no window..."]

and every consumer keying on the status read a successful analysis that happened
to find nothing. The warning was there, but a warning is prose beside the result
rather than a field describing it.

Three questions were collapsed into one field, and they are now separate:

    transport success     - the HTTP request succeeded
    execution success     - the windows were retrieved (QueryExecutionResult.status)
    analysis completeness - the requested analysis was produced (HERE)

``status`` keeps its exact previous values, so nothing that reads it breaks.
"""

from __future__ import annotations

import asyncio

from app.services.analysis import AnalysisRequest, AnalysisService

from tests.test_analysis import (
    FakeImageryService,
    analyze_ndwi,
    make_execution,
    ndwi_execution,
)


def outcomes_by_name(result) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {outcome.name: outcome for outcome in result.analysis_outcomes}


# --------------------------------------------------------------------------- #
# The defect
# --------------------------------------------------------------------------- #


def test_a_requested_analysis_that_produced_nothing_is_not_complete() -> None:
    result, imagery = analyze_ndwi(make_execution(windows=[]))

    # The status is unchanged - the task really does have an engine.
    assert result.status == "ok"
    # ...and the new field says what actually happened.
    assert result.completeness == "none"
    ndwi = outcomes_by_name(result)["ndwi"]
    assert ndwi.status == "unavailable"  # type: ignore[attr-defined]
    # The reason is the service's own words, not a second explanation.
    assert "no Sentinel-2 optical window" in (ndwi.reason or "")  # type: ignore[attr-defined]
    assert imagery.calls == []  # nothing was read, so nothing is claimed


def test_a_produced_analysis_is_complete() -> None:
    """Non-vacuity: completeness must not read "none" for everything."""

    result, _ = analyze_ndwi(ndwi_execution())

    assert result.status == "ok"
    assert result.completeness == "complete"
    assert outcomes_by_name(result)["ndwi"].status == "completed"  # type: ignore[attr-defined]


def test_one_produced_and_one_unavailable_is_partial() -> None:
    """A run can honestly be half an answer.

    NDWI has an optical window and is computed; Sentinel-1 backscatter was also
    asked for and there is no SAR window, so it cannot be. Reporting either
    "ok" or "failed" for that run would be wrong in opposite directions.
    """

    service = AnalysisService(imagery_service=FakeImageryService())  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                execution=ndwi_execution(),
                include_ndwi=True,
                include_sar_backscatter=True,
            )
        )
    )

    assert result.completeness == "partial"
    outcomes = outcomes_by_name(result)
    assert outcomes["ndwi"].status == "completed"  # type: ignore[attr-defined]
    assert outcomes["sar_backscatter"].status == "unavailable"  # type: ignore[attr-defined]
    assert "no SAR window" in (outcomes["sar_backscatter"].reason or "")  # type: ignore[attr-defined]


def test_asking_for_nothing_is_not_an_incomplete_answer() -> None:
    """``not_requested`` is deliberately distinct from ``none``.

    A plain ``visualize`` request asks for no analysis, and getting none is a
    complete answer to the question it asked.
    """

    service = AnalysisService(imagery_service=FakeImageryService())  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(AnalysisRequest(execution=ndwi_execution()))
    )

    assert result.status == "ok"
    assert result.analysis_outcomes == []
    assert result.completeness == "not_requested"


# --------------------------------------------------------------------------- #
# One entry per analysis
# --------------------------------------------------------------------------- #


def test_the_same_index_asked_for_twice_is_reported_once() -> None:
    """``include_ndwi`` and ``indices=["ndwi"]`` ask for the same measurements."""

    service = AnalysisService(imagery_service=FakeImageryService())  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                execution=ndwi_execution(), include_ndwi=True, indices=["ndwi"]
            )
        )
    )

    names = [outcome.name for outcome in result.analysis_outcomes]
    assert names.count("ndwi") == 1
    assert result.completeness == "complete"


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #


def test_the_new_fields_are_additive_on_the_wire() -> None:
    result, _ = analyze_ndwi(ndwi_execution())
    body = result.model_dump(mode="json")

    # Everything a previous client read is still there, unchanged in meaning.
    for field in ("status", "task", "answer", "windows_considered", "warnings",
                  "measurements"):
        assert field in body
    assert body["status"] == "ok"
    # ...beside the two new ones.
    assert body["completeness"] == "complete"
    assert body["analysis_outcomes"][0]["name"] == "ndwi"
