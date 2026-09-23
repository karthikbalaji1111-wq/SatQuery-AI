"""One run, one id, and where the time went.

An agent run crosses a planner, a catalog, several windowed raster reads, a
vision model, a synthesizer and the grounding checks, each logging from its own
module. Under any concurrency those lines interleave and nothing said which
belonged to which request, so "the planner failed" could not be tied to the run
being investigated. Slowness was worse: a run taking two minutes gave no
indication of which stage spent them.

What these pin: every line emitted during a run carries that run's id; each
stage reports its own duration and outcome; a failing stage is recorded and the
exception still propagates untouched; concurrent runs never see each other's
id; and the fields written are names, ids and durations - never a question, a
place name or a credential, because a log line is the artefact most likely to
be shipped somewhere else.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from app.core.observability import (
    RUN_ID,
    RunIdFilter,
    current_run_id,
    stage,
    workflow,
)

ROUTES = Path(__file__).resolve().parents[1] / "app" / "api" / "routes"


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #


def test_a_run_has_an_id_and_gives_it_back() -> None:
    with workflow("agent") as run_id:
        assert run_id == current_run_id()
        assert len(run_id) == 12  # short enough to read off a terminal


def test_the_id_is_cleared_when_the_run_ends() -> None:
    assert current_run_id() == "-"
    with workflow("agent"):
        pass
    assert current_run_id() == "-"


def test_the_id_survives_a_failure() -> None:
    """A run that raises must not leak its id into the next one."""

    with pytest.raises(ValueError), workflow("agent"):
        raise ValueError("boom")

    assert current_run_id() == "-"


def test_every_record_is_stamped_with_the_run_in_scope() -> None:
    """The filter is what makes an interleaved log readable."""

    record = logging.LogRecord(
        "satquery.anything", logging.INFO, __file__, 1, "msg", None, None
    )
    token = RUN_ID.set("abc123abc123")
    try:
        assert RunIdFilter().filter(record) is True
        assert record.run_id == "abc123abc123"  # type: ignore[attr-defined]
    finally:
        RUN_ID.reset(token)


def test_a_record_outside_any_run_is_still_stamped() -> None:
    """The formatter interpolates run_id on EVERY line, so it must always exist."""

    record = logging.LogRecord(
        "satquery.anything", logging.INFO, __file__, 1, "msg", None, None
    )
    RunIdFilter().filter(record)

    assert record.run_id == "-"  # type: ignore[attr-defined]


def test_two_concurrent_runs_keep_separate_ids() -> None:
    """The reason this is a ContextVar and not a global."""

    observed: list[str] = []

    async def run_once() -> str:
        with workflow("agent") as run_id:
            # Yield control so the other run is definitely in flight.
            await asyncio.sleep(0)
            observed.append(current_run_id())
            return run_id

    async def both() -> list[str]:
        return list(await asyncio.gather(run_once(), run_once()))

    ids = asyncio.run(both())

    assert ids[0] != ids[1]
    # Each task read back its OWN id, never the other's.
    assert sorted(observed) == sorted(ids)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def test_a_stage_reports_its_duration_and_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="satquery.run"), stage("planning"):
        pass

    [line] = [m for m in messages(caplog) if m.startswith("stage=planning")]
    assert "outcome=ok" in line
    assert "ms=" in line


def test_a_failing_stage_is_recorded_and_the_error_still_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CatalogDownError(Exception):
        pass

    with (
        caplog.at_level(logging.INFO, logger="satquery.run"),
        pytest.raises(CatalogDownError),
        stage("execution"),
    ):
        raise CatalogDownError("upstream said no")

    [line] = [m for m in messages(caplog) if m.startswith("stage=execution")]
    # The exception CLASS, so the line is diagnostic...
    assert "outcome=error:CatalogDownError" in line
    # ...and NOT its message, which can carry upstream content.
    assert "upstream said no" not in line


def test_a_workflow_reports_its_own_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="satquery.run"), workflow(
        "agent", provider="local"
    ):
        pass

    [line] = [m for m in messages(caplog) if m.startswith("workflow=agent")]
    assert "outcome=ok" in line
    assert "provider=local" in line


def test_a_workflow_records_a_failure_as_such(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.INFO, logger="satquery.run"),
        pytest.raises(RuntimeError),
        workflow("agent"),
    ):
        raise RuntimeError("boom")

    [line] = [m for m in messages(caplog) if m.startswith("workflow=agent")]
    assert "outcome=error" in line


# --------------------------------------------------------------------------- #
# What must never be logged
# --------------------------------------------------------------------------- #


def test_the_agent_route_logs_the_backend_and_not_the_question() -> None:
    """A question is user content; the provider is configuration.

    Asserted at the call site, because that is where the decision is made about
    what a run is described BY.
    """

    source = (ROUTES / "query.py").read_text()
    opened = source.split("with workflow(", 1)[1].split(")", 1)[0]

    assert "provider" in opened
    assert "question" not in opened


def test_no_field_with_a_falsey_value_is_written(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Absent context is omitted rather than written as an empty claim."""

    with caplog.at_level(logging.INFO, logger="satquery.run"), workflow(
        "agent", provider="", model=None
    ):
        pass

    [line] = [m for m in messages(caplog) if m.startswith("workflow=agent")]
    assert "provider=" not in line
    assert "model=" not in line
