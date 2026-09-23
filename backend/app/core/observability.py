"""Correlation and timing for one workflow.

One agent run crosses a planner, a catalog, several windowed raster reads, a
vision model, a synthesizer and the grounding checks - each logging from its own
module. Under any concurrency those lines interleave, and there was nothing to
tell which belonged to which request. Diagnosing "the planner failed" meant
guessing which of the interleaved failures belonged to the run being
investigated.

Two things fix that, and nothing more is added here:

* a **run id** carried in a :class:`~contextvars.ContextVar`, so every line
  emitted anywhere during that run is stamped with it automatically - no
  threading of a parameter through services that have no business knowing about
  logging;
* a **stage** timer, so where a slow run spent its time is a fact rather than a
  hypothesis.

``ContextVar`` rather than a thread local because this is asyncio: a task
inherits the current context at creation, so a value set here follows the run
into everything it awaits, while concurrent runs stay isolated. Work handed to
``run_in_threadpool`` inherits the context too, so a raster read logs under its
own run.

**No secrets, no payloads.** Fields are names, ids, durations and outcomes -
never a question, a place name, a credential or a response body. A log line is
the one artefact most likely to be shipped somewhere else.

This module imports nothing from the application, so ``core.logging`` can use
its filter without a circular import.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

#: The run currently in scope. ``"-"`` outside any workflow, so every record has
#: the attribute and the formatter never has to guard for it.
RUN_ID: ContextVar[str] = ContextVar("satquery_run_id", default="-")

logger = logging.getLogger("satquery.run")


class RunIdFilter(logging.Filter):
    """Stamp every record with the run in scope when it was emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID.get()
        return True


def current_run_id() -> str:
    """The run in scope, or ``"-"``. For a response header or an error body."""

    return RUN_ID.get()


def _fields(values: dict[str, object]) -> str:
    return " ".join(f"{key}={value}" for key, value in values.items() if value)


@contextmanager
def workflow(name: str, **fields: object) -> Iterator[str]:
    """Open a correlated workflow and time it end to end.

    Yields the run id so a caller can return it to the client. The id is short
    on purpose: it is read off a terminal and typed into a search, not parsed.
    """

    run_id = uuid.uuid4().hex[:12]
    token = RUN_ID.set(run_id)
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield run_id
    except BaseException:
        # Including cancellation: a client that disconnects mid-run is a fact
        # worth seeing, and it is not the same as a failure.
        outcome = "error"
        raise
    finally:
        logger.info(
            "workflow=%s outcome=%s ms=%.0f %s",
            name,
            outcome,
            (time.perf_counter() - started) * 1000,
            _fields(fields),
        )
        RUN_ID.reset(token)


@contextmanager
def stage(name: str, **fields: object) -> Iterator[None]:
    """Time one stage of the workflow in scope.

    A failing stage is logged with ``outcome=error`` and the exception is
    re-raised untouched: this records what happened, it never handles it.
    """

    started = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except BaseException as exc:
        # The exception CLASS, never its message: an upstream message can carry
        # content this module has no business writing to a log.
        outcome = f"error:{type(exc).__name__}"
        raise
    finally:
        logger.info(
            "stage=%s outcome=%s ms=%.0f %s",
            name,
            outcome,
            (time.perf_counter() - started) * 1000,
            _fields(fields),
        )
