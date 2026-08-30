"""Analysis boundary contracts.

This layer interprets an *already-computed* :class:`QueryExecutionResult`. It
introduces no geometry, scene, imagery, plan, or intent shape of its own -
``QueryExecutionResult`` (and everything nested inside it) is reused verbatim.

The analysis task is **derived** from ``execution.plan.intent.task`` rather than
being passed separately, so there is exactly one source of truth for it.

Deliberate status convention: a task this phase does not implement is reported
as ``status="not_implemented"`` inside a normal HTTP 200 body, *not* as an
:class:`app.core.errors.NotImplementedFeatureError` (HTTP 501). The analysis did
run - it inspected the execution result and produced traceability
(``windows_considered``) and ``warnings`` - so returning an error body would
discard information the caller needs. Genuine failures (malformed request)
still surface through the existing error handlers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.services.query.schemas import (
    Modality,
    QueryExecutionResult,
    QueryTask,
    TimeRange,
)

#: ``ok``              - the requested analysis ran and produced an answer.
#: ``not_implemented`` - the task is recognised but no engine exists for it yet.
#: Only values this phase can actually produce are listed; more are added when
#: an engine can emit them.
AnalysisStatus = Literal["ok", "not_implemented"]


class Measurement(BaseModel):
    """A single quantitative result produced by a deterministic engine.

    ``unit`` is free-form (e.g. ``"km^2"``, ``"count"``, ``"%"``) - the repo has
    no unit vocabulary and this phase does not invent one. No measurement is
    produced in this phase; the model exists so the contract is stable.
    """

    name: str
    value: float
    unit: str


class AnalysisWindowRef(BaseModel):
    """Slim, traceable reference to one executed (modality, window) pair.

    Deliberately *not* an :class:`ExecutedWindow`: the discovered ``scenes``
    list and the bounded ``imagery`` payload are never echoed back.
    """

    modality: Modality
    label: str
    time_range: TimeRange
    selected_scene_id: str | None


class AnalysisRequest(BaseModel):
    """Input to the analysis boundary.

    ``execution`` is the contract ``/query/execute`` produces, reused verbatim
    and re-validated. There is no separate ``task`` field - see the module
    docstring.

    ``include_ndwi`` opts in to single-scene Sentinel-2 NDWI statistics. It is a
    flag rather than a new :data:`QueryTask` value deliberately: a spectral
    index is a descriptive add-on, not one of the three user intents, and a new
    task value would have to propagate into ``SatQueryIntent``, the intent
    parser's instructions and the frontend task list. Defaulting to ``False``
    keeps every existing request byte-identical in behaviour.
    """

    execution: QueryExecutionResult
    include_ndwi: bool = False


class AnalysisResult(BaseModel):
    """Structured, deterministic interpretation of a query execution."""

    status: AnalysisStatus
    task: QueryTask
    answer: str
    windows_considered: list[AnalysisWindowRef]
    warnings: list[str] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
