"""Analysis service: interpretation of an already-computed query execution.

    QueryExecutionResult -> AnalysisService.analyze() -> AnalysisResult

This is a *pure* boundary. It performs no discovery, no STAC calls, no imagery
retrieval, no raster I/O, and no LLM/VLM inference; it holds no collaborators
and touches no network. It only reads the supplied execution result.

This phase implements no analysis engine. ``visualize`` is answered with a
deterministic, templated summary of what the execution actually retrieved;
every other task is reported as ``not_implemented`` (see
``analysis/schemas.py`` for why that is a 200 body rather than a 501 error).
Future engines (change detection, object identification, fusion) are dispatched
from here without changing this contract.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.services.analysis.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisWindowRef,
)
from app.services.base import DomainService
from app.services.query.schemas import ExecutedWindow, QueryExecutionResult, QueryTask

logger = get_logger("analysis")

_IMPLEMENTED_TASK: QueryTask = "visualize"


def _window_ref(window: ExecutedWindow) -> AnalysisWindowRef:
    """Project an executed window onto its slim, traceable reference."""

    return AnalysisWindowRef(
        modality=window.modality,
        label=window.label,
        time_range=window.time_range,
        selected_scene_id=window.selected_scene_id,
    )


def _collect_warnings(execution: QueryExecutionResult) -> list[str]:
    """Surface execution defects that limit any interpretation of the result.

    Ordered by window so the output is deterministic.
    """

    warnings: list[str] = []

    if not execution.windows:
        warnings.append(
            "The execution produced no windows; there is nothing to interpret."
        )

    for window in execution.windows:
        if window.selected_scene_id is None:
            warnings.append(
                f"No scene was selected for the {window.modality} window "
                f"{window.label!r}; that window contributes no observation."
            )
        if window.imagery_error is not None:
            warnings.append(
                f"Imagery was unavailable for the {window.modality} window "
                f"{window.label!r}: {window.imagery_error}"
            )

    for skipped in execution.skipped_modalities:
        warnings.append(f"Modality {skipped.modality} was skipped: {skipped.reason}")

    return warnings


def _describe_window(ref: AnalysisWindowRef) -> str:
    scene = ref.selected_scene_id if ref.selected_scene_id is not None else "no scene"
    return (
        f"{ref.modality} {ref.label} "
        f"({ref.time_range.start_date.isoformat()} to "
        f"{ref.time_range.end_date.isoformat()}) -> {scene}"
    )


def _visualize_answer(
    execution: QueryExecutionResult, refs: list[AnalysisWindowRef]
) -> str:
    """Deterministic templated summary of what the execution retrieved.

    Derived only from the supplied execution result - no inference, no model.
    """

    location = execution.plan.intent.location_query
    if not refs:
        return f"No imagery windows were executed for {location!r}."

    parts = "; ".join(_describe_window(ref) for ref in refs)
    return (
        f"Retrieved {len(refs)} window(s) for {location!r} from "
        f"{execution.catalog}: {parts}."
    )


def _not_implemented_answer(task: QueryTask, window_count: int) -> str:
    return (
        f"The {task!r} analysis is not implemented in this phase. The query "
        f"executed and {window_count} window(s) were retrieved, but no "
        f"{task!r} was performed and no result is claimed."
    )


class AnalysisService(DomainService):
    """Interprets a :class:`QueryExecutionResult` into an :class:`AnalysisResult`.

    The generic :meth:`run` hook stays unimplemented; :meth:`analyze` is the
    typed entry point. Constructed with no arguments - this service has no
    collaborators, no settings, and no external dependencies.
    """

    name = "analysis"

    def describe(self) -> str:
        return (
            "Deterministic interpretation of an executed SatQuery result: "
            "status, templated answer, window traceability, and warnings. "
            "No analysis engine, no imagery processing, no model inference."
        )

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Interpret ``request.execution``; the task comes from its intent."""

        execution = request.execution
        task = execution.plan.intent.task
        refs = [_window_ref(window) for window in execution.windows]
        warnings = _collect_warnings(execution)

        if task == _IMPLEMENTED_TASK:
            status = "ok"
            answer = _visualize_answer(execution, refs)
        else:
            status = "not_implemented"
            answer = _not_implemented_answer(task, len(refs))

        logger.info(
            "Analysed execution (task=%s, status=%s, windows=%d, warnings=%d)",
            task,
            status,
            len(refs),
            len(warnings),
        )

        return AnalysisResult(
            status=status,
            task=task,
            answer=answer,
            windows_considered=refs,
            warnings=warnings,
            measurements=[],
        )
