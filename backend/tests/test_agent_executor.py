"""Phase 15 Commit 2 - deterministic executor tests.

The executor is the only place a validated plan turns into real work. It holds
exactly two service handles and nothing else: no imagery service, no raster
handle, no HTTP client, no provider. There is no model call anywhere in this
commit.

Two invariants get the most attention here because they are architectural
rather than incidental:

* **Analysis calls are coalesced.** However many analysis tools a plan names,
  the executor makes exactly ONE ``AnalysisService.analyze`` call with the
  union of their flags. Calling analyze once per tool would re-run discovery
  interpretation and double the band reads.
* **``limit`` is server-controlled.** It was removed from the model-facing
  parameters in the Commit 1 hardening pass; the executor injects it. A planner
  cannot influence the resource budget.

Recording fakes only - no ``unittest.mock``, matching the repository's
convention. No test contacts Gemini, STAC, Nominatim or real imagery.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from typing import Any

import pytest
from app.core.errors import (
    ImageryError,
    InvalidInputError,
    NotFoundError,
    UpstreamServiceError,
)
from app.services.agent import executor as executor_mod
from app.services.agent.executor import SERVER_QUERY_LIMIT, AgentExecutor
from app.services.agent.schemas import (
    AgentEvidence,
    AgentPlan,
    AgentToolStep,
    ExecuteQueryParams,
    NdwiParams,
)
from app.services.analysis.schemas import (
    AnalysisRequest,
    AnalysisResult,
    Measurement,
    ObservationIndexResult,
    TemporalIndexComparison,
)
from app.services.geospatial.schemas import BoundingBox
from app.services.query.compatibility import compute_compatibility
from app.services.query.schemas import (
    ExecutedWindow,
    Observation,
    QueryExecutionRequest,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    TimeRange,
)
from app.services.satellite.schemas import Scene

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
CATALOG = "https://earth-search.aws.element84.com/v1"
S2 = "sentinel-2-optical"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def intent_dict(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
        "modalities": [S2],
        "task": "visualize",
    }
    body.update(overrides)
    return body


def execute_params(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool": "execute_query", "intent": intent_dict()}
    payload.update(overrides)
    return payload


def make_plan(*steps: dict[str, Any]) -> AgentPlan:
    return AgentPlan.model_validate(
        {"steps": list(steps) if steps else [execute_params()]}
    )


def make_scene(scene_id: str, *, datetime_: str = "2024-01-15T05:00:00Z") -> Scene:
    return Scene(
        id=scene_id,
        datetime=datetime_,
        bbox=DEFAULT_BBOX,
        geometry=None,
        cloud_cover=1.0,
        collection="sentinel-2-l2a",
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
    )


def make_window(*, label: str = "single", scene: Scene) -> ExecutedWindow:
    return ExecutedWindow(
        modality=S2,
        label=label,
        time_range=TimeRange.model_validate(
            {"start_date": "2024-01-01", "end_date": "2024-01-31"}
        ),
        scene_count=3,
        scenes=[scene],
        selected_scene_id=scene.id,
        imagery=None,
        imagery_error=None,
    )


def make_execution_result() -> QueryExecutionResult:
    intent = SatQueryIntent.model_validate(intent_dict())
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=intent, bbox=DEFAULT_BBOX),
        executed_modalities=[S2],
        skipped_modalities=[],
        windows=[make_window(scene=make_scene("scene-a"))],
        catalog=CATALOG,
    )


def make_observation(scene_id: str, label: str) -> Observation:
    return Observation(
        modality=S2,
        window_label=label,
        requested_window=TimeRange.model_validate(
            {"start_date": "2024-01-01", "end_date": "2024-01-31"}
        ),
        scene=make_scene(scene_id),
        imagery=None,
    )


def make_index_result(label: str, scene_id: str, mean: float) -> ObservationIndexResult:
    return ObservationIndexResult(
        window_label=label,
        scene_id=scene_id,
        acquired_at=None,
        cloud_cover=1.0,
        measurements=[
            Measurement(name="ndwi_valid_pixel_count", value=100.0, unit="pixels"),
            Measurement(name="ndwi_mean", value=mean, unit="index"),
        ],
        crs="EPSG:32644",
        resolution=10.0,
        window_pixel_count=121,
    )


def make_temporal_comparison() -> TemporalIndexComparison:
    return TemporalIndexComparison(
        first=make_index_result("baseline", "scene-a", 0.2),
        second=make_index_result("target", "scene-b", 0.5),
        compatibility=compute_compatibility(
            make_observation("scene-a", "baseline"),
            make_observation("scene-b", "target"),
        ),
        differences=[
            Measurement(name="mean_ndwi_difference", value=0.3, unit="index")
        ],
        warnings=["mean_ndwi_difference is a difference between aggregates."],
    )


def make_analysis_result(
    *,
    measurements: list[Measurement] | None = None,
    warnings: list[str] | None = None,
    temporal: TemporalIndexComparison | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        status="ok",
        task="visualize",
        answer="Retrieved 1 window(s).",
        windows_considered=[],
        warnings=warnings or [],
        measurements=measurements or [],
        temporal_comparison=temporal,
    )


class FakeQueryExecutionService:
    """Records the QueryExecutionRequest it is handed."""

    def __init__(self, *, result: QueryExecutionResult | None = None,
                 error: Exception | None = None) -> None:
        self.calls: list[QueryExecutionRequest] = []
        self._result = result if result is not None else make_execution_result()
        self._error = error

    async def execute(self, request: QueryExecutionRequest) -> QueryExecutionResult:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return self._result


class FakeAnalysisService:
    """Records every AnalysisRequest, so coalescing is directly observable."""

    def __init__(self, *, result: AnalysisResult | None = None,
                 error: Exception | None = None) -> None:
        self.calls: list[AnalysisRequest] = []
        self._result = result if result is not None else make_analysis_result()
        self._error = error

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return self._result


def run(
    plan: AgentPlan,
    *,
    query: FakeQueryExecutionService | None = None,
    analysis: FakeAnalysisService | None = None,
) -> tuple[Any, FakeQueryExecutionService, FakeAnalysisService]:
    query = query or FakeQueryExecutionService()
    analysis = analysis or FakeAnalysisService()
    executor = AgentExecutor(
        query_execution_service=query,  # type: ignore[arg-type]
        analysis_service=analysis,  # type: ignore[arg-type]
    )
    return asyncio.run(executor.execute(plan)), query, analysis


# =========================================================================== #
# A. execute_query
# =========================================================================== #


def test_execute_query_invokes_the_query_execution_service() -> None:
    outcome, query, analysis = run(make_plan())

    assert len(query.calls) == 1
    assert analysis.calls == []
    assert outcome.steps[0].status == "ok"
    assert outcome.steps[0].tool == "execute_query"


def test_the_validated_intent_reaches_the_service_unchanged() -> None:
    _, query, _ = run(make_plan(execute_params()))

    sent = query.calls[0].intent
    assert isinstance(sent, SatQueryIntent)
    assert sent.location_query == "Chennai"
    assert sent.modalities == [S2]
    assert sent.task == "visualize"


def test_include_imagery_and_max_cloud_cover_reach_the_service() -> None:
    _, query, _ = run(
        make_plan(execute_params(include_imagery=True, max_cloud_cover=20.0))
    )

    assert query.calls[0].include_imagery is True
    assert query.calls[0].max_cloud_cover == 20.0


def test_the_executor_supplies_the_server_side_limit() -> None:
    """``limit`` is a resource budget, never a planner decision."""

    _, query, _ = run(make_plan())

    assert query.calls[0].limit == SERVER_QUERY_LIMIT
    # And the model-facing contract has no way to express it.
    assert "limit" not in ExecuteQueryParams.model_fields


def test_the_server_limit_matches_the_existing_default() -> None:
    """No new budget is invented; the executor reuses the repository's."""

    assert QueryExecutionRequest.model_fields["limit"].default == SERVER_QUERY_LIMIT


def test_the_execution_result_is_carried_into_evidence() -> None:
    outcome, _, _ = run(make_plan())

    assert isinstance(outcome.evidence, AgentEvidence)
    assert outcome.evidence.execution is not None
    assert outcome.evidence.execution.catalog == CATALOG


# =========================================================================== #
# B/C/D. Analysis dispatch and coalescing
# =========================================================================== #


def test_ndwi_only_sets_only_the_ndwi_flag() -> None:
    _, _, analysis = run(make_plan(execute_params(), {"tool": "ndwi_statistics"}))

    assert len(analysis.calls) == 1
    assert analysis.calls[0].include_ndwi is True
    assert analysis.calls[0].include_temporal_ndwi is False


def test_temporal_only_sets_only_the_temporal_flag() -> None:
    _, _, analysis = run(
        make_plan(execute_params(), {"tool": "temporal_ndwi_statistics"})
    )

    assert len(analysis.calls) == 1
    assert analysis.calls[0].include_ndwi is False
    assert analysis.calls[0].include_temporal_ndwi is True


def test_both_analysis_tools_coalesce_into_exactly_one_analyze_call() -> None:
    """The architectural invariant: one analyze call, union of the flags."""

    _, _, analysis = run(
        make_plan(
            execute_params(),
            {"tool": "ndwi_statistics"},
            {"tool": "temporal_ndwi_statistics"},
        )
    )

    assert len(analysis.calls) == 1, "analyze() must not be called per tool"
    assert analysis.calls[0].include_ndwi is True
    assert analysis.calls[0].include_temporal_ndwi is True


def test_analysis_receives_the_execution_result_from_the_discovery_step() -> None:
    result = make_execution_result()
    _, _, analysis = run(
        make_plan(execute_params(), {"tool": "ndwi_statistics"}),
        query=FakeQueryExecutionService(result=result),
    )

    assert analysis.calls[0].execution == result


def test_no_analysis_call_when_the_plan_has_no_analysis_tool() -> None:
    _, _, analysis = run(make_plan())
    assert analysis.calls == []


# =========================================================================== #
# E. Safe failure on a plan that bypassed validation
# =========================================================================== #


def test_a_plan_without_a_discovery_step_executes_nothing() -> None:
    """model_construct bypasses validation; the executor must still be safe."""

    malformed = AgentPlan.model_construct(steps=[NdwiParams()])
    outcome, query, analysis = run(malformed)

    assert query.calls == []
    assert analysis.calls == []
    assert [step.status for step in outcome.steps] == ["skipped"]
    assert outcome.steps[0].rejection_reason


def test_an_unregistered_tool_is_refused_before_anything_executes() -> None:
    """A tool absent from the allowlist can never be dispatched.

    Classification completes before any service call, so a plan carrying an
    unpermitted tool executes NOTHING - not even its otherwise-valid discovery
    step. The refusal is raised rather than recorded because such a step is
    unrepresentable in the trace: ``AgentToolStep.parameters`` is the closed
    ``ToolCall`` union, so a rogue tool cannot be described by the contract.
    """

    class Rogue(NdwiParams):
        tool: Any = "rs_model_analysis"  # type: ignore[assignment]

    malformed = AgentPlan.model_construct(
        steps=[ExecuteQueryParams.model_validate(execute_params()), Rogue()]
    )
    query = FakeQueryExecutionService()
    analysis = FakeAnalysisService()

    with pytest.raises(InvalidInputError) as exc:
        run(malformed, query=query, analysis=analysis)

    assert "rs_model_analysis" in exc.value.message
    assert query.calls == []  # nothing ran at all
    assert analysis.calls == []


def test_a_rogue_tool_cannot_even_be_recorded_in_the_trace() -> None:
    """The contract, not the executor, is what makes this unrepresentable."""

    from pydantic import ValidationError

    class Rogue(NdwiParams):
        tool: Any = "rs_model_analysis"  # type: ignore[assignment]

    with pytest.raises(ValidationError):
        AgentToolStep(status="rejected", parameters=Rogue())  # type: ignore[arg-type]


# =========================================================================== #
# F. Trace
# =========================================================================== #


def test_every_planned_step_becomes_a_trace_step_in_order() -> None:
    outcome, _, _ = run(
        make_plan(
            execute_params(),
            {"tool": "ndwi_statistics"},
            {"tool": "temporal_ndwi_statistics"},
        )
    )

    assert [step.tool for step in outcome.steps] == [
        "execute_query",
        "ndwi_statistics",
        "temporal_ndwi_statistics",
    ]
    assert all(isinstance(step, AgentToolStep) for step in outcome.steps)
    assert all(step.status == "ok" for step in outcome.steps)


def test_trace_parameters_are_the_validated_call_actually_executed() -> None:
    plan = make_plan(execute_params(include_imagery=True))
    outcome, _, _ = run(plan)

    assert outcome.steps[0].parameters is plan.steps[0]
    assert isinstance(outcome.steps[0].parameters, ExecuteQueryParams)
    assert outcome.steps[0].parameters.include_imagery is True


def test_trace_steps_carry_no_second_tool_name_or_reasoning() -> None:
    outcome, _, _ = run(make_plan())
    step = outcome.steps[0]

    assert "tool" not in AgentToolStep.model_fields  # derived, not stored
    for banned in ("reasoning", "thoughts", "thinking", "chain_of_thought"):
        assert banned not in AgentToolStep.model_fields
        assert not hasattr(step, banned)


# =========================================================================== #
# G. Errors
# =========================================================================== #


@pytest.mark.parametrize(
    "error",
    [
        NotFoundError("Scene 'x' was not found in the catalog."),
        InvalidInputError("The requested bbox does not intersect the scene."),
        UpstreamServiceError("The satellite catalog is unavailable."),
        ImageryError("Failed to read the requested raster window."),
    ],
)
def test_a_discovery_failure_is_recorded_and_later_steps_are_skipped(
    error: Exception,
) -> None:
    outcome, _, analysis = run(
        make_plan(execute_params(), {"tool": "ndwi_statistics"}),
        query=FakeQueryExecutionService(error=error),
    )

    assert outcome.steps[0].status == "failed"
    assert outcome.steps[0].error_message == str(error)
    assert outcome.steps[1].status == "skipped"
    assert analysis.calls == []  # nothing to analyse
    assert outcome.evidence.execution is None


def test_an_analysis_failure_marks_every_analysis_step_failed() -> None:
    outcome, query, _ = run(
        make_plan(
            execute_params(),
            {"tool": "ndwi_statistics"},
            {"tool": "temporal_ndwi_statistics"},
        ),
        analysis=FakeAnalysisService(error=UpstreamServiceError("provider down")),
    )

    assert len(query.calls) == 1
    assert outcome.steps[0].status == "ok"  # discovery still succeeded
    assert [s.status for s in outcome.steps[1:]] == ["failed", "failed"]
    assert all(s.error_message == "provider down" for s in outcome.steps[1:])
    # The discovery evidence survives an analysis failure.
    assert outcome.evidence.execution is not None
    assert outcome.evidence.analysis is None


def test_an_unexpected_exception_is_not_swallowed() -> None:
    """Only handled AppErrors become steps; a bug must surface, not vanish."""

    class BugError(RuntimeError):
        pass

    with pytest.raises(BugError):
        run(make_plan(), query=FakeQueryExecutionService(error=BugError("bug")))


def test_error_messages_are_service_messages_not_invented_text() -> None:
    outcome, _, _ = run(
        make_plan(),
        query=FakeQueryExecutionService(
            error=UpstreamServiceError("The satellite catalog timed out.")
        ),
    )
    assert outcome.steps[0].error_message == "The satellite catalog timed out."


# =========================================================================== #
# H. Deterministic service behaviour is preserved, not reinterpreted
# =========================================================================== #


def test_analysis_warnings_are_preserved_verbatim() -> None:
    warnings = [
        "Temporal NDWI statistics were requested but no Sentinel-2 pair could "
        "be formed: only 1 sentinel-2-optical observation is available.",
    ]
    outcome, _, _ = run(
        make_plan(execute_params(), {"tool": "temporal_ndwi_statistics"}),
        analysis=FakeAnalysisService(
            result=make_analysis_result(warnings=warnings)
        ),
    )

    assert outcome.evidence.analysis is not None
    assert outcome.evidence.analysis.warnings == warnings


def test_a_semantically_impossible_step_still_reports_ok_with_the_warning() -> None:
    """The executor does not second-guess the analysis service.

    A temporal request against a single-window intent is structurally valid;
    the service answers with a warning and no comparison. The executor must
    report what happened, not invent a rejection of its own.
    """

    outcome, _, analysis = run(
        make_plan(execute_params(), {"tool": "temporal_ndwi_statistics"}),
        analysis=FakeAnalysisService(
            result=make_analysis_result(warnings=["no pair could be formed"])
        ),
    )

    assert analysis.calls[0].include_temporal_ndwi is True
    assert outcome.steps[1].status == "ok"
    assert outcome.evidence.analysis is not None
    assert outcome.evidence.analysis.temporal_comparison is None


# =========================================================================== #
# I. Evidence assembly
# =========================================================================== #


def test_measurements_become_citable_evidence_items() -> None:
    outcome, _, _ = run(
        make_plan(execute_params(), {"tool": "ndwi_statistics"}),
        analysis=FakeAnalysisService(
            result=make_analysis_result(
                measurements=[
                    Measurement(name="ndwi_mean", value=0.25, unit="index"),
                    Measurement(
                        name="ndwi_valid_pixel_count", value=1234.0, unit="pixels"
                    ),
                ]
            )
        ),
    )

    by_id = {item.id: item for item in outcome.evidence.items}
    assert "ndwi.ndwi_mean" in by_id
    assert by_id["ndwi.ndwi_mean"].source == "ndwi"
    assert by_id["ndwi.ndwi_mean"].measurement is not None
    assert by_id["ndwi.ndwi_mean"].measurement.value == 0.25
    assert by_id["ndwi.ndwi_mean"].produced_by


def test_temporal_evidence_covers_both_observations_and_the_difference() -> None:
    outcome, _, _ = run(
        make_plan(execute_params(), {"tool": "temporal_ndwi_statistics"}),
        analysis=FakeAnalysisService(
            result=make_analysis_result(temporal=make_temporal_comparison())
        ),
    )

    ids = {item.id for item in outcome.evidence.items}
    assert "temporal_ndwi.first.ndwi_mean" in ids
    assert "temporal_ndwi.second.ndwi_mean" in ids
    assert "temporal_ndwi.difference.mean_ndwi_difference" in ids
    assert any(i.startswith("compatibility.limitation.") for i in ids)


def test_evidence_ids_are_unique_across_every_source() -> None:
    outcome, _, _ = run(
        make_plan(
            execute_params(),
            {"tool": "ndwi_statistics"},
            {"tool": "temporal_ndwi_statistics"},
        ),
        analysis=FakeAnalysisService(
            result=make_analysis_result(
                measurements=[Measurement(name="ndwi_mean", value=0.1, unit="index")],
                warnings=["w1", "w2"],
                temporal=make_temporal_comparison(),
            )
        ),
    )

    ids = [item.id for item in outcome.evidence.items]
    assert len(ids) == len(set(ids))
    # Constructing AgentEvidence would have raised otherwise.
    assert outcome.evidence.ids() == set(ids)


def test_every_evidence_item_carries_content_and_provenance() -> None:
    outcome, _, _ = run(
        make_plan(execute_params(), {"tool": "ndwi_statistics"}),
        analysis=FakeAnalysisService(
            result=make_analysis_result(
                measurements=[Measurement(name="ndwi_mean", value=0.1, unit="index")],
                warnings=["a warning"],
            )
        ),
    )

    for item in outcome.evidence.items:
        assert item.measurement is not None or item.text is not None
        assert item.produced_by


def test_scene_counts_are_recorded_as_execution_evidence() -> None:
    outcome, _, _ = run(make_plan())

    by_id = {item.id: item for item in outcome.evidence.items}
    key = "execution.single.scene_count"
    assert key in by_id
    assert by_id[key].source == "execution"
    assert by_id[key].measurement is not None
    assert by_id[key].measurement.value == 3.0


def test_no_evidence_is_produced_when_discovery_fails() -> None:
    outcome, _, _ = run(
        make_plan(),
        query=FakeQueryExecutionService(error=UpstreamServiceError("down")),
    )

    assert outcome.evidence.items == []
    assert outcome.evidence.execution is None
    assert outcome.evidence.analysis is None


# =========================================================================== #
# J. No arbitrary execution, no forbidden collaborators
# =========================================================================== #


def _executor_tree() -> ast.Module:
    return ast.parse(pathlib.Path(executor_mod.__file__).read_text())


def test_executor_uses_no_dynamic_execution_primitive() -> None:
    forbidden = {
        "eval",
        "exec",
        "getattr",
        "setattr",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "input",
    }
    for node in ast.walk(_executor_tree()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden, (
                f"executor.py calls {node.func.id}()"
            )


def test_executor_performs_no_dynamic_import() -> None:
    for node in ast.walk(_executor_tree()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"importlib", "subprocess", "os"}
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in {
                "importlib",
                "subprocess",
                "os",
            }


def test_executor_imports_no_provider_raster_or_transport() -> None:
    roots: set[str] = set()
    for node in ast.walk(_executor_tree()):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    for forbidden in ("google", "genai", "httpx", "rasterio", "numpy", "PIL", "fastapi"):
        assert forbidden not in roots, f"{forbidden!r} must not be imported"


def test_executor_holds_only_the_two_deterministic_service_handles() -> None:
    executor = AgentExecutor(
        query_execution_service=FakeQueryExecutionService(),  # type: ignore[arg-type]
        analysis_service=FakeAnalysisService(),  # type: ignore[arg-type]
    )

    for banned in ("_imagery", "_raster", "_client", "_planner", "_synthesizer"):
        assert not hasattr(executor, banned)
    assert hasattr(executor, "_query")
    assert hasattr(executor, "_analysis")


def test_executor_requires_both_services_explicitly() -> None:
    with pytest.raises(TypeError):
        AgentExecutor()  # type: ignore[call-arg]


def test_no_lower_layer_imports_the_agent_package() -> None:
    """Nothing BELOW the agent may import it.

    ``app/api`` is deliberately excluded: the dependency direction is
    ``api -> agent -> {analysis, query} -> satellite``, so the API layer
    importing the agent is correct and is asserted positively below. This test
    previously scanned it too, which only passed because no route existed yet.
    """

    backend = pathlib.Path(__file__).resolve().parents[1]
    for package in ("analysis", "query", "satellite", "geospatial", "ai"):
        for path in (backend / "app" / "services" / package).rglob("*.py"):
            assert "services.agent" not in path.read_text(), (
                f"{path} imports the agent package - dependency direction violated"
            )
    for path in (backend / "app" / "core").rglob("*.py"):
        assert "services.agent" not in path.read_text()


def test_the_api_layer_is_the_one_that_composes_the_agent() -> None:
    """The direction is real: api imports agent, never the reverse."""

    backend = pathlib.Path(__file__).resolve().parents[1]
    routes = (backend / "app" / "api" / "routes" / "query.py").read_text()
    assert "services.agent" in routes
