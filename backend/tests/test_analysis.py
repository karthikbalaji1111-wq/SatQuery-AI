"""Analysis boundary tests.

The analysis layer is pure: no test contacts Gemini, Nominatim, the STAC API,
or imagery, because the service itself never does. Execution results are built
in memory; the route is exercised through ``dependency_overrides``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from app.api.routes.query import (
    get_ai_service,
    get_analysis_service,
    get_query_execution_service,
    get_query_service,
)
from app.main import create_app
from app.services.ai import AiService, MockIntentParser
from app.services.analysis import AnalysisRequest, AnalysisResult, AnalysisService
from app.services.geospatial import ResolveRequest, ResolveResponse
from app.services.geospatial.schemas import BoundingBox
from app.services.query import QueryService
from app.services.query.schemas import (
    ExecutedWindow,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    SkippedModality,
    TimeRange,
)
from app.services.satellite.schemas import ImageryResponse, WindowInfo
from fastapi.testclient import TestClient

ANALYZE_URL = "/api/v1/query/analyze"
EXECUTE_URL = "/api/v1/query/execute"
PARSE_URL = "/api/v1/query/parse"
BUILD_PLAN_URL = "/api/v1/query/build-plan"

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
CATALOG = "https://earth-search.aws.element84.com/v1"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def intent_dict(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    body.update(overrides)
    return body


def make_intent(**overrides: Any) -> SatQueryIntent:
    return SatQueryIntent.model_validate(intent_dict(**overrides))


def make_imagery(scene_id: str, *, asset: str = "visual") -> ImageryResponse:
    return ImageryResponse(
        scene_id=scene_id,
        bbox=DEFAULT_BBOX,
        asset=asset,
        asset_href="https://example.test/TCI.tif",
        width=4,
        height=4,
        format="png",
        media_type="image/png",
        bands=["red", "green", "blue"],
        crs="EPSG:32644",
        resolution=10.0,
        normalization="none (source is 8-bit RGB)",
        window=WindowInfo(col_off=0, row_off=0, width=4, height=4),
        source_shape=[10, 10],
        image_base64="AAAA",
    )


def make_window(
    *,
    modality: str = "sentinel-2-optical",
    label: str = "single",
    start: str = "2024-01-01",
    end: str = "2024-01-31",
    selected_scene_id: str | None = "scene-a",
    scene_count: int = 1,
    imagery: ImageryResponse | None = None,
    imagery_error: str | None = None,
) -> ExecutedWindow:
    return ExecutedWindow(
        modality=modality,  # type: ignore[arg-type]
        label=label,
        time_range=TimeRange.model_validate({"start_date": start, "end_date": end}),
        scene_count=scene_count,
        scenes=[],
        selected_scene_id=selected_scene_id,
        imagery=imagery,
        imagery_error=imagery_error,
    )


def make_execution(
    *,
    intent: SatQueryIntent | None = None,
    windows: list[ExecutedWindow] | None = None,
    executed_modalities: list[str] | None = None,
    skipped_modalities: list[SkippedModality] | None = None,
    catalog: str = CATALOG,
) -> QueryExecutionResult:
    intent = intent if intent is not None else make_intent()
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=intent, bbox=DEFAULT_BBOX),
        executed_modalities=executed_modalities  # type: ignore[arg-type]
        or list(intent.modalities),
        skipped_modalities=skipped_modalities or [],
        windows=windows if windows is not None else [make_window()],
        catalog=catalog,
    )


def analyze(execution: QueryExecutionResult) -> AnalysisResult:
    return asyncio.run(
        AnalysisService().analyze(AnalysisRequest(execution=execution))
    )


def make_client(service: AnalysisService | None = None) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_analysis_service] = lambda: (
        service or AnalysisService()
    )
    return TestClient(app)


def analyze_body(execution: QueryExecutionResult) -> dict[str, Any]:
    """The whole request body - note there is no top-level ``task`` field."""

    return {"execution": execution.model_dump(mode="json")}


class FakeGeospatialService:
    """Records ``resolve`` calls; returns a canned bbox (mirrors test_query.py)."""

    def __init__(self) -> None:
        self.calls: list[ResolveRequest] = []

    async def resolve(self, request: ResolveRequest) -> ResolveResponse:
        self.calls.append(request)
        return ResolveResponse(
            query_type="place",
            display_name="Chennai, Tamil Nadu, India",
            center=DEFAULT_BBOX.center,
            bbox=DEFAULT_BBOX,
            source="nominatim",
        )


# --------------------------------------------------------------------------- #
# Service contract
# --------------------------------------------------------------------------- #


def test_analysis_service_is_zero_arg_constructible_and_describes_itself() -> None:
    service = AnalysisService()
    assert service.name == "analysis"
    assert isinstance(service.describe(), str) and service.describe()


# --------------------------------------------------------------------------- #
# visualize -> ok
# --------------------------------------------------------------------------- #


def test_visualize_returns_ok_with_a_templated_answer() -> None:
    result = analyze(make_execution())

    assert result.status == "ok"
    assert result.task == "visualize"
    assert "Chennai" in result.answer
    assert "scene-a" in result.answer
    assert CATALOG in result.answer
    assert result.warnings == []
    assert result.measurements == []


def test_visualize_answer_is_deterministic() -> None:
    execution = make_execution(
        windows=[
            make_window(label="series[0]", selected_scene_id="s2-a"),
            make_window(
                modality="sentinel-1-sar", label="series[0]", selected_scene_id="s1-a"
            ),
        ]
    )
    first = analyze(execution)
    second = analyze(execution)

    assert first.model_dump() == second.model_dump()


def test_visualize_with_no_windows_says_so_and_warns() -> None:
    result = analyze(make_execution(windows=[]))

    assert result.status == "ok"
    assert result.windows_considered == []
    assert "No imagery windows were executed" in result.answer
    assert any("no windows" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# Task derivation (no duplicated task field)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("task", "expected_status"),
    [
        ("visualize", "ok"),
        ("change_detection", "not_implemented"),
        ("object_identification", "not_implemented"),
    ],
)
def test_task_is_derived_from_the_execution_intent(
    task: str, expected_status: str
) -> None:
    result = analyze(make_execution(intent=make_intent(task=task)))

    assert result.task == task
    assert result.status == expected_status


def test_request_needs_only_the_execution_field() -> None:
    request = AnalysisRequest.model_validate(analyze_body(make_execution()))

    assert set(AnalysisRequest.model_fields) == {"execution"}
    assert request.execution.plan.intent.task == "visualize"


# --------------------------------------------------------------------------- #
# not_implemented tasks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task", ["change_detection", "object_identification"])
def test_unimplemented_task_claims_no_result(task: str) -> None:
    result = analyze(make_execution(intent=make_intent(task=task)))

    assert result.status == "not_implemented"
    assert "not implemented" in result.answer.lower()
    assert "no result is claimed" in result.answer.lower()
    assert result.measurements == []
    # Traceability is still returned - that is why this is not an error body.
    assert len(result.windows_considered) == 1


def test_unimplemented_task_still_reports_warnings() -> None:
    result = analyze(
        make_execution(
            intent=make_intent(task="change_detection"),
            windows=[make_window(selected_scene_id=None)],
        )
    )

    assert result.status == "not_implemented"
    assert any("No scene was selected" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


def test_missing_selected_scene_produces_a_warning() -> None:
    result = analyze(make_execution(windows=[make_window(selected_scene_id=None)]))

    assert result.windows_considered[0].selected_scene_id is None
    assert any("No scene was selected" in warning for warning in result.warnings)
    assert any("'single'" in warning for warning in result.warnings)


def test_imagery_error_produces_a_warning() -> None:
    result = analyze(
        make_execution(windows=[make_window(imagery_error="SAR render failed")])
    )

    assert any(
        "Imagery was unavailable" in warning and "SAR render failed" in warning
        for warning in result.warnings
    )


def test_skipped_modality_produces_a_warning() -> None:
    result = analyze(
        make_execution(
            skipped_modalities=[
                SkippedModality(modality="sentinel-1-sar", reason="not available")
            ]
        )
    )

    assert any("was skipped" in warning for warning in result.warnings)


def test_a_healthy_execution_produces_no_warnings() -> None:
    result = analyze(
        make_execution(windows=[make_window(imagery=make_imagery("scene-a"))])
    )

    assert result.warnings == []


# --------------------------------------------------------------------------- #
# Window references (slim projection)
# --------------------------------------------------------------------------- #


def test_multiple_windows_are_represented_in_order() -> None:
    execution = make_execution(
        intent=make_intent(
            modalities=["sentinel-2-optical", "sentinel-1-sar"],
            temporal_mode="compare",
            time_windows={
                "baseline": {"start_date": "2023-01-01", "end_date": "2023-03-31"},
                "target": {"start_date": "2024-01-01", "end_date": "2024-03-31"},
            },
        ),
        windows=[
            make_window(label="baseline", start="2023-01-01", end="2023-03-31"),
            make_window(label="target", selected_scene_id="scene-b"),
            make_window(
                modality="sentinel-1-sar",
                label="baseline",
                start="2023-01-01",
                end="2023-03-31",
                selected_scene_id="s1-a",
            ),
            make_window(
                modality="sentinel-1-sar", label="target", selected_scene_id="s1-b"
            ),
        ],
    )
    result = analyze(execution)

    assert [(ref.modality, ref.label) for ref in result.windows_considered] == [
        ("sentinel-2-optical", "baseline"),
        ("sentinel-2-optical", "target"),
        ("sentinel-1-sar", "baseline"),
        ("sentinel-1-sar", "target"),
    ]
    assert [ref.selected_scene_id for ref in result.windows_considered] == [
        "scene-a",
        "scene-b",
        "s1-a",
        "s1-b",
    ]
    assert result.windows_considered[0].time_range.start_date.isoformat() == "2023-01-01"


def test_window_reference_never_echoes_scenes_or_imagery() -> None:
    execution = make_execution(
        windows=[make_window(imagery=make_imagery("scene-a"), scene_count=7)]
    )
    ref = analyze(execution).windows_considered[0]

    assert set(type(ref).model_fields) == {
        "modality",
        "label",
        "time_range",
        "selected_scene_id",
    }
    assert not hasattr(ref, "scenes")
    assert not hasattr(ref, "imagery")


def test_measurements_default_to_empty() -> None:
    result = AnalysisResult(
        status="ok", task="visualize", answer="x", windows_considered=[]
    )

    assert result.measurements == []
    assert result.warnings == []


# --------------------------------------------------------------------------- #
# POST /api/v1/query/analyze
# --------------------------------------------------------------------------- #


def test_analyze_endpoint_returns_the_expected_contract() -> None:
    response = make_client().post(ANALYZE_URL, json=analyze_body(make_execution()))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "status",
        "task",
        "answer",
        "windows_considered",
        "warnings",
        "measurements",
    }
    assert body["status"] == "ok"
    assert body["task"] == "visualize"
    assert body["measurements"] == []
    window = body["windows_considered"][0]
    assert set(window) == {"modality", "label", "time_range", "selected_scene_id"}
    assert window["modality"] == "sentinel-2-optical"


@pytest.mark.parametrize("task", ["change_detection", "object_identification"])
def test_analyze_endpoint_returns_200_for_unimplemented_tasks(task: str) -> None:
    execution = make_execution(intent=make_intent(task=task))
    response = make_client().post(ANALYZE_URL, json=analyze_body(execution))

    assert response.status_code == 200  # deliberately not 501
    assert response.json()["status"] == "not_implemented"


def test_analyze_endpoint_uses_the_dependency_override() -> None:
    class RecordingAnalysisService(AnalysisService):
        def __init__(self) -> None:
            self.requests: list[AnalysisRequest] = []

        async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
            self.requests.append(request)
            return await super().analyze(request)

    service = RecordingAnalysisService()
    response = make_client(service).post(
        ANALYZE_URL, json=analyze_body(make_execution())
    )

    assert response.status_code == 200
    assert len(service.requests) == 1
    assert service.requests[0].execution.plan.intent.location_query == "Chennai"


@pytest.mark.parametrize(
    "bad_body",
    [
        {},
        {"execution": {}},
        {"execution": {"plan": {"intent": intent_dict(task="segment")}}},
        {"task": "visualize"},
    ],
)
def test_analyze_endpoint_rejects_an_invalid_request(bad_body: dict[str, Any]) -> None:
    response = make_client().post(ANALYZE_URL, json=bad_body)
    assert response.status_code == 422


def test_analyze_endpoint_rejects_an_invalid_nested_intent() -> None:
    body = analyze_body(make_execution())
    body["execution"]["plan"]["intent"]["modalities"] = []

    response = make_client().post(ANALYZE_URL, json=body)
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Existing routes remain unaffected
# --------------------------------------------------------------------------- #


def test_parse_endpoint_is_unaffected() -> None:
    app = create_app()
    app.dependency_overrides[get_ai_service] = lambda: AiService(
        parser=MockIntentParser()
    )
    response = TestClient(app).post(PARSE_URL, json={"prompt": "show me Chennai"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "location_query",
        "temporal_mode",
        "time_windows",
        "modalities",
        "task",
    }
    for leaked in ("status", "answer", "windows_considered", "measurements"):
        assert leaked not in body


def test_build_plan_endpoint_is_unaffected() -> None:
    fake_geo = FakeGeospatialService()
    app = create_app()
    app.dependency_overrides[get_query_service] = lambda: QueryService(
        geospatial_service=fake_geo  # type: ignore[arg-type]
    )
    response = TestClient(app).post(BUILD_PLAN_URL, json=intent_dict())

    assert response.status_code == 200
    assert set(response.json()) == {"intent", "bbox"}
    assert len(fake_geo.calls) == 1


def test_execute_endpoint_is_still_mounted_and_validating() -> None:
    class UnreachableExecutionService:
        async def execute(self, request: Any) -> Any:  # pragma: no cover
            raise AssertionError("invalid intent must not reach the service")

    app = create_app()
    app.dependency_overrides[get_query_execution_service] = (
        lambda: UnreachableExecutionService()
    )
    response = TestClient(app).post(
        EXECUTE_URL, json={"intent": intent_dict(task="segment")}
    )

    assert response.status_code == 422


def test_analysis_route_is_registered_alongside_the_existing_query_routes() -> None:
    paths = set(create_app().openapi()["paths"])

    assert {
        "/api/v1/query/parse",
        "/api/v1/query/build-plan",
        "/api/v1/query/execute",
        "/api/v1/query/analyze",
    } <= paths
