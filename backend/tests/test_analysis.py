"""Analysis boundary tests.

The analysis layer is pure: no test contacts Gemini, Nominatim, the STAC API,
or imagery, because the service itself never does. Execution results are built
in memory; the route is exercised through ``dependency_overrides``.
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import numpy as np
import pytest
from app.api.routes.query import (
    get_ai_service,
    get_analysis_service,
    get_query_execution_service,
    get_query_service,
)
from app.core.errors import ImageryError, NotFoundError
from app.main import create_app
from app.services.ai import AiService, MockIntentParser
from app.services.analysis import AnalysisRequest, AnalysisResult, AnalysisService, engines
from app.services.analysis import service as service_mod
from app.services.analysis.engines import compute_ndwi_measurements
from app.services.analysis.schemas import Measurement
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
from app.services.satellite.raster import BandWindow
from app.services.satellite.schemas import ImageryResponse, Scene, WindowInfo
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

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
    scenes_override: list[Scene] | None = None,
) -> ExecutedWindow:
    return ExecutedWindow(
        modality=modality,  # type: ignore[arg-type]
        label=label,
        time_range=TimeRange.model_validate({"start_date": start, "end_date": end}),
        scene_count=scene_count,
        scenes=scenes_override or [],
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

    # ``execution`` remains the ONLY required field - there is still no
    # duplicated ``task``, and Phase 11's ``include_ndwi`` is opt-in.
    required = {
        name for name, field in AnalysisRequest.model_fields.items() if field.is_required()
    }
    assert required == {"execution"}
    assert request.execution.plan.intent.task == "visualize"
    assert request.include_ndwi is False


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
        # Phase 14, additive by design: the field is always serialized and is
        # null whenever temporal NDWI was not requested.
        "temporal_comparison",
    }
    assert body["status"] == "ok"
    assert body["task"] == "visualize"
    assert body["measurements"] == []
    assert body["temporal_comparison"] is None
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


# =========================================================================== #
# Phase 11: pure NDWI engine
#
# Small deterministic arrays only - the engine never touches STAC, COGs, HTTP
# or the filesystem, so nothing here needs a fake service.
# =========================================================================== #


def band(
    values: list[float] | np.ndarray,
    *,
    dtype: str = "uint16",
    nodata: float | None = 0.0,
) -> BandWindow:
    """A one-row BandWindow with an explicit nodata-derived validity mask."""

    array = np.asarray(values, dtype=dtype).reshape(1, -1)
    valid = np.isfinite(array) if np.issubdtype(array.dtype, np.floating) else (
        np.ones(array.shape, dtype=bool)
    )
    if nodata is not None:
        valid = valid & (array != nodata)
    return BandWindow(
        values=array,
        valid=valid,
        width=array.shape[1],
        height=1,
        crs="EPSG:32644",
        transform=from_origin(399960.0, 1500000.0, 10.0, 10.0),
        resolution=10.0,
        nodata=nodata,
        window={"col_off": 0, "row_off": 0, "width": array.shape[1], "height": 1},
        source_shape=[1, array.shape[1]],
    )


def named(measurements: list[Measurement]) -> dict[str, float]:
    return {m.name: m.value for m in measurements}


def test_ndwi_normal_case_matches_hand_computed_values() -> None:
    # (3-1)/(3+1)=0.5 ; (4-2)/(4+2)=1/3 ; (5-5)/(5+5)=0.0
    values = named(compute_ndwi_measurements(band([3, 4, 5]), band([1, 2, 5])))

    assert values["ndwi_valid_pixel_count"] == 3
    assert values["ndwi_max"] == pytest.approx(0.5)
    assert values["ndwi_min"] == pytest.approx(0.0)
    assert values["ndwi_mean"] == pytest.approx((0.5 + 1 / 3 + 0.0) / 3)


def test_ndwi_zero_numerator_is_valid_data_not_nodata() -> None:
    # green == nir gives NDWI 0.0; that pixel must still be counted.
    values = named(compute_ndwi_measurements(band([5]), band([5])))

    assert values["ndwi_valid_pixel_count"] == 1
    assert values["ndwi_mean"] == pytest.approx(0.0)


def test_ndwi_masks_green_nodata() -> None:
    values = named(compute_ndwi_measurements(band([0, 3]), band([7, 1])))

    assert values["ndwi_valid_pixel_count"] == 1  # first pixel is green nodata
    assert values["ndwi_mean"] == pytest.approx(0.5)


def test_ndwi_masks_nir_nodata() -> None:
    values = named(compute_ndwi_measurements(band([3, 3]), band([0, 1])))

    assert values["ndwi_valid_pixel_count"] == 1
    assert values["ndwi_mean"] == pytest.approx(0.5)


def test_ndwi_excludes_zero_denominator() -> None:
    # Both bands zero -> nodata AND a zero denominator; excluded either way.
    values = named(compute_ndwi_measurements(band([0, 4]), band([0, 2])))

    assert values["ndwi_valid_pixel_count"] == 1
    assert values["ndwi_mean"] == pytest.approx(1 / 3)


def test_ndwi_with_no_valid_pixels_reports_a_zero_count_and_nothing_else() -> None:
    measurements = compute_ndwi_measurements(band([0, 0]), band([0, 0]))

    assert len(measurements) == 1
    assert measurements[0].name == "ndwi_valid_pixel_count"
    assert measurements[0].value == 0.0
    # No fabricated statistics for an empty sample.
    assert "ndwi_mean" not in named(measurements)


def test_ndwi_excludes_nan_and_inf() -> None:
    green = band([np.nan, np.inf, 3.0, 4.0], dtype="float64", nodata=None)
    nir = band([1.0, 1.0, 1.0, 2.0], dtype="float64", nodata=None)
    values = named(compute_ndwi_measurements(green, nir))

    assert values["ndwi_valid_pixel_count"] == 2
    assert values["ndwi_mean"] == pytest.approx((0.5 + 1 / 3) / 2)


def test_ndwi_does_not_overflow_uint16_arithmetic() -> None:
    # 65535 + 1 wraps to 0 in uint16 (zero denominator); 65535 - 1 is fine.
    # Correct float64 promotion gives 65534 / 65536.
    values = named(compute_ndwi_measurements(band([65535]), band([1])))

    assert values["ndwi_valid_pixel_count"] == 1
    assert values["ndwi_mean"] == pytest.approx(65534 / 65536)


def test_ndwi_does_not_underflow_uint16_subtraction() -> None:
    # 3 - 4 wraps to 65535 in uint16; float64 promotion gives -1/7.
    values = named(compute_ndwi_measurements(band([3]), band([4])))

    assert values["ndwi_mean"] == pytest.approx(-1 / 7)


def test_ndwi_values_stay_within_the_valid_index_range() -> None:
    rng = np.random.default_rng(0)
    green = rng.integers(1, 65535, size=512, dtype=np.uint16)
    nir = rng.integers(1, 65535, size=512, dtype=np.uint16)
    values = named(compute_ndwi_measurements(band(green), band(nir)))

    assert -1.0 <= values["ndwi_min"] <= 1.0
    assert -1.0 <= values["ndwi_max"] <= 1.0


def test_ndwi_uses_raw_dn_and_ignores_the_advertised_scale_offset() -> None:
    green_dn, nir_dn = 3000, 1000
    values = named(compute_ndwi_measurements(band([green_dn]), band([nir_dn])))

    raw = (green_dn - nir_dn) / (green_dn + nir_dn)
    scaled = (green_dn * 0.0001 - 0.1 - (nir_dn * 0.0001 - 0.1)) / (
        (green_dn * 0.0001 - 0.1) + (nir_dn * 0.0001 - 0.1)
    )
    assert values["ndwi_mean"] == pytest.approx(raw)
    assert values["ndwi_mean"] != pytest.approx(scaled)
    assert engines.STAC_SCALE_OFFSET_APPLIED is False


def test_ndwi_percent_above_threshold_is_named_as_an_index_threshold() -> None:
    measurements = compute_ndwi_measurements(band([3, 4, 5]), band([1, 2, 5]))
    names = [m.name for m in measurements]
    threshold_name = (
        f"ndwi_percent_above_index_threshold_{engines.NDWI_INDEX_THRESHOLD}"
    )

    assert threshold_name in names
    # Honesty: nothing here may be presented as water or flood.
    assert not any("water" in n or "flood" in n for n in names)
    assert named(measurements)[threshold_name] == pytest.approx(200 / 3)


def test_ndwi_is_deterministic() -> None:
    green, nir = band([3, 4, 5]), band([1, 2, 5])
    first = compute_ndwi_measurements(green, nir)
    second = compute_ndwi_measurements(green, nir)

    assert [m.model_dump() for m in first] == [m.model_dump() for m in second]


def test_ndwi_rejects_mismatched_window_shapes() -> None:
    with pytest.raises(ImageryError, match="different"):
        compute_ndwi_measurements(band([1, 2, 3]), band([1, 2]))


# =========================================================================== #
# Phase 11: AnalysisService NDWI dispatch
# =========================================================================== #


class FakeImageryService:
    """Records read_band calls; returns canned bands or raises."""

    def __init__(
        self,
        *,
        bands: dict[str, BandWindow] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._bands = bands or {"green": band([3, 4, 5]), "nir": band([1, 2, 5])}
        self._error = error

    def read_band(
        self,
        *,
        scene_id: str,
        bbox: Any,
        asset: str,
        collection: str | None = None,
    ) -> BandWindow:
        self.calls.append(
            {
                "scene_id": scene_id,
                "bbox": bbox,
                "asset": asset,
                "collection": collection,
            }
        )
        if self._error is not None:
            raise self._error
        return self._bands[asset]


def make_scene(scene_id: str, *, collection: str | None = "sentinel-2-l2a") -> Scene:
    return Scene(
        id=scene_id,
        datetime="2024-01-15T05:00:00Z",
        bbox=None,
        geometry=None,
        cloud_cover=1.0,
        collection=collection,
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
    )


def analyze_ndwi(
    execution: QueryExecutionResult, imagery: FakeImageryService | None = None
) -> tuple[AnalysisResult, FakeImageryService]:
    imagery = imagery or FakeImageryService()
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(AnalysisRequest(execution=execution, include_ndwi=True))
    )
    return result, imagery


def ndwi_execution(**overrides: Any) -> QueryExecutionResult:
    window = make_window(scenes_override=[make_scene("scene-a")])
    return make_execution(windows=[window], **overrides)


def test_analysis_service_accepts_an_injected_imagery_service() -> None:
    imagery = FakeImageryService()
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]

    assert service._imagery is imagery
    assert AnalysisService() is not None  # zero-arg construction still works


def test_ndwi_is_not_computed_unless_requested() -> None:
    imagery = FakeImageryService()
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(AnalysisRequest(execution=ndwi_execution()))
    )

    assert imagery.calls == []  # include_ndwi defaults to False
    assert result.measurements == []
    assert result.status == "ok"


def test_ndwi_dispatch_reads_green_and_nir_and_returns_scalars() -> None:
    result, imagery = analyze_ndwi(ndwi_execution())

    assert [c["asset"] for c in imagery.calls] == ["green", "nir"]
    values = named(result.measurements)
    assert values["ndwi_valid_pixel_count"] == 3
    assert values["ndwi_mean"] == pytest.approx((0.5 + 1 / 3 + 0.0) / 3)
    assert result.status == "ok"  # visualize behaviour preserved
    assert result.task == "visualize"


def test_ndwi_propagates_the_selected_scene_id_and_collection() -> None:
    _, imagery = analyze_ndwi(ndwi_execution())

    assert {c["scene_id"] for c in imagery.calls} == {"scene-a"}
    assert {c["collection"] for c in imagery.calls} == {"sentinel-2-l2a"}
    assert all(c["bbox"] == DEFAULT_BBOX for c in imagery.calls)


def test_ndwi_collection_falls_back_to_none_when_the_scene_is_unknown() -> None:
    window = make_window(scenes_override=[make_scene("other-scene")])
    _, imagery = analyze_ndwi(make_execution(windows=[window]))

    # No matching Scene -> None -> ImageryService uses its configured default.
    assert {c["collection"] for c in imagery.calls} == {None}


def test_ndwi_skips_sar_windows() -> None:
    windows = [
        make_window(
            modality="sentinel-1-sar",
            selected_scene_id="s1-a",
            scenes_override=[make_scene("s1-a", collection="sentinel-1-grd")],
        ),
        make_window(scenes_override=[make_scene("scene-a")]),
    ]
    _, imagery = analyze_ndwi(make_execution(windows=windows))

    assert {c["scene_id"] for c in imagery.calls} == {"scene-a"}


def test_ndwi_with_no_selected_scene_warns_and_does_not_crash() -> None:
    window = make_window(selected_scene_id=None)
    result, imagery = analyze_ndwi(make_execution(windows=[window]))

    assert imagery.calls == []  # never invents a scene id
    assert result.measurements == []
    assert result.status == "ok"
    assert any("no Sentinel-2 optical window" in w for w in result.warnings)
    # The pre-existing missing-scene warning convention is still used too.
    assert any("No scene was selected" in w for w in result.warnings)


def test_ndwi_is_single_scene_and_says_so_when_more_windows_exist() -> None:
    windows = [
        make_window(label="baseline", scenes_override=[make_scene("scene-a")]),
        make_window(
            label="target",
            selected_scene_id="scene-b",
            scenes_override=[make_scene("scene-b")],
        ),
    ]
    result, imagery = analyze_ndwi(make_execution(windows=windows))

    assert {c["scene_id"] for c in imagery.calls} == {"scene-a"}  # first only
    assert any("single-scene" in w for w in result.warnings)


def test_ndwi_imagery_failure_is_isolated_to_a_warning() -> None:
    imagery = FakeImageryService(error=NotFoundError("Asset 'green' is missing."))
    result, _ = analyze_ndwi(ndwi_execution(), imagery)

    assert result.measurements == []
    assert result.status == "ok"  # the analysis itself still succeeds
    assert any("NDWI could not be computed" in w for w in result.warnings)


def test_ndwi_result_carries_an_honesty_warning() -> None:
    result, _ = analyze_ndwi(ndwi_execution())

    assert any(
        "not a validated water or flood classification" in w for w in result.warnings
    )
    assert "water" not in result.answer.lower()
    assert "flood" not in result.answer.lower()


def test_ndwi_runs_for_an_unimplemented_task_without_claiming_it() -> None:
    execution = ndwi_execution(intent=make_intent(task="change_detection"))
    result, imagery = analyze_ndwi(execution)

    assert result.status == "not_implemented"  # the TASK is still unimplemented
    assert "not implemented" in result.answer.lower()
    assert len(imagery.calls) == 2  # but the opt-in index still ran
    assert named(result.measurements)["ndwi_valid_pixel_count"] == 3


def test_ndwi_returns_only_scalars_and_never_arrays() -> None:
    result, _ = analyze_ndwi(ndwi_execution())
    payload = result.model_dump(mode="json")

    for measurement in payload["measurements"]:
        assert set(measurement) == {"name", "value", "unit"}
        assert isinstance(measurement["value"], float)
    # Nothing array-shaped anywhere in the serialised result.
    assert not any(
        isinstance(v, list) and v and isinstance(v[0], list)
        for v in payload.values()
    )
    assert "mask" not in payload and "array" not in payload


def test_ndwi_does_not_mutate_the_execution_result() -> None:
    execution = ndwi_execution()
    before = execution.model_dump(mode="json")
    analyze_ndwi(execution)

    assert execution.model_dump(mode="json") == before


def test_analysis_service_performs_no_pixel_arithmetic() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()

    # All arithmetic lives in engines.py; the service only dispatches.
    assert "import numpy" not in source
    assert "np." not in source
    assert "compute_ndwi_measurements" in source
