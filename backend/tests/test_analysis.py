"""Analysis boundary tests.

The analysis layer is pure: no test contacts Gemini, Nominatim, the STAC API,
or imagery, because the service itself never does. Execution results are built
in memory; the route is exercised through ``dependency_overrides``.
"""

from __future__ import annotations

import asyncio
import pathlib
from datetime import UTC, datetime
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
from app.services.analysis.engines import (
    compute_ndwi_measurements,
    compute_ndwi_threshold_measurement,
    render_ndwi_overlay,
)
from app.services.analysis.schemas import (
    Measurement,
    NdwiOverlay,
)
from app.services.geospatial import ResolveRequest, ResolveResponse
from app.services.geospatial.schemas import BoundingBox
from app.services.query import QueryService
from app.services.query.schemas import (
    ExecutedWindow,
    NdwiThreshold,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    SkippedModality,
    TimeRange,
)
from app.services.satellite.raster import BandWindow, image_corners_wgs84
from app.services.satellite.schemas import ImageryResponse, Scene, WindowInfo
from fastapi.testclient import TestClient
from pydantic import ValidationError
from rasterio.transform import Affine, from_origin

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
        # Phase 17.1, additive on the same principle: always serialized, null
        # unless an NDWI overlay was requested AND could be positioned.
        "ndwi_overlay",
        # Phase 17.2, additive: null unless the intent stated a threshold and
        # there were valid pixels to count.
        "spatial_measurement",
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
        # Phase 17.2, additive: null unless the request stated a threshold.
        "ndwi_threshold",
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


# =========================================================================== #
# Phase 17.1 - the NDWI overlay raster
# =========================================================================== #
#
# The index becomes a picture. The picture is positioned by the SAME affine the
# analysis was computed on - never by the requested bbox - so an overlay can
# never claim ground it did not measure. Invalid pixels are transparent, so the
# basemap stays readable where the index says nothing.


def grid_band(
    values: list[list[float]],
    *,
    dtype: str = "uint16",
    nodata: float | None = 0.0,
    transform: Any = None,
    crs: str | None = "EPSG:32644",
) -> BandWindow:
    """A 2-D BandWindow - the overlay needs a real grid, not one row."""

    array = np.asarray(values, dtype=dtype)
    valid = np.ones(array.shape, dtype=bool)
    if nodata is not None:
        valid = valid & (array != nodata)
    return BandWindow(
        values=array,
        valid=valid,
        width=array.shape[1],
        height=array.shape[0],
        crs=crs,
        transform=transform or from_origin(399960.0, 1500000.0, 10.0, 10.0),
        resolution=10.0,
        nodata=nodata,
        window={
            "col_off": 0,
            "row_off": 0,
            "width": array.shape[1],
            "height": array.shape[0],
        },
        source_shape=[array.shape[0], array.shape[1]],
    )


def decode_overlay(overlay: Any) -> np.ndarray:
    """The overlay PNG back as an RGBA array, so pixels can be asserted."""

    import base64
    import io

    from PIL import Image

    raw = base64.b64decode(overlay.image_base64)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))


GREEN_GRID = [[3, 4], [5, 0]]
NIR_GRID = [[1, 2], [5, 7]]


# --- contract -------------------------------------------------------------- #


def test_overlay_carries_the_full_georeferencing_contract() -> None:
    overlay = render_ndwi_overlay(
        grid_band(GREEN_GRID), grid_band(NIR_GRID), scene_id="S2X", window_label="single"
    )

    assert overlay is not None
    assert overlay.media_type == "image/png"
    assert overlay.width == 2
    assert overlay.height == 2
    assert overlay.crs == "EPSG:32644"
    assert overlay.transform == [10.0, 0.0, 399960.0, 0.0, -10.0, 1500000.0]
    assert overlay.corners_wgs84 is not None
    assert len(overlay.corners_wgs84) == 4
    assert overlay.scene_id == "S2X"
    assert overlay.window_label == "single"
    assert overlay.image_base64


def test_overlay_serializes_and_round_trips() -> None:
    overlay = render_ndwi_overlay(
        grid_band(GREEN_GRID), grid_band(NIR_GRID), scene_id="S2X", window_label="single"
    )
    assert overlay is not None
    restored = NdwiOverlay.model_validate_json(overlay.model_dump_json())

    assert restored.transform == overlay.transform
    assert restored.corners_wgs84 == overlay.corners_wgs84
    assert restored.image_base64 == overlay.image_base64


# --- dimensions match the analysed grid ------------------------------------ #


def test_overlay_dimensions_match_the_returned_raster() -> None:
    green = grid_band([[3, 4, 5], [6, 7, 8]])
    nir = grid_band([[1, 2, 3], [4, 5, 6]])
    overlay = render_ndwi_overlay(green, nir, scene_id="S2X", window_label="single")

    assert overlay is not None
    assert (overlay.width, overlay.height) == (green.width, green.height)
    assert decode_overlay(overlay).shape == (green.height, green.width, 4)


# --- georeferencing comes from the affine, never the bbox ------------------ #


def test_corners_are_derived_from_the_band_affine() -> None:
    green = grid_band(GREEN_GRID)
    overlay = render_ndwi_overlay(
        green, grid_band(NIR_GRID), scene_id="S2X", window_label="single"
    )

    assert overlay is not None
    expected = image_corners_wgs84(
        green.transform, width=green.width, height=green.height, crs=green.crs
    )
    assert overlay.corners_wgs84 == expected


def test_a_shifted_affine_moves_the_overlay() -> None:
    here = render_ndwi_overlay(
        grid_band(GREEN_GRID), grid_band(NIR_GRID), scene_id="S", window_label="w"
    )
    there = render_ndwi_overlay(
        grid_band(GREEN_GRID, transform=from_origin(500000.0, 1600000.0, 10.0, 10.0)),
        grid_band(NIR_GRID, transform=from_origin(500000.0, 1600000.0, 10.0, 10.0)),
        scene_id="S",
        window_label="w",
    )

    assert here is not None and there is not None
    assert here.corners_wgs84 != there.corners_wgs84


def test_a_coarser_grid_keeps_its_own_georeferencing() -> None:
    """A 20 m grid is a different grid; the overlay must say so."""

    coarse = from_origin(399960.0, 1500000.0, 20.0, 20.0)
    overlay = render_ndwi_overlay(
        grid_band(GREEN_GRID, transform=coarse),
        grid_band(NIR_GRID, transform=coarse),
        scene_id="S",
        window_label="w",
    )

    assert overlay is not None
    assert overlay.transform[0] == 20.0
    assert overlay.transform[4] == -20.0


# --- fail closed ----------------------------------------------------------- #


def test_no_overlay_without_a_crs() -> None:
    assert (
        render_ndwi_overlay(
            grid_band(GREEN_GRID, crs=None),
            grid_band(NIR_GRID, crs=None),
            scene_id="S",
            window_label="w",
        )
        is None
    )


def test_no_overlay_for_a_rotated_grid() -> None:
    rotated = Affine(10.0, 0.5, 399960.0, 0.0, -10.0, 1500000.0)
    assert (
        render_ndwi_overlay(
            grid_band(GREEN_GRID, transform=rotated),
            grid_band(NIR_GRID, transform=rotated),
            scene_id="S",
            window_label="w",
        )
        is None
    )


def test_no_overlay_when_no_pixel_is_valid() -> None:
    """Nothing measured means nothing to draw - not a fully transparent lie."""

    assert (
        render_ndwi_overlay(
            grid_band([[0, 0], [0, 0]]),
            grid_band([[0, 0], [0, 0]]),
            scene_id="S",
            window_label="w",
        )
        is None
    )


def test_mismatched_band_shapes_are_refused() -> None:
    with pytest.raises(ImageryError):
        render_ndwi_overlay(
            grid_band([[3, 4]]),
            grid_band([[1, 2], [3, 4]]),
            scene_id="S",
            window_label="w",
        )


# --- pixel semantics ------------------------------------------------------- #


def test_invalid_pixels_are_transparent() -> None:
    """green==0 is nodata, so that pixel must not paint over the basemap."""

    overlay = render_ndwi_overlay(
        grid_band(GREEN_GRID), grid_band(NIR_GRID), scene_id="S", window_label="w"
    )
    assert overlay is not None
    rgba = decode_overlay(overlay)

    assert rgba[1, 1, 3] == 0  # green==0 -> nodata -> fully transparent
    assert rgba[0, 0, 3] > 0  # a measured pixel is drawn


def test_the_overlay_is_not_the_rgb_source_image() -> None:
    """Distinct NDWI values must produce distinct colours."""

    overlay = render_ndwi_overlay(
        grid_band([[3, 9], [7, 2]]),
        grid_band([[9, 1], [7, 2]]),
        scene_id="S",
        window_label="w",
    )
    assert overlay is not None
    rgba = decode_overlay(overlay)

    # NDWI: (3-9)/12=-0.5 ; (9-1)/10=+0.8 ; 0.0 ; 0.0
    assert tuple(rgba[0, 0][:3]) != tuple(rgba[0, 1][:3])
    assert tuple(rgba[1, 0][:3]) == tuple(rgba[1, 1][:3])  # equal index, equal colour


def test_the_colour_ramp_is_monotonic_in_the_index() -> None:
    """Wetter pixels move consistently along the ramp - a readable legend."""

    overlay = render_ndwi_overlay(
        grid_band([[1, 5, 9]]), grid_band([[9, 5, 1]]), scene_id="S", window_label="w"
    )
    assert overlay is not None
    blue = decode_overlay(overlay)[0, :, 2].astype(int)

    # NDWI -0.8, 0.0, +0.8 -> increasing blue
    assert blue[0] < blue[1] < blue[2]


# --- the numbers themselves are unchanged ---------------------------------- #


def test_the_overlay_does_not_change_the_measurements() -> None:
    green, nir = grid_band(GREEN_GRID), grid_band(NIR_GRID)
    before = named(compute_ndwi_measurements(green, nir))
    render_ndwi_overlay(green, nir, scene_id="S", window_label="w")
    after = named(compute_ndwi_measurements(green, nir))

    assert before == after


def test_overlay_values_agree_with_the_measured_range() -> None:
    green, nir = grid_band(GREEN_GRID), grid_band(NIR_GRID)
    overlay = render_ndwi_overlay(green, nir, scene_id="S", window_label="w")
    stats = named(compute_ndwi_measurements(green, nir))

    assert overlay is not None
    assert overlay.value_min == pytest.approx(stats["ndwi_min"])
    assert overlay.value_max == pytest.approx(stats["ndwi_max"])
    assert overlay.valid_pixel_count == int(stats["ndwi_valid_pixel_count"])


# --- service wiring: the overlay is opt-in and rides the same band reads ---- #


def analyze_with_overlay(
    execution: QueryExecutionResult, imagery: FakeImageryService | None = None
) -> tuple[AnalysisResult, FakeImageryService]:
    imagery = imagery or FakeImageryService(
        bands={"green": grid_band(GREEN_GRID), "nir": grid_band(NIR_GRID)}
    )
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                execution=execution, include_ndwi=True, include_ndwi_overlay=True
            )
        )
    )
    return result, imagery


def test_no_overlay_unless_requested() -> None:
    result, _ = analyze_ndwi(ndwi_execution())
    assert result.ndwi_overlay is None


def test_the_overlay_is_produced_when_requested() -> None:
    result, _ = analyze_with_overlay(ndwi_execution())

    assert result.ndwi_overlay is not None
    assert result.ndwi_overlay.scene_id == "scene-a"
    assert result.ndwi_overlay.corners_wgs84 is not None


def test_the_overlay_reuses_the_analysis_band_reads() -> None:
    """No second retrieval pipeline: still exactly the two reads NDWI needs."""

    _, imagery = analyze_with_overlay(ndwi_execution())

    assert [c["asset"] for c in imagery.calls] == ["green", "nir"]


def test_the_overlay_does_not_disturb_the_measurements() -> None:
    plain, _ = analyze_ndwi(
        ndwi_execution(),
        FakeImageryService(
            bands={"green": grid_band(GREEN_GRID), "nir": grid_band(NIR_GRID)}
        ),
    )
    with_overlay, _ = analyze_with_overlay(ndwi_execution())

    assert named(with_overlay.measurements) == named(plain.measurements)


def test_the_overlay_needs_the_statistics_too() -> None:
    """Asking for a picture without the numbers is not a supported shape."""

    imagery = FakeImageryService(
        bands={"green": grid_band(GREEN_GRID), "nir": grid_band(NIR_GRID)}
    )
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                execution=ndwi_execution(),
                include_ndwi=False,
                include_ndwi_overlay=True,
            )
        )
    )

    assert result.ndwi_overlay is None
    assert imagery.calls == []


def test_an_unusable_grid_yields_no_overlay_but_keeps_the_numbers() -> None:
    imagery = FakeImageryService(
        bands={
            "green": grid_band(GREEN_GRID, crs=None),
            "nir": grid_band(NIR_GRID, crs=None),
        }
    )
    result, _ = analyze_with_overlay(ndwi_execution(), imagery)

    assert result.ndwi_overlay is None
    assert named(result.measurements)["ndwi_valid_pixel_count"] == 3


def test_the_analysis_result_serializes_with_the_overlay() -> None:
    result, _ = analyze_with_overlay(ndwi_execution())
    restored = AnalysisResult.model_validate_json(result.model_dump_json())

    assert restored.ndwi_overlay is not None
    assert restored.ndwi_overlay.transform == result.ndwi_overlay.transform


def test_the_result_is_backward_compatible_without_an_overlay() -> None:
    result, _ = analyze_ndwi(ndwi_execution())
    assert "ndwi_overlay" in result.model_dump()
    assert result.model_dump()["ndwi_overlay"] is None


# =========================================================================== #
# Phase 17.2 - spatially grounded NDWI threshold measurements
# =========================================================================== #
#
# The number is computed from the analysed pixels, never by a language model.
# Percentage is matching / VALID pixels: nodata, non-finite samples and
# zero-denominator pixels are excluded from BOTH sides of the ratio, because a
# pixel that was never measured cannot count as "not matching".
#
# Comparisons are exact: `gt` means `>`, never `>=`.


def measure(
    green_grid: list[list[float]],
    nir_grid: list[list[float]],
    operator: str,
    value: float,
    **kwargs: Any,
) -> Any:
    return compute_ndwi_threshold_measurement(
        grid_band(green_grid),
        grid_band(nir_grid),
        threshold=NdwiThreshold(operator=operator, value=value),  # type: ignore[arg-type]
        scene_id=kwargs.pop("scene_id", "scene-a"),
        window_label=kwargs.pop("window_label", "single"),
        acquired_at=kwargs.pop("acquired_at", None),
    )


# A 2x2 grid whose NDWI values are exactly -0.5, 0.3, 0.0 and nodata.
#   (1-3)/4 = -0.5 ; (13-7)/20 = 0.3 ; (5-5)/10 = 0.0 ; green 0 -> nodata
BOUNDARY_GREEN = [[1, 13], [5, 0]]
BOUNDARY_NIR = [[3, 7], [5, 7]]


def test_ndwi_values_of_the_boundary_grid_are_what_the_tests_assume() -> None:
    """Anchor the fixture: every later assertion depends on these three values."""

    stats = named(
        compute_ndwi_measurements(grid_band(BOUNDARY_GREEN), grid_band(BOUNDARY_NIR))
    )
    assert stats["ndwi_valid_pixel_count"] == 3
    assert stats["ndwi_min"] == pytest.approx(-0.5)
    assert stats["ndwi_max"] == pytest.approx(0.3)


# --- 1-4. the four operators ----------------------------------------------- #


def test_gt_is_strictly_greater() -> None:
    """0.3 must NOT match `> 0.3`."""

    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gt", 0.3)

    assert m is not None
    assert m.matching_pixel_count == 0
    assert m.valid_pixel_count == 3
    assert m.percentage == pytest.approx(0.0)


def test_gte_includes_the_boundary() -> None:
    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gte", 0.3)

    assert m is not None
    assert m.matching_pixel_count == 1
    assert m.percentage == pytest.approx(100.0 / 3)


def test_lt_is_strictly_less() -> None:
    """-0.5 and 0.0 are below 0.3; 0.3 itself is not."""

    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "lt", 0.3)

    assert m is not None
    assert m.matching_pixel_count == 2
    assert m.percentage == pytest.approx(200.0 / 3)


def test_lte_includes_the_boundary() -> None:
    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "lte", 0.3)

    assert m is not None
    assert m.matching_pixel_count == 3
    assert m.percentage == pytest.approx(100.0)


# --- 5-6. negative thresholds and exact boundaries ------------------------- #


def test_a_negative_threshold_is_supported() -> None:
    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gt", -0.5)

    assert m is not None
    assert m.threshold == -0.5
    assert m.matching_pixel_count == 2  # 0.0 and 0.3, not -0.5 itself


def test_the_boundary_pixel_is_decided_by_the_operator_alone() -> None:
    gt = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gt", -0.5)
    gte = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gte", -0.5)

    assert gt is not None and gte is not None
    assert gte.matching_pixel_count - gt.matching_pixel_count == 1


# --- 7-8. all / none matching ---------------------------------------------- #


def test_all_valid_pixels_can_match() -> None:
    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gte", -1.0)

    assert m is not None
    assert m.matching_pixel_count == m.valid_pixel_count == 3
    assert m.percentage == pytest.approx(100.0)


def test_no_pixel_matching_is_zero_percent_not_an_absence() -> None:
    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gt", 0.99)

    assert m is not None
    assert m.matching_pixel_count == 0
    assert m.percentage == 0.0


# --- 9-11. what is excluded from the denominator --------------------------- #


def test_nodata_is_excluded_from_both_sides_of_the_ratio() -> None:
    """The 2x2 grid has 4 pixels but only 3 valid ones."""

    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "lte", 1.0)

    assert m is not None
    assert m.valid_pixel_count == 3  # not 4
    assert m.matching_pixel_count == 3
    assert m.percentage == pytest.approx(100.0)


def test_non_finite_samples_are_excluded() -> None:
    green = grid_band(
        [[float("nan"), 13.0], [float("inf"), 5.0]], dtype="float32", nodata=None
    )
    nir = grid_band([[3.0, 7.0], [1.0, 5.0]], dtype="float32", nodata=None)
    m = compute_ndwi_threshold_measurement(
        green,
        nir,
        threshold=NdwiThreshold(operator="gte", value=-1.0),
        scene_id="s",
        window_label="w",
        acquired_at=None,
    )

    assert m is not None
    assert m.valid_pixel_count == 2  # NaN and inf both dropped


def test_a_zero_denominator_pixel_is_excluded() -> None:
    """green + nir == 0 has no defined index, so it is not measured."""

    green = grid_band([[-5.0, 13.0]], dtype="float32", nodata=None)
    nir = grid_band([[5.0, 7.0]], dtype="float32", nodata=None)
    m = compute_ndwi_threshold_measurement(
        green,
        nir,
        threshold=NdwiThreshold(operator="gte", value=-1.0),
        scene_id="s",
        window_label="w",
        acquired_at=None,
    )

    assert m is not None
    assert m.valid_pixel_count == 1


# --- 11. zero valid pixels fails closed ------------------------------------ #


def test_zero_valid_pixels_yields_no_measurement() -> None:
    """No denominator means no percentage - not 0%, and not 100%."""

    assert measure([[0, 0], [0, 0]], [[1, 2], [3, 4]], "gt", 0.3) is None


# --- 12-13. the percentage itself ------------------------------------------ #


def test_percentage_is_matching_over_valid() -> None:
    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "lt", 0.3)

    assert m is not None
    assert m.percentage == pytest.approx(
        m.matching_pixel_count / m.valid_pixel_count * 100.0
    )


def test_percentage_never_divides_by_the_raster_size() -> None:
    """4 pixels, 3 valid, 2 matching -> 66.67%, never 50%."""

    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "lt", 0.3)

    assert m is not None
    assert m.percentage == pytest.approx(66.6666666, abs=1e-6)
    assert m.percentage != pytest.approx(50.0)


def test_the_measurement_is_deterministic() -> None:
    first = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gte", 0.3)
    second = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gte", 0.3)

    assert first is not None and second is not None
    assert first.model_dump() == second.model_dump()


# --- provenance ------------------------------------------------------------ #


def test_the_measurement_carries_its_provenance_and_footprint() -> None:
    acquired = datetime(2025, 1, 4, 5, 0, tzinfo=UTC)
    m = measure(
        BOUNDARY_GREEN,
        BOUNDARY_NIR,
        "gt",
        0.0,
        scene_id="S2B_44PMV_20250104_0_L2A",
        window_label="single",
        acquired_at=acquired,
    )

    assert m is not None
    assert m.metric == "ndwi"
    assert m.scene_id == "S2B_44PMV_20250104_0_L2A"
    assert m.window_label == "single"
    assert m.acquired_at == acquired
    assert m.crs == "EPSG:32644"
    assert m.corners_wgs84 is not None
    assert len(m.corners_wgs84) == 4
    assert m.corners_wgs84 == image_corners_wgs84(
        grid_band(BOUNDARY_GREEN).transform, width=2, height=2, crs="EPSG:32644"
    )


def test_the_measurement_carries_no_raster_data() -> None:
    """Provenance and counts only - the pixels stay in the overlay."""

    m = measure(BOUNDARY_GREEN, BOUNDARY_NIR, "gt", 0.0)

    assert m is not None
    dumped = m.model_dump()
    assert "image_base64" not in dumped
    assert not any(isinstance(v, (bytes, bytearray)) for v in dumped.values())


def test_an_ungeoreferenced_grid_still_reports_the_number() -> None:
    """The count is the evidence; a missing footprint must not suppress it."""

    m = compute_ndwi_threshold_measurement(
        grid_band(BOUNDARY_GREEN, crs=None),
        grid_band(BOUNDARY_NIR, crs=None),
        threshold=NdwiThreshold(operator="lte", value=1.0),
        scene_id="s",
        window_label="w",
        acquired_at=None,
    )

    assert m is not None
    assert m.matching_pixel_count == 3
    assert m.crs is None
    assert m.corners_wgs84 is None


# --- 15. malformed thresholds are refused at the contract ------------------ #


@pytest.mark.parametrize("value", [1.5, -1.5, float("nan"), float("inf")])
def test_an_out_of_range_threshold_is_refused(value: float) -> None:
    with pytest.raises(ValidationError):
        NdwiThreshold(operator="gt", value=value)


def test_an_unknown_operator_is_refused() -> None:
    with pytest.raises(ValidationError):
        NdwiThreshold(operator="approximately", value=0.3)  # type: ignore[arg-type]


def test_the_threshold_contract_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        NdwiThreshold.model_validate({"operator": "gt", "value": 0.3, "unit": "m"})


# --- service wiring: the threshold rides the intent ------------------------ #


def threshold_execution(operator: str = "gt", value: float = 0.3) -> Any:
    """An execution whose intent carries an explicit NDWI threshold."""

    execution = ndwi_execution()
    intent = execution.plan.intent.model_copy(
        update={"ndwi_threshold": NdwiThreshold(operator=operator, value=value)}  # type: ignore[arg-type]
    )
    return execution.model_copy(
        update={"plan": execution.plan.model_copy(update={"intent": intent})}
    )


def analyze_threshold(
    execution: Any, imagery: FakeImageryService | None = None
) -> tuple[AnalysisResult, FakeImageryService]:
    imagery = imagery or FakeImageryService(
        bands={"green": grid_band(BOUNDARY_GREEN), "nir": grid_band(BOUNDARY_NIR)}
    )
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(AnalysisRequest(execution=execution, include_ndwi=True))
    )
    return result, imagery


def test_no_threshold_in_the_intent_means_no_spatial_measurement() -> None:
    result, _ = analyze_ndwi(ndwi_execution())
    assert result.spatial_measurement is None


def test_a_threshold_in_the_intent_produces_the_measurement() -> None:
    result, _ = analyze_threshold(threshold_execution("gte", 0.3))

    assert result.spatial_measurement is not None
    assert result.spatial_measurement.operator == "gte"
    assert result.spatial_measurement.threshold == 0.3
    assert result.spatial_measurement.matching_pixel_count == 1
    assert result.spatial_measurement.valid_pixel_count == 3


def test_the_measurement_reuses_the_same_two_band_reads() -> None:
    _, imagery = analyze_threshold(threshold_execution())
    assert [c["asset"] for c in imagery.calls] == ["green", "nir"]


def test_the_threshold_does_not_change_the_statistics() -> None:
    plain, _ = analyze_ndwi(
        ndwi_execution(),
        FakeImageryService(
            bands={"green": grid_band(BOUNDARY_GREEN), "nir": grid_band(BOUNDARY_NIR)}
        ),
    )
    thresholded, _ = analyze_threshold(threshold_execution())

    assert named(thresholded.measurements) == named(plain.measurements)


def test_the_measurement_needs_the_ndwi_opt_in() -> None:
    """Without include_ndwi nothing is read, so nothing can be counted."""

    imagery = FakeImageryService(
        bands={"green": grid_band(BOUNDARY_GREEN), "nir": grid_band(BOUNDARY_NIR)}
    )
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(execution=threshold_execution(), include_ndwi=False)
        )
    )

    assert result.spatial_measurement is None
    assert imagery.calls == []


def test_no_measurement_when_no_pixel_is_valid_but_no_fabrication() -> None:
    imagery = FakeImageryService(
        bands={"green": grid_band([[0, 0]]), "nir": grid_band([[1, 2]])}
    )
    result, _ = analyze_threshold(threshold_execution(), imagery)

    assert result.spatial_measurement is None
    assert named(result.measurements)["ndwi_valid_pixel_count"] == 0


def test_the_measurement_carries_the_scene_and_window() -> None:
    result, _ = analyze_threshold(threshold_execution())

    assert result.spatial_measurement is not None
    assert result.spatial_measurement.scene_id == "scene-a"
    assert result.spatial_measurement.window_label == "single"


def test_the_analysis_result_serializes_with_the_measurement() -> None:
    result, _ = analyze_threshold(threshold_execution())
    restored = AnalysisResult.model_validate_json(result.model_dump_json())

    assert restored.spatial_measurement is not None
    assert (
        restored.spatial_measurement.percentage
        == result.spatial_measurement.percentage
    )


def test_the_measurement_is_separate_from_the_overlay() -> None:
    """Two contracts: the number and the picture never merge."""

    imagery = FakeImageryService(
        bands={"green": grid_band(BOUNDARY_GREEN), "nir": grid_band(BOUNDARY_NIR)}
    )
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                execution=threshold_execution(),
                include_ndwi=True,
                include_ndwi_overlay=True,
            )
        )
    )

    assert result.spatial_measurement is not None
    assert result.ndwi_overlay is not None
    assert not hasattr(result.spatial_measurement, "image_base64")
