"""Query execution orchestration tests.

The orchestrator is exercised with injected fakes only: no test contacts
Gemini, Nominatim, the STAC API, or real imagery. Grounding runs through the
real ``QueryService`` wired to a fake Geospatial Service (mirroring
``test_query.py``); discovery and bounded imagery are fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from app.api.routes.query import (
    get_ai_service,
    get_query_execution_service,
    get_query_service,
)
from app.core.errors import (
    ImageryError,
    InvalidInputError,
    NotFoundError,
    UpstreamServiceError,
)
from app.main import create_app
from app.services.ai import AiService, MockIntentParser
from app.services.geospatial import ResolveRequest, ResolveResponse
from app.services.geospatial.schemas import BoundingBox
from app.services.query import (
    QueryExecutionRequest,
    QueryExecutionService,
    QueryService,
)
from app.services.query.execution import _expand_windows, _select_scene
from app.services.query.schemas import SatQueryIntent
from app.services.satellite.schemas import (
    ImageryResponse,
    QueryEcho,
    Scene,
    SceneSearchResponse,
    WindowInfo,
)
from fastapi.testclient import TestClient

EXECUTE_URL = "/api/v1/query/execute"
PARSE_URL = "/api/v1/query/parse"
BUILD_PLAN_URL = "/api/v1/query/build-plan"

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
CATALOG = "https://earth-search.aws.element84.com/v1"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeGeospatialService:
    """Records ``resolve`` calls; returns a canned bbox or raises."""

    def __init__(
        self, *, bbox: BoundingBox | None = None, error: Exception | None = None
    ) -> None:
        self.calls: list[ResolveRequest] = []
        self._bbox = bbox or DEFAULT_BBOX
        self._error = error

    async def resolve(self, request: ResolveRequest) -> ResolveResponse:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return ResolveResponse(
            query_type="place",
            display_name="Chennai, Tamil Nadu, India",
            center=self._bbox.center,
            bbox=self._bbox,
            source="nominatim",
        )


class FakeSatelliteService:
    """Records search requests; returns canned responses (in order) or raises."""

    def __init__(
        self,
        *,
        responses: list[SceneSearchResponse] | SceneSearchResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[Any] = []
        self._responses = responses
        self._error = error
        self._index = 0

    async def search(self, request: Any) -> SceneSearchResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if isinstance(self._responses, list):
            response = self._responses[self._index]
            self._index += 1
            return response
        if self._responses is not None:
            return self._responses
        return make_search_response()


class FakeImageryService:
    """Records retrieve requests; returns a canned response or raises.

    ``errors`` gives a per-call outcome (an exception or ``None``); ``error``
    raises the same exception on every call.
    """

    def __init__(
        self,
        *,
        response: ImageryResponse | None = None,
        errors: list[Exception | None] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[Any] = []
        self._response = response
        self._errors = errors
        self._error = error
        self._index = 0

    def retrieve(self, request: Any) -> ImageryResponse:
        self.requests.append(request)
        if self._errors is not None:
            exc = self._errors[self._index]
            self._index += 1
            if exc is not None:
                raise exc
        elif self._error is not None:
            raise self._error
        return self._response or make_imagery_response(request.scene_id)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_scene(
    scene_id: str,
    *,
    cloud_cover: float | None = None,
    moment: str | None = None,
) -> Scene:
    return Scene(
        id=scene_id,
        datetime=moment,
        bbox=None,
        geometry=None,
        cloud_cover=cloud_cover,
        collection="sentinel-2-l2a",
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
    )


def make_search_response(
    *scenes: Scene, catalog: str = CATALOG
) -> SceneSearchResponse:
    return SceneSearchResponse(
        query=QueryEcho(
            collections=["sentinel-2-l2a"],
            bbox=[80.10, 12.90, 80.30, 13.20],
            datetime="2024-01-01T00:00:00Z/2024-01-31T23:59:59Z",
            max_cloud_cover=None,
            limit=10,
            filter=None,
        ),
        scene_count=len(scenes),
        scenes=list(scenes),
        catalog=catalog,
    )


def make_imagery_response(scene_id: str) -> ImageryResponse:
    return ImageryResponse(
        scene_id=scene_id,
        bbox=DEFAULT_BBOX,
        asset="visual",
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


COMPARISON = {
    "baseline": {"start_date": "2023-01-01", "end_date": "2023-03-31"},
    "target": {"start_date": "2024-01-01", "end_date": "2024-03-31"},
}

TIMESERIES = [
    {"start_date": "2024-01-01", "end_date": "2024-01-31"},
    {"start_date": "2024-02-01", "end_date": "2024-02-29"},
    {"start_date": "2024-03-01", "end_date": "2024-03-31"},
]


def build_service(
    *,
    geo: FakeGeospatialService | None = None,
    satellite: FakeSatelliteService | None = None,
    imagery: FakeImageryService | None = None,
) -> QueryExecutionService:
    geo = geo or FakeGeospatialService()
    return QueryExecutionService(
        query_service=QueryService(geospatial_service=geo),  # type: ignore[arg-type]
        satellite_service=satellite or FakeSatelliteService(),  # type: ignore[arg-type]
        imagery_service=imagery or FakeImageryService(),  # type: ignore[arg-type]
    )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def make_client(service: QueryExecutionService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_query_execution_service] = lambda: service
    return TestClient(app)


# --------------------------------------------------------------------------- #
# 1-3: temporal window expansion (pure)
# --------------------------------------------------------------------------- #


def test_expand_single_window() -> None:
    windows = _expand_windows(make_intent())
    assert [label for label, _ in windows] == ["single"]
    assert windows[0][1].start_date.isoformat() == "2024-01-01"
    assert windows[0][1].end_date.isoformat() == "2024-01-31"


def test_expand_timeseries_windows() -> None:
    windows = _expand_windows(
        make_intent(temporal_mode="timeseries", time_windows=TIMESERIES)
    )
    assert [label for label, _ in windows] == ["series[0]", "series[1]", "series[2]"]
    assert [w.start_date.isoformat() for _, w in windows] == [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
    ]


def test_expand_compare_windows() -> None:
    windows = _expand_windows(
        make_intent(temporal_mode="compare", time_windows=COMPARISON)
    )
    assert [label for label, _ in windows] == ["baseline", "target"]
    assert windows[0][1].start_date.isoformat() == "2023-01-01"
    assert windows[1][1].end_date.isoformat() == "2024-03-31"


# --------------------------------------------------------------------------- #
# 4-8: deterministic scene selection (pure)
# --------------------------------------------------------------------------- #


def test_select_scene_prefers_lowest_cloud_cover() -> None:
    scenes = [
        make_scene("a", cloud_cover=40.0),
        make_scene("b", cloud_cover=5.0),
        make_scene("c", cloud_cover=20.0),
    ]
    picked = _select_scene(scenes)
    assert picked is not None and picked.id == "b"


def test_select_scene_datetime_tie_break() -> None:
    scenes = [
        make_scene("a", cloud_cover=10.0, moment="2024-07-15T05:00:00Z"),
        make_scene("b", cloud_cover=10.0, moment="2024-07-01T05:00:00Z"),
    ]
    picked = _select_scene(scenes)
    assert picked is not None and picked.id == "b"


def test_select_scene_id_tie_break() -> None:
    scenes = [
        make_scene("scene-z", cloud_cover=10.0, moment="2024-07-01T05:00:00Z"),
        make_scene("scene-a", cloud_cover=10.0, moment="2024-07-01T05:00:00Z"),
    ]
    picked = _select_scene(scenes)
    assert picked is not None and picked.id == "scene-a"


def test_select_scene_none_cloud_cover_sorts_after_numeric() -> None:
    scenes = [
        make_scene("a", cloud_cover=None, moment="2024-01-01T00:00:00Z"),
        make_scene("b", cloud_cover=90.0, moment="2024-12-31T00:00:00Z"),
    ]
    picked = _select_scene(scenes)
    assert picked is not None and picked.id == "b"


def test_select_scene_all_none_cloud_cover_is_deterministic() -> None:
    scenes = [
        make_scene("b", cloud_cover=None, moment="2024-07-01T00:00:00Z"),
        make_scene("a", cloud_cover=None, moment="2024-07-01T00:00:00Z"),
    ]
    picked = _select_scene(scenes)
    assert picked is not None and picked.id == "a"


def test_select_scene_empty_list_returns_none() -> None:
    assert _select_scene([]) is None


# --------------------------------------------------------------------------- #
# 8-9: service-level execution (fakes)
# --------------------------------------------------------------------------- #


def test_empty_scene_list_is_not_an_execution_failure() -> None:
    satellite = FakeSatelliteService(responses=make_search_response())
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent())
        )
    )

    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.scene_count == 0
    assert window.scenes == []
    assert window.selected_scene_id is None
    assert window.imagery is None and window.imagery_error is None


def test_optical_modality_executes_discovery_and_selection() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(
            make_scene("scene-a", cloud_cover=30.0),
            make_scene("scene-b", cloud_cover=8.0),
        )
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent())
        )
    )

    assert result.executed_modalities == ["sentinel-2-optical"]
    assert result.skipped_modalities == []
    assert result.catalog == CATALOG
    assert result.plan.bbox == DEFAULT_BBOX

    assert len(satellite.requests) == 1
    request = satellite.requests[0]
    assert request.bbox == DEFAULT_BBOX
    assert request.start_date.isoformat() == "2024-01-01"
    assert request.end_date.isoformat() == "2024-01-31"

    window = result.windows[0]
    assert window.label == "single"
    assert window.scene_count == 2
    assert window.selected_scene_id == "scene-b"


def test_timeseries_runs_one_search_per_window() -> None:
    satellite = FakeSatelliteService(
        responses=[
            make_search_response(make_scene("w0", cloud_cover=1.0)),
            make_search_response(make_scene("w1", cloud_cover=1.0)),
            make_search_response(make_scene("w2", cloud_cover=1.0)),
        ]
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(
                intent=make_intent(temporal_mode="timeseries", time_windows=TIMESERIES)
            )
        )
    )

    assert [w.label for w in result.windows] == ["series[0]", "series[1]", "series[2]"]
    assert [r.start_date.isoformat() for r in satellite.requests] == [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
    ]
    assert [w.selected_scene_id for w in result.windows] == ["w0", "w1", "w2"]


# --------------------------------------------------------------------------- #
# 10: Sentinel-1 SAR is skipped, never executed
# --------------------------------------------------------------------------- #


def test_sar_modality_is_skipped_not_executed() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(make_scene("s", cloud_cover=1.0))
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(
                intent=make_intent(
                    modalities=["sentinel-2-optical", "sentinel-1-sar"]
                )
            )
        )
    )

    assert result.executed_modalities == ["sentinel-2-optical"]
    assert [s.modality for s in result.skipped_modalities] == ["sentinel-1-sar"]
    assert "not implemented" in result.skipped_modalities[0].reason.lower()
    assert len(satellite.requests) == 1  # optical discovery still ran
    assert len(result.windows) == 1


def test_sar_only_intent_executes_nothing_but_still_succeeds() -> None:
    satellite = FakeSatelliteService()
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent(modalities=["sentinel-1-sar"]))
        )
    )

    assert result.executed_modalities == []
    assert [s.modality for s in result.skipped_modalities] == ["sentinel-1-sar"]
    assert result.windows == []
    assert satellite.requests == []
    assert result.catalog  # well-defined: the configured catalog


# --------------------------------------------------------------------------- #
# 11-13: optional bounded imagery
# --------------------------------------------------------------------------- #


def test_include_imagery_false_never_calls_imagery_service() -> None:
    imagery = FakeImageryService()
    satellite = FakeSatelliteService(
        responses=make_search_response(make_scene("s", cloud_cover=2.0))
    )
    result = run(
        build_service(satellite=satellite, imagery=imagery).execute(
            QueryExecutionRequest(intent=make_intent(), include_imagery=False)
        )
    )

    assert imagery.requests == []
    assert result.windows[0].imagery is None
    assert result.windows[0].imagery_error is None


def test_include_imagery_true_retrieves_for_the_selected_scene() -> None:
    imagery = FakeImageryService()
    satellite = FakeSatelliteService(
        responses=make_search_response(
            make_scene("high", cloud_cover=50.0),
            make_scene("low", cloud_cover=3.0),
        )
    )
    result = run(
        build_service(satellite=satellite, imagery=imagery).execute(
            QueryExecutionRequest(intent=make_intent(), include_imagery=True)
        )
    )

    assert len(imagery.requests) == 1
    request = imagery.requests[0]
    assert request.scene_id == "low"
    assert request.bbox == DEFAULT_BBOX
    assert request.asset == "visual"

    window = result.windows[0]
    assert window.imagery is not None
    assert window.imagery.scene_id == "low"
    assert window.imagery_error is None


def test_imagery_failure_is_isolated_to_its_window() -> None:
    satellite = FakeSatelliteService(
        responses=[
            make_search_response(make_scene("w0", cloud_cover=1.0)),
            make_search_response(make_scene("w1", cloud_cover=1.0)),
            make_search_response(make_scene("w2", cloud_cover=1.0)),
        ]
    )
    imagery = FakeImageryService(
        errors=[
            InvalidInputError("bbox does not intersect the selected scene"),
            None,
            None,
        ]
    )
    result = run(
        build_service(satellite=satellite, imagery=imagery).execute(
            QueryExecutionRequest(
                intent=make_intent(
                    temporal_mode="timeseries", time_windows=TIMESERIES
                ),
                include_imagery=True,
            )
        )
    )

    assert len(result.windows) == 3
    assert result.windows[0].imagery is None
    assert (
        result.windows[0].imagery_error
        == "bbox does not intersect the selected scene"
    )
    assert result.windows[0].selected_scene_id == "w0"
    assert result.windows[1].imagery is not None
    assert result.windows[2].imagery is not None


def test_imagery_upstream_error_does_not_abort_execution() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(make_scene("s", cloud_cover=1.0))
    )
    imagery = FakeImageryService(
        error=ImageryError("Failed to encode the bounded image as PNG.")
    )
    result = run(
        build_service(satellite=satellite, imagery=imagery).execute(
            QueryExecutionRequest(intent=make_intent(), include_imagery=True)
        )
    )

    assert result.windows[0].imagery is None
    assert "PNG" in (result.windows[0].imagery_error or "")


# --------------------------------------------------------------------------- #
# 14-15: geospatial / STAC failures propagate unchanged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "error",
    [
        NotFoundError("No matching location was found."),
        UpstreamServiceError("Nominatim is unavailable."),
    ],
)
def test_geospatial_failure_propagates(error: Exception) -> None:
    service = build_service(geo=FakeGeospatialService(error=error))
    with pytest.raises(type(error)):
        run(service.execute(QueryExecutionRequest(intent=make_intent())))


def test_stac_failure_propagates() -> None:
    satellite = FakeSatelliteService(
        error=UpstreamServiceError("The satellite catalog timed out.")
    )
    service = build_service(satellite=satellite)
    with pytest.raises(UpstreamServiceError):
        run(service.execute(QueryExecutionRequest(intent=make_intent())))


# --------------------------------------------------------------------------- #
# 16-17: POST /api/v1/query/execute
# --------------------------------------------------------------------------- #


def test_execute_endpoint_with_dependency_override() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(make_scene("scene-a", cloud_cover=12.0))
    )
    client = make_client(build_service(satellite=satellite))

    response = client.post(EXECUTE_URL, json={"intent": intent_dict()})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "plan",
        "executed_modalities",
        "skipped_modalities",
        "windows",
        "catalog",
    }
    assert body["plan"]["bbox"] == {
        "west": pytest.approx(80.10),
        "south": pytest.approx(12.90),
        "east": pytest.approx(80.30),
        "north": pytest.approx(13.20),
    }
    assert body["executed_modalities"] == ["sentinel-2-optical"]
    assert body["skipped_modalities"] == []
    window = body["windows"][0]
    assert window["label"] == "single"
    assert window["selected_scene_id"] == "scene-a"
    assert window["imagery"] is None


def test_execute_endpoint_include_imagery_true() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(make_scene("scene-a", cloud_cover=12.0))
    )
    client = make_client(
        build_service(satellite=satellite, imagery=FakeImageryService())
    )

    response = client.post(
        EXECUTE_URL, json={"intent": intent_dict(), "include_imagery": True}
    )

    assert response.status_code == 200
    window = response.json()["windows"][0]
    assert window["imagery"]["scene_id"] == "scene-a"
    assert window["imagery"]["media_type"] == "image/png"


def test_execute_endpoint_reports_skipped_sar() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(make_scene("scene-a", cloud_cover=12.0))
    )
    client = make_client(build_service(satellite=satellite))

    response = client.post(
        EXECUTE_URL,
        json={
            "intent": intent_dict(modalities=["sentinel-2-optical", "sentinel-1-sar"])
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["executed_modalities"] == ["sentinel-2-optical"]
    assert body["skipped_modalities"] == [
        {
            "modality": "sentinel-1-sar",
            "reason": body["skipped_modalities"][0]["reason"],
        }
    ]
    assert "not implemented" in body["skipped_modalities"][0]["reason"].lower()


@pytest.mark.parametrize(
    "bad_intent",
    [
        intent_dict(task="segment"),
        intent_dict(modalities=[]),
        intent_dict(modalities=["radar"]),
        intent_dict(
            temporal_mode="compare",
            time_windows=[{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
        ),
        intent_dict(
            time_windows=[{"start_date": "2024-08-31", "end_date": "2024-01-01"}]
        ),
    ],
)
def test_execute_endpoint_rejects_invalid_intent(bad_intent: dict[str, Any]) -> None:
    satellite = FakeSatelliteService()
    client = make_client(build_service(satellite=satellite))

    response = client.post(EXECUTE_URL, json={"intent": bad_intent})

    assert response.status_code == 422
    assert satellite.requests == []  # never reached the service


def test_execute_endpoint_rejects_out_of_range_cloud_cover() -> None:
    client = make_client(build_service())
    response = client.post(
        EXECUTE_URL, json={"intent": intent_dict(), "max_cloud_cover": 150}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 18-19: existing endpoints remain unchanged
# --------------------------------------------------------------------------- #


def test_parse_endpoint_still_returns_only_the_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_ai_service] = lambda: AiService(
        parser=MockIntentParser()
    )
    client = TestClient(app)

    response = client.post(PARSE_URL, json={"prompt": "show me Chennai"})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    for leaked in ("plan", "windows", "executed_modalities", "catalog", "bbox"):
        assert leaked not in body


def test_build_plan_endpoint_still_returns_only_plan_fields() -> None:
    fake_geo = FakeGeospatialService()
    app = create_app()
    app.dependency_overrides[get_query_service] = lambda: QueryService(
        geospatial_service=fake_geo  # type: ignore[arg-type]
    )
    client = TestClient(app)

    response = client.post(BUILD_PLAN_URL, json=intent_dict())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"intent", "bbox"}
    assert body["intent"]["location_query"] == "Chennai"
    assert len(fake_geo.calls) == 1
