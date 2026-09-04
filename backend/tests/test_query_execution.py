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
    ExecutedWindow,
    Observation,
    ObservationSet,
    QueryExecutionRequest,
    QueryExecutionResult,
    QueryExecutionService,
    QueryService,
)
from app.services.query.execution import (
    _expand_windows,
    _select_scene,
    _select_scene_sar,
)
from app.services.query.schemas import ResolvedQueryPlan, SatQueryIntent, TimeRange
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
    """Records search requests; returns canned responses or raises.

    ``by_collection`` routes the response by ``request.collection`` (``None`` for
    optical, the S1 collection for SAR); a mapped value that is an ``Exception``
    is raised. ``responses`` (ordered list or single) is the collection-agnostic
    fallback.
    """

    def __init__(
        self,
        *,
        responses: list[SceneSearchResponse] | SceneSearchResponse | None = None,
        by_collection: dict[str | None, SceneSearchResponse | Exception] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[Any] = []
        self._responses = responses
        self._by_collection = by_collection
        self._error = error
        self._index = 0

    async def search(self, request: Any) -> SceneSearchResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._by_collection is not None:
            outcome = self._by_collection[request.collection]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
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
        return self._response or make_imagery_response(
            request.scene_id, asset=request.asset
        )


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_scene(
    scene_id: str,
    *,
    cloud_cover: float | None = None,
    moment: str | None = None,
    collection: str = "sentinel-2-l2a",
    processing_level: str | None = "L2A",
) -> Scene:
    return Scene(
        id=scene_id,
        datetime=moment,
        bbox=None,
        geometry=None,
        cloud_cover=cloud_cover,
        collection=collection,
        platform=None,
        processing_level=processing_level,
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


def make_imagery_response(scene_id: str, *, asset: str = "visual") -> ImageryResponse:
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
    assert request.collection is None  # SatelliteService falls back to its S2 default

    window = result.windows[0]
    assert window.modality == "sentinel-2-optical"
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
# Sentinel-1 SAR is now executed (discovery + selection), never skipped
# --------------------------------------------------------------------------- #


def _s1_scene(scene_id: str, *, moment: str) -> Scene:
    """A Sentinel-1 scene fixture: no cloud cover, no processing level, no assets."""

    return make_scene(
        scene_id,
        moment=moment,
        collection="sentinel-1-grd",
        processing_level=None,
    )


def test_sar_modality_is_executed_alongside_optical() -> None:
    satellite = FakeSatelliteService(
        by_collection={
            None: make_search_response(make_scene("s2", cloud_cover=5.0)),
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1", moment="2024-01-10T00:00:00Z")
            ),
        }
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

    assert result.executed_modalities == ["sentinel-2-optical", "sentinel-1-sar"]
    assert result.skipped_modalities == []
    assert [(w.modality, w.label) for w in result.windows] == [
        ("sentinel-2-optical", "single"),
        ("sentinel-1-sar", "single"),
    ]
    assert result.windows[0].selected_scene_id == "s2"
    assert result.windows[1].selected_scene_id == "s1"
    assert [r.collection for r in satellite.requests] == [None, "sentinel-1-grd"]
    assert satellite.requests[1].max_cloud_cover is None
    assert result.windows[1].imagery is None
    assert result.windows[1].imagery_error is None


def test_sar_only_intent_runs_s1_discovery() -> None:
    satellite = FakeSatelliteService(
        by_collection={
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1-a", moment="2024-01-03T00:00:00Z")
            )
        }
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent(modalities=["sentinel-1-sar"]))
        )
    )

    assert result.executed_modalities == ["sentinel-1-sar"]
    assert result.skipped_modalities == []
    assert len(result.windows) == 1
    assert result.windows[0].modality == "sentinel-1-sar"
    assert result.windows[0].label == "single"
    assert result.windows[0].selected_scene_id == "s1-a"
    assert result.windows[0].imagery is None
    assert satellite.requests[0].collection == "sentinel-1-grd"
    assert satellite.requests[0].max_cloud_cover is None
    assert result.catalog  # well-defined: the configured catalog


def test_s1_and_s2_use_correct_collection_and_select_independently() -> None:
    satellite = FakeSatelliteService(
        by_collection={
            None: make_search_response(
                make_scene("s2-hi", cloud_cover=40.0),
                make_scene("s2-lo", cloud_cover=3.0),
            ),
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1-late", moment="2024-02-01T00:00:00Z"),
                _s1_scene("s1-early", moment="2024-01-01T00:00:00Z"),
            ),
        }
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(
                intent=make_intent(
                    modalities=["sentinel-2-optical", "sentinel-1-sar"]
                ),
                max_cloud_cover=20.0,
            )
        )
    )

    # S2 -> lowest cloud; S1 -> earliest datetime (cloud is irrelevant for SAR)
    assert result.windows[0].selected_scene_id == "s2-lo"
    assert result.windows[1].selected_scene_id == "s1-early"
    assert satellite.requests[0].collection is None
    assert satellite.requests[0].max_cloud_cover == 20.0
    assert satellite.requests[1].collection == "sentinel-1-grd"
    assert satellite.requests[1].max_cloud_cover is None


def test_timeseries_runs_each_modality_per_window() -> None:
    satellite = FakeSatelliteService(
        by_collection={
            None: make_search_response(make_scene("s2", cloud_cover=1.0)),
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1", moment="2024-01-01T00:00:00Z")
            ),
        }
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(
                intent=make_intent(
                    temporal_mode="timeseries",
                    time_windows=TIMESERIES,
                    modalities=["sentinel-2-optical", "sentinel-1-sar"],
                )
            )
        )
    )

    assert [(w.modality, w.label) for w in result.windows] == [
        ("sentinel-2-optical", "series[0]"),
        ("sentinel-2-optical", "series[1]"),
        ("sentinel-2-optical", "series[2]"),
        ("sentinel-1-sar", "series[0]"),
        ("sentinel-1-sar", "series[1]"),
        ("sentinel-1-sar", "series[2]"),
    ]
    assert [r.collection for r in satellite.requests] == [
        None,
        None,
        None,
        "sentinel-1-grd",
        "sentinel-1-grd",
        "sentinel-1-grd",
    ]


def test_compare_runs_each_modality_per_window() -> None:
    satellite = FakeSatelliteService(
        by_collection={
            None: make_search_response(make_scene("s2", cloud_cover=1.0)),
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1", moment="2024-01-01T00:00:00Z")
            ),
        }
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(
                intent=make_intent(
                    temporal_mode="compare",
                    time_windows=COMPARISON,
                    modalities=["sentinel-2-optical", "sentinel-1-sar"],
                )
            )
        )
    )

    assert [(w.modality, w.label) for w in result.windows] == [
        ("sentinel-2-optical", "baseline"),
        ("sentinel-2-optical", "target"),
        ("sentinel-1-sar", "baseline"),
        ("sentinel-1-sar", "target"),
    ]


def test_s1_discovery_request_arguments() -> None:
    satellite = FakeSatelliteService(
        by_collection={"sentinel-1-grd": make_search_response()}
    )
    run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(
                intent=make_intent(modalities=["sentinel-1-sar"]),
                max_cloud_cover=25.0,
                limit=7,
            )
        )
    )

    req = satellite.requests[0]
    assert req.bbox == DEFAULT_BBOX
    assert req.start_date.isoformat() == "2024-01-01"
    assert req.end_date.isoformat() == "2024-01-31"
    assert req.collection == "sentinel-1-grd"
    assert req.max_cloud_cover is None  # never forwarded for S1
    assert req.limit == 7


def test_s2_discovery_request_arguments() -> None:
    satellite = FakeSatelliteService(
        by_collection={None: make_search_response(make_scene("s2", cloud_cover=5.0))}
    )
    run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(
                intent=make_intent(modalities=["sentinel-2-optical"]),
                max_cloud_cover=25.0,
                limit=7,
            )
        )
    )

    req = satellite.requests[0]
    assert req.collection is None  # SatelliteService falls back to its S2 default
    assert req.max_cloud_cover == 25.0  # forwarded for optical
    assert req.limit == 7


# --------------------------------------------------------------------------- #
# Deterministic Sentinel-1 scene selection (pure)
# --------------------------------------------------------------------------- #


def test_select_scene_sar_earliest_datetime_wins() -> None:
    scenes = [
        _s1_scene("late", moment="2024-07-15T00:00:00Z"),
        _s1_scene("early", moment="2024-07-01T00:00:00Z"),
    ]
    picked = _select_scene_sar(scenes)
    assert picked is not None and picked.id == "early"


def test_select_scene_sar_id_tie_break() -> None:
    scenes = [
        _s1_scene("s1-z", moment="2024-07-01T00:00:00Z"),
        _s1_scene("s1-a", moment="2024-07-01T00:00:00Z"),
    ]
    picked = _select_scene_sar(scenes)
    assert picked is not None and picked.id == "s1-a"


def test_select_scene_sar_none_datetime_sorts_last() -> None:
    scenes = [
        make_scene("no-date", moment=None, collection="sentinel-1-grd", processing_level=None),
        _s1_scene("dated", moment="2024-12-31T00:00:00Z"),
    ]
    picked = _select_scene_sar(scenes)
    assert picked is not None and picked.id == "dated"


def test_select_scene_sar_ignores_cloud_cover() -> None:
    scenes = [
        make_scene(
            "early-cloudy",
            moment="2024-01-01T00:00:00Z",
            cloud_cover=99.0,
            collection="sentinel-1-grd",
            processing_level=None,
        ),
        make_scene(
            "late-clear",
            moment="2024-06-01T00:00:00Z",
            cloud_cover=0.0,
            collection="sentinel-1-grd",
            processing_level=None,
        ),
    ]
    picked = _select_scene_sar(scenes)
    assert picked is not None and picked.id == "early-cloudy"


def test_select_scene_sar_empty_list_returns_none() -> None:
    assert _select_scene_sar([]) is None


def test_optical_selector_is_not_used_for_sar_windows() -> None:
    # An S1 scene with high cloud_cover set: the SAR selector must ignore it and
    # pick by datetime; the optical selector would have de-prioritised it.
    satellite = FakeSatelliteService(
        by_collection={
            "sentinel-1-grd": make_search_response(
                make_scene(
                    "s1-early",
                    moment="2024-01-01T00:00:00Z",
                    cloud_cover=90.0,
                    collection="sentinel-1-grd",
                    processing_level=None,
                ),
                make_scene(
                    "s1-late",
                    moment="2024-01-20T00:00:00Z",
                    cloud_cover=1.0,
                    collection="sentinel-1-grd",
                    processing_level=None,
                ),
            )
        }
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent(modalities=["sentinel-1-sar"]))
        )
    )
    assert result.windows[0].selected_scene_id == "s1-early"


# --------------------------------------------------------------------------- #
# Sentinel-1 metadata gaps, empty results, failure propagation, imagery guard
# --------------------------------------------------------------------------- #


def test_s1_scene_with_missing_metadata_is_discoverable_and_selectable() -> None:
    bare = Scene(
        id="s1-bare",
        datetime="2024-01-05T00:00:00Z",
        bbox=None,
        geometry=None,
        cloud_cover=None,
        collection="sentinel-1-grd",
        platform=None,
        processing_level=None,
        thumbnail_url=None,
        assets=[],
    )
    satellite = FakeSatelliteService(
        by_collection={"sentinel-1-grd": make_search_response(bare)}
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent(modalities=["sentinel-1-sar"]))
        )
    )
    assert result.windows[0].selected_scene_id == "s1-bare"
    assert result.windows[0].scene_count == 1


def test_empty_s1_result_is_not_a_failure() -> None:
    satellite = FakeSatelliteService(
        by_collection={"sentinel-1-grd": make_search_response()}
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent(modalities=["sentinel-1-sar"]))
        )
    )
    assert result.windows[0].modality == "sentinel-1-sar"
    assert result.windows[0].scene_count == 0
    assert result.windows[0].selected_scene_id is None
    assert result.windows[0].imagery is None
    assert result.windows[0].imagery_error is None


def test_s1_upstream_failure_propagates() -> None:
    satellite = FakeSatelliteService(
        by_collection={"sentinel-1-grd": UpstreamServiceError("S1 catalog down")}
    )
    service = build_service(satellite=satellite)
    with pytest.raises(UpstreamServiceError):
        run(
            service.execute(
                QueryExecutionRequest(intent=make_intent(modalities=["sentinel-1-sar"]))
            )
        )


def test_mixed_s1_s2_failure_propagates_not_soft_fail() -> None:
    satellite = FakeSatelliteService(
        by_collection={
            None: make_search_response(make_scene("s2", cloud_cover=5.0)),
            "sentinel-1-grd": UpstreamServiceError("S1 catalog down"),
        }
    )
    service = build_service(satellite=satellite)
    with pytest.raises(UpstreamServiceError):
        run(
            service.execute(
                QueryExecutionRequest(
                    intent=make_intent(
                        modalities=["sentinel-2-optical", "sentinel-1-sar"]
                    )
                )
            )
        )


def test_include_imagery_true_retrieves_vv_for_s1_and_visual_for_s2() -> None:
    imagery = FakeImageryService()
    satellite = FakeSatelliteService(
        by_collection={
            None: make_search_response(make_scene("s2", cloud_cover=5.0)),
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1", moment="2024-01-01T00:00:00Z")
            ),
        }
    )
    result = run(
        build_service(satellite=satellite, imagery=imagery).execute(
            QueryExecutionRequest(
                intent=make_intent(
                    modalities=["sentinel-2-optical", "sentinel-1-sar"]
                ),
                include_imagery=True,
            )
        )
    )

    # One retrieve per modality window; S2 -> "visual"/default collection,
    # S1 -> "vv"/sentinel-1-grd collection.
    assert [
        (r.scene_id, r.asset, r.collection) for r in imagery.requests
    ] == [
        ("s2", "visual", None),
        ("s1", "vv", "sentinel-1-grd"),
    ]
    s2_window = next(w for w in result.windows if w.modality == "sentinel-2-optical")
    s1_window = next(w for w in result.windows if w.modality == "sentinel-1-sar")
    assert s2_window.imagery is not None and s2_window.imagery.asset == "visual"
    assert s1_window.imagery is not None and s1_window.imagery.asset == "vv"
    assert s1_window.imagery_error is None
    # The S1 imagery request carries the grounded plan bbox, same as S2.
    s1_req = next(r for r in imagery.requests if r.scene_id == "s1")
    assert s1_req.bbox == DEFAULT_BBOX


def test_include_imagery_false_does_not_retrieve_for_s1() -> None:
    imagery = FakeImageryService()
    satellite = FakeSatelliteService(
        by_collection={
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1", moment="2024-01-01T00:00:00Z")
            )
        }
    )
    result = run(
        build_service(satellite=satellite, imagery=imagery).execute(
            QueryExecutionRequest(
                intent=make_intent(modalities=["sentinel-1-sar"]),
                include_imagery=False,
            )
        )
    )
    assert imagery.requests == []
    assert result.windows[0].imagery is None
    assert result.windows[0].imagery_error is None


def test_s1_imagery_failure_populates_error_without_aborting_execution() -> None:
    # S2 imagery succeeds; S1 imagery fails -> isolated to the S1 window.
    imagery = FakeImageryService(errors=[None, ImageryError("SAR render failed")])
    satellite = FakeSatelliteService(
        by_collection={
            None: make_search_response(make_scene("s2", cloud_cover=5.0)),
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1", moment="2024-01-01T00:00:00Z")
            ),
        }
    )
    result = run(
        build_service(satellite=satellite, imagery=imagery).execute(
            QueryExecutionRequest(
                intent=make_intent(
                    modalities=["sentinel-2-optical", "sentinel-1-sar"]
                ),
                include_imagery=True,
            )
        )
    )
    s2_window = next(w for w in result.windows if w.modality == "sentinel-2-optical")
    s1_window = next(w for w in result.windows if w.modality == "sentinel-1-sar")
    assert s2_window.imagery is not None
    assert s2_window.imagery_error is None
    assert s1_window.imagery is None
    assert s1_window.imagery_error == "SAR render failed"


def test_s1_discovery_failure_propagates_even_with_include_imagery() -> None:
    satellite = FakeSatelliteService(
        by_collection={"sentinel-1-grd": UpstreamServiceError("S1 catalog down")}
    )
    service = build_service(satellite=satellite, imagery=FakeImageryService())
    with pytest.raises(UpstreamServiceError):
        run(
            service.execute(
                QueryExecutionRequest(
                    intent=make_intent(modalities=["sentinel-1-sar"]),
                    include_imagery=True,
                )
            )
        )


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
        "observations",  # Phase 12: derived from windows, additive
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
    assert window["modality"] == "sentinel-2-optical"
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


def test_execute_endpoint_executes_s1_alongside_s2() -> None:
    satellite = FakeSatelliteService(
        by_collection={
            None: make_search_response(make_scene("s2-a", cloud_cover=12.0)),
            "sentinel-1-grd": make_search_response(
                _s1_scene("s1-a", moment="2024-01-02T00:00:00Z")
            ),
        }
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
    assert body["executed_modalities"] == ["sentinel-2-optical", "sentinel-1-sar"]
    assert body["skipped_modalities"] == []
    assert [(w["modality"], w["label"]) for w in body["windows"]] == [
        ("sentinel-2-optical", "single"),
        ("sentinel-1-sar", "single"),
    ]
    assert body["windows"][1]["selected_scene_id"] == "s1-a"
    assert body["windows"][1]["imagery"] is None


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
        # Phase 17.2, additive: always serialized, null unless the
        # request stated an explicit NDWI threshold.
        "ndwi_threshold": None,
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


# =========================================================================== #
# Phase 12: temporal observation model
#
# A requested window and an actual observation are different things. These
# tests pin that distinction and prove the model carries enough metadata for a
# later co-registration / comparison phase WITHOUT asserting any alignment.
# =========================================================================== #


def observed_window(
    *,
    modality: str = "sentinel-2-optical",
    label: str = "single",
    scene_id: str | None = "scene-a",
    moment: str | None = "2024-01-15T05:12:34Z",
    collection: str = "sentinel-2-l2a",
    start: str = "2024-01-01",
    end: str = "2024-01-31",
    imagery: ImageryResponse | None = None,
) -> ExecutedWindow:
    """An executed window that did (or did not) select a scene."""

    scenes = (
        []
        if scene_id is None
        else [make_scene(scene_id, moment=moment, collection=collection)]
    )
    return ExecutedWindow(
        modality=modality,  # type: ignore[arg-type]
        label=label,
        time_range=TimeRange.model_validate({"start_date": start, "end_date": end}),
        scene_count=len(scenes),
        scenes=scenes,
        selected_scene_id=scene_id,
        imagery=imagery,
    )


def observation_set(*windows: ExecutedWindow) -> ObservationSet:
    return ObservationSet.from_windows(DEFAULT_BBOX, list(windows))


def test_a_single_observation_can_be_represented() -> None:
    result = observation_set(observed_window())

    assert len(result.observations) == 1
    observation = result.observations[0]
    assert observation.scene_id == "scene-a"
    assert observation.modality == "sentinel-2-optical"
    assert observation.window_label == "single"


def test_multiple_observations_can_be_represented() -> None:
    result = observation_set(
        observed_window(label="series[0]", scene_id="s0", moment="2024-01-15T05:00:00Z"),
        observed_window(label="series[1]", scene_id="s1", moment="2024-02-15T05:00:00Z"),
        observed_window(label="series[2]", scene_id="s2", moment="2024-03-15T05:00:00Z"),
    )

    assert [o.scene_id for o in result.observations] == ["s0", "s1", "s2"]
    assert [o.window_label for o in result.observations] == [
        "series[0]",
        "series[1]",
        "series[2]",
    ]


def test_observation_preserves_the_acquired_scene_metadata() -> None:
    observation = observation_set(observed_window()).observations[0]

    # The Scene is embedded verbatim - not copied field by field.
    assert observation.scene.id == "scene-a"
    assert observation.scene.datetime == "2024-01-15T05:12:34Z"
    assert observation.scene.collection == "sentinel-2-l2a"
    assert observation.scene.processing_level == "L2A"
    assert observation.collection == "sentinel-2-l2a"  # convenience accessor


def test_observation_carries_its_imagery_when_retrieved() -> None:
    imagery = make_imagery_response("scene-a")
    with_imagery = observation_set(observed_window(imagery=imagery)).observations[0]
    without = observation_set(observed_window()).observations[0]

    assert with_imagery.imagery is not None
    assert with_imagery.imagery.scene_id == "scene-a"
    assert without.imagery is None


def test_sentinel_1_and_sentinel_2_observations_coexist() -> None:
    result = observation_set(
        observed_window(scene_id="s2-a", collection="sentinel-2-l2a"),
        observed_window(
            modality="sentinel-1-sar",
            scene_id="s1-a",
            collection="sentinel-1-grd",
            moment="2024-01-12T00:15:00Z",
        ),
    )

    assert len(result.observations) == 2
    assert [o.scene_id for o in result.for_modality("sentinel-2-optical")] == ["s2-a"]
    assert [o.scene_id for o in result.for_modality("sentinel-1-sar")] == ["s1-a"]
    # Each keeps its own collection - the two are not merged or paired.
    assert {o.collection for o in result.observations} == {
        "sentinel-2-l2a",
        "sentinel-1-grd",
    }


def test_different_acquisition_times_coexist_and_can_be_ordered() -> None:
    result = observation_set(
        observed_window(label="target", scene_id="late", moment="2024-06-20T05:00:00Z"),
        observed_window(label="baseline", scene_id="early", moment="2024-01-12T05:00:00Z"),
        observed_window(label="unknown", scene_id="undated", moment=None),
    )

    assert [o.scene_id for o in result.ordered_by_acquisition()] == [
        "early",
        "late",
        "undated",  # unknown acquisition time sorts last
    ]
    early, late, undated = (
        result.observations[1],
        result.observations[0],
        result.observations[2],
    )
    assert early.acquired_at is not None and late.acquired_at is not None
    assert early.acquired_at < late.acquired_at
    assert undated.acquired_at is None


def test_requested_window_is_distinct_from_the_acquisition_time() -> None:
    observation = observation_set(
        observed_window(
            start="2024-01-01", end="2024-01-31", moment="2024-01-15T05:12:34Z"
        )
    ).observations[0]

    # The window is what was ASKED FOR; the acquisition is what was RECEIVED.
    assert observation.requested_window.start_date.isoformat() == "2024-01-01"
    assert observation.requested_window.end_date.isoformat() == "2024-01-31"
    assert observation.acquired_at is not None
    assert observation.acquired_at.date().isoformat() == "2024-01-15"
    assert observation.acquired_at.date() != observation.requested_window.start_date


def test_a_window_with_no_selected_scene_yields_no_observation() -> None:
    result = observation_set(
        observed_window(label="found", scene_id="scene-a"),
        observed_window(label="empty", scene_id=None),
    )

    # A requested window without data is a legitimate outcome, not an observation.
    assert [o.window_label for o in result.observations] == ["found"]


def test_observations_are_grouped_per_window_across_modalities() -> None:
    result = observation_set(
        observed_window(label="baseline", scene_id="s2-base"),
        observed_window(
            modality="sentinel-1-sar", label="baseline", scene_id="s1-base",
            collection="sentinel-1-grd",
        ),
        observed_window(label="target", scene_id="s2-target"),
    )

    baseline = result.for_window_label("baseline")
    assert {(o.modality, o.scene_id) for o in baseline} == {
        ("sentinel-2-optical", "s2-base"),
        ("sentinel-1-sar", "s1-base"),
    }
    assert [o.scene_id for o in result.for_window_label("target")] == ["s2-target"]


def test_observation_set_records_the_requested_aoi_not_an_alignment_claim() -> None:
    result = observation_set(
        observed_window(scene_id="s2-a"),
        observed_window(
            modality="sentinel-1-sar", scene_id="s1-a", collection="sentinel-1-grd"
        ),
    )

    assert result.requested_bbox == DEFAULT_BBOX
    # Nothing in the model asserts a shared grid, CRS, resolution or footprint.
    for field in ("crs", "resolution", "transform", "aligned", "common_grid"):
        assert field not in ObservationSet.model_fields
        assert field not in Observation.model_fields


def test_observation_contract_carries_no_raster_arrays() -> None:
    imagery = make_imagery_response("scene-a")
    result = observation_set(observed_window(imagery=imagery))
    payload = result.model_dump(mode="json")

    def has_nested_list(value: Any) -> bool:
        if isinstance(value, list):
            return any(isinstance(item, list) for item in value) or any(
                has_nested_list(item) for item in value
            )
        if isinstance(value, dict):
            return any(has_nested_list(item) for item in value.values())
        return False

    # source_shape is a flat [h, w] pair; nothing 2-D, no arrays, no masks.
    assert not has_nested_list(payload)
    assert "array" not in str(payload) and "mask" not in str(payload)
    # Imagery is still the existing display PNG contract, not raw pixels.
    observation = payload["observations"][0]
    assert observation["imagery"]["media_type"] == "image/png"


# --- integration with the existing execution result ------------------------ #


def test_execution_result_derives_observations_from_its_windows() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(
            make_scene("scene-a", cloud_cover=5.0, moment="2024-01-15T05:00:00Z")
        )
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent())
        )
    )

    assert len(result.observations.observations) == 1
    observation = result.observations.observations[0]
    assert observation.scene_id == "scene-a"
    assert observation.window_label == result.windows[0].label
    assert observation.modality == result.windows[0].modality
    assert result.observations.requested_bbox == result.plan.bbox


def test_execution_result_observations_cannot_drift_from_windows() -> None:
    intent = make_intent()
    plan = ResolvedQueryPlan(intent=intent, bbox=DEFAULT_BBOX)
    result = QueryExecutionResult(
        plan=plan,
        executed_modalities=["sentinel-2-optical"],
        skipped_modalities=[],
        windows=[observed_window()],
        catalog=CATALOG,
        # A caller cannot inject a contradictory set - it is always derived.
        observations={"requested_bbox": DEFAULT_BBOX, "observations": []},  # type: ignore[call-arg]
    )

    assert len(result.observations.observations) == 1
    assert result.observations.observations[0].scene_id == "scene-a"


def test_existing_single_window_execution_behaviour_is_unchanged() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(make_scene("scene-a", cloud_cover=5.0))
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent())
        )
    )

    # Every pre-Phase-12 field keeps its exact meaning.
    assert result.executed_modalities == ["sentinel-2-optical"]
    assert result.skipped_modalities == []
    assert result.catalog == CATALOG
    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.label == "single"
    assert window.selected_scene_id == "scene-a"
    assert window.scene_count == 1


def test_execution_result_round_trips_through_json_with_observations() -> None:
    satellite = FakeSatelliteService(
        responses=make_search_response(
            make_scene("scene-a", cloud_cover=5.0, moment="2024-01-15T05:00:00Z")
        )
    )
    result = run(
        build_service(satellite=satellite).execute(
            QueryExecutionRequest(intent=make_intent())
        )
    )

    payload = result.model_dump(mode="json")
    assert "observations" in payload

    # A client echoing the whole result back (which the frontend does) still
    # validates, and observations are recomputed rather than trusted.
    restored = QueryExecutionResult.model_validate(payload)
    assert restored.model_dump(mode="json") == payload
    assert restored.observations.observations[0].scene_id == "scene-a"
