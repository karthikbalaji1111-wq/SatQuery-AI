"""Sentinel-2 STAC scene-discovery tests.

Every STAC HTTP call is stubbed with ``httpx.MockTransport`` - no test touches
the live Earth Search API, and no test downloads assets, COGs, or pixels.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from app.api.routes.satellite import get_satellite_service
from app.main import create_app
from app.services.satellite import SatelliteService
from fastapi.testclient import TestClient

SEARCH_URL = "/api/v1/satellite/search"

VALID_REQUEST: dict[str, Any] = {
    "bbox": {"west": 80.15, "south": 12.90, "east": 80.35, "north": 13.15},
    "start_date": "2024-06-01",
    "end_date": "2024-08-31",
    "max_cloud_cover": 30,
    "limit": 10,
}

SCENE_FEATURE: dict[str, Any] = {
    "type": "Feature",
    "stac_version": "1.0.0",
    "id": "S2B_44PLA_20240715_0_L2A",
    "collection": "sentinel-2-l2a",
    "bbox": [80.10, 12.85, 80.42, 13.22],
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [[80.10, 12.85], [80.42, 12.85], [80.42, 13.22], [80.10, 13.22], [80.10, 12.85]]
        ],
    },
    "properties": {
        "datetime": "2024-07-15T05:12:34.567Z",
        "eo:cloud_cover": 12.34,
        "platform": "sentinel-2b",
        "processing:level": "L2A",
        "s2:product_type": "S2MSI2A",
        "view:sun_elevation": 61.2,
    },
    "assets": {
        "thumbnail": {
            "href": "https://example.test/thumb.jpg",
            "type": "image/jpeg",
            "title": "Thumbnail image",
            "roles": ["thumbnail"],
        },
        "visual": {
            "href": "https://example.test/TCI.tif",
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "roles": ["visual"],
        },
        "red": {"href": "https://example.test/B04.tif", "type": "image/tiff"},
        "nir": {"href": "https://example.test/B08.tif", "type": "image/tiff"},
    },
    "links": [{"rel": "self", "href": "https://example.test/items/abc"}],
}

Handler = Callable[[httpx.Request], httpx.Response]


def feature_collection(*features: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def make_client(handler: Handler) -> TestClient:
    transport = httpx.MockTransport(handler)
    app = create_app()
    app.dependency_overrides[get_satellite_service] = lambda: SatelliteService(
        transport=transport
    )
    return TestClient(app)


def ok_handler(payload: dict[str, Any]) -> Handler:
    return lambda _request: httpx.Response(200, json=payload)


def _unreachable(_: httpx.Request) -> httpx.Response:  # pragma: no cover
    raise AssertionError("STAC must not be called for invalid input")


# --------------------------------------------------------------------------- #
# Happy path + request construction
# --------------------------------------------------------------------------- #


def test_search_success_and_normalization() -> None:
    response = make_client(ok_handler(feature_collection(SCENE_FEATURE))).post(
        SEARCH_URL, json=VALID_REQUEST
    )

    assert response.status_code == 200
    body = response.json()

    assert body["catalog"] == "https://earth-search.aws.element84.com/v1"
    assert body["scene_count"] == 1
    assert len(body["scenes"]) == 1

    scene = body["scenes"][0]
    assert scene["id"] == "S2B_44PLA_20240715_0_L2A"
    assert scene["datetime"] == "2024-07-15T05:12:34.567Z"
    assert scene["cloud_cover"] == pytest.approx(12.34)
    assert scene["collection"] == "sentinel-2-l2a"
    assert scene["platform"] == "sentinel-2b"
    assert scene["processing_level"] == "L2A"
    assert scene["thumbnail_url"] == "https://example.test/thumb.jpg"
    assert scene["bbox"] == {
        "west": pytest.approx(80.10),
        "south": pytest.approx(12.85),
        "east": pytest.approx(80.42),
        "north": pytest.approx(13.22),
    }
    assert scene["geometry"]["type"] == "Polygon"

    # Only curated assets are surfaced; raw band rasters are dropped.
    asset_keys = {asset["key"] for asset in scene["assets"]}
    assert asset_keys == {"thumbnail", "visual"}
    assert "red" not in asset_keys and "nir" not in asset_keys
    # No raw STAC passthrough.
    assert "links" not in scene
    assert "properties" not in scene


def test_stac_request_construction() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=feature_collection())

    make_client(handler).post(SEARCH_URL, json=VALID_REQUEST)

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/search"
    body = captured["body"]
    assert body["collections"] == ["sentinel-2-l2a"]
    assert body["bbox"] == [80.15, 12.90, 80.35, 13.15]
    assert body["datetime"] == "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z"
    assert body["query"] == {"eo:cloud_cover": {"lte": 30}}
    assert body["limit"] == 10


def test_query_echo_reports_actual_parameters() -> None:
    body = (
        make_client(ok_handler(feature_collection(SCENE_FEATURE)))
        .post(SEARCH_URL, json=VALID_REQUEST)
        .json()
    )
    query = body["query"]
    assert query["collections"] == ["sentinel-2-l2a"]
    assert query["bbox"] == [80.15, 12.90, 80.35, 13.15]
    assert query["datetime"] == "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z"
    assert query["max_cloud_cover"] == pytest.approx(30)
    assert query["limit"] == 10
    assert query["filter"] == {"eo:cloud_cover": {"lte": 30}}


def test_cloud_cover_filter_omitted_when_absent() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=feature_collection())

    request = {k: v for k, v in VALID_REQUEST.items() if k != "max_cloud_cover"}
    body = make_client(handler).post(SEARCH_URL, json=request).json()

    assert "query" not in captured["body"]
    assert body["query"]["filter"] is None
    assert body["query"]["max_cloud_cover"] is None


def test_limit_is_propagated_and_enforced() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        # Catalog hands back more than requested; service must clamp.
        return httpx.Response(
            200,
            json=feature_collection(
                {**SCENE_FEATURE, "id": "scene-a"},
                {**SCENE_FEATURE, "id": "scene-b"},
                {**SCENE_FEATURE, "id": "scene-c"},
            ),
        )

    request = {**VALID_REQUEST, "limit": 2}
    body = make_client(handler).post(SEARCH_URL, json=request).json()

    assert captured["body"]["limit"] == 2
    assert body["scene_count"] == 2
    assert len(body["scenes"]) == 2


# --------------------------------------------------------------------------- #
# Normalization edge cases
# --------------------------------------------------------------------------- #


def test_normalization_tolerates_missing_optional_fields() -> None:
    minimal = {"type": "Feature", "id": "bare-scene", "collection": "sentinel-2-l2a"}
    body = (
        make_client(ok_handler(feature_collection(minimal)))
        .post(SEARCH_URL, json=VALID_REQUEST)
        .json()
    )

    scene = body["scenes"][0]
    assert scene["id"] == "bare-scene"
    assert scene["datetime"] is None
    assert scene["bbox"] is None
    assert scene["geometry"] is None
    assert scene["cloud_cover"] is None
    assert scene["platform"] is None
    assert scene["thumbnail_url"] is None
    assert scene["assets"] == []
    # Processing level is still derived from the known collection.
    assert scene["processing_level"] == "L2A"


def test_zero_results_is_a_valid_response() -> None:
    response = make_client(ok_handler(feature_collection())).post(
        SEARCH_URL, json=VALID_REQUEST
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_count"] == 0
    assert body["scenes"] == []


# --------------------------------------------------------------------------- #
# Upstream / payload failures -> 502
# --------------------------------------------------------------------------- #


def test_malformed_stac_json() -> None:
    response = make_client(lambda _: httpx.Response(200, text="not json at all")).post(
        SEARCH_URL, json=VALID_REQUEST
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_missing_features_key() -> None:
    response = make_client(ok_handler({"type": "FeatureCollection"})).post(
        SEARCH_URL, json=VALID_REQUEST
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_malformed_scene_not_an_object() -> None:
    response = make_client(
        ok_handler({"type": "FeatureCollection", "features": ["nope"]})
    ).post(SEARCH_URL, json=VALID_REQUEST)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_malformed_scene_without_id() -> None:
    response = make_client(
        ok_handler(feature_collection({"type": "Feature", "properties": {}}))
    ).post(SEARCH_URL, json=VALID_REQUEST)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_upstream_non_2xx() -> None:
    response = make_client(lambda _: httpx.Response(503, text="unavailable")).post(
        SEARCH_URL, json=VALID_REQUEST
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_upstream_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    response = make_client(handler).post(SEARCH_URL, json=VALID_REQUEST)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_network_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    response = make_client(handler).post(SEARCH_URL, json=VALID_REQUEST)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


# --------------------------------------------------------------------------- #
# Input validation -> 422 (project convention: pydantic request validation)
# --------------------------------------------------------------------------- #


def test_malformed_request_missing_fields() -> None:
    assert make_client(_unreachable).post(SEARCH_URL, json={}).status_code == 422


def test_invalid_bbox_reversed() -> None:
    request = {
        **VALID_REQUEST,
        "bbox": {"west": 80.35, "south": 12.90, "east": 80.15, "north": 13.15},
    }
    assert make_client(_unreachable).post(SEARCH_URL, json=request).status_code == 422


def test_invalid_bbox_out_of_range() -> None:
    request = {
        **VALID_REQUEST,
        "bbox": {"west": -181.0, "south": 12.90, "east": 80.15, "north": 13.15},
    }
    assert make_client(_unreachable).post(SEARCH_URL, json=request).status_code == 422


def test_invalid_date_format() -> None:
    request = {**VALID_REQUEST, "start_date": "01-06-2024"}
    assert make_client(_unreachable).post(SEARCH_URL, json=request).status_code == 422


def test_start_date_after_end_date() -> None:
    request = {**VALID_REQUEST, "start_date": "2024-09-01", "end_date": "2024-06-01"}
    assert make_client(_unreachable).post(SEARCH_URL, json=request).status_code == 422


def test_invalid_cloud_cover_range() -> None:
    high = {**VALID_REQUEST, "max_cloud_cover": 150}
    low = {**VALID_REQUEST, "max_cloud_cover": -5}
    assert make_client(_unreachable).post(SEARCH_URL, json=high).status_code == 422
    assert make_client(_unreachable).post(SEARCH_URL, json=low).status_code == 422


def test_invalid_limit() -> None:
    zero = {**VALID_REQUEST, "limit": 0}
    huge = {**VALID_REQUEST, "limit": 9999}
    assert make_client(_unreachable).post(SEARCH_URL, json=zero).status_code == 422
    assert make_client(_unreachable).post(SEARCH_URL, json=huge).status_code == 422


# --------------------------------------------------------------------------- #
# Guarantee: metadata only, no asset/imagery requests
# --------------------------------------------------------------------------- #


def test_no_asset_or_imagery_downloads() -> None:
    requests_seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append((request.method, request.url.path))
        return httpx.Response(200, json=feature_collection(SCENE_FEATURE))

    make_client(handler).post(SEARCH_URL, json=VALID_REQUEST)

    # Exactly one request, the STAC search POST - never a GET to a thumbnail,
    # COG, JP2, or any asset href.
    assert len(requests_seen) == 1
    method, path = requests_seen[0]
    assert method == "POST"
    assert path.endswith("/search")


# --------------------------------------------------------------------------- #
# Collection override (Sentinel-1) - additive, default preserves S2
# --------------------------------------------------------------------------- #


def test_default_collection_is_sentinel_2() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=feature_collection())

    make_client(handler).post(SEARCH_URL, json=VALID_REQUEST)

    assert captured["body"]["collections"] == ["sentinel-2-l2a"]


def test_collection_override_produces_sentinel_1_stac_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=feature_collection())

    request = {
        "bbox": VALID_REQUEST["bbox"],
        "start_date": VALID_REQUEST["start_date"],
        "end_date": VALID_REQUEST["end_date"],
        "collection": "sentinel-1-grd",
        "limit": 10,
    }
    body = make_client(handler).post(SEARCH_URL, json=request).json()

    assert captured["body"]["collections"] == ["sentinel-1-grd"]
    assert body["query"]["collections"] == ["sentinel-1-grd"]
    assert body["catalog"] == "https://earth-search.aws.element84.com/v1"


def test_cloud_cover_filter_not_applied_to_non_optical_collection() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=feature_collection())

    # max_cloud_cover supplied *and* a non-optical collection: no eo:cloud_cover.
    request = {**VALID_REQUEST, "collection": "sentinel-1-grd", "max_cloud_cover": 30}
    body = make_client(handler).post(SEARCH_URL, json=request).json()

    assert "query" not in captured["body"]
    assert body["query"]["filter"] is None
    assert body["query"]["max_cloud_cover"] == pytest.approx(30)


def test_cloud_cover_filter_still_applied_to_default_collection() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=feature_collection())

    # Explicitly passing the S2 collection keeps the optical filter.
    request = {**VALID_REQUEST, "collection": "sentinel-2-l2a", "max_cloud_cover": 30}
    make_client(handler).post(SEARCH_URL, json=request)

    assert captured["body"]["query"] == {"eo:cloud_cover": {"lte": 30}}
