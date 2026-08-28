"""Geospatial resolve endpoint tests.

The Nominatim HTTP call is stubbed with ``httpx.MockTransport`` - no test
touches the live API.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from app.api.routes.geospatial import get_geospatial_service
from app.main import create_app
from app.services.geospatial import GeospatialService
from fastapi.testclient import TestClient

RESOLVE_URL = "/api/v1/geospatial/resolve"

CHENNAI_RESULT = [
    {
        "lat": "13.0836939",
        "lon": "80.270186",
        "display_name": "Chennai, Tamil Nadu, India",
        "boundingbox": ["12.9", "13.2", "80.1", "80.3"],
    }
]

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler) -> TestClient:
    transport = httpx.MockTransport(handler)
    app = create_app()
    app.dependency_overrides[get_geospatial_service] = lambda: GeospatialService(
        transport=transport
    )
    return TestClient(app)


def _unreachable(_: httpx.Request) -> httpx.Response:  # pragma: no cover
    raise AssertionError("Nominatim must not be called for bbox / invalid input")


def test_resolve_place_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["q"] == "Chennai"
        assert request.headers["user-agent"].startswith("SatQuery/0.1")
        return httpx.Response(200, json=CHENNAI_RESULT)

    response = make_client(handler).post(RESOLVE_URL, json={"place": "Chennai"})

    assert response.status_code == 200
    body = response.json()
    assert body["query_type"] == "place"
    assert body["source"] == "nominatim"
    assert body["display_name"] == "Chennai, Tamil Nadu, India"
    assert body["center"]["lat"] == pytest.approx(13.0836939)
    assert body["center"]["lon"] == pytest.approx(80.270186)
    assert body["bbox"] == {
        "west": pytest.approx(80.1),
        "south": pytest.approx(12.9),
        "east": pytest.approx(80.3),
        "north": pytest.approx(13.2),
    }


def test_resolve_place_not_found() -> None:
    response = make_client(lambda _: httpx.Response(200, json=[])).post(
        RESOLVE_URL, json={"place": "qwertzuiop noplace"}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_resolve_upstream_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    response = make_client(handler).post(RESOLVE_URL, json={"place": "Chennai"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_resolve_upstream_5xx() -> None:
    response = make_client(lambda _: httpx.Response(503, text="down")).post(
        RESOLVE_URL, json={"place": "Chennai"}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_resolve_malformed_payload() -> None:
    response = make_client(lambda _: httpx.Response(200, json=[{"lat": "1.0"}])).post(
        RESOLVE_URL, json={"place": "Chennai"}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_resolve_bbox_passthrough() -> None:
    response = make_client(_unreachable).post(
        RESOLVE_URL,
        json={"bbox": {"west": 80.1, "south": 12.9, "east": 80.3, "north": 13.2}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query_type"] == "bbox"
    assert body["source"] == "input"
    assert body["display_name"] is None
    assert body["center"]["lat"] == pytest.approx(13.05)
    assert body["center"]["lon"] == pytest.approx(80.2)


def test_resolve_requires_exactly_one_input() -> None:
    client = make_client(_unreachable)

    both = client.post(
        RESOLVE_URL,
        json={
            "place": "Chennai",
            "bbox": {"west": 1, "south": 1, "east": 2, "north": 2},
        },
    )
    neither = client.post(RESOLVE_URL, json={})

    assert both.status_code == 422
    assert neither.status_code == 422


def test_resolve_rejects_reversed_bbox() -> None:
    response = make_client(_unreachable).post(
        RESOLVE_URL,
        json={"bbox": {"west": 80.3, "south": 12.9, "east": 80.1, "north": 13.2}},
    )

    assert response.status_code == 422


def test_resolve_rejects_out_of_range_bbox() -> None:
    response = make_client(_unreachable).post(
        RESOLVE_URL,
        json={"bbox": {"west": -181, "south": 12.9, "east": 80.1, "north": 13.2}},
    )

    assert response.status_code == 422


def test_resolve_rejects_blank_place() -> None:
    response = make_client(_unreachable).post(RESOLVE_URL, json={"place": "   "})

    assert response.status_code == 422
