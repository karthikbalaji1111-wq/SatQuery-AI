"""Structured query-intent and plan-resolution tests.

The Geospatial Service is replaced with an in-memory fake - no test performs
geocoding, STAC discovery, or imagery retrieval.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.api.routes.query import get_query_service
from app.core.errors import NotFoundError, UpstreamServiceError
from app.main import create_app
from app.services.geospatial import ResolveRequest, ResolveResponse
from app.services.geospatial.schemas import BoundingBox
from app.services.query import QueryService, SatQueryIntent
from fastapi.testclient import TestClient

BUILD_PLAN_URL = "/api/v1/query/build-plan"

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)


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


def make_client(
    fake: FakeGeospatialService | None = None,
) -> tuple[TestClient, FakeGeospatialService]:
    fake = fake or FakeGeospatialService()
    app = create_app()
    app.dependency_overrides[get_query_service] = lambda: QueryService(
        geospatial_service=fake  # type: ignore[arg-type]
    )
    return TestClient(app), fake


def intent_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2024-06-01", "end_date": "2024-08-31"}],
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    body.update(overrides)
    return body


COMPARISON = {
    "baseline": {"start_date": "2023-01-01", "end_date": "2023-03-31"},
    "target": {"start_date": "2024-01-01", "end_date": "2024-03-31"},
}


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


def test_build_plan_valid_single() -> None:
    client, fake = make_client()
    response = client.post(BUILD_PLAN_URL, json=intent_body())

    assert response.status_code == 200
    plan = response.json()
    assert plan["intent"]["location_query"] == "Chennai"
    assert plan["intent"]["temporal_mode"] == "single"
    assert plan["intent"]["task"] == "visualize"
    assert plan["intent"]["modalities"] == ["sentinel-2-optical"]
    assert plan["bbox"] == {
        "west": pytest.approx(80.10),
        "south": pytest.approx(12.90),
        "east": pytest.approx(80.30),
        "north": pytest.approx(13.20),
    }
    assert len(fake.calls) == 1


def test_build_plan_compare() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL,
        json=intent_body(temporal_mode="compare", time_windows=COMPARISON),
    )

    assert response.status_code == 200
    windows = response.json()["intent"]["time_windows"]
    assert windows["baseline"]["start_date"] == "2023-01-01"
    assert windows["target"]["end_date"] == "2024-03-31"


def test_build_plan_timeseries() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL,
        json=intent_body(
            temporal_mode="timeseries",
            time_windows=[
                {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                {"start_date": "2024-02-01", "end_date": "2024-02-29"},
                {"start_date": "2024-03-01", "end_date": "2024-03-31"},
            ],
        ),
    )

    assert response.status_code == 200
    assert len(response.json()["intent"]["time_windows"]) == 3


def test_build_plan_preserves_multiple_modalities_and_task() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL,
        json=intent_body(
            modalities=["sentinel-2-optical", "sentinel-1-sar"],
            task="change_detection",
        ),
    )

    assert response.status_code == 200
    intent = response.json()["intent"]
    assert intent["modalities"] == ["sentinel-2-optical", "sentinel-1-sar"]
    assert intent["task"] == "change_detection"


# --------------------------------------------------------------------------- #
# Intent validation -> 422
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_location_is_rejected(value: str) -> None:
    client, fake = make_client()
    response = client.post(BUILD_PLAN_URL, json=intent_body(location_query=value))
    assert response.status_code == 422
    assert fake.calls == []


def test_invalid_date_range_is_rejected() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL,
        json=intent_body(
            time_windows=[{"start_date": "2024-08-31", "end_date": "2024-06-01"}]
        ),
    )
    assert response.status_code == 422


def test_invalid_date_range_in_comparison_is_rejected() -> None:
    client, _ = make_client()
    bad = {
        "baseline": {"start_date": "2023-03-31", "end_date": "2023-01-01"},
        "target": {"start_date": "2024-01-01", "end_date": "2024-03-31"},
    }
    response = client.post(
        BUILD_PLAN_URL, json=intent_body(temporal_mode="compare", time_windows=bad)
    )
    assert response.status_code == 422


def test_single_mode_with_comparison_structure_is_rejected() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL,
        json=intent_body(temporal_mode="single", time_windows=COMPARISON),
    )
    assert response.status_code == 422


def test_single_mode_with_two_windows_is_rejected() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL,
        json=intent_body(
            time_windows=[
                {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                {"start_date": "2024-02-01", "end_date": "2024-02-28"},
            ]
        ),
    )
    assert response.status_code == 422


def test_compare_mode_with_list_is_rejected() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL,
        json=intent_body(
            temporal_mode="compare",
            time_windows=[{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
        ),
    )
    assert response.status_code == 422


def test_timeseries_mode_with_single_window_is_rejected() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL, json=intent_body(temporal_mode="timeseries")
    )
    assert response.status_code == 422


def test_empty_modalities_is_rejected() -> None:
    client, fake = make_client()
    response = client.post(BUILD_PLAN_URL, json=intent_body(modalities=[]))
    assert response.status_code == 422
    assert fake.calls == []


def test_invalid_modality_is_rejected() -> None:
    client, _ = make_client()
    response = client.post(BUILD_PLAN_URL, json=intent_body(modalities=["radar"]))
    assert response.status_code == 422


def test_duplicate_modalities_is_rejected() -> None:
    client, _ = make_client()
    response = client.post(
        BUILD_PLAN_URL,
        json=intent_body(modalities=["sentinel-2-optical", "sentinel-2-optical"]),
    )
    assert response.status_code == 422


def test_invalid_task_is_rejected() -> None:
    client, _ = make_client()
    response = client.post(BUILD_PLAN_URL, json=intent_body(task="segment"))
    assert response.status_code == 422


def test_malformed_request_missing_fields() -> None:
    client, _ = make_client()
    assert client.post(BUILD_PLAN_URL, json={}).status_code == 422


# --------------------------------------------------------------------------- #
# Geospatial Service integration
# --------------------------------------------------------------------------- #


def test_geospatial_service_is_called_with_location_query() -> None:
    client, fake = make_client()
    client.post(BUILD_PLAN_URL, json=intent_body(location_query="  Paris  "))

    assert len(fake.calls) == 1
    request = fake.calls[0]
    assert isinstance(request, ResolveRequest)
    assert request.place == "Paris"  # trimmed by the intent, forwarded as-is
    assert request.bbox is None


def test_resolved_bbox_becomes_plan_bbox() -> None:
    custom = BoundingBox(west=2.20, south=48.80, east=2.45, north=48.92)
    client, _ = make_client(FakeGeospatialService(bbox=custom))
    response = client.post(BUILD_PLAN_URL, json=intent_body(location_query="Paris"))

    assert response.json()["bbox"] == {
        "west": pytest.approx(2.20),
        "south": pytest.approx(48.80),
        "east": pytest.approx(2.45),
        "north": pytest.approx(48.92),
    }


def test_geospatial_not_found_propagates_as_404() -> None:
    client, _ = make_client(
        FakeGeospatialService(error=NotFoundError("No matching location was found."))
    )
    response = client.post(BUILD_PLAN_URL, json=intent_body())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_geospatial_upstream_failure_propagates_as_502() -> None:
    client, _ = make_client(
        FakeGeospatialService(error=UpstreamServiceError("Nominatim is unavailable."))
    )
    response = client.post(BUILD_PLAN_URL, json=intent_body())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_response_contains_only_plan_fields() -> None:
    client, _ = make_client()
    body = client.post(BUILD_PLAN_URL, json=intent_body()).json()
    assert set(body) == {"intent", "bbox"}
    assert "scenes" not in body
    assert "image_base64" not in body


# --------------------------------------------------------------------------- #
# Service-level
# --------------------------------------------------------------------------- #


def test_service_build_plan_returns_intent_and_bbox() -> None:
    import asyncio

    fake = FakeGeospatialService()
    service = QueryService(geospatial_service=fake)  # type: ignore[arg-type]
    intent = SatQueryIntent.model_validate(intent_body())

    plan = asyncio.run(service.build_plan(intent))

    assert plan.intent == intent
    assert plan.bbox == DEFAULT_BBOX
    assert fake.calls[0].place == "Chennai"
