"""Stage 1 of the scientific pipeline: the request gate.

Two entry points share one set of rules (``app/services/analysis/validation.py``):

* ``validate_analysis_request`` - the direct contract (operation, dataset, AOI,
  Date 1 / Date 2, parameters). Pure: no network, no raster, no catalog.
* ``validate_execution_analysis`` - the same rules applied to the live
  ``/query/analyze`` contract, called by ``AnalysisService`` before its first
  read.

A request that passes is ELIGIBLE to enter the pipeline. Nothing here asserts
that a scene exists, that its pixels are usable, or that a measurement will be
valid; those are later stages.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
import socket
from datetime import date
from typing import Any

import pytest
from app.api.routes.query import get_analysis_service
from app.core.config import get_settings
from app.core.errors import ImageryError
from app.main import create_app
from app.services.analysis import AnalysisRequest, AnalysisService
from app.services.analysis.validation import (
    OPERATIONS,
    AnalysisOperation,
    AnalysisRequestRejectedError,
    ValidatedAnalysisRequest,
    estimated_window_pixels,
    validate_analysis_request,
    validate_execution_analysis,
)
from app.services.geospatial.schemas import BoundingBox
from app.services.query.schemas import (
    ExecutedWindow,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    TimeRange,
)
from app.services.satellite.imagery import ImageryService, QuantitativeReadLimits
from app.services.satellite.rtc import RTC_COLLECTION
from app.services.satellite.schemas import Scene
from fastapi.testclient import TestClient
from pydantic import ValidationError
from rasterio.warp import transform_bounds

TODAY = date(2026, 9, 23)
#: The real reader's bounds (``imagery_hard_max_dimension`` /
#: ``imagery_max_window_pixels`` defaults): 2048 px across at 10 m ~ 20 km.
LIMITS = QuantitativeReadLimits(max_dimension=2048, max_window_pixels=50_000_000)
OPTICAL = "sentinel-2-l2a"

SMALL_BBOX = [80.25, 13.00, 80.30, 13.05]  # ~5.4 x 5.5 km, Chennai
LARGE_BBOX = [79.0, 12.0, 81.0, 14.0]  # ~217 x 221 km


def body(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": "ndvi",
        "aoi": {"bbox": list(SMALL_BBOX)},
        "time_window": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not _DROP}


def temporal_body(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": "temporal_ndwi",
        "aoi": {"bbox": list(SMALL_BBOX)},
        "date1": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        "date2": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not _DROP}


_DROP = object()


def validate(payload: object, *, limits: QuantitativeReadLimits | None = LIMITS):
    return validate_analysis_request(payload, limits=limits, today=TODAY)


def rejected(
    payload: object,
    code: str,
    field: str | None = None,
    *,
    limits: QuantitativeReadLimits | None = LIMITS,
) -> AnalysisRequestRejectedError:
    with pytest.raises(AnalysisRequestRejectedError) as info:
        validate(payload, limits=limits)
    assert info.value.code == code, info.value.message
    assert info.value.status_code == 422
    if field is not None:
        assert info.value.field == field
    return info.value


def polygon(ring: list[list[float]]) -> dict[str, Any]:
    return {"geometry": {"type": "Polygon", "coordinates": [ring]}}


# =========================================================================== #
# Analysis allowlist
# =========================================================================== #


@pytest.mark.parametrize(
    ("operation", "extra"),
    [
        ("ndvi", {}),
        ("ndwi", {}),
        ("ndbi", {}),
        ("sar_backscatter", {}),
    ],
)
def test_every_implemented_single_operation_is_accepted(
    operation: str, extra: dict[str, Any]
) -> None:
    result = validate(body(operation=operation, **extra))
    assert result.operation == AnalysisOperation(operation)
    assert not result.is_temporal


def test_temporal_ndwi_is_accepted() -> None:
    result = validate(temporal_body())
    assert result.operation is AnalysisOperation.TEMPORAL_NDWI
    assert result.is_temporal


def test_operation_names_are_case_and_whitespace_insensitive() -> None:
    assert validate(body(operation=" NDVI ")).operation is AnalysisOperation.NDVI


@pytest.mark.parametrize("operation", ["evi", "ndvi; drop", "", "shell", "rs_model_analysis"])
def test_unknown_operation_is_refused(operation: str) -> None:
    rejected(body(operation=operation), "analysis_unknown", "operation")


@pytest.mark.parametrize("operation", [42, ["ndvi"], {"name": "ndvi"}, True])
def test_non_string_operation_is_refused(operation: object) -> None:
    rejected(body(operation=operation), "analysis_unknown", "operation")


def test_missing_operation_is_refused() -> None:
    rejected(body(operation=_DROP), "analysis_missing", "operation")


@pytest.mark.parametrize("operation", ["temporal_ndvi", "temporal_ndbi"])
def test_recognised_but_unimplemented_operation_says_so(operation: str) -> None:
    """Distinguishes "fix your request" from "no engine yet"."""

    error = rejected(temporal_body(operation=operation), "analysis_not_implemented")
    assert "no engine" in error.message


def test_a_long_unknown_value_is_never_echoed_in_full() -> None:
    error = rejected(body(operation="x" * 5000), "analysis_unknown")
    assert "x" * 50 not in error.message


def test_the_allowlist_matches_the_engines_that_exist() -> None:
    """Pins the allowlist: adding an engine must be a deliberate edit here."""

    implemented = {op.value for op, spec in OPERATIONS.items() if spec.implemented}
    assert implemented == {"ndvi", "ndwi", "ndbi", "sar_backscatter", "temporal_ndwi"}


# =========================================================================== #
# Dataset / sensor compatibility
# =========================================================================== #


@pytest.mark.parametrize("operation", ["ndvi", "ndwi", "ndbi"])
def test_optical_index_defaults_to_and_accepts_the_optical_dataset(operation: str) -> None:
    assert validate(body(operation=operation)).dataset == OPTICAL
    result = validate(body(operation=operation, dataset=OPTICAL))
    assert result.dataset == OPTICAL
    assert result.modality == "sentinel-2-optical"


@pytest.mark.parametrize("operation", ["ndvi", "ndwi", "ndbi"])
def test_optical_index_on_a_sar_dataset_is_refused(operation: str) -> None:
    error = rejected(
        body(operation=operation, dataset=RTC_COLLECTION), "dataset_incompatible", "dataset"
    )
    assert "optical" in error.message


def test_sar_defaults_to_and_accepts_rtc() -> None:
    result = validate(body(operation="sar_backscatter"))
    assert result.dataset == RTC_COLLECTION
    assert result.modality == "sentinel-1-sar"
    assert validate(body(operation="sar_backscatter", dataset=RTC_COLLECTION)).dataset == (
        RTC_COLLECTION
    )


def test_sar_on_an_optical_dataset_is_refused() -> None:
    rejected(
        body(operation="sar_backscatter", dataset=OPTICAL), "dataset_incompatible", "dataset"
    )


def test_temporal_optical_on_sar_dataset_is_refused() -> None:
    rejected(temporal_body(dataset=RTC_COLLECTION), "dataset_incompatible", "dataset")


@pytest.mark.parametrize("dataset", ["landsat-c2-l2", "sentinel-1-grd", "", 7])
def test_unknown_dataset_is_refused(dataset: object) -> None:
    rejected(body(dataset=dataset), "dataset_unknown", "dataset")


# =========================================================================== #
# AOI
# =========================================================================== #


def test_valid_bbox_is_normalised_with_ground_extent() -> None:
    aoi = validate(body()).aoi
    assert aoi.bbox == BoundingBox(west=80.25, south=13.00, east=80.30, north=13.05)
    assert aoi.crs == "EPSG:4326"
    assert aoi.source == "bbox"
    assert 5_000 < aoi.width_m < 6_000
    assert 5_000 < aoi.height_m < 6_000


@pytest.mark.parametrize("aoi", [_DROP, None])
def test_missing_aoi_is_refused(aoi: object) -> None:
    rejected(body(aoi=aoi), "aoi_missing", "aoi")


def test_aoi_with_neither_bbox_nor_geometry_is_refused() -> None:
    rejected(body(aoi={}), "aoi_missing", "aoi")


def test_aoi_with_both_bbox_and_geometry_is_refused() -> None:
    aoi = {"bbox": list(SMALL_BBOX), **polygon([[80, 13], [81, 13], [81, 14], [80, 14], [80, 13]])}
    rejected(body(aoi=aoi), "aoi_invalid", "aoi")


@pytest.mark.parametrize(
    "bbox",
    [
        [80.25, 13.0, 80.3],  # too short
        [80.25, 13.0, 80.3, 13.05, 1.0],  # too long
        "80.25,13,80.3,13.05",  # not a list
        [80.25, "13", 80.3, 13.05],  # non-numeric
        [80.25, True, 80.3, 13.05],  # bool is not a coordinate
        [float("nan"), 13.0, 80.3, 13.05],
        [80.25, 13.0, float("inf"), 13.05],
        [-181.0, 13.0, 80.3, 13.05],  # longitude out of range
        [80.25, 13.0, 80.3, 90.5],  # latitude out of range
    ],
)
def test_malformed_bbox_is_refused(bbox: object) -> None:
    rejected(body(aoi={"bbox": bbox}), "aoi_invalid", "aoi.bbox")


@pytest.mark.parametrize(
    "bbox", [[80.25, 13.0, 80.25, 13.05], [80.25, 13.0, 80.3, 13.0]]
)
def test_zero_area_bbox_is_empty(bbox: list[float]) -> None:
    rejected(body(aoi={"bbox": bbox}), "aoi_empty", "aoi.bbox")


def test_swapped_bbox_is_refused_not_repaired() -> None:
    error = rejected(body(aoi={"bbox": [80.30, 13.0, 80.25, 13.05]}), "aoi_invalid")
    assert "not corrected" in error.message
    rejected(body(aoi={"bbox": [80.25, 13.05, 80.30, 13.0]}), "aoi_invalid")


@pytest.mark.parametrize("crs", ["EPSG:4326", "OGC:CRS84", " EPSG:4326 "])
def test_wgs84_crs_names_are_accepted(crs: str) -> None:
    assert validate(body(aoi={"bbox": list(SMALL_BBOX), "crs": crs})).aoi.crs == "EPSG:4326"


@pytest.mark.parametrize("crs", ["EPSG:32644", "EPSG:3857", "wgs84-ish", 4326])
def test_other_crs_is_refused_never_reprojected(crs: object) -> None:
    rejected(body(aoi={"bbox": list(SMALL_BBOX), "crs": crs}), "aoi_crs_unsupported", "aoi.crs")


def test_unknown_aoi_field_is_refused() -> None:
    rejected(body(aoi={"bbox": list(SMALL_BBOX), "buffer_m": 500}), "field_unknown")


def test_axis_aligned_rectangle_polygon_is_converted_losslessly() -> None:
    ring = [[80.25, 13.0], [80.30, 13.0], [80.30, 13.05], [80.25, 13.05], [80.25, 13.0]]
    aoi = validate(body(aoi=polygon(ring))).aoi
    assert aoi.source == "polygon"
    assert aoi.bbox == validate(body()).aoi.bbox


def test_rectangle_polygon_in_clockwise_order_is_accepted() -> None:
    ring = [[80.25, 13.0], [80.25, 13.05], [80.30, 13.05], [80.30, 13.0], [80.25, 13.0]]
    assert validate(body(aoi=polygon(ring))).aoi.source == "polygon"


def test_non_rectangular_polygon_is_unsupported_not_enveloped() -> None:
    triangle = [[80.25, 13.0], [80.30, 13.0], [80.27, 13.05], [80.25, 13.0]]
    error = rejected(body(aoi=polygon(triangle)), "aoi_geometry_unsupported")
    assert "envelope" in error.message


def test_self_intersecting_bow_tie_is_invalid() -> None:
    bow_tie = [[80.25, 13.0], [80.30, 13.05], [80.30, 13.0], [80.25, 13.05], [80.25, 13.0]]
    rejected(body(aoi=polygon(bow_tie)), "aoi_invalid")


def test_unclosed_ring_is_invalid() -> None:
    ring = [[80.25, 13.0], [80.30, 13.0], [80.30, 13.05], [80.25, 13.05]]
    rejected(body(aoi=polygon(ring)), "aoi_invalid")


def test_polygon_with_hole_is_unsupported() -> None:
    outer = [[80.2, 13.0], [80.3, 13.0], [80.3, 13.1], [80.2, 13.1], [80.2, 13.0]]
    hole = [[80.24, 13.04], [80.26, 13.04], [80.26, 13.06], [80.24, 13.06], [80.24, 13.04]]
    aoi = {"geometry": {"type": "Polygon", "coordinates": [outer, hole]}}
    rejected(body(aoi=aoi), "aoi_geometry_unsupported")


@pytest.mark.parametrize(
    "coordinates",
    [
        [],  # no rings
        [[]],  # empty ring
    ],
)
def test_empty_polygon_is_empty(coordinates: list[Any]) -> None:
    aoi = {"geometry": {"type": "Polygon", "coordinates": coordinates}}
    rejected(body(aoi=aoi), "aoi_empty")


def test_collinear_polygon_has_zero_area() -> None:
    ring = [[80.25, 13.0], [80.26, 13.0], [80.27, 13.0], [80.28, 13.0], [80.25, 13.0]]
    rejected(body(aoi=polygon(ring)), "aoi_empty")


def test_polygon_with_non_finite_coordinate_is_invalid() -> None:
    ring = [[80.25, 13.0], [float("nan"), 13.0], [80.3, 13.05], [80.25, 13.05], [80.25, 13.0]]
    rejected(body(aoi=polygon(ring)), "aoi_invalid")


@pytest.mark.parametrize("kind", ["Point", "LineString", "MultiPolygon", "GeometryCollection"])
def test_other_geojson_types_are_unsupported(kind: str) -> None:
    aoi = {"geometry": {"type": kind, "coordinates": [80.25, 13.0]}}
    rejected(body(aoi=aoi), "aoi_geometry_unsupported", "aoi.geometry.type")


@pytest.mark.parametrize("geometry", [{"type": "Circle"}, {"coordinates": []}, "POLYGON(...)"])
def test_non_geojson_geometry_is_invalid(geometry: object) -> None:
    rejected(body(aoi={"geometry": geometry}), "aoi_invalid")


def test_oversized_aoi_is_refused_with_its_extent() -> None:
    error = rejected(body(aoi={"bbox": list(LARGE_BBOX)}), "aoi_too_large", "aoi")
    assert "km" in error.message
    assert "never downsampled" in error.message


def test_oversized_temporal_and_sar_aoi_are_refused() -> None:
    rejected(temporal_body(aoi={"bbox": list(LARGE_BBOX)}), "aoi_too_large")
    rejected(body(operation="sar_backscatter", aoi={"bbox": list(LARGE_BBOX)}), "aoi_too_large")


def test_size_limit_belongs_to_the_reader() -> None:
    """With no declared limits there is no early size rule - it is the reader's."""

    assert validate(body(aoi={"bbox": list(LARGE_BBOX)}), limits=None).aoi.width_m > 200_000


def test_pixel_budget_is_enforced_as_well_as_the_side_length() -> None:
    tight = QuantitativeReadLimits(max_dimension=10_000, max_window_pixels=100_000)
    rejected(body(), "aoi_too_large", limits=tight)  # ~550 x 550 px > 100k px


@pytest.mark.parametrize(
    ("bbox", "epsg"),
    [
        # Chennai, Sentinel-2 tile zone 44N (central meridian 81E).
        ((80.10, 12.90, 80.30, 13.20), 32644),
        # On the central meridian.
        ((80.90, 20.00, 81.10, 20.15), 32644),
        # At the zone edge, where UTM distortion and convergence are largest.
        ((83.80, 30.00, 83.99, 30.15), 32644),
        # High latitude.
        ((10.90, 60.00, 11.20, 60.15), 32632),
        # Southern hemisphere.
        ((18.40, -34.00, 18.60, -33.85), 32734),
    ],
)
def test_early_size_estimate_never_exceeds_the_readers_projected_window(
    bbox: tuple[float, float, float, float], epsg: int
) -> None:
    """The gate may only move the reader's rule earlier, never make it stricter.

    The reader projects the box into the scene's UTM zone and takes the pixel
    envelope. This compares the gate's estimate against rasterio's own
    projection of the same box on a 10 m grid.
    """

    box = BoundingBox(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3])
    left, bottom, right, top = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox)
    columns, rows = estimated_window_pixels(box, 10.0)
    assert columns <= (right - left) / 10.0
    assert rows <= (top - bottom) / 10.0
    # And it is a close estimate, not a vacuous one.
    assert columns >= 0.95 * (right - left) / 10.0
    assert rows >= 0.95 * (top - bottom) / 10.0


# =========================================================================== #
# Dates
# =========================================================================== #


def test_valid_single_window_is_normalised_to_dates() -> None:
    (observation,) = validate(body()).observations
    assert observation.role == "single"
    assert observation.period == TimeRange(
        start_date=date(2025, 1, 1), end_date=date(2025, 1, 31)
    )
    assert observation.requested_scene_id is None


def test_single_day_window_is_accepted() -> None:
    window = {"start_date": "2025-01-04", "end_date": "2025-01-04"}
    assert validate(body(time_window=window)).observations[0].period.start_date == date(
        2025, 1, 4
    )


@pytest.mark.parametrize(
    "value",
    [
        "2025-13-01",  # no month 13
        "2025-02-30",  # no such day
        "01/02/2025",  # not ISO
        "2025-1-1",  # not zero-padded
        "2025-01-01T00:00:00Z",  # a timestamp, not a date
        "yesterday",
        20250101,
    ],
)
def test_invalid_date_is_refused(value: object) -> None:
    window = {"start_date": value, "end_date": "2025-01-31"}
    rejected(body(time_window=window), "date_invalid", "time_window.start_date")


def test_missing_time_window_is_refused() -> None:
    rejected(body(time_window=_DROP), "date_missing", "time_window")


def test_missing_end_date_is_refused() -> None:
    rejected(
        body(time_window={"start_date": "2025-01-01"}), "date_missing", "time_window.end_date"
    )


def test_window_ending_before_it_starts_is_refused() -> None:
    window = {"start_date": "2025-01-31", "end_date": "2025-01-01"}
    rejected(body(time_window=window), "date_order_invalid", "time_window.end_date")


def test_window_before_the_first_observation_is_out_of_range() -> None:
    window = {"start_date": "2010-01-01", "end_date": "2010-01-31"}
    rejected(body(time_window=window), "date_out_of_range", "time_window")


def test_window_in_the_future_is_out_of_range() -> None:
    window = {"start_date": "2027-01-01", "end_date": "2027-01-31"}
    rejected(body(time_window=window), "date_out_of_range", "time_window")


def test_client_scene_metadata_is_not_accepted() -> None:
    """The catalog is authoritative for a scene's date, cloud and footprint."""

    window = {"start_date": "2025-01-01", "end_date": "2025-01-31", "cloud_cover": 0.0}
    error = rejected(body(time_window=window), "field_unknown")
    assert "catalog" in error.message


# =========================================================================== #
# Temporal: Date 1 is EARLIER, Date 2 is LATER
# =========================================================================== #


def test_valid_temporal_request_orders_date1_before_date2() -> None:
    first, second = validate(temporal_body()).observations
    assert (first.role, second.role) == ("date1", "date2")
    assert first.period.end_date < second.period.start_date


@pytest.mark.parametrize("missing", ["date1", "date2"])
def test_missing_date_is_refused(missing: str) -> None:
    rejected(temporal_body(**{missing: _DROP}), "date_missing", missing)


def test_date1_after_date2_is_refused_not_swapped() -> None:
    error = rejected(
        temporal_body(
            date1={"start_date": "2025-01-01", "end_date": "2025-01-31"},
            date2={"start_date": "2024-01-01", "end_date": "2024-01-31"},
        ),
        "date_order_invalid",
        "date1",
    )
    assert "EARLIER" in error.message
    assert "not swapped" in error.message


def test_identical_dates_are_refused() -> None:
    same = {"start_date": "2025-01-04", "end_date": "2025-01-04"}
    rejected(temporal_body(date1=same, date2=dict(same)), "temporal_windows_overlap", "date2")


def test_overlapping_windows_are_refused() -> None:
    rejected(
        temporal_body(
            date1={"start_date": "2025-01-01", "end_date": "2025-01-20"},
            date2={"start_date": "2025-01-15", "end_date": "2025-01-31"},
        ),
        "temporal_windows_overlap",
    )


def test_adjacent_windows_are_accepted() -> None:
    first, second = validate(
        temporal_body(
            date1={"start_date": "2025-01-01", "end_date": "2025-01-15"},
            date2={"start_date": "2025-01-16", "end_date": "2025-01-31"},
        )
    ).observations
    assert first.period.end_date < second.period.start_date


def test_temporal_request_given_a_single_window_is_refused() -> None:
    rejected(
        temporal_body(time_window={"start_date": "2025-01-01", "end_date": "2025-01-31"}),
        "temporal_structure_invalid",
        "time_window",
    )


@pytest.mark.parametrize("extra", ["date1", "date2"])
def test_single_request_given_a_comparison_date_is_refused(extra: str) -> None:
    window = {"start_date": "2024-01-01", "end_date": "2024-01-31"}
    rejected(body(**{extra: window}), "temporal_structure_invalid", extra)


def test_date_errors_in_either_side_are_named() -> None:
    rejected(
        temporal_body(date2={"start_date": "2025-02-30", "end_date": "2025-03-01"}),
        "date_invalid",
        "date2.start_date",
    )


# =========================================================================== #
# Scenes
# =========================================================================== #


def test_named_scene_id_is_kept_as_a_request_not_as_metadata() -> None:
    window = {
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "scene_id": "S2B_44PMV_20250104_0_L2A",
    }
    (observation,) = validate(body(time_window=window)).observations
    assert observation.requested_scene_id == "S2B_44PMV_20250104_0_L2A"


@pytest.mark.parametrize(
    "scene_id", ["../../search", "a/b", "id?x=1", "..", "", "x" * 201, 123]
)
def test_malformed_scene_id_is_refused(scene_id: object) -> None:
    window = {"start_date": "2025-01-01", "end_date": "2025-01-31", "scene_id": scene_id}
    rejected(body(time_window=window), "scene_invalid", "time_window.scene_id")


def test_same_scene_on_both_sides_is_refused() -> None:
    scene = "S2B_44PMV_20250104_0_L2A"
    rejected(
        temporal_body(
            date1={"start_date": "2024-01-01", "end_date": "2024-01-31", "scene_id": scene},
            date2={"start_date": "2025-01-01", "end_date": "2025-01-31", "scene_id": scene},
        ),
        "scene_invalid",
        "date2.scene_id",
    )


# =========================================================================== #
# Parameters
# =========================================================================== #


def test_default_parameters() -> None:
    parameters = validate(body()).parameters
    assert parameters.resolution == "native"
    assert parameters.polarizations is None
    assert parameters.max_cloud_cover is None


@pytest.mark.parametrize("resolution", ["20m", 20, "coarse", "10m"])
def test_non_native_resolution_is_refused(resolution: object) -> None:
    rejected(
        body(parameters={"resolution": resolution}),
        "parameter_unsupported",
        "parameters.resolution",
    )


def test_native_resolution_is_accepted() -> None:
    assert validate(body(parameters={"resolution": "native"})).parameters.resolution == "native"


@pytest.mark.parametrize("resampling", ["bilinear", "nearest", "cubic"])
def test_resampling_cannot_be_chosen(resampling: str) -> None:
    rejected(
        body(parameters={"resampling": resampling}),
        "parameter_unsupported",
        "parameters.resampling",
    )


def test_sar_polarizations_default_to_both_and_normalise() -> None:
    assert validate(body(operation="sar_backscatter")).parameters.polarizations == ("vv", "vh")
    result = validate(
        body(operation="sar_backscatter", parameters={"polarizations": [" VH", "vv"]})
    )
    assert result.parameters.polarizations == ("vv", "vh")


@pytest.mark.parametrize(
    "polarizations",
    [["hh"], ["vv", "hv"], ["vv"], ["vv", "vv"], "vv", [1, 2]],
)
def test_unsupported_polarization_is_refused(polarizations: object) -> None:
    rejected(
        body(operation="sar_backscatter", parameters={"polarizations": polarizations}),
        "parameter_unsupported",
        "parameters.polarizations",
    )


def test_polarization_on_optical_analysis_is_not_applicable() -> None:
    rejected(
        body(parameters={"polarizations": ["vv", "vh"]}),
        "parameter_not_applicable",
        "parameters.polarizations",
    )


def test_cloud_cover_bound_is_validated() -> None:
    assert validate(body(parameters={"max_cloud_cover": 20})).parameters.max_cloud_cover == 20.0
    for bad in (-1, 100.5, float("nan"), "20", True):
        rejected(
            body(parameters={"max_cloud_cover": bad}),
            "parameter_unsupported",
            "parameters.max_cloud_cover",
        )


def test_cloud_cover_on_sar_is_not_applicable() -> None:
    rejected(
        body(operation="sar_backscatter", parameters={"max_cloud_cover": 20}),
        "parameter_not_applicable",
    )


@pytest.mark.parametrize("name", ["GDAL_HTTP_PROXY", "gdal_config", "nodata", "scale"])
def test_arbitrary_processing_parameters_are_refused(name: str) -> None:
    rejected(body(parameters={name: "x"}), "parameter_unsupported")


def test_parameters_must_be_an_object() -> None:
    rejected(body(parameters=["native"]), "parameter_unsupported", "parameters")


# =========================================================================== #
# The request as a whole, and the normalised output
# =========================================================================== #


@pytest.mark.parametrize("payload", [None, [], "ndvi", 1])
def test_body_must_be_an_object(payload: object) -> None:
    rejected(payload, "field_unknown", "body")


def test_unknown_top_level_field_is_refused() -> None:
    rejected(body(provider="gemini"), "field_unknown")


def test_rules_are_applied_in_a_fixed_order() -> None:
    """The same bad request always produces the same code: what, then where, then when."""

    payload = body(operation="evi", aoi={"bbox": [1, 2]}, time_window={"start_date": "x"})
    rejected(payload, "analysis_unknown")
    payload = body(aoi={"bbox": [1, 2]}, time_window={"start_date": "x"})
    rejected(payload, "aoi_invalid")


def test_validated_request_is_frozen_and_closed() -> None:
    result = validate(body())
    assert result.validation_stage == "request"
    with pytest.raises(ValidationError):
        result.operation = AnalysisOperation.NDWI  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ValidatedAnalysisRequest.model_validate({**result.model_dump(), "note": "x"})


def test_validated_request_carries_no_raw_input() -> None:
    dumped = validate(body(operation=" NDVI ")).model_dump(mode="json")
    assert set(dumped) == {
        "operation", "modality", "dataset", "aoi", "observations",
        "parameters", "validation_stage",
    }
    assert dumped["operation"] == "ndvi"


def test_input_is_not_mutated() -> None:
    payload = body(operation=" NDVI ")
    snapshot = repr(payload)
    validate(payload)
    assert repr(payload) == snapshot


def test_validation_performs_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("request validation opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    validate(body())
    validate(temporal_body())
    validate(body(operation="sar_backscatter"))


def test_validation_module_imports_no_io_library() -> None:
    source = pathlib.Path(__file__).parents[1] / "app/services/analysis/validation.py"
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"httpx", "rasterio", "numpy", "requests", "urllib"})


# =========================================================================== #
# The live contract: /query/analyze stops before any read
# =========================================================================== #


def make_scene(scene_id: str, collection: str | None) -> Scene:
    return Scene(
        id=scene_id,
        datetime="2025-01-04T05:00:00Z",
        bbox=None,
        geometry=None,
        cloud_cover=1.0,
        collection=collection,
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
    )


def make_execution(
    bbox: list[float],
    *,
    collection: str | None = OPTICAL,
    modality: str = "sentinel-2-optical",
) -> QueryExecutionResult:
    intent = SatQueryIntent.model_validate(
        {
            "location_query": "Chennai",
            "temporal_mode": "single",
            "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
            "modalities": [modality],
            "task": "visualize",
        }
    )
    window = ExecutedWindow(
        modality=modality,  # type: ignore[arg-type]
        label="single",
        time_range=TimeRange(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31)),
        scene_count=1,
        scenes=[make_scene("scene-a", collection)],
        selected_scene_id="scene-a",
    )
    west, south, east, north = bbox
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(
            intent=intent,
            bbox=BoundingBox(west=west, south=south, east=east, north=north),
        ),
        executed_modalities=[modality],  # type: ignore[list-item]
        skipped_modalities=[],
        windows=[window],
        catalog="https://earth-search.aws.element84.com/v1",
    )


class RecordingStac:
    """Stands in for the STAC item fetch inside the REAL ImageryService."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, scene_id: str, collection: str) -> dict[str, Any]:
        self.calls.append((scene_id, collection))
        # A handled failure, so a request that legitimately passes the gate
        # ends as a per-operation warning rather than a crash.
        raise ImageryError("catalog stub: no item")


class RecordingBandReader:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_: object, **__: object) -> Any:
        self.calls += 1
        raise ImageryError("raster stub: no read")


def real_reader() -> tuple[ImageryService, RecordingStac, RecordingBandReader]:
    stac, bands = RecordingStac(), RecordingBandReader()
    return ImageryService(stac_item_fetcher=stac, band_reader=bands), stac, bands


@pytest.mark.parametrize(
    "flags",
    [
        {"indices": ["ndvi"]},
        {"indices": ["ndbi", "ndwi"]},
        {"include_ndwi": True},
    ],
)
def test_oversized_analysis_reaches_neither_stac_nor_raster(flags: dict[str, Any]) -> None:
    imagery, stac, bands = real_reader()
    request = AnalysisRequest(execution=make_execution(LARGE_BBOX), **flags)
    with pytest.raises(AnalysisRequestRejectedError) as info:
        asyncio.run(AnalysisService(imagery_service=imagery).analyze(request))
    assert info.value.code == "aoi_too_large"
    assert info.value.field == "execution.plan.bbox"
    assert stac.calls == []
    assert bands.calls == 0


def test_oversized_sar_analysis_reaches_neither_stac_nor_raster() -> None:
    imagery, stac, bands = real_reader()
    execution = make_execution(
        LARGE_BBOX, collection=RTC_COLLECTION, modality="sentinel-1-sar"
    )
    request = AnalysisRequest(execution=execution, include_sar_backscatter=True)
    with pytest.raises(AnalysisRequestRejectedError):
        asyncio.run(AnalysisService(imagery_service=imagery).analyze(request))
    assert (stac.calls, bands.calls) == ([], 0)


def test_optical_window_holding_a_sar_scene_is_refused_before_any_read() -> None:
    imagery, stac, bands = real_reader()
    request = AnalysisRequest(
        execution=make_execution(SMALL_BBOX, collection=RTC_COLLECTION), indices=["ndvi"]
    )
    with pytest.raises(AnalysisRequestRejectedError) as info:
        asyncio.run(AnalysisService(imagery_service=imagery).analyze(request))
    assert info.value.code == "dataset_incompatible"
    assert (stac.calls, bands.calls) == ([], 0)


def test_window_from_an_unserved_collection_is_refused_before_any_read() -> None:
    imagery, stac, bands = real_reader()
    request = AnalysisRequest(
        execution=make_execution(SMALL_BBOX, collection="landsat-c2-l2"), indices=["ndvi"]
    )
    with pytest.raises(AnalysisRequestRejectedError) as info:
        asyncio.run(AnalysisService(imagery_service=imagery).analyze(request))
    assert info.value.code == "dataset_unknown"
    assert (stac.calls, bands.calls) == ([], 0)


def test_a_valid_small_request_passes_the_gate_and_reaches_the_reader() -> None:
    """Not over-strict: a normal area gets as far as the catalog lookup."""

    imagery, stac, _ = real_reader()
    request = AnalysisRequest(execution=make_execution(SMALL_BBOX), indices=["ndvi"])
    asyncio.run(AnalysisService(imagery_service=imagery).analyze(request))
    assert stac.calls  # the fake refuses, which the service records as a warning


def test_an_analysis_with_nothing_to_measure_is_not_size_checked() -> None:
    """The gate protects reads; a visualize-only interpretation reads nothing."""

    request = AnalysisRequest(execution=make_execution(LARGE_BBOX))
    gate = validate_execution_analysis(request, limits=LIMITS)
    assert gate.operations == ()
    assert gate.size_checked is False


def test_the_gate_uses_the_readers_own_limits() -> None:
    settings = get_settings()
    limits = ImageryService().quantitative_read_limits()
    assert limits.max_dimension == settings.imagery_hard_max_dimension
    assert limits.max_window_pixels == settings.imagery_max_window_pixels


def test_analyze_endpoint_returns_a_structured_422_without_reading() -> None:
    imagery, stac, bands = real_reader()
    app = create_app()
    app.dependency_overrides[get_analysis_service] = lambda: AnalysisService(
        imagery_service=imagery
    )
    client = TestClient(app, raise_server_exceptions=False)
    execution = make_execution(LARGE_BBOX)
    response = client.post(
        f"{get_settings().api_v1_prefix}/query/analyze",
        json={"execution": execution.model_dump(mode="json"), "indices": ["ndvi"]},
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "aoi_too_large"
    assert "execution.plan.bbox" in error["message"]
    assert "Traceback" not in response.text
    assert (stac.calls, bands.calls) == ([], 0)
