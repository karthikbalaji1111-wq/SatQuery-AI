"""Stage 3 of the scientific pipeline: pixel quality control.

SCENE VALID IS NOT PIXEL VALID. The Sentinel-2 Scene Classification Layer is
placed on the final analysis grid by whole-cell assignment (never blended), each
pixel is counted into exactly one quality category, and the band pair is masked
BEFORE any statistic. These tests pin the classification, the accounting, the
mask's reach into every statistic, the categorical grid handling, the temporal
pairing rule, the evidence, and the unknown/failure semantics.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import numpy as np
import pytest
from app.core.errors import ImageryError, NotFoundError
from app.services.agent.executor import _analysis_items
from app.services.analysis import AnalysisRequest, AnalysisService
from app.services.analysis.engines import compute_index_measurements
from app.services.analysis.indices import resolve_index
from app.services.analysis.pixel_quality import (
    QUALITY_PRECEDENCE,
    SCL_CATEGORY,
    SCL_SOURCE,
    USABLE_SCL_CLASSES,
    align_scl_to_grid,
    mask_with_scl,
    quality_measurements,
)
from app.services.analysis.schemas import Measurement, PixelQuality
from app.services.geospatial.schemas import BoundingBox
from app.services.satellite.imagery import ImageryService
from app.services.satellite.raster import BandWindow
from app.services.satellite.rtc import RTC_COLLECTION
from pydantic import ValidationError
from rasterio.transform import from_origin

from tests import test_temporal_ndwi as temporal
from tests.test_analysis import FakeImageryService, make_execution, make_intent
from tests.test_scene_validation import (
    SCENE_ID,
    Catalog,
    client_scene,
    execution,
    s2_item,
    window,
)

ORIGIN = (399960.0, 1500000.0)
AOI_BOX = BoundingBox(west=80.25, south=13.00, east=80.30, north=13.05)


def grid(
    rows: list[list[float]],
    *,
    res: float = 10.0,
    dtype: str = "uint16",
    nodata: float | None = 0.0,
    origin: tuple[float, float] = ORIGIN,
    crs: str = "EPSG:32644",
) -> BandWindow:
    values = np.asarray(rows, dtype=dtype)
    valid = np.isfinite(values) if np.issubdtype(values.dtype, np.floating) else (
        np.ones(values.shape, dtype=bool)
    )
    if nodata is not None:
        valid &= values != nodata
    height, width = values.shape
    return BandWindow(
        values=values,
        valid=valid,
        width=width,
        height=height,
        crs=crs,
        transform=from_origin(origin[0], origin[1], res, res),
        resolution=res,
        nodata=nodata,
        window={"col_off": 0, "row_off": 0, "width": width, "height": height},
        source_shape=[height, width],
    )


def scl(rows: list[list[int]], *, res: float = 10.0, **kwargs: Any) -> BandWindow:
    return grid(rows, res=res, dtype="uint8", nodata=0.0, **kwargs)


def masked(
    high: BandWindow, low: BandWindow, layer: BandWindow, *, index: str = "ndvi",
    status: str = "known",
):
    return mask_with_scl(
        high, low, layer, index=index, label=index.upper(), scene_id="scene-a",
        window_label="single", scl_metadata_status=status,
    )


def ones(n: int, value: float = 100.0) -> BandWindow:
    return grid([[value] * n])


def named(measurements: list[Measurement]) -> dict[str, float]:
    return {m.name: m.value for m in measurements}


# =========================================================================== #
# SCL classification
# =========================================================================== #


def test_the_class_table_is_the_documented_one_and_cites_its_source() -> None:
    assert sorted(int(v) for v in SCL_CATEGORY) == list(range(12))
    assert USABLE_SCL_CLASSES == (4, 5, 6)
    assert "sentiwiki.copernicus.eu" in SCL_SOURCE


@pytest.mark.parametrize(
    ("value", "category"),
    [
        (0, "nodata"),
        (1, "saturated_or_defective"),
        (2, "other_masked"),  # cast shadows (dark features before PB 05.11)
        (3, "cloud_shadow"),
        (4, None),
        (5, None),
        (6, None),
        (7, "other_masked"),  # unclassified
        (8, "cloud"),
        (9, "cloud"),
        (10, "cloud"),  # thin cirrus
        (11, "snow"),
    ],
)
def test_each_documented_class_lands_in_its_category(value: int, category: str | None) -> None:
    result = masked(ones(1), ones(1, 50.0), scl([[value]]))
    q = result.quality
    if category is None:
        assert q.valid_pixels == 1
        assert q.masked_pixels == 0
    else:
        assert q.valid_pixels == 0
        assert getattr(q, f"{category}_pixels") == 1


@pytest.mark.parametrize("value", [12, 99, 255])
def test_an_undocumented_class_is_excluded_and_reported(value: int) -> None:
    q = masked(ones(2), ones(2, 50.0), scl([[4, value]])).quality
    assert q.valid_pixels == 1
    assert q.unknown_class_pixels == 1
    assert q.unknown_scl_values == [value]
    assert any("outside the documented SCL classes" in n for n in q.quality_notes)


# =========================================================================== #
# Mask accounting
# =========================================================================== #


def test_all_valid() -> None:
    q = masked(ones(4), ones(4, 50.0), scl([[4, 5, 6, 4]])).quality
    assert (q.total_pixels, q.valid_pixels, q.masked_pixels) == (4, 4, 0)
    assert (q.valid_fraction, q.contamination_fraction) == (1.0, 0.0)
    assert q.quality_notes == []
    assert q.scl_class_counts == {"4": 2, "5": 1, "6": 1}


def test_all_cloud() -> None:
    q = masked(ones(3), ones(3, 50.0), scl([[8, 9, 10]])).quality
    assert (q.valid_pixels, q.cloud_pixels) == (0, 3)
    assert (q.valid_fraction, q.contamination_fraction) == (0.0, 1.0)
    assert any(n.startswith("No pixel was usable") for n in q.quality_notes)


def test_all_band_nodata() -> None:
    q = masked(grid([[0, 0, 0]]), ones(3, 50.0), scl([[4, 4, 4]])).quality
    assert (q.valid_pixels, q.nodata_pixels) == (0, 3)


def test_mixed_contamination_is_counted_class_by_class() -> None:
    classes = [[4, 8, 9, 10, 3, 11, 1, 2, 7, 0, 5, 6, 12, 4]]
    q = masked(ones(14), ones(14, 50.0), scl(classes)).quality
    assert q.total_pixels == 14
    assert q.valid_pixels == 4  # 4, 5, 6, 4
    assert q.cloud_pixels == 3
    assert q.cloud_shadow_pixels == 1
    assert q.snow_pixels == 1
    assert q.saturated_or_defective_pixels == 1
    assert q.other_masked_pixels == 2  # cast shadow + unclassified
    assert q.nodata_pixels == 1
    assert q.unknown_class_pixels == 1
    assert q.masked_pixels == 10
    assert q.valid_fraction == pytest.approx(4 / 14)
    assert q.contamination_fraction == pytest.approx(10 / 14)


def test_overlapping_conditions_count_once_by_precedence() -> None:
    """Band nodata under a cloud is nodata - once - never nodata AND cloud."""

    high = grid([[0, 100, 100, 100]])  # pixel 0 has no band data
    q = masked(high, ones(4, 50.0), scl([[9, 9, 3, 4]])).quality
    assert q.nodata_pixels == 1
    assert q.cloud_pixels == 1  # pixel 1 only
    assert q.cloud_shadow_pixels == 1
    assert q.valid_pixels == 1
    categories = sum(getattr(q, f"{c}_pixels") for c in QUALITY_PRECEDENCE)
    assert categories + q.valid_pixels == q.total_pixels


def test_nodata_is_the_first_precedence() -> None:
    assert QUALITY_PRECEDENCE[0] == "nodata"
    assert len(set(QUALITY_PRECEDENCE)) == len(QUALITY_PRECEDENCE)


def test_the_record_refuses_inconsistent_counts() -> None:
    q = masked(ones(2), ones(2, 50.0), scl([[4, 9]])).quality
    fields = q.model_dump()
    with pytest.raises(ValidationError):
        PixelQuality.model_validate({**fields, "cloud_pixels": 0})
    with pytest.raises(ValidationError):
        PixelQuality.model_validate({**fields, "valid_pixels": 2})
    with pytest.raises(ValidationError):
        PixelQuality.model_validate({**fields, "grid_width": 5})


def test_valid_fraction_uses_the_whole_grid_as_denominator() -> None:
    """Nodata pixels are in the denominator: they are part of the area asked about."""

    high = grid([[0, 0, 100, 100]])
    q = masked(high, ones(4, 50.0), scl([[4, 4, 4, 9]])).quality
    assert q.valid_pixels == 1
    assert q.valid_fraction == pytest.approx(1 / 4)
    assert q.contamination_fraction == pytest.approx(3 / 4)


def test_masked_pixels_are_marked_invalid_not_zeroed() -> None:
    high = ones(3, 100.0)
    result = masked(high, ones(3, 50.0), scl([[4, 9, 4]]))
    assert result.high.valid.tolist() == [[True, False, True]]
    # The value is left as read; only validity changes.
    assert result.high.values.tolist() == high.values.tolist()
    assert result.high.transform == high.transform


# =========================================================================== #
# The mask reaches every statistic
# =========================================================================== #


def test_ndvi_statistics_use_only_usable_pixels() -> None:
    # nir/red: clear pixels give NDVI 0.5, the clouded one would give -0.8.
    nir = grid([[300, 300, 10]])
    red = grid([[100, 100, 90]])
    result = masked(nir, red, scl([[4, 4, 9]]))
    stats = named(compute_index_measurements(resolve_index("ndvi"), result.high, result.low))
    assert stats["ndvi_valid_pixel_count"] == 2
    assert stats["ndvi_mean"] == pytest.approx(0.5)
    assert stats["ndvi_min"] == pytest.approx(0.5)
    assert stats["ndvi_max"] == pytest.approx(0.5)


def test_zero_usable_pixels_yield_no_statistic_but_an_honest_count() -> None:
    result = masked(ones(2), ones(2, 50.0), scl([[9, 3]]))
    stats = named(compute_index_measurements(resolve_index("ndvi"), result.high, result.low))
    assert stats == {"ndvi_valid_pixel_count": 0.0}


def service_result(bands: dict[str, BandWindow], **request: Any):
    imagery = FakeImageryService(bands=bands)
    result = asyncio.run(
        AnalysisService(imagery_service=imagery).analyze(  # type: ignore[arg-type]
            AnalysisRequest(execution=make_execution(), **request)
        )
    )
    return result, imagery


def test_ndvi_through_the_service_is_masked() -> None:
    result, _ = service_result(
        {
            "nir": grid([[300, 300, 10]]),
            "red": grid([[100, 100, 90]]),
            "scl": scl([[4, 4, 9]]),
        },
        indices=["ndvi"],
    )
    stats = named(result.measurements)
    assert stats["ndvi_mean"] == pytest.approx(0.5)
    assert stats["ndvi_valid_pixel_count"] == 2
    assert stats["ndvi_quality_cloud_pixel_count"] == 1
    (quality,) = result.pixel_quality
    assert (quality.index, quality.valid_pixels, quality.cloud_pixels) == ("ndvi", 2, 1)


def test_ndwi_statistics_overlay_and_threshold_share_the_masked_pixels() -> None:
    bands = {
        "green": grid([[300, 300, 10, 300]]),
        "nir": grid([[100, 100, 90, 100]]),
        "scl": scl([[4, 4, 8, 11]]),
    }
    imagery = FakeImageryService(bands=bands)
    exec_ = make_execution(
        intent=make_intent(ndwi_threshold={"operator": "gt", "value": 0.3})
    )
    result = asyncio.run(
        AnalysisService(imagery_service=imagery).analyze(  # type: ignore[arg-type]
            AnalysisRequest(execution=exec_, include_ndwi=True, include_ndwi_overlay=True)
        )
    )
    stats = named(result.measurements)
    assert stats["ndwi_valid_pixel_count"] == 2
    assert stats["ndwi_mean"] == pytest.approx(0.5)
    assert stats["ndwi_quality_snow_pixel_count"] == 1
    assert result.ndwi_overlay is not None
    assert result.ndwi_overlay.valid_pixel_count == 2
    # The threshold count's denominator is the masked pixels too.
    assert result.spatial_measurement is not None
    assert result.spatial_measurement.valid_pixel_count == 2
    assert result.pixel_quality[0].valid_pixels == 2


def test_ndbi_masks_on_the_final_ten_metre_grid() -> None:
    """SWIR and SCL are both 20 m; both reach the 10 m NIR grid by whole cells."""

    nir = grid([[100, 100, 100, 100], [100, 100, 100, 100]], res=10.0)
    swir = grid([[300, 20]], res=20.0)  # left cell NDBI 0.5, right cell would be -0.67
    layer = scl([[4, 9]], res=20.0)  # the right 20 m cell is cloud
    result, _ = service_result({"nir": nir, "swir16": swir, "scl": layer}, indices=["ndbi"])
    stats = named(result.measurements)
    assert stats["ndbi_valid_pixel_count"] == 4  # the four 10 m pixels under the clear cell
    assert stats["ndbi_mean"] == pytest.approx(0.5)
    (quality,) = result.pixel_quality
    assert (quality.grid_width, quality.grid_height, quality.grid_resolution) == (4, 2, 10.0)
    assert (quality.total_pixels, quality.cloud_pixels) == (8, 4)


def test_invalid_band_in_the_pair_excludes_the_pixel() -> None:
    result, _ = service_result(
        {"nir": grid([[300, 300, 300]]), "red": grid([[100, 0, 100]]), "scl": scl([[4, 4, 4]])},
        indices=["ndvi"],
    )
    stats = named(result.measurements)
    assert stats["ndvi_valid_pixel_count"] == 2
    assert stats["ndvi_quality_nodata_pixel_count"] == 1


def test_fully_masked_index_is_an_unavailable_outcome_not_a_clean_result() -> None:
    result, _ = service_result(
        {"nir": ones(2, 300.0), "red": ones(2), "scl": scl([[9, 9]])}, indices=["ndvi"]
    )
    (outcome,) = result.analysis_outcomes
    assert outcome.status == "unavailable"
    assert "ndvi_mean" not in named(result.measurements)
    assert any("No pixel was usable" in w for w in result.warnings)


def test_partially_masked_index_completes_and_says_what_was_excluded() -> None:
    result, _ = service_result(
        {"nir": ones(3, 300.0), "red": ones(3), "scl": scl([[4, 4, 3]])}, indices=["ndvi"]
    )
    (outcome,) = result.analysis_outcomes
    assert outcome.status == "completed"
    assert any("1 of 3 pixels were excluded" in w for w in result.warnings)


# =========================================================================== #
# The SCL on the analysis grid
# =========================================================================== #


def test_twenty_metre_scl_lands_on_the_ten_metre_grid() -> None:
    fine = grid([[1, 1, 1, 1], [1, 1, 1, 1]], res=10.0)
    aligned = align_scl_to_grid(scl([[4, 8]], res=20.0), fine)
    assert aligned.values.shape == fine.values.shape
    assert aligned.transform == fine.transform
    assert aligned.values.tolist() == [[4, 4, 8, 8], [4, 4, 8, 8]]


def test_class_labels_are_never_blended() -> None:
    """Averaging vegetation (4) and cloud (8) would invent water (6)."""

    fine = grid([[1] * 6, [1] * 6], res=10.0)
    aligned = align_scl_to_grid(scl([[4, 8, 4]], res=20.0), fine)
    assert set(np.unique(aligned.values).tolist()) <= {4, 8}
    assert aligned.values.dtype == np.uint8
    q = masked(fine, fine, scl([[4, 8, 4]], res=20.0)).quality
    assert (q.valid_pixels, q.cloud_pixels) == (8, 4)
    assert "6" not in q.scl_class_counts


def test_float_valued_scl_is_refused() -> None:
    with pytest.raises(ImageryError, match="not integer-valued"):
        align_scl_to_grid(grid([[4.0, 8.0]], dtype="float32"), ones(2))


def test_scl_in_another_crs_is_refused_not_resampled() -> None:
    with pytest.raises(ImageryError, match="coordinate reference"):
        align_scl_to_grid(scl([[4]], crs="EPSG:32643"), ones(1))


def test_grid_pixels_without_an_scl_cell_are_no_data() -> None:
    # A 20 m SCL window one cell wide over a 10 m grid four pixels wide.
    fine = grid([[100, 100, 100, 100]], res=10.0)
    q = masked(fine, ones(4, 50.0), scl([[4]], res=20.0)).quality
    assert (q.valid_pixels, q.nodata_pixels) == (2, 2)
    assert any("no value in the scene classification layer" in n for n in q.quality_notes)


def test_scl_failure_to_align_leaves_the_index_uncomputed() -> None:
    result, _ = service_result(
        {"nir": ones(2, 300.0), "red": ones(2), "scl": scl([[4, 4]], crs="EPSG:32643")},
        indices=["ndvi"],
    )
    assert "ndvi_mean" not in named(result.measurements)
    assert any("could not be computed" in w for w in result.warnings)


# =========================================================================== #
# Temporal: each date masked by its own classification
# =========================================================================== #


def temporal_bands(first_scl: list[int], second_scl: list[int]) -> temporal.FakeImageryService:
    return temporal.FakeImageryService(
        bands={
            "scene-a": {
                "green": temporal.band([3, 3, 3]),
                "nir": temporal.band([1, 1, 1]),
                "scl": scl([first_scl]),
            },
            "scene-b": {
                "green": temporal.band([5, 5, 5]),
                "nir": temporal.band([5, 5, 5]),
                "scl": scl([second_scl]),
            },
        }
    )


def test_each_observation_carries_its_own_quality() -> None:
    result, _ = temporal.analyze_temporal(
        temporal.two_window_execution(), temporal_bands([4, 9, 4], [4, 4, 3])
    )
    comparison = result.temporal_comparison
    assert comparison is not None
    first, second = comparison.first.pixel_quality, comparison.second.pixel_quality
    assert first is not None and second is not None
    assert (first.valid_pixels, first.cloud_pixels, first.cloud_shadow_pixels) == (2, 1, 0)
    assert (second.valid_pixels, second.cloud_pixels, second.cloud_shadow_pixels) == (2, 0, 1)
    assert first.scene_id == "scene-a" and second.scene_id == "scene-b"


def test_paired_change_uses_only_pixels_usable_on_both_dates() -> None:
    result, _ = temporal.analyze_temporal(
        temporal.two_window_execution(), temporal_bands([4, 9, 4], [4, 4, 3])
    )
    comparison = result.temporal_comparison
    assert comparison is not None and comparison.change is not None
    assert comparison.change.paired_valid_pixel_count == 1  # only pixel 0
    assert comparison.change.change_mean == pytest.approx(0.0 - 0.5)
    assert any(
        "1 pixels are usable on both dates" in w for w in comparison.warnings
    )


def test_no_usable_pair_means_no_change_and_says_so() -> None:
    result, _ = temporal.analyze_temporal(
        temporal.two_window_execution(), temporal_bands([4, 9, 9], [9, 4, 9])
    )
    comparison = result.temporal_comparison
    assert comparison is not None
    assert comparison.change is None
    assert any("no paired NDWI change was computed" in w for w in comparison.warnings)


def test_different_quality_between_dates_is_reported_not_equalised() -> None:
    result, _ = temporal.analyze_temporal(
        temporal.two_window_execution(), temporal_bands([4, 4, 4], [9, 9, 4])
    )
    comparison = result.temporal_comparison
    assert comparison is not None
    first = named(comparison.first.measurements)
    second = named(comparison.second.measurements)
    assert first["ndwi_valid_pixel_count"] == 3
    assert second["ndwi_valid_pixel_count"] == 1
    assert second["ndwi_quality_cloud_pixel_count"] == 2
    assert first["ndwi_quality_valid_percent"] == pytest.approx(100.0)
    assert second["ndwi_quality_valid_percent"] == pytest.approx(100.0 / 3)


def test_the_aggregate_difference_uses_masked_means() -> None:
    bands = temporal_bands([4, 4, 4], [4, 4, 4])
    bands._bands["scene-b"]["green"] = temporal.band([5, 5, 50])  # type: ignore[index]
    bands._bands["scene-b"]["scl"] = scl([[4, 4, 9]])  # type: ignore[index]
    result, _ = temporal.analyze_temporal(temporal.two_window_execution(), bands)
    comparison = result.temporal_comparison
    assert comparison is not None
    diff = named(comparison.differences)["mean_ndwi_difference"]
    assert diff == pytest.approx(0.0 - 0.5)  # the clouded pixel's 0.82 never counted


# =========================================================================== #
# Evidence
# =========================================================================== #


def test_quality_counts_survive_into_citable_evidence_exactly() -> None:
    result, _ = service_result(
        {"nir": ones(4, 300.0), "red": ones(4), "scl": scl([[4, 8, 3, 4]])}, indices=["ndvi"]
    )
    items = {item.id: item for item in _analysis_items(result)}
    cloud = items["ndvi.ndvi_quality_cloud_pixel_count"]
    assert cloud.source == "ndvi" and cloud.measurement is not None
    assert cloud.measurement.value == 1.0
    assert items["ndvi.ndvi_quality_valid_percent"].measurement.value == 50.0  # type: ignore[union-attr]
    assert items["ndvi.ndvi_quality_total_pixel_count"].measurement.value == 4.0  # type: ignore[union-attr]


def test_temporal_quality_reaches_evidence_per_observation() -> None:
    result, _ = temporal.analyze_temporal(
        temporal.two_window_execution(), temporal_bands([4, 9, 4], [4, 4, 4])
    )
    ids = {item.id: item for item in _analysis_items(result)}
    assert ids["temporal_ndwi.first.ndwi_quality_cloud_pixel_count"].measurement.value == 1.0  # type: ignore[union-attr]
    assert ids["temporal_ndwi.second.ndwi_quality_cloud_pixel_count"].measurement.value == 0.0  # type: ignore[union-attr]


def test_no_percentage_is_fabricated_without_a_denominator() -> None:
    empty = PixelQuality(
        index="ndvi", scene_id="s", window_label="w", usable_scl_classes=[4, 5, 6],
        scl_metadata_status="known", grid_width=0, grid_height=0, total_pixels=0,
        valid_pixels=0, masked_pixels=0, nodata_pixels=0,
        saturated_or_defective_pixels=0, cloud_pixels=0, cloud_shadow_pixels=0,
        snow_pixels=0, unknown_class_pixels=0, other_masked_pixels=0,
    )
    names = {m.name for m in quality_measurements(empty)}
    assert "ndvi_quality_valid_percent" not in names
    assert "ndvi_quality_contamination_percent" not in names
    assert empty.valid_fraction is None


def test_quality_record_holds_no_raster() -> None:
    q = masked(ones(3), ones(3, 50.0), scl([[4, 9, 4]])).quality
    for value in q.model_dump().values():
        assert not isinstance(value, np.ndarray)
        if isinstance(value, list):
            assert len(value) <= 12


# =========================================================================== #
# Unknown / failure semantics
# =========================================================================== #


def test_scl_asset_missing_from_the_catalog_item_reads_no_band() -> None:
    item = s2_item()
    del item["assets"]["scl"]
    catalog = Catalog({SCENE_ID: item})
    result = asyncio.run(
        catalog.service().analyze(
            AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), indices=["ndvi"])
        )
    )
    assert catalog.reads == []
    (outcome,) = result.analysis_outcomes
    assert outcome.status == "unavailable"
    assert "scene classification layer (SCL) is unavailable" in (outcome.reason or "")


def test_scl_read_failure_reads_no_spectral_band() -> None:
    class SclFails(FakeImageryService):
        def read_band(self, **kwargs: Any) -> BandWindow:
            if kwargs["asset"] == "scl":
                self.calls.append(kwargs)
                raise NotFoundError("SCL not readable")
            return super().read_band(**kwargs)

    imagery = SclFails(bands={"nir": ones(2, 300.0), "red": ones(2)})
    result = asyncio.run(
        AnalysisService(imagery_service=imagery).analyze(  # type: ignore[arg-type]
            AnalysisRequest(execution=make_execution(), indices=["ndvi"])
        )
    )
    assert [c["asset"] for c in imagery.calls] == ["scl"]
    assert result.measurements == []
    assert any("(SCL) is unavailable" in w for w in result.warnings)


def test_ndwi_without_scl_is_not_a_clean_result() -> None:
    item = s2_item()
    del item["assets"]["scl"]
    catalog = Catalog({SCENE_ID: item})
    result = asyncio.run(
        catalog.service().analyze(
            AnalysisRequest(
                execution=execution([window(client_scene(SCENE_ID))]), include_ndwi=True
            )
        )
    )
    assert catalog.reads == []
    assert result.measurements == []
    assert result.analysis_outcomes[0].status == "unavailable"


def test_scl_metadata_confirmed_by_the_catalog_is_recorded_known() -> None:
    catalog = Catalog({SCENE_ID: s2_item()})
    result = asyncio.run(
        catalog.service().analyze(
            AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), indices=["ndvi"])
        )
    )
    assert result.pixel_quality[0].scl_metadata_status == "known"


def test_scl_metadata_absent_from_the_catalog_is_recorded_unknown() -> None:
    item = s2_item()
    del item["assets"]["scl"]["raster:bands"]
    catalog = Catalog({SCENE_ID: item})
    result = asyncio.run(
        catalog.service().analyze(
            AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), indices=["ndvi"])
        )
    )
    (quality,) = result.pixel_quality
    assert quality.scl_metadata_status == "unknown"
    assert any("no raster metadata for the SCL" in n for n in quality.quality_notes)


def test_scl_declared_with_the_wrong_data_type_is_unsupported() -> None:
    item = s2_item()
    item["assets"]["scl"]["raster:bands"][0]["data_type"] = "uint16"
    catalog = Catalog({SCENE_ID: item})
    result = asyncio.run(
        catalog.service().analyze(
            AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), indices=["ndvi"])
        )
    )
    assert catalog.reads == []
    assert "unsupported_asset_encoding" in (result.analysis_outcomes[0].reason or "")


def test_unvalidated_scl_is_recorded_as_such() -> None:
    result, _ = service_result({"nir": ones(1, 300.0), "red": ones(1)}, indices=["ndvi"])
    assert result.pixel_quality[0].scl_metadata_status == "not_validated"


# =========================================================================== #
# SAR stays SAR
# =========================================================================== #


def test_sar_reads_no_scene_classification() -> None:
    from tests.test_sar_backscatter import analyze_sar, sar_execution

    result, imagery = analyze_sar(sar_execution())
    assert "scl" not in [c["asset"] for c in imagery.calls]
    assert result.pixel_quality == []


def test_scl_is_readable_from_optical_and_refused_from_rtc() -> None:
    item = s2_item()
    reads: list[str] = []
    imagery = ImageryService(
        stac_item_fetcher=lambda *_: item,
        band_reader=lambda href, *a, **k: reads.append(href) or scl([[4]]),
    )
    imagery.read_band(scene_id=SCENE_ID, bbox=AOI_BOX, asset="scl")
    assert reads and reads[0].endswith("/SCL.tif")
    from app.core.errors import InvalidInputError

    with pytest.raises(InvalidInputError, match="quality layer"):
        imagery.read_band(
            scene_id=SCENE_ID, bbox=AOI_BOX, asset="scl",
            collection=RTC_COLLECTION,
        )


def test_the_masked_bands_are_views_of_one_mask_not_copies_of_the_data() -> None:
    """Memory: masking replaces the validity array only; values are shared."""

    high = ones(3, 100.0)
    result = masked(high, ones(3, 50.0), scl([[4, 9, 4]]))
    assert result.high.values is high.values
    assert dataclasses.is_dataclass(result.high)
