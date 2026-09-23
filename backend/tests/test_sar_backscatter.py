"""Quantitative Sentinel-1 RTC backscatter statistics.

Everything here runs on small deterministic synthetic arrays. No test contacts
the Planetary Computer, the STAC API, the signing endpoint or a raster: the
engine is pure, and the service is exercised through a fake imagery service.

The reference values in the docstrings come from the live scene
``S1A_IW_GRDH_1SDV_20250324T003151_20250324T003216_058439_073A5F_rtc`` and are
reproduced end to end in the live validation script; the assertions below are
independent of it and derive their expectations by hand.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
import pytest
from app.core.errors import InvalidInputError, NotFoundError
from app.services.analysis import AnalysisRequest, AnalysisService
from app.services.analysis.sar import (
    DECIBEL_UNIT,
    RTC_NODATA,
    compare_polarizations,
    compute_backscatter_measurements,
    compute_polarization_statistics,
    compute_sar_backscatter,
    to_decibels,
    valid_linear_power,
)
from app.services.satellite.raster import BandWindow
from app.services.satellite.schemas import (
    ANALYSIS_BAND_ASSETS,
    SAR_ANALYSIS_BAND_ASSETS,
    SUPPORTED_IMAGERY_ASSETS,
)
from rasterio.transform import from_origin

from tests.test_analysis import (  # reuse the established fixtures verbatim
    DEFAULT_BBOX,
    make_execution,
    make_scene,
    make_window,
    named,
)

RTC_COLLECTION = "sentinel-1-rtc"


def sar_band(
    values: list[float],
    *,
    nodata: float | None = RTC_NODATA,
    origin: tuple[float, float] = (399960.0, 1500000.0),
    crs: str = "EPSG:32644",
    resolution: float = 10.0,
) -> BandWindow:
    """A one-row float32 BandWindow shaped like a real RTC read.

    ``float32``/``nodata = -32768``/10 m are exactly what the RTC assets
    publish, so the arithmetic under test sees the dtype it will see live.
    """

    array = np.asarray(values, dtype="float32").reshape(1, -1)
    valid = np.isfinite(array)
    if nodata is not None:
        valid = valid & (array != np.float32(nodata))
    return BandWindow(
        values=array,
        valid=valid,
        width=array.shape[1],
        height=1,
        crs=crs,
        transform=from_origin(origin[0], origin[1], resolution, resolution),
        resolution=resolution,
        nodata=nodata,
        window={"col_off": 0, "row_off": 0, "width": array.shape[1], "height": 1},
        source_shape=[1, array.shape[1]],
    )


# =========================================================================== #
# A. The conversion: dB = 10 * log10(linear power)
# =========================================================================== #


def test_known_linear_power_converts_to_known_decibels() -> None:
    """The one conversion, on values whose decibels are exact by construction."""

    assert to_decibels(1.0) == pytest.approx(0.0)
    assert to_decibels(10.0) == pytest.approx(10.0)
    assert to_decibels(0.1) == pytest.approx(-10.0)
    assert to_decibels(0.01) == pytest.approx(-20.0)


def test_the_conversion_matches_the_measured_rtc_medians() -> None:
    """RTC medians convert to textbook coastal/urban gamma naught."""

    assert float(to_decibels(0.027164)) == pytest.approx(10 * math.log10(0.027164), abs=1e-12)
    assert float(to_decibels(0.008323)) == pytest.approx(10 * math.log10(0.008323), abs=1e-12)


def test_no_amplitude_squaring_is_applied() -> None:
    """The product is already power, so the conversion is log-only.

    An amplitude convention would report ``20*log10`` - exactly twice these
    decibels. Pinning the factor keeps a future 'completeness' amplitude path
    from being wired into the live one.
    """

    values = named(compute_backscatter_measurements(sar_band([0.01]), polarization="vv"))

    assert values["vv_mean_db"] == pytest.approx(-20.0)  # not -40.0


# =========================================================================== #
# B. THE AVERAGING RULE - mean in linear power, converted once
#
# The single most important correctness property here. Averaging per-pixel
# decibels yields the GEOMETRIC mean, which is biased low and is not mean
# backscatter. These tests are built so that the two conventions cannot agree.
# =========================================================================== #


def test_the_mean_is_computed_in_linear_power_then_converted() -> None:
    """10*log10(mean(x)) - never mean(10*log10(x)).

    Powers 0.001 / 0.1 / 10 have linear mean 3.367, i.e. +5.2724 dB. The
    per-pixel-decibel average of the same three samples is (-30 - 10 + 10)/3 =
    -10 dB. A 15 dB gap: no tolerance can confuse them.
    """

    values = named(
        compute_backscatter_measurements(
            sar_band([0.001, 0.1, 10.0]), polarization="vv"
        )
    )
    linear_mean = (0.001 + 0.1 + 10.0) / 3.0

    assert values["vv_mean_db"] == pytest.approx(10.0 * math.log10(linear_mean))
    assert values["vv_mean_db"] == pytest.approx(5.27239, abs=1e-4)
    assert values["vv_mean_db"] != pytest.approx(-10.0, abs=1.0)


def test_the_geometric_mean_convention_is_not_what_the_engine_reports() -> None:
    """Stated as its own assertion so the wrong answer is written down.

    On a 1/10/100 sample the linear-power mean is +15.7 dB while the mean of
    per-pixel decibels is exactly +10 dB.
    """

    band = sar_band([1.0, 10.0, 100.0])
    reported = named(compute_backscatter_measurements(band, polarization="vv"))
    geometric_db = float(np.mean(10.0 * np.log10(np.asarray([1.0, 10.0, 100.0]))))

    assert geometric_db == pytest.approx(10.0)
    assert reported["vv_mean_db"] == pytest.approx(
        10.0 * math.log10((1.0 + 10.0 + 100.0) / 3.0)
    )
    assert reported["vv_mean_db"] == pytest.approx(15.6820172407, abs=1e-9)


def test_min_and_max_convert_directly_because_log10_is_monotonic() -> None:
    """Ordering survives the conversion, so the extremes need no averaging."""

    values = named(
        compute_backscatter_measurements(
            sar_band([0.001, 0.1, 10.0]), polarization="vv"
        )
    )

    assert values["vv_min_db"] == pytest.approx(-30.0)
    assert values["vv_max_db"] == pytest.approx(10.0)


# =========================================================================== #
# C. Exclusions - nodata, non-finite, non-positive. Counted, never repaired.
# =========================================================================== #


def test_nodata_is_excluded_from_the_statistics() -> None:
    """-32768 is the RTC nodata and must not reach the mean."""

    values = named(
        compute_backscatter_measurements(
            sar_band([RTC_NODATA, 1.0, RTC_NODATA]), polarization="vv"
        )
    )

    assert values["vv_valid_pixel_count"] == 1
    assert values["vv_mean_db"] == pytest.approx(0.0)


def test_zero_and_negative_samples_are_excluded_not_clamped() -> None:
    """log10 is undefined there; a floor would fabricate a measurement."""

    band = sar_band([0.0, -0.5, 1.0, 10.0], nodata=None)
    values = named(compute_backscatter_measurements(band, polarization="vv"))
    statistics = compute_polarization_statistics(band, polarization="vv")

    assert values["vv_valid_pixel_count"] == 2
    assert values["vv_mean_db"] == pytest.approx(10.0 * math.log10((1.0 + 10.0) / 2.0))
    assert statistics.nonpositive_pixel_count == 2
    assert statistics.window_pixel_count == 4


def test_nan_and_infinity_are_excluded() -> None:
    band = sar_band([float("nan"), float("inf"), float("-inf"), 1.0], nodata=None)
    values = named(compute_backscatter_measurements(band, polarization="vv"))

    assert values["vv_valid_pixel_count"] == 1
    assert values["vv_mean_db"] == pytest.approx(0.0)


def test_already_decibel_data_is_not_converted_a_second_time() -> None:
    """The guard against a double conversion, stated as data rather than a flag.

    Land gamma naught in decibels is negative, so a band already in dB has no
    strictly positive sample. It yields zero valid pixels and NO statistics -
    never a second 10*log10 of a negative number, and never a clamped stand-in.
    """

    already_db = sar_band([-15.7, -20.8, -12.4, -18.8], nodata=None)
    statistics = compute_polarization_statistics(already_db, polarization="vv")
    values = named(statistics.measurements)

    assert values["vv_valid_pixel_count"] == 0
    assert "vv_mean_db" not in values
    assert statistics.nonpositive_pixel_count == 4

    result = compute_sar_backscatter(
        vv=already_db, vh=None, scene_id="s", window_label="single"
    )
    assert any("not power" in warning for warning in result.warnings)


def test_a_band_with_no_valid_pixel_reports_the_count_and_nothing_else() -> None:
    values = named(
        compute_backscatter_measurements(sar_band([RTC_NODATA]), polarization="vh")
    )

    assert values == {"vh_valid_pixel_count": 0.0}


def test_valid_linear_power_returns_values_and_the_excluded_count() -> None:
    values, nonpositive = valid_linear_power(
        sar_band([RTC_NODATA, 0.0, -1.0, 4.0], nodata=RTC_NODATA)
    )

    assert values.tolist() == [4.0]
    assert nonpositive == 2  # the nodata sample is not counted as non-positive


# =========================================================================== #
# D. Per-polarization statistics and the contract they are published under
# =========================================================================== #


def test_vv_statistics_carry_the_agreed_names_and_the_db_unit() -> None:
    measurements = compute_backscatter_measurements(
        sar_band([0.01, 1.0]), polarization="vv"
    )
    units = {m.name: m.unit for m in measurements}

    assert set(units) == {
        "vv_valid_pixel_count",
        "vv_mean_db",
        "vv_min_db",
        "vv_max_db",
    }
    assert units["vv_mean_db"] == units["vv_min_db"] == units["vv_max_db"] == "dB"
    assert DECIBEL_UNIT == "dB"
    assert units["vv_valid_pixel_count"] == "pixels"


def test_vh_statistics_are_named_separately_from_vv() -> None:
    """Metric identity is what makes a mis-attributed claim fail structurally."""

    vv = named(compute_backscatter_measurements(sar_band([1.0]), polarization="vv"))
    vh = named(compute_backscatter_measurements(sar_band([0.1]), polarization="vh"))

    assert vv["vv_mean_db"] == pytest.approx(0.0)
    assert vh["vh_mean_db"] == pytest.approx(-10.0)
    assert set(vv) & set(vh) == set()


def test_no_measurement_is_reported_in_index_units() -> None:
    """"index" belongs to the normalised-difference engines and means something else."""

    result = compute_sar_backscatter(
        vv=sar_band([1.0]), vh=sar_band([0.1]), scene_id="s", window_label="single"
    )

    assert {m.unit for m in result.measurements} == {"dB", "pixels"}


def test_polarization_statistics_carry_the_read_grid_as_evidence() -> None:
    statistics = compute_polarization_statistics(sar_band([1.0, 2.0]), polarization="vv")

    assert statistics.crs == "EPSG:32644"
    assert statistics.resolution == 10.0
    assert statistics.transform is not None and len(statistics.transform) == 6
    assert statistics.window_pixel_count == 2


# =========================================================================== #
# E. VV/VH comparison - same scene, same grid, paired pixels
# =========================================================================== #


def test_vv_minus_vh_is_the_difference_of_the_two_linear_means_in_db() -> None:
    """Both means over the SAME pixels, each converted once."""

    vv = sar_band([1.0, 10.0])
    vh = sar_band([0.1, 1.0])
    difference, warnings = compare_polarizations(vv, vh)

    assert warnings == []
    assert difference is not None
    assert difference.vv_mean_db == pytest.approx(10.0 * math.log10(5.5))
    assert difference.vh_mean_db == pytest.approx(10.0 * math.log10(0.55))
    assert difference.vv_minus_vh_mean_db == pytest.approx(10.0)
    assert difference.paired_valid_pixel_count == 2


def test_the_difference_is_published_as_one_measurement_in_db() -> None:
    result = compute_sar_backscatter(
        vv=sar_band([1.0, 10.0]),
        vh=sar_band([0.1, 1.0]),
        scene_id="scene-s1",
        window_label="single",
    )
    values = named(result.measurements)
    units = {m.name: m.unit for m in result.measurements}

    assert values["vv_minus_vh_mean_db"] == pytest.approx(10.0)
    assert units["vv_minus_vh_mean_db"] == "dB"
    assert result.difference is not None


def test_the_difference_uses_only_pixels_valid_in_both_polarizations() -> None:
    """A pixel measured once says nothing about the ratio of the two."""

    vv = sar_band([1.0, 10.0])
    vh = sar_band([0.1, RTC_NODATA])
    difference, _ = compare_polarizations(vv, vh)

    assert difference is not None
    assert difference.paired_valid_pixel_count == 1
    assert difference.vv_mean_db == pytest.approx(0.0)
    assert difference.vh_mean_db == pytest.approx(-10.0)
    assert difference.vv_minus_vh_mean_db == pytest.approx(10.0)


def test_a_different_grid_refuses_the_comparison() -> None:
    """Grid identity, checked exactly as the temporal path checks it."""

    vv = sar_band([1.0, 10.0])
    shifted = sar_band([0.1, 1.0], origin=(399970.0, 1500000.0))
    difference, warnings = compare_polarizations(vv, shifted)

    assert difference is None
    assert warnings and "different pixel grids" in warnings[0]


def test_a_different_crs_refuses_the_comparison() -> None:
    difference, warnings = compare_polarizations(
        sar_band([1.0]), sar_band([1.0], crs="EPSG:32643")
    )

    assert difference is None
    assert warnings and "different CRS" in warnings[0]


def test_different_dimensions_refuse_the_comparison() -> None:
    difference, warnings = compare_polarizations(sar_band([1.0, 2.0]), sar_band([1.0]))

    assert difference is None
    assert warnings and "different pixel dimensions" in warnings[0]


def test_no_co_valid_pixel_refuses_the_comparison() -> None:
    difference, warnings = compare_polarizations(
        sar_band([1.0, RTC_NODATA]), sar_band([RTC_NODATA, 1.0])
    )

    assert difference is None
    assert warnings and "both polarizations" in warnings[0]


def test_a_refused_comparison_still_reports_each_polarization() -> None:
    result = compute_sar_backscatter(
        vv=sar_band([1.0]),
        vh=sar_band([0.1], crs="EPSG:32643"),
        scene_id="s",
        window_label="single",
    )
    values = named(result.measurements)

    assert result.difference is None
    assert values["vv_mean_db"] == pytest.approx(0.0)
    assert values["vh_mean_db"] == pytest.approx(-10.0)
    assert "vv_minus_vh_mean_db" not in values


def test_a_single_polarization_reports_why_there_is_no_difference() -> None:
    result = compute_sar_backscatter(
        vv=sar_band([1.0]), vh=None, scene_id="s", window_label="single"
    )

    assert result.difference is None
    assert [p.polarization for p in result.polarizations] == ["vv"]
    assert any("VH was not read" in warning for warning in result.warnings)


# =========================================================================== #
# F. Honesty - what the numbers are, and what the product does not provide
# =========================================================================== #


def test_the_result_states_the_averaging_convention() -> None:
    result = compute_sar_backscatter(
        vv=sar_band([1.0]), vh=sar_band([0.1]), scene_id="s", window_label="single"
    )

    assert any("geometric mean" in warning for warning in result.warnings)


def test_the_absence_of_a_quality_mask_is_stated_plainly() -> None:
    result = compute_sar_backscatter(
        vv=sar_band([1.0]), vh=sar_band([0.1]), scene_id="s", window_label="single"
    )
    text = " ".join(result.warnings)

    assert "no quality score is reported" in text
    assert "cloud" in text.lower()  # said to NOT apply, never applied


def test_no_classification_or_threshold_is_claimed() -> None:
    result = compute_sar_backscatter(
        vv=sar_band([1.0]), vh=sar_band([0.1]), scene_id="s", window_label="single"
    )
    text = " ".join(result.warnings).lower()

    assert "not a land-cover" in text
    assert "no threshold was applied" in text


def test_the_provider_is_credited_for_the_terrain_correction() -> None:
    result = compute_sar_backscatter(
        vv=sar_band([1.0]), vh=None, scene_id="s", window_label="single"
    )

    assert any("terrain-corrected gamma naught" in w for w in result.warnings)


# =========================================================================== #
# G. The quantitative SAR read path
# =========================================================================== #


def test_the_sar_analysis_allowlist_is_separate_from_the_optical_one() -> None:
    assert SAR_ANALYSIS_BAND_ASSETS == ("vv", "vh")
    assert set(SAR_ANALYSIS_BAND_ASSETS) & set(ANALYSIS_BAND_ASSETS) == set()
    # The display whitelist is untouched and still carries its own entries.
    assert "visual" in SUPPORTED_IMAGERY_ASSETS


def _imagery_service(**kwargs: Any):
    from app.services.satellite.imagery import ImageryService

    return ImageryService(**kwargs)


def test_read_band_refuses_a_polarization_outside_the_rtc_collection() -> None:
    """A polarization is gamma naught only in the collection that publishes it."""

    service = _imagery_service(
        stac_item_fetcher=lambda scene_id, collection: {"assets": {}},
        band_reader=lambda *a, **k: pytest.fail("must not read"),
    )

    with pytest.raises(InvalidInputError) as excinfo:
        service.read_band(
            scene_id="scene-x",
            bbox=DEFAULT_BBOX,
            asset="vv",
            collection="sentinel-2-l2a",
        )

    assert "sentinel-1-rtc" in str(excinfo.value)


def test_read_band_still_refuses_an_unknown_asset() -> None:
    service = _imagery_service(
        stac_item_fetcher=lambda scene_id, collection: {"assets": {}},
        band_reader=lambda *a, **k: pytest.fail("must not read"),
    )

    with pytest.raises(InvalidInputError):
        # "scl" was the example here until pixel quality control made it a
        # deliberate quality asset; "aot" is still outside every allowlist.
        service.read_band(scene_id="s", bbox=DEFAULT_BBOX, asset="aot")


def test_read_band_signs_the_rtc_asset_and_never_returns_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTC blobs refuse anonymous access, so the read href must be signed."""

    import app.services.satellite.imagery as imagery_mod

    unsigned = (
        "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/"
        "GRD/2025/3/24/IW/DV/S1A_x/measurement/iw-vv.rtc.tiff"
    )
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        imagery_mod, "sign_rtc_asset", lambda href, **kw: f"{href}?se=TOKEN"
    )

    def fake_band_reader(href: str, bbox: Any, **kwargs: Any) -> BandWindow:
        seen["href"] = href
        seen["kwargs"] = kwargs
        return sar_band([1.0])

    service = _imagery_service(
        stac_item_fetcher=lambda scene_id, collection: {
            "assets": {
                "vv": {
                    "href": unsigned,
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                }
            }
        },
        band_reader=fake_band_reader,
    )
    band = service.read_band(
        scene_id="scene-s1", bbox=DEFAULT_BBOX, asset="vv", collection=RTC_COLLECTION
    )

    assert seen["href"] == f"{unsigned}?se=TOKEN"
    assert band.values.tolist() == [[1.0]]


def test_the_quantitative_sar_read_keeps_the_existing_size_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oversized-AOI bounds are passed through unchanged - never relaxed."""

    import app.services.satellite.imagery as imagery_mod
    from app.core.config import get_settings

    monkeypatch.setattr(imagery_mod, "sign_rtc_asset", lambda href, **kw: href)
    seen: dict[str, Any] = {}

    def fake_band_reader(href: str, bbox: Any, **kwargs: Any) -> BandWindow:
        seen.update(kwargs)
        return sar_band([1.0])

    service = _imagery_service(
        stac_item_fetcher=lambda scene_id, collection: {
            "assets": {
                "vh": {
                    "href": "https://example.invalid/iw-vh.rtc.tiff",
                    "type": "image/tiff; application=geotiff",
                }
            }
        },
        band_reader=fake_band_reader,
    )
    service.read_band(
        scene_id="s", bbox=DEFAULT_BBOX, asset="vh", collection=RTC_COLLECTION
    )
    settings = get_settings()

    assert seen["max_dimension"] == settings.imagery_hard_max_dimension
    assert seen["max_window_pixels"] == settings.imagery_max_window_pixels


# =========================================================================== #
# H. AnalysisService dispatch - the service orchestrates, the engine computes
# =========================================================================== #


class FakeSarImagery:
    """Records read_band calls; returns canned polarizations or raises."""

    def __init__(
        self,
        *,
        bands: dict[str, BandWindow] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._bands = bands or {
            "vv": sar_band([1.0, 10.0]),
            "vh": sar_band([0.1, 1.0]),
        }
        self._errors = errors or {}

    def read_band(
        self,
        *,
        scene_id: str,
        bbox: Any,
        asset: str,
        collection: str | None = None,
    ) -> BandWindow:
        self.calls.append(
            {"scene_id": scene_id, "bbox": bbox, "asset": asset, "collection": collection}
        )
        if asset in self._errors:
            raise self._errors[asset]
        return self._bands[asset]


def sar_execution(**overrides: Any):
    window = make_window(
        modality="sentinel-1-sar",
        label="single",
        selected_scene_id="scene-s1",
        scenes_override=[make_scene("scene-s1", collection=RTC_COLLECTION)],
    )
    return make_execution(windows=[window], **overrides)


def analyze_sar(execution: Any, imagery: FakeSarImagery | None = None):
    imagery = imagery or FakeSarImagery()
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(execution=execution, include_sar_backscatter=True)
        )
    )
    return result, imagery


def test_backscatter_is_not_computed_unless_requested() -> None:
    imagery = FakeSarImagery()
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(AnalysisRequest(execution=sar_execution()))
    )

    assert imagery.calls == []
    assert result.sar_backscatter is None
    assert result.measurements == []


def test_the_flag_reads_both_polarizations_of_the_selected_scene() -> None:
    result, imagery = analyze_sar(sar_execution())

    assert [c["asset"] for c in imagery.calls] == ["vv", "vh"]
    assert {c["collection"] for c in imagery.calls} == {RTC_COLLECTION}
    assert {c["scene_id"] for c in imagery.calls} == {"scene-s1"}
    assert result.sar_backscatter is not None
    assert result.sar_backscatter.scene_id == "scene-s1"
    assert result.sar_backscatter.collection == RTC_COLLECTION


def test_the_measurements_reach_the_top_level_result() -> None:
    result, _ = analyze_sar(sar_execution())
    values = named(result.measurements)

    assert values["vv_mean_db"] == pytest.approx(10.0 * math.log10(5.5))
    assert values["vh_mean_db"] == pytest.approx(10.0 * math.log10(0.55))
    assert values["vv_minus_vh_mean_db"] == pytest.approx(10.0)
    assert values["vv_valid_pixel_count"] == 2
    assert values["vh_valid_pixel_count"] == 2


def test_an_optical_only_execution_reports_that_nothing_was_measured() -> None:
    execution = make_execution(
        windows=[make_window(scenes_override=[make_scene("scene-a")])]
    )
    result, imagery = analyze_sar(execution)

    assert imagery.calls == []
    assert result.sar_backscatter is None
    assert any("no SAR window" in warning for warning in result.warnings)


def test_a_failed_polarization_read_degrades_to_a_warning() -> None:
    imagery = FakeSarImagery(errors={"vh": NotFoundError("Asset 'vh' is missing.")})
    result, _ = analyze_sar(sar_execution(), imagery)

    assert result.sar_backscatter is not None
    assert named(result.measurements)["vv_mean_db"] == pytest.approx(
        10.0 * math.log10(5.5)
    )
    assert any("vh polarization could not be read" in w for w in result.warnings)


def test_both_reads_failing_produces_no_result_and_no_claim() -> None:
    imagery = FakeSarImagery(
        errors={
            "vv": NotFoundError("Asset 'vv' is missing."),
            "vh": NotFoundError("Asset 'vh' is missing."),
        }
    )
    result, _ = analyze_sar(sar_execution(), imagery)

    assert result.sar_backscatter is None
    assert result.measurements == []
    assert len([w for w in result.warnings if "could not be read" in w]) == 2


def test_backscatter_does_not_disturb_the_optical_measurements() -> None:
    """SAR is additive: an optical request keeps its exact previous behaviour."""

    from tests.test_analysis import FakeImageryService, ndwi_execution

    optical = FakeImageryService()
    service = AnalysisService(imagery_service=optical)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(execution=ndwi_execution(), include_ndwi=True)
        )
    )

    assert result.sar_backscatter is None
    assert named(result.measurements)["ndwi_valid_pixel_count"] == 3


def test_the_analysis_service_still_performs_no_pixel_arithmetic() -> None:
    import pathlib

    from app.services.analysis import service as service_mod

    source = pathlib.Path(service_mod.__file__).read_text()

    assert "import numpy" not in source
    assert "np." not in source
    assert "compute_sar_backscatter" in source


@pytest.mark.parametrize("encoding", [
    {"scale": 0.01}, {"offset": 1}, {"unit": "dB"},
])
def test_sar_rejects_stac_encoding_before_raster_read(encoding) -> None:
    service = _imagery_service(
        stac_item_fetcher=lambda *args: {"assets": {"vv": {
            "href": "https://example.invalid/vv.tif", "type": "image/tiff; application=geotiff",
            "raster:bands": [encoding],
        }}},
        band_reader=lambda *args, **kwargs: pytest.fail("must not read encoded SAR"),
    )
    with pytest.raises(InvalidInputError, match="unscaled linear"):
        service.read_band(scene_id="s", bbox=DEFAULT_BBOX, asset="vv", collection=RTC_COLLECTION)


@pytest.mark.parametrize("encoding", [
    {"source_scale": 0.1}, {"source_offset": 1}, {"source_unit": "dB"},
])
def test_sar_rejects_cog_encoding_even_when_stac_has_no_unit(monkeypatch, encoding) -> None:
    from dataclasses import replace

    import app.services.satellite.imagery as imagery_mod
    monkeypatch.setattr(imagery_mod, "sign_rtc_asset", lambda href, **kw: href)
    band = replace(sar_band([1.0]), **encoding)
    service = _imagery_service(
        stac_item_fetcher=lambda *args: {"assets": {"vv": {
            "href": "https://example.invalid/vv.tif", "type": "image/tiff; application=geotiff",
        }}}, band_reader=lambda *args, **kwargs: band,
    )
    with pytest.raises(InvalidInputError, match="unscaled linear"):
        service.read_band(scene_id="s", bbox=DEFAULT_BBOX, asset="vv", collection=RTC_COLLECTION)


def test_a_missing_rtc_asset_remains_a_typed_failure() -> None:
    service = _imagery_service(stac_item_fetcher=lambda *args: {"assets": {}})
    with pytest.raises(NotFoundError):
        service.read_band(scene_id="s", bbox=DEFAULT_BBOX, asset="vh", collection=RTC_COLLECTION)


def test_zero_valid_pixels_do_not_claim_decibel_statistics_were_computed() -> None:
    result, _ = analyze_sar(sar_execution(), FakeSarImagery(bands={
        "vv": sar_band([0, RTC_NODATA]), "vh": sar_band([0, RTC_NODATA]),
    }))
    assert result.sar_backscatter.difference is None
    assert not any(m.unit == "dB" for m in result.measurements)
    assert "no backscatter statistics in decibels were computed" in result.answer
