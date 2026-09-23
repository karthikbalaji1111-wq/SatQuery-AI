"""Stage 4 of the scientific pipeline: radiometric validation.

VALID SCENE -> VALID PIXELS -> VALID RADIOMETRIC REPRESENTATION -> ONLY THEN THE
INDEX. These tests pin the central Sentinel-2 offset rule (baseline x provider
flag x declared offset), the scale/unit/dtype handling, the absence of any
invented valid range or second saturation mask, the SAR linear-power check,
cross-date comparability, evidence, and - above all - that a refused
representation is refused BEFORE any raster is read, and never corrected.

Fixture items follow the live Earth Search / Planetary Computer items read on
2026-09-23 (see ``tests/test_scene_validation.py``). The offset cases mirror
real scenes of tile 44PMV:

    S2B_44PMV_20220120  baseline 03.01  flag false  declared offset 0
    S2B_44PMV_20220331  baseline 04.00  flag false  declared offset -0.1
    S2A_44PMV_20230629  baseline 05.09  flag true   declared offset -0.1
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import numpy as np
import pytest
from app.services.agent.executor import _analysis_items
from app.services.analysis import AnalysisRequest
from app.services.satellite import imagery as imagery_mod
from app.services.satellite.imagery import ImageryService
from app.services.satellite.radiometry import (
    OFFSET_INTRODUCED_BASELINE,
    RadiometricValidationError,
    assess_radiometry,
    parse_baseline,
    radiometric_pair_problem,
    require_usable,
)
from app.services.satellite.rtc import RTC_COLLECTION
from app.services.satellite.scene_validation import (
    SceneValidationError,
    ValidatedScene,
    validate_scene_pair,
)

from tests.test_pixel_quality import grid, scl
from tests.test_sar_backscatter import sar_band
from tests.test_scene_validation import (
    LATER_ID,
    RTC_ID,
    SAR,
    SCENE_ID,
    Catalog,
    client_scene,
    execution,
    rtc_item,
    s2_item,
    validate,
    window,
)

NDVI = ("nir", "red")
NDWI = ("green", "nir")


def optical(
    *,
    baseline: str | None = "05.11",
    flag: Any = True,
    offset: float | None = -0.1,
    scene_id: str = SCENE_ID,
    when: str = "2025-01-04T05:15:39.024000Z",
    drop_flag: bool = False,
) -> dict:
    item = s2_item(scene_id, when=when)
    props = item["properties"]
    if baseline is None:
        del props["s2:processing_baseline"]
    else:
        props["s2:processing_baseline"] = baseline
    if drop_flag:
        del props["earthsearch:boa_offset_applied"]
    else:
        props["earthsearch:boa_offset_applied"] = flag
    for key in ("red", "green", "nir", "swir16"):
        band = item["assets"][key]["raster:bands"][0]
        if offset is None:
            band.pop("offset", None)
        else:
            band["offset"] = offset
    return item


def state_of(item: dict, assets: tuple[str, ...] = NDVI):
    scene = validate(item, scene_id=item["id"], assets=(*assets, "scl"))
    return assess_radiometry(scene, assets)


# =========================================================================== #
# Processing baseline
# =========================================================================== #


@pytest.mark.parametrize(
    ("text", "parsed"),
    [("05.11", (5, 11)), ("04.00", (4, 0)), ("03.01", (3, 1)), (" 05.09 ", (5, 9))],
)
def test_baselines_parse(text: str, parsed: tuple[int, int]) -> None:
    assert parse_baseline(text) == parsed


@pytest.mark.parametrize("text", ["5.11", "05.1", "baseline 05.11", "", None, "N0511"])
def test_malformed_baselines_do_not_parse(text: str | None) -> None:
    assert parse_baseline(text) is None


def test_the_offset_baseline_is_stated_once() -> None:
    assert OFFSET_INTRODUCED_BASELINE == (4, 0)
    assert parse_baseline("04.00") >= OFFSET_INTRODUCED_BASELINE  # type: ignore[operator]
    assert parse_baseline("03.01") < OFFSET_INTRODUCED_BASELINE  # type: ignore[operator]


def test_an_unreadable_baseline_leaves_the_representation_undetermined() -> None:
    scene = validate(optical(), assets=(*NDVI, "scl"))
    broken = scene.model_copy(
        update={"processing": scene.processing.model_copy(update={"processing_baseline": None})}
    )
    state = assess_radiometry(broken, NDVI)
    assert state.status == "undetermined"
    assert any("processing baseline is missing" in n for n in state.notes)


def test_a_malformed_baseline_is_already_refused_by_scene_validation() -> None:
    with pytest.raises(SceneValidationError) as info:
        validate(optical(baseline="5.11"), assets=(*NDVI, "scl"))
    assert info.value.code == "unknown_processing_baseline"


# =========================================================================== #
# The BOA offset - the four cases, and the baselines before the offset
# =========================================================================== #


def test_a_offset_removed_by_the_provider_is_usable() -> None:
    state = state_of(optical(baseline="05.11", flag=True))
    assert state.offset_state == "removed_by_provider"
    assert state.boa_offset_applied is True
    assert state.status == "verified_with_unknown_metadata"
    assert state.representation is not None and "no additive offset" in state.representation
    # The declared -0.1 is recorded as contradicted, never applied.
    assert state.metadata_conflicts and "NOT applied" in state.metadata_conflicts[0]


def test_b_offset_not_removed_is_undetermined_and_refused() -> None:
    state = state_of(optical(baseline="04.00", flag=False))
    assert state.offset_state == "not_removed"
    assert state.status == "undetermined"
    assert state.representation is None
    with pytest.raises(RadiometricValidationError) as info:
        require_usable(state)
    assert info.value.code == "radiometric_undetermined"
    assert "boa_offset_applied=false" in info.value.message


def test_c_offset_state_unparseable_is_undetermined() -> None:
    state = state_of(optical(baseline="05.11", flag="yes"))
    assert state.offset_state == "unparseable"
    assert state.status == "undetermined"
    assert state.boa_offset_applied is None


def test_d_offset_flag_absent_is_undetermined() -> None:
    state = state_of(optical(baseline="05.11", drop_flag=True))
    assert state.offset_state == "not_published"
    assert state.status == "undetermined"


@pytest.mark.parametrize("flag", [None, "true"])
def test_a_null_or_string_flag_is_not_a_boolean(flag: Any) -> None:
    assert state_of(optical(flag=flag)).offset_state == "unparseable"


def test_a_baseline_before_the_offset_needs_no_removal() -> None:
    state = state_of(optical(baseline="03.01", flag=False, offset=0.0))
    assert state.offset_state == "not_introduced"
    assert state.status == "verified_with_unknown_metadata"
    assert state.metadata_conflicts == []


def test_a_baseline_before_the_offset_with_no_flag_is_fine() -> None:
    state = state_of(optical(baseline="03.01", drop_flag=True, offset=0.0))
    assert state.offset_state == "not_introduced"
    assert state.status == "verified_with_unknown_metadata"


def test_a_declared_offset_on_an_old_baseline_is_not_ignored() -> None:
    state = state_of(optical(baseline="03.01", flag=False, offset=-0.1))
    assert state.status == "undetermined"
    assert any("declare a non-zero offset" in n for n in state.notes)


def test_offset_removed_from_a_baseline_that_never_had_one_is_contradictory() -> None:
    state = state_of(optical(baseline="03.01", flag=True, offset=0.0))
    assert (state.offset_state, state.status) == ("contradictory", "undetermined")


# =========================================================================== #
# Scale / offset / unit / dtype - recorded as declared, never invented
# =========================================================================== #


def test_declared_scale_and_offset_are_recorded_as_declared() -> None:
    state = state_of(optical())
    nir = next(e for e in state.encodings if e.key == "nir")
    assert (nir.scale, nir.offset, nir.data_type, nir.nodata) == (0.0001, -0.1, "uint16", 0.0)
    assert nir.bits_per_sample == 15


def test_an_undeclared_scale_or_offset_is_never_filled_in() -> None:
    item = optical()
    for key in NDVI:
        item["assets"][key]["raster:bands"][0].pop("scale")
        item["assets"][key]["raster:bands"][0].pop("offset")
    state = state_of(item)
    nir = next(e for e in state.encodings if e.key == "nir")
    assert (nir.scale, nir.offset) == (None, None)
    assert {"nir.scale", "nir.offset", "red.scale", "red.offset"} <= set(state.unknown_fields)
    assert state.status == "verified_with_unknown_metadata"


def test_an_undeclared_unit_is_unknown_not_assumed() -> None:
    state = state_of(optical())
    assert {"nir.unit", "red.unit"} <= set(state.unknown_fields)
    assert all(e.unit is None for e in state.encodings)


def test_a_declared_unit_is_recorded() -> None:
    item = optical()
    for key in NDVI:
        item["assets"][key]["raster:bands"][0]["unit"] = "1"
    state = state_of(item)
    assert {e.unit for e in state.encodings} == {"1"}
    assert "nir.unit" not in state.unknown_fields


def test_bands_declaring_different_units_are_incompatible() -> None:
    item = optical()
    item["assets"]["nir"]["raster:bands"][0]["unit"] = "1"
    item["assets"]["red"]["raster:bands"][0]["unit"] = "W m-2 sr-1 um-1"
    state = state_of(item)
    assert state.status == "incompatible"
    with pytest.raises(RadiometricValidationError) as info:
        require_usable(state)
    assert info.value.code == "radiometric_incompatible"


def test_bands_declaring_different_scales_are_incompatible() -> None:
    item = optical()
    item["assets"]["red"]["raster:bands"][0]["scale"] = 0.001
    assert state_of(item).status == "incompatible"


def test_everything_declared_and_consistent_is_verified() -> None:
    item = optical(offset=0.0)
    for key in NDVI:
        item["assets"][key]["raster:bands"][0]["unit"] = "1"
    state = state_of(item)
    assert state.status == "verified"
    assert state.unknown_fields == [] and state.metadata_conflicts == []


def test_a_compatible_dtype_is_recorded_and_an_incompatible_one_refused_earlier() -> None:
    assert {e.data_type for e in state_of(optical()).encodings} == {"uint16"}
    item = optical()
    item["assets"]["nir"]["raster:bands"][0]["data_type"] = "float32"
    with pytest.raises(SceneValidationError) as info:
        validate(item, assets=(*NDVI, "scl"))
    assert info.value.code == "unsupported_asset_encoding"


def test_the_categorical_scl_is_not_a_radiometric_asset() -> None:
    scene = validate(optical(), assets=(*NDVI, "scl"))
    assert "scl" not in assess_radiometry(scene, (*NDVI, "scl")).assets


# =========================================================================== #
# Valid range and saturation - nothing invented, nothing counted twice
# =========================================================================== #


def test_no_valid_range_is_invented_from_bits_per_sample() -> None:
    state = state_of(optical())
    assert state.valid_range is None
    assert {e.bits_per_sample for e in state.encodings} == {15}


def test_a_missing_valid_range_does_not_refuse_the_scene() -> None:
    item = optical()
    for key in NDVI:
        item["assets"][key]["raster:bands"][0].pop("bits_per_sample")
    state = state_of(item)
    assert state.valid_range is None
    assert state.usable


def test_saturation_is_the_scl_mask_of_stage_three_only() -> None:
    assert state_of(optical()).saturation_source == "scl_class_1"


def test_no_second_saturation_mask_is_built_from_radiometry() -> None:
    """A DN at the 15-bit ceiling on a clear SCL pixel stays valid."""

    catalog = Catalog({SCENE_ID: optical()})
    catalog.read = lambda href, *a, **k: (  # type: ignore[method-assign]
        catalog.events.append(f"read:{href.rsplit('/', 1)[-1]}")
        or (scl([[4, 4]]) if href.endswith("SCL.tif") else grid([[32767, 300]]))
    )
    result = analyze(catalog, indices=["ndvi"])
    (quality,) = result.pixel_quality
    assert (quality.valid_pixels, quality.saturated_or_defective_pixels) == (2, 0)


# =========================================================================== #
# Temporal comparability
# =========================================================================== #


def pair(first: dict, second: dict) -> tuple[ValidatedScene, ValidatedScene]:
    return (
        validate(first, scene_id=first["id"], assets=(*NDWI, "scl")),
        validate(second, scene_id=second["id"], assets=(*NDWI, "scl")),
    )


def later(**kwargs: Any) -> dict:
    return optical(scene_id=LATER_ID, when="2025-01-19T05:15:41Z", **kwargs)


def test_a_compatible_pair_passes_both_stages() -> None:
    a, b = pair(optical(), later(baseline="05.09"))
    validate_scene_pair(a, b)
    first, second = assess_radiometry(a, NDWI), assess_radiometry(b, NDWI)
    assert radiometric_pair_problem(first, second) is None


def test_different_baselines_with_one_representation_are_comparable() -> None:
    """03.01 (no offset ever) and 05.09 (offset removed) are both offset-free.

    Comparing the provider FLAGS refused this pair (false vs true); comparing
    the REPRESENTATION the pixels carry accepts it.
    """

    a, b = pair(
        optical(baseline="03.01", flag=False, offset=0.0, when="2022-01-20T05:13:00Z"),
        later(baseline="05.09", flag=True),
    )
    pair_result = validate_scene_pair(a, b)
    assert any("different baselines" in n for n in pair_result.notes)
    first, second = assess_radiometry(a, NDWI), assess_radiometry(b, NDWI)
    assert radiometric_pair_problem(first, second) is None


def test_matching_offset_states_are_comparable() -> None:
    a, b = pair(optical(flag=True), later(flag=True))
    assert validate_scene_pair(a, b).status == "compatible"


def test_mismatched_offset_states_are_refused() -> None:
    a, b = pair(optical(flag=True), later(flag=False))
    with pytest.raises(SceneValidationError) as info:
        validate_scene_pair(a, b)
    assert info.value.code == "temporal_scene_incompatible"


def test_a_missing_offset_state_on_one_side_is_refused() -> None:
    a, b = pair(optical(flag=True), later(drop_flag=True))
    with pytest.raises(SceneValidationError):
        validate_scene_pair(a, b)


def test_a_scale_declared_differently_on_the_two_dates_is_incompatible() -> None:
    second = later()
    for key in NDWI:
        second["assets"][key]["raster:bands"][0]["scale"] = 0.001
    a, b = pair(optical(), second)
    problem = radiometric_pair_problem(assess_radiometry(a, NDWI), assess_radiometry(b, NDWI))
    assert problem is not None and "scales" in problem


def test_an_unusable_state_is_never_paired() -> None:
    a, b = pair(optical(), later(baseline="04.00", flag=False))
    assert radiometric_pair_problem(assess_radiometry(a, NDWI), assess_radiometry(b, NDWI))


def temporal(first: dict, second: dict) -> tuple[Catalog, Any]:
    catalog = Catalog({first["id"]: first, second["id"]: second})
    windows = [
        window(client_scene(first["id"], datetime="2025-01-04T05:15:39Z"),
               label="baseline", start="2025-01-01", end="2025-01-10"),
        window(client_scene(second["id"], datetime="2025-01-19T05:15:41Z"),
               label="target", start="2025-01-11", end="2025-01-31"),
    ]
    result = asyncio.run(
        catalog.service().analyze(
            AnalysisRequest(execution=execution(windows, temporal=True), include_temporal_ndwi=True)
        )
    )
    return catalog, result


def test_temporal_compatible_pair_is_compared_and_carries_both_states() -> None:
    catalog, result = temporal(optical(), later(baseline="05.09"))
    comparison = result.temporal_comparison
    assert comparison is not None
    assert comparison.first.radiometry is not None
    assert comparison.second.radiometry is not None
    assert comparison.second.radiometry.processing_baseline == "05.09"
    assert len(catalog.reads) == 6


def test_temporal_offset_mismatch_reads_nothing() -> None:
    catalog, result = temporal(optical(), later(flag=False))
    assert result.temporal_comparison is None
    assert catalog.reads == []


def test_temporal_missing_offset_state_reads_nothing() -> None:
    catalog, result = temporal(optical(), later(drop_flag=True))
    assert result.temporal_comparison is None
    assert catalog.reads == []


def test_temporal_cross_date_scale_mismatch_reads_nothing() -> None:
    second = later()
    for key in NDWI:
        second["assets"][key]["raster:bands"][0]["scale"] = 0.001
    catalog, result = temporal(optical(), second)
    assert result.temporal_comparison is None
    assert catalog.reads == []
    assert any("radiometric_incompatible" in w for w in result.warnings)


def test_temporal_pair_across_the_offset_change_is_compared() -> None:
    first = optical(baseline="03.01", flag=False, offset=0.0, when="2025-01-04T05:15:39Z")
    catalog, result = temporal(first, later(baseline="05.09"))
    assert result.temporal_comparison is not None
    assert len(catalog.reads) == 6


# =========================================================================== #
# Integration: NDVI / NDWI / NDBI through the real ImageryService
# =========================================================================== #


def analyze(catalog: Catalog, **request: Any):
    return asyncio.run(
        catalog.service().analyze(
            AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), **request)
        )
    )


def test_ndvi_on_an_undetermined_scene_reads_nothing() -> None:
    catalog = Catalog({SCENE_ID: optical(baseline="04.00", flag=False)})
    result = analyze(catalog, indices=["ndvi"])
    assert catalog.reads == []
    (outcome,) = result.analysis_outcomes
    assert outcome.status == "unavailable"
    assert "radiometric_undetermined" in (outcome.reason or "")
    (state,) = result.radiometry
    assert state.status == "undetermined"  # the refused state is kept as evidence


def test_ndwi_on_an_undetermined_scene_reads_nothing() -> None:
    catalog = Catalog({SCENE_ID: optical(drop_flag=True)})
    result = analyze(catalog, include_ndwi=True)
    assert catalog.reads == []
    assert result.measurements == []
    assert result.radiometry[0].offset_state == "not_published"


def test_ndbi_refused_alone_leaves_ndvi_and_never_reads_swir() -> None:
    item = optical()
    item["assets"]["swir16"]["raster:bands"][0]["scale"] = 0.001
    catalog = Catalog({SCENE_ID: item})
    result = analyze(catalog, indices=["ndvi", "ndbi"])
    statuses = {o.name: o.status for o in result.analysis_outcomes}
    assert statuses == {"ndvi": "completed", "ndbi": "unavailable"}
    assert "read:B11.tif" not in catalog.reads
    assert [s.status for s in result.radiometry] == [
        "verified_with_unknown_metadata", "incompatible",
    ]


def test_the_formula_consumes_raw_values_unchanged() -> None:
    """Declared scale 0.0001 and offset -0.1 are recorded, never applied."""

    catalog = Catalog({SCENE_ID: optical()})
    catalog.read = lambda href, *a, **k: (  # type: ignore[method-assign]
        catalog.events.append(href)
        or (scl([[4]]) if href.endswith("SCL.tif")
            else grid([[300]]) if href.endswith("B08.tif") else grid([[100]]))
    )
    result = analyze(catalog, indices=["ndvi"])
    ndvi = {m.name: m.value for m in result.measurements}["ndvi_mean"]
    assert ndvi == pytest.approx((300 - 100) / (300 + 100))
    applied = ((300 * 1e-4 - 0.1) - (100 * 1e-4 - 0.1)) / ((300 * 1e-4 - 0.1) + (100 * 1e-4 - 0.1))
    assert not np.isclose(ndvi, applied)


def test_an_old_baseline_scene_is_measured() -> None:
    catalog = Catalog({SCENE_ID: optical(baseline="03.01", flag=False, offset=0.0)})
    result = analyze(catalog, indices=["ndvi"])
    assert result.analysis_outcomes[0].status == "completed"
    assert result.radiometry[0].offset_state == "not_introduced"


# =========================================================================== #
# SAR
# =========================================================================== #


def sar_state(item: dict):
    scene = validate(item, scene_id=RTC_ID, collection=RTC_COLLECTION, modality=SAR,
                     assets=("vv", "vh"))
    return assess_radiometry(scene, ("vv", "vh"))


def test_rtc_as_published_is_linear_power_with_undeclared_encoding() -> None:
    state = sar_state(rtc_item())
    assert state.status == "verified_with_unknown_metadata"
    assert state.offset_state == "not_applicable"
    assert state.representation == "provider RTC gamma naught, linear power, read as-is"
    assert "provider" in state.notes[0] and "not by SatQuery" in state.notes[0]
    assert {"vv.scale", "vv.offset", "vv.unit"} <= set(state.unknown_fields)


def test_rtc_declared_as_linear_power_is_verified() -> None:
    item = rtc_item()
    for pol in ("vv", "vh"):
        item["assets"][pol]["raster:bands"][0].update(unit="linear", scale=1.0, offset=0.0)
    assert sar_state(item).status == "verified"


@pytest.mark.parametrize("unit", ["dB", "decibel", "DB"])
def test_decibel_values_are_incompatible(unit: str) -> None:
    item = rtc_item()
    item["assets"]["vv"]["raster:bands"][0]["unit"] = unit
    state = sar_state(item)
    assert state.status == "incompatible"
    assert any("decibels" in n for n in state.notes)


@pytest.mark.parametrize(("field", "value"), [("scale", 0.5), ("offset", 1.0)])
def test_an_encoding_scale_or_offset_is_incompatible(field: str, value: float) -> None:
    item = rtc_item()
    item["assets"]["vh"]["raster:bands"][0][field] = value
    assert sar_state(item).status == "incompatible"


def test_rtc_with_integer_pixels_is_refused_by_scene_validation() -> None:
    item = rtc_item()
    item["assets"]["vv"]["raster:bands"][0]["data_type"] = "uint16"
    with pytest.raises(SceneValidationError) as info:
        validate(item, scene_id=RTC_ID, collection=RTC_COLLECTION, modality=SAR,
                 assets=("vv", "vh"))
    assert info.value.code == "unsupported_asset_encoding"


def sar_analysis(item: dict, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], Any]:
    signed: list[str] = []
    monkeypatch.setattr(
        imagery_mod, "sign_rtc_asset", lambda href, **kw: signed.append(href) or f"{href}?se=T"
    )
    reads: list[str] = []
    imagery = ImageryService(
        stac_item_fetcher=lambda *_: copy.deepcopy(item),
        band_reader=lambda href, *a, **k: reads.append(href) or sar_band([0.03, 0.02]),
    )
    from app.services.analysis import AnalysisService

    sar_scene = client_scene(RTC_ID, collection=RTC_COLLECTION, platform="sentinel-1a")
    result = asyncio.run(
        AnalysisService(imagery_service=imagery).analyze(
            AnalysisRequest(
                execution=execution([window(sar_scene, modality=SAR)]),
                include_sar_backscatter=True,
            )
        )
    )
    return reads + signed, result


def test_sar_in_decibels_is_refused_before_any_read_or_signing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = rtc_item()
    item["assets"]["vv"]["raster:bands"][0]["unit"] = "dB"
    touched, result = sar_analysis(item, monkeypatch)
    assert touched == []
    assert result.sar_backscatter is None
    assert result.radiometry[0].status == "incompatible"
    assert any("radiometric_incompatible" in w for w in result.warnings)


def test_sar_as_published_is_measured_with_its_state_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched, result = sar_analysis(rtc_item(), monkeypatch)
    assert touched  # both polarizations were signed and read
    assert result.sar_backscatter is not None
    assert result.radiometry[0].modality == SAR
    assert any("Sentinel-1 backscatter radiometric state" in w for w in result.warnings)


# =========================================================================== #
# Evidence
# =========================================================================== #


def test_the_radiometric_facts_reach_citable_evidence() -> None:
    catalog = Catalog({SCENE_ID: optical()})
    result = analyze(catalog, indices=["ndvi"])
    texts = [i.text for i in _analysis_items(result) if i.text]
    line = next(t for t in texts if t.startswith("NDVI radiometric state"))
    assert "verified_with_unknown_metadata" in line
    assert "Processing baseline 05.11" in line
    assert "reflectance offset removed by provider" in line
    assert "declared scale 0.0001" in line
    assert "unit not declared" in line
    assert "source: the catalog item" in line


def test_the_structured_state_names_its_authority_and_holds_no_catalog_blob() -> None:
    state = state_of(optical())
    dumped = state.model_dump(mode="json")
    assert "earthsearch:boa_offset_applied" in state.authority
    assert "href" not in str(dumped)
    assert dumped["validation_stage"] == "radiometric"


def test_an_unvalidated_reader_gets_no_radiometric_state() -> None:
    """Test doubles that validate no scene are not given a verdict they did not earn."""

    from tests.test_pixel_quality import ones, service_result

    result, _ = service_result({"nir": ones(1, 300.0), "red": ones(1)}, indices=["ndvi"])
    assert result.radiometry == []
