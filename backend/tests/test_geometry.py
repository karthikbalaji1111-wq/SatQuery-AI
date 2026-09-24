"""Stage 5 of the scientific pipeline: geometric validation.

The one authoritative statement of when rasters may be combined
(``app/services/analysis/geometry.py``): identical grids for a band pair, a
paired temporal change and a VV-VH difference; EXACT nesting (same CRS,
north-up, integer ratio per axis, cell edges on pixel edges, overlap) for a
coarser raster placed on the 10 m analysis grid. Anything else is refused -
never shifted, snapped, cropped, padded, resampled or reprojected.

Grids here follow the live Sentinel-2 tile 44PMV (EPSG:32644, origin
399960 / 1500000): 10 m bands 10980 px square, 20 m SWIR and SCL 5490 px.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from typing import Any

import pytest
from affine import Affine
from app.services.agent.executor import _analysis_items
from app.services.analysis import AnalysisRequest, AnalysisService
from app.services.analysis import service as service_mod
from app.services.analysis.engines import coregister_to_finer_grid
from app.services.analysis.geometry import (
    TOLERANCE_PIXELS,
    GeometryError,
    Grid,
    nesting,
    same_grid_problem,
    source_problem,
)
from app.services.analysis.pixel_quality import align_scl_to_grid
from app.services.satellite.raster import BandWindow
from rasterio import crs as rasterio_crs

from tests import test_temporal_ndwi as temporal
from tests.test_analysis import FakeImageryService, make_execution
from tests.test_pixel_quality import grid, ones, scl
from tests.test_scene_validation import (
    SCENE_ID,
    Catalog,
    client_scene,
    execution,
    s2_item,
    window,
)

X0, Y0 = 399960.0, 1500000.0


def g(
    *,
    res: float = 10.0,
    res_y: float | None = None,
    x0: float = X0,
    y0: float = Y0,
    width: int = 4,
    height: int = 4,
    crs: str | None = "EPSG:32644",
    b: float = 0.0,
    d: float = 0.0,
) -> Grid:
    ry = res if res_y is None else res_y
    return Grid(crs=crs, transform=(res, b, x0, d, -ry, y0), width=width, height=height)


def code(problem: Any) -> str | None:
    return None if problem is None else problem[0]


def moved(band: BandWindow, *, dx: float = 0.0, dy: float = 0.0, **changes: Any) -> BandWindow:
    t = band.transform
    return dataclasses.replace(
        band, transform=Affine(t.a, t.b, t.c + dx, t.d, t.e, t.f + dy), **changes
    )


def named(items: list[Any]) -> dict[str, float]:
    return {m.name: m.value for m in items}


# =========================================================================== #
# Basic
# =========================================================================== #


def test_identical_grids_are_the_same_grid() -> None:
    assert same_grid_problem(g(), g()) is None


def test_identical_crs_transform_and_resolution_nest_at_ratio_one() -> None:
    placed, problem = nesting(g(), g())
    assert problem is None and placed is not None
    assert (placed.ratio_x, placed.ratio_y, placed.offset_x, placed.offset_y) == (1, 1, 0, 0)


def test_the_tolerance_is_a_millionth_of_a_pixel() -> None:
    assert TOLERANCE_PIXELS == 1e-6


# =========================================================================== #
# CRS
# =========================================================================== #


def test_a_crs_mismatch_is_refused() -> None:
    assert code(same_grid_problem(g(), g(crs="EPSG:32643"))) == "grid_crs_mismatch"
    assert code(nesting(g(res=20.0), g(crs="EPSG:32643"))[1]) == "grid_crs_mismatch"


@pytest.mark.parametrize("which", ["first", "second"])
def test_a_missing_crs_is_never_valid(which: str) -> None:
    a, b = (g(crs=None), g()) if which == "first" else (g(), g(crs=None))
    assert code(same_grid_problem(a, b)) == "grid_crs_unknown"
    assert code(nesting(a, b)[1]) == "grid_crs_unknown"
    assert code(same_grid_problem(g(crs=None), g(crs=None))) == "grid_crs_unknown"


@pytest.mark.parametrize(
    "other", ["epsg:32644", " EPSG:32644 ", rasterio_crs.CRS.from_epsg(32644).to_wkt()]
)
def test_equivalent_crs_representations_are_the_same_crs(other: str) -> None:
    assert same_grid_problem(g(), g(crs=other)) is None


# =========================================================================== #
# Transform
# =========================================================================== #


def test_exact_alignment() -> None:
    assert same_grid_problem(g(x0=X0 + 40.0), g(x0=X0 + 40.0)) is None


def test_floating_point_representation_noise_is_the_same_grid() -> None:
    assert same_grid_problem(g(), g(x0=X0 + 1e-9, y0=Y0 - 1e-9)) is None
    assert same_grid_problem(g(), g(res=10.0 + 1e-12)) is None


def test_a_shift_just_beyond_the_tolerance_is_a_different_grid() -> None:
    assert code(same_grid_problem(g(), g(x0=X0 + 1e-3))) == "grid_misaligned"


@pytest.mark.parametrize(("dx", "dy"), [(5.0, 0.0), (0.0, -5.0), (5.0, -5.0)])
def test_a_half_pixel_shift_is_misalignment(dx: float, dy: float) -> None:
    assert code(same_grid_problem(g(), g(x0=X0 + dx, y0=Y0 + dy))) == "grid_misaligned"


@pytest.mark.parametrize(("dx", "dy"), [(10.0, 0.0), (0.0, 10.0)])
def test_a_one_pixel_shift_is_misalignment(dx: float, dy: float) -> None:
    assert code(same_grid_problem(g(), g(x0=X0 + dx, y0=Y0 + dy))) == "grid_misaligned"


def test_a_different_origin_is_a_different_grid() -> None:
    assert code(same_grid_problem(g(), g(x0=500000.0))) == "grid_misaligned"


@pytest.mark.parametrize(
    "rotated",
    [dict(b=0.5), dict(d=0.5), dict(res_y=-10.0)],
    ids=["x-shear", "y-shear", "south-up"],
)
def test_rotated_sheared_or_south_up_grids_are_unsupported(rotated: dict) -> None:
    assert code(same_grid_problem(g(**rotated), g(**rotated))) == "grid_orientation_unsupported"
    assert code(nesting(g(res=20.0), g(**rotated))[1]) == "grid_orientation_unsupported"


# =========================================================================== #
# Resolution
# =========================================================================== #


def test_identical_resolution() -> None:
    assert same_grid_problem(g(res=20.0), g(res=20.0)) is None


def test_a_different_resolution_is_not_the_same_grid() -> None:
    assert code(same_grid_problem(g(), g(res=20.0))) == "grid_resolution_mismatch"


def test_a_valid_two_to_one_nesting() -> None:
    placed, problem = nesting(g(res=20.0, width=2, height=2), g())
    assert problem is None and placed is not None
    assert (placed.ratio_x, placed.ratio_y) == (2, 2)


@pytest.mark.parametrize("res", [15.0, 25.0, 5.0], ids=["1.5x", "2.5x", "finer"])
def test_an_invalid_ratio_is_refused(res: float) -> None:
    assert code(nesting(g(res=res), g())[1]) == "grid_ratio_invalid"


def test_a_wrong_x_ratio_is_refused() -> None:
    assert code(nesting(g(res=15.0, res_y=20.0), g())[1]) == "grid_ratio_invalid"


def test_a_wrong_y_ratio_is_refused() -> None:
    """A right x ratio does not hide a wrong y ratio."""

    assert code(nesting(g(res=20.0, res_y=15.0), g())[1]) == "grid_ratio_invalid"


# =========================================================================== #
# Bounds
# =========================================================================== #


def test_identical_bounds() -> None:
    assert g().bounds == g().bounds == (X0, Y0 - 40.0, X0 + 40.0, Y0)


def test_compatible_nested_bounds_with_a_whole_cell_offset() -> None:
    """A 20 m window clamped one cell further in is still exactly nested."""

    placed, problem = nesting(g(res=20.0, x0=X0 + 20.0, width=1, height=2), g())
    assert problem is None and placed is not None
    assert (placed.offset_x, placed.offset_y) == (2, 0)


def test_a_coarse_grid_shifted_by_half_its_fine_pixel_is_not_nested() -> None:
    """20 m, correct ratio, shifted 5 m: the right ratio and the wrong ground."""

    assert code(nesting(g(res=20.0, x0=X0 + 5.0), g())[1]) == "grid_misaligned"
    assert code(nesting(g(res=20.0, y0=Y0 - 5.0), g())[1]) == "grid_misaligned"


def test_a_coarse_grid_shifted_by_a_whole_fine_pixel_is_still_nested() -> None:
    """Cell edges on 10 m pixel edges: a 10 m offset is a legitimate nesting."""

    assert nesting(g(res=20.0, x0=X0 + 10.0), g())[1] is None


def test_partial_overlap_is_not_a_common_grid() -> None:
    assert code(same_grid_problem(g(), g(width=3))) == "grid_shape_mismatch"


def test_no_overlap_is_refused() -> None:
    assert code(nesting(g(res=20.0, x0=900000.0), g())[1]) == "grid_no_overlap"


# =========================================================================== #
# The existing transformations now PROVE their relationship first
# =========================================================================== #


def fine_band() -> BandWindow:
    return grid([[100] * 4] * 4, res=10.0)


def coarse_band(**kwargs: Any) -> BandWindow:
    return grid([[300, 20], [300, 20]], res=20.0, **kwargs)


def test_coregistration_refuses_a_swir_band_shifted_by_five_metres() -> None:
    with pytest.raises(GeometryError) as info:
        coregister_to_finer_grid(moved(coarse_band(), dx=5.0), fine_band())
    assert info.value.code == "grid_misaligned"


def test_coregistration_still_accepts_the_valid_nesting() -> None:
    aligned = coregister_to_finer_grid(coarse_band(), fine_band())
    assert aligned.values.tolist() == [[300, 300, 20, 20]] * 4


@pytest.mark.parametrize(
    ("layer", "expected"),
    [
        (lambda: moved(scl([[4, 9]], res=20.0), dx=5.0), "grid_misaligned"),
        (lambda: scl([[4, 9]], res=20.0, crs="EPSG:32643"), "grid_crs_mismatch"),
        (lambda: scl([[4, 9]], res=15.0), "grid_ratio_invalid"),
        (lambda: dataclasses.replace(
            scl([[4, 9]], res=20.0), transform=Affine(20.0, 1.0, X0, 0.0, -20.0, Y0)
        ), "grid_orientation_unsupported"),
    ],
    ids=["shifted", "wrong-crs", "wrong-resolution", "wrong-transform"],
)
def test_scl_geometry_is_validated_before_classification(layer: Any, expected: str) -> None:
    with pytest.raises(GeometryError) as info:
        align_scl_to_grid(layer(), grid([[100] * 4], res=10.0))
    assert info.value.code == expected


def test_a_valid_twenty_metre_scl_lands_on_the_ten_metre_grid() -> None:
    aligned = align_scl_to_grid(scl([[4, 9]], res=20.0), grid([[100] * 4] * 2, res=10.0))
    assert aligned.values.tolist() == [[4, 4, 9, 9]] * 2


# =========================================================================== #
# Service: NDBI, SCL and the pair, with the GridState that results
# =========================================================================== #


def analyze(bands: dict[str, BandWindow], **request: Any):
    imagery = FakeImageryService(bands=bands)
    return asyncio.run(
        AnalysisService(imagery_service=imagery).analyze(  # type: ignore[arg-type]
            AnalysisRequest(execution=make_execution(), **request)
        )
    )


def ndbi_bands(**swir: Any) -> dict[str, BandWindow]:
    return {
        "nir": grid([[100] * 4] * 2, res=10.0),
        "swir16": moved(grid([[300, 20]], res=20.0), **swir),
        "scl": scl([[4, 4]], res=20.0),
    }


def test_valid_ndbi_nesting_is_computed_and_recorded() -> None:
    result = analyze(ndbi_bands(), indices=["ndbi"])
    assert "ndbi_mean" in named(result.measurements)
    (state,) = result.grids
    assert (state.status, state.analysis, state.stage) == ("valid", "ndbi", "post_read")
    assert (state.crs, state.width, state.height) == ("EPSG:32644", 4, 2)
    assert (state.resolution_x, state.resolution_y) == (10.0, 10.0)
    assert state.origin == [X0, Y0]
    assert state.bounds == [X0, Y0 - 20.0, X0 + 40.0, Y0]
    roles = {i.role: i for i in state.inputs}
    assert roles["swir16"].relationship == "nested"
    assert (roles["swir16"].ratio_x, roles["swir16"].ratio_y) == (2, 2)
    assert (roles["swir16"].offset_x, roles["swir16"].offset_y) == (0, 0)
    assert roles["nir"].relationship == "identical"
    assert roles["scl"].relationship == "nested"


def test_shifted_swir_refuses_ndbi_with_no_number_and_a_recorded_reason() -> None:
    result = analyze(ndbi_bands(dx=5.0), indices=["ndbi"])
    assert "ndbi_mean" not in named(result.measurements)
    (state,) = result.grids
    assert state.status == "refused"
    assert state.refusal is not None and "grid_misaligned" in state.refusal
    assert result.analysis_outcomes[0].status == "unavailable"
    assert any("grid_misaligned" in w for w in result.warnings)


def test_shifted_scl_refuses_the_index() -> None:
    bands = {
        "nir": grid([[300] * 4] * 2, res=10.0),
        "red": grid([[100] * 4] * 2, res=10.0),
        "scl": moved(scl([[4, 4]], res=20.0), dx=5.0),
    }
    result = analyze(bands, indices=["ndvi"])
    assert "ndvi_mean" not in named(result.measurements)
    assert result.grids[0].status == "refused"


def test_a_band_pair_on_shifted_ground_is_refused() -> None:
    bands = {
        "nir": grid([[300] * 4], res=10.0),
        "red": moved(grid([[100] * 4], res=10.0), dx=10.0),
        "scl": scl([[4] * 4], res=10.0),
    }
    result = analyze(bands, indices=["ndvi"])
    assert "ndvi_mean" not in named(result.measurements)
    assert result.grids[0].status == "refused"


def test_valid_geometry_leaves_the_numbers_unchanged() -> None:
    """M5 validates; it does not alter a valid computation's arithmetic."""

    bands = {"nir": grid([[300, 300, 10]]), "red": grid([[100, 100, 90]])}
    result = analyze(bands, indices=["ndvi"])
    stats = named(result.measurements)
    assert stats["ndvi_mean"] == pytest.approx((0.5 + 0.5 + (10 - 90) / 100) / 3)
    assert result.grids[0].status == "valid"


# -- temporal ----------------------------------------------------------------- #


def temporal_result(second_grid: dict[str, Any] | None = None, **second: Any):
    bands_b = {
        "green": temporal.band([5, 5, 5]),
        "nir": temporal.band([5, 5, 5]),
    }
    if second_grid:
        bands_b = {k: moved(v, **second_grid) for k, v in bands_b.items()}
    if second:
        bands_b = {k: dataclasses.replace(v, **second) for k, v in bands_b.items()}
    bands_b["scl"] = dataclasses.replace(
        scl([[4, 4, 4]]), transform=bands_b["green"].transform, crs=bands_b["green"].crs
    )
    imagery = temporal.FakeImageryService(
        bands={
            "scene-a": {"green": temporal.band([3, 3, 3]), "nir": temporal.band([1, 1, 1])},
            "scene-b": bands_b,
        }
    )
    result, _ = temporal.analyze_temporal(temporal.two_window_execution(), imagery)
    return result


def test_temporal_matching_geometry_has_a_valid_pair_grid() -> None:
    comparison = temporal_result().temporal_comparison
    assert comparison is not None and comparison.change is not None
    pair = comparison.pair_grid
    assert pair is not None and pair.status == "valid"
    assert [i.relationship for i in pair.inputs] == ["identical", "identical"]
    assert comparison.first.grid is not None and comparison.first.grid.status == "valid"
    assert comparison.second.grid is not None


@pytest.mark.parametrize(
    ("shift", "replace", "expected"),
    [
        (None, {"crs": "EPSG:32643"}, "different coordinate reference systems"),
        ({"dx": 10.0}, None, "different pixel grids"),
        ({"dx": 5.0}, None, "different pixel grids"),
        (None, {"resolution": 20.0,
                "transform": Affine(20.0, 0.0, X0, 0.0, -20.0, Y0)}, "different pixel grids"),
    ],
    ids=["different-crs", "shifted-one-pixel", "shifted-half-pixel", "different-resolution"],
)
def test_temporal_incompatible_grids_withhold_only_the_paired_change(
    shift: Any, replace: Any, expected: str
) -> None:
    comparison = temporal_result(shift, **(replace or {})).temporal_comparison
    assert comparison is not None
    assert comparison.change is None
    assert comparison.pair_grid is not None and comparison.pair_grid.status == "refused"
    assert expected in (comparison.pair_grid.refusal or "")
    # Each side's own statistics never needed a common grid.
    assert "ndwi_mean" in named(comparison.first.measurements)
    assert "ndwi_mean" in named(comparison.second.measurements)


def test_an_incompatible_pair_is_refused_before_the_change_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(**_: Any) -> None:
        raise AssertionError("the paired change was computed on incompatible grids")

    monkeypatch.setattr(service_mod, "compute_ndwi_temporal_change", never)
    comparison = temporal_result({"dx": 10.0}).temporal_comparison
    assert comparison is not None and comparison.change is None


# =========================================================================== #
# Pre-read: the catalog's source grids, before any pixel is read
# =========================================================================== #


def with_grids(item: dict, **shift: dict[str, float]) -> dict:
    """Publish tile 44PMV's real source grids on the fixture item."""

    item = copy.deepcopy(item)
    item["properties"]["proj:epsg"] = 32644
    for key in ("red", "green", "nir"):
        item["assets"][key]["proj:transform"] = [10, 0, X0, 0, -10, Y0]
        item["assets"][key]["proj:shape"] = [10980, 10980]
    for key in ("swir16", "scl"):
        dx = shift.get(key, {}).get("dx", 0.0)
        item["assets"][key]["proj:transform"] = [20, 0, X0 + dx, 0, -20, Y0]
        item["assets"][key]["proj:shape"] = [5490, 5490]
    return item


def run(item: dict, **request: Any) -> tuple[Catalog, Any]:
    catalog = Catalog({SCENE_ID: item})
    result = asyncio.run(
        catalog.service().analyze(
            AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), **request)
        )
    )
    return catalog, result


def test_the_real_tile_geometry_passes_before_the_read() -> None:
    catalog, result = run(with_grids(s2_item()), indices=["ndvi", "ndbi"])
    assert {s.status for s in result.grids} == {"valid"}
    assert "read:B11.tif" in catalog.reads


def test_a_shifted_swir_source_refuses_ndbi_before_it_is_read() -> None:
    catalog, result = run(with_grids(s2_item(), swir16={"dx": 5.0}), indices=["ndvi", "ndbi"])
    statuses = {o.name: o.status for o in result.analysis_outcomes}
    assert statuses == {"ndvi": "completed", "ndbi": "unavailable"}
    assert "read:B11.tif" not in catalog.reads
    refused = [s for s in result.grids if s.status == "refused"]
    assert [(s.analysis, s.stage) for s in refused] == [("ndbi", "pre_read")]


def test_a_shifted_scl_source_refuses_every_index_before_any_read() -> None:
    catalog, result = run(with_grids(s2_item(), scl={"dx": 5.0}), indices=["ndvi", "ndwi"])
    assert catalog.reads == []
    assert {(s.status, s.stage) for s in result.grids} == {("refused", "pre_read")}


def test_ndwi_source_refusal_reads_nothing() -> None:
    catalog, result = run(with_grids(s2_item(), scl={"dx": 5.0}), include_ndwi=True)
    assert catalog.reads == []
    assert result.grids[0].status == "refused" and result.grids[0].stage == "pre_read"


def test_unpublished_source_grids_are_validated_after_the_read() -> None:
    catalog, result = run(s2_item(), indices=["ndvi"])
    assert result.grids[0].status == "valid" and result.grids[0].stage == "post_read"
    assert catalog.reads


def test_source_problem_notes_unpublished_grids() -> None:
    problem, notes = source_problem({"nir": object()}, None, scientific=("nir",))
    assert problem is None
    assert notes and "publishes no grid" in notes[0]


def test_temporal_source_refusal_reads_nothing() -> None:
    first = with_grids(s2_item())
    second = with_grids(
        s2_item("S2A_44PMV_20250119_0_L2A", when="2025-01-19T05:15:41Z"), scl={"dx": 5.0}
    )
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
    assert result.temporal_comparison is None
    assert catalog.reads == []
    assert any("grid_misaligned" in w for w in result.warnings)


# =========================================================================== #
# Evidence
# =========================================================================== #


def test_grid_facts_reach_citable_evidence() -> None:
    result = analyze(ndbi_bands(), indices=["ndbi"])
    texts = [i.text for i in _analysis_items(result) if i.text]
    line = next(t for t in texts if t.startswith("NDBI grid:"))
    assert "valid - EPSG:32644, 10 x 10 m, 4x2 px" in line
    assert "swir16 nested 2:1" in line and "scl nested 2:1" in line


def test_a_refusal_reason_reaches_evidence() -> None:
    result = analyze(ndbi_bands(dx=5.0), indices=["ndbi"])
    texts = [i.text for i in _analysis_items(result) if i.text]
    assert any("grid_misaligned" in t for t in texts)


def test_the_grid_record_is_compact_and_exact() -> None:
    state = analyze(ndbi_bands(), indices=["ndbi"]).grids[0]
    dumped = state.model_dump(mode="json")
    assert dumped["transform"] == [10.0, 0.0, X0, 0.0, -10.0, Y0]
    assert dumped["grid_id"] == state.grid_id and "EPSG:32644" in state.grid_id
    assert all(not isinstance(v, bytes) for v in dumped.values())


def test_a_single_scene_grid_record_is_on_every_optical_result() -> None:
    result = analyze({"nir": ones(2, 300.0), "red": ones(2)}, indices=["ndvi"])
    assert len(result.grids) == 1
    result = analyze({"green": ones(2, 300.0), "nir": ones(2)}, include_ndwi=True)
    assert len(result.grids) == 1 and result.grids[0].analysis == "ndwi"
