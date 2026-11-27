"""An area measured in tiles gives the result of one read of the same area.

Real GeoTIFFs on disk, read through the REAL raster layer and the REAL
ImageryService (catalog validation, radiometry, grid checks): only the STAC
fetch and the file location are substituted. Each scenario runs twice - once
with tiles of 128 px and once in a single read - and the two must agree:
pixel counts and quality categories exactly, means to floating-point
rounding, minima and maxima exactly.
"""

from __future__ import annotations

import asyncio
import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine
from app.core.config import Settings
from app.services.analysis import AnalysisRequest, AnalysisService
from app.services.analysis.tiling import plan_tiles
from app.services.satellite.imagery import ImageryService
from app.services.satellite.raster import aoi_window, read_band_window

from tests.test_scene_validation import AOI, SCENE_ID, client_scene, execution, s2_item, window

X0, Y0 = 399960.0, 1500000.0
CRS = "EPSG:32644"
TILE = 128


def _local_grid() -> tuple[Affine, int, int]:
    """A 10 m grid on tile 44PMV's lattice that covers the AOI with a margin."""

    full = Affine(10, 0, X0, 0, -10, Y0)
    win = aoi_window(AOI, crs=CRS, transform=full, width=10980, height=10980)
    col0 = (int(win.col_off) - 40) // 2 * 2
    row0 = (int(win.row_off) - 40) // 2 * 2
    width = (int(win.width) + 80) // 2 * 2
    height = (int(win.height) + 80) // 2 * 2
    return full * Affine.translation(col0, row0), width, height


def _write(path: Path, data: np.ndarray, transform: Affine, nodata: float) -> None:
    profile = {
        "driver": "GTiff", "width": data.shape[1], "height": data.shape[0], "count": 1,
        "dtype": str(data.dtype), "crs": CRS, "transform": transform, "nodata": nodata,
        "tiled": True, "blockxsize": 64, "blockysize": 64,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture(scope="module")
def scene_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("tiles")
    t10, w, h = _local_grid()
    t20 = Affine(20, 0, t10.c, 0, -20, t10.f)
    rng = np.random.default_rng(42)
    files: dict[str, Path] = {}
    for name in ("B03", "B04", "B08"):
        data = rng.integers(1, 6000, (h, w), dtype=np.uint16)
        data[rng.random((h, w)) < 0.01] = 0  # scattered band nodata
        files[name] = root / f"{name}.tif"
        _write(files[name], data, t10, 0)
    swir = rng.integers(1, 6000, (h // 2, w // 2), dtype=np.uint16)
    files["B11"] = root / "B11.tif"
    _write(files["B11"], swir, t20, 0)
    # Clear land, water, bare soil, cloud, shadow, snow, and some no-data.
    classes = rng.choice(
        np.array([4, 5, 6, 8, 9, 3, 11, 0], dtype=np.uint8),
        (h // 2, w // 2),
        p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05],
    )
    files["SCL"] = root / "SCL.tif"
    _write(files["SCL"], classes, t20, 0)

    item = copy.deepcopy(s2_item())
    item["properties"]["proj:epsg"] = 32644
    for key in ("red", "green", "nir"):
        item["assets"][key]["proj:transform"] = list(t10)[:6]
        item["assets"][key]["proj:shape"] = [h, w]
    for key in ("swir16", "scl"):
        item["assets"][key]["proj:transform"] = list(t20)[:6]
        item["assets"][key]["proj:shape"] = [h // 2, w // 2]
    return {"files": files, "item": item}


def _service(scene_files: dict[str, Any], tile: int) -> AnalysisService:
    files = scene_files["files"]
    item = scene_files["item"]

    def fetch(scene_id: str, collection: str) -> dict:
        return copy.deepcopy(item)

    def read(href: str, bbox: Any, **kwargs: Any) -> Any:
        name = href.rsplit("/", 1)[-1].removesuffix(".tif")
        return read_band_window(str(files[name]), bbox, **kwargs)

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        imagery_hard_max_dimension=tile,
        imagery_tiled_max_dimension=5000,
    )
    return AnalysisService(
        imagery_service=ImageryService(settings, stac_item_fetcher=fetch, band_reader=read)
    )


def _run(service: AnalysisService, **request: Any):
    exec_result = execution([window(client_scene(SCENE_ID))])
    return asyncio.run(service.analyze(AnalysisRequest(execution=exec_result, **request)))


def _named(result) -> dict[str, float]:
    return {m.name: m.value for m in result.measurements}


def test_the_tile_plan_partitions_the_area_exactly() -> None:
    t10, w, h = _local_grid()
    plan = plan_tiles(bbox=AOI, epsg=32644, transform=list(t10)[:6], shape=[h, w], tile_size=TILE)
    assert plan is not None and plan.tiled
    assert sum(t.width * t.height for t in plan.tiles) == plan.width * plan.height
    covered = np.zeros((plan.height, plan.width), dtype=np.int8)
    for tile in plan.tiles:
        rows = slice(tile.row_off, tile.row_off + tile.height)
        cols = slice(tile.col_off, tile.col_off + tile.width)
        covered[rows, cols] += 1
    assert covered.min() == 1 and covered.max() == 1
    assert all(t.width <= TILE and t.height <= TILE for t in plan.tiles)


def test_tiled_indices_equal_one_read(scene_files: dict[str, Any]) -> None:
    tiled = _run(_service(scene_files, TILE), indices=["ndvi", "ndwi", "ndbi"])
    whole = _run(_service(scene_files, 2048), indices=["ndvi", "ndwi", "ndbi"])

    assert {o.name: o.status for o in tiled.analysis_outcomes} == {
        "ndvi": "completed", "ndwi": "completed", "ndbi": "completed",
    }
    a, b = _named(tiled), _named(whole)
    assert a.keys() == b.keys()
    for name, value in b.items():
        if name.endswith(("_count", "_min", "_max")):
            assert a[name] == value, name
        else:
            assert math.isclose(a[name], value, rel_tol=1e-12, abs_tol=1e-12), name

    for q_tiled, q_whole in zip(tiled.pixel_quality, whole.pixel_quality, strict=True):
        assert q_tiled.tile_count > 1 and q_whole.tile_count == 1
        for field in (
            "total_pixels", "valid_pixels", "masked_pixels", "nodata_pixels", "cloud_pixels",
            "cloud_shadow_pixels", "snow_pixels", "other_masked_pixels",
            "unclassified_by_scl_pixels", "grid_width", "grid_height",
        ):
            assert getattr(q_tiled, field) == getattr(q_whole, field), field
        assert q_tiled.scl_class_counts == q_whole.scl_class_counts
        assert q_tiled.quality_notes == q_whole.quality_notes

    assert {g.status for g in tiled.grids} == {"valid"}
    assert [(g.width, g.height) for g in tiled.grids] == [
        (g.width, g.height) for g in whole.grids
    ]
    assert any("native-resolution tiles" in w for w in tiled.warnings)
    assert not any("native-resolution tiles" in w for w in whole.warnings)


def test_a_failing_tile_withholds_the_index_rather_than_reporting_part(
    scene_files: dict[str, Any],
) -> None:
    service = _service(scene_files, TILE)
    reader = service._imagery._read_band  # type: ignore[attr-defined]
    calls = {"n": 0}

    def flaky(href: str, bbox: Any, **kwargs: Any) -> Any:
        if href.endswith("B04.tif"):
            calls["n"] += 1
            if calls["n"] == 3:
                from app.core.errors import UpstreamServiceError

                raise UpstreamServiceError("Failed to read the requested raster window.")
        return reader(href, bbox, **kwargs)

    service._imagery._read_band = flaky  # type: ignore[attr-defined]
    result = _run(service, indices=["ndvi", "ndwi"])
    outcomes = {o.name: o.status for o in result.analysis_outcomes}
    assert outcomes == {"ndvi": "unavailable", "ndwi": "completed"}
    assert "ndvi_mean" not in _named(result)


# --------------------------------------------------------------------------- #
# Sentinel-1 RTC backscatter
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sar_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from tests.test_scene_validation import RTC_ID, rtc_item

    root = tmp_path_factory.mktemp("sar")
    t10, w, h = _local_grid()
    rng = np.random.default_rng(7)
    files: dict[str, Path] = {}
    for pol, scale in (("vv", 0.05), ("vh", 0.01)):
        data = rng.gamma(2.0, scale, (h, w)).astype(np.float32)
        data[rng.random((h, w)) < 0.01] = -32768.0  # nodata
        data[rng.random((h, w)) < 0.002] = 0.0  # zero power: excluded, counted
        files[pol] = root / f"{pol}.tif"
        _write(files[pol], data, t10, -32768.0)
    item = copy.deepcopy(rtc_item())
    item["properties"]["proj:epsg"] = 32644
    item["bbox"] = [79.9, 12.6, 80.9, 13.6]
    for pol in ("vv", "vh"):
        item["assets"][pol]["proj:transform"] = list(t10)[:6]
        item["assets"][pol]["proj:shape"] = [h, w]
    return {"files": files, "item": item, "scene_id": RTC_ID}


def _sar_run(sar_files: dict[str, Any], tile: int, monkeypatch: pytest.MonkeyPatch):
    import app.services.satellite.imagery as imagery_mod
    from app.services.satellite.rtc import RTC_COLLECTION
    from app.services.satellite.schemas import Scene

    monkeypatch.setattr(imagery_mod, "sign_rtc_asset", lambda href, **_: href)
    files, item = sar_files["files"], sar_files["item"]

    def read(href: str, bbox: Any, **kwargs: Any) -> Any:
        return read_band_window(str(files[href.rsplit("/", 1)[-1][:2]]), bbox, **kwargs)

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, imagery_hard_max_dimension=tile, imagery_tiled_max_dimension=5000
    )
    service = AnalysisService(
        imagery_service=ImageryService(
            settings, stac_item_fetcher=lambda *_: copy.deepcopy(item), band_reader=read
        )
    )
    scene = Scene(
        id=sar_files["scene_id"], datetime="2025-01-23T00:32:04Z", bbox=None, geometry=None,
        cloud_cover=None, collection=RTC_COLLECTION, platform="sentinel-1a",
        processing_level="L1", thumbnail_url=None, assets=[],
    )
    exec_result = execution([window(scene, modality="sentinel-1-sar")])
    return asyncio.run(
        service.analyze(AnalysisRequest(execution=exec_result, include_sar_backscatter=True))
    )


def test_tiled_sar_equals_one_read(
    sar_files: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tiled_result = _sar_run(sar_files, TILE, monkeypatch)
    tiled = tiled_result.sar_backscatter
    whole = _sar_run(sar_files, 2048, monkeypatch).sar_backscatter
    assert tiled is not None and whole is not None
    assert any("native-resolution tiles" in w for w in tiled_result.warnings)

    a = {m.name: m.value for m in tiled.measurements}
    b = {m.name: m.value for m in whole.measurements}
    assert a.keys() == b.keys()
    for name, value in b.items():
        if name.endswith(("_count", "_min_db", "_max_db")):
            assert a[name] == value, name
        else:
            assert math.isclose(a[name], value, rel_tol=1e-9, abs_tol=1e-9), name
    for x, y in zip(tiled.polarizations, whole.polarizations, strict=True):
        assert (x.valid_pixel_count, x.nonpositive_pixel_count, x.window_pixel_count) == (
            y.valid_pixel_count, y.nonpositive_pixel_count, y.window_pixel_count,
        )
    assert tiled.difference is not None and whole.difference is not None
    assert tiled.difference.paired_valid_pixel_count == whole.difference.paired_valid_pixel_count
    assert [w for w in tiled.warnings if "tiles" not in w] == whole.warnings


# --------------------------------------------------------------------------- #
# Two dates: both sides' statistics and the paired change
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pair_files(tmp_path_factory: pytest.TempPathFactory, scene_files: dict[str, Any]) -> dict:
    from tests.test_scene_validation import LATER_ID

    root = tmp_path_factory.mktemp("later")
    t10, w, h = _local_grid()
    t20 = Affine(20, 0, t10.c, 0, -20, t10.f)
    rng = np.random.default_rng(99)
    later: dict[str, Path] = {}
    for name in ("B03", "B08"):
        data = rng.integers(1, 6000, (h, w), dtype=np.uint16)
        later[name] = root / f"{name}.tif"
        _write(later[name], data, t10, 0)
    classes = rng.choice(np.array([4, 5, 6, 8], dtype=np.uint8), (h // 2, w // 2))
    later["SCL"] = root / "SCL.tif"
    _write(later["SCL"], classes, t20, 0)
    item = copy.deepcopy(scene_files["item"])
    item["id"] = LATER_ID
    item["properties"]["datetime"] = "2025-01-19T05:15:39.024000Z"
    for key in item["assets"]:
        item["assets"][key]["href"] = item["assets"][key]["href"].replace("/MV/", "/MV-later/")
    return {"later": later, "later_item": item, "later_id": LATER_ID}


def _pair_service(scene_files: dict, pair_files: dict, tile: int) -> AnalysisService:
    earlier_files, later_files = scene_files["files"], pair_files["later"]
    items = {SCENE_ID: scene_files["item"], pair_files["later_id"]: pair_files["later_item"]}

    def read(href: str, bbox: Any, **kwargs: Any) -> Any:
        files = later_files if "/MV-later/" in href else earlier_files
        return read_band_window(str(files[href.rsplit("/", 1)[-1][:-4]]), bbox, **kwargs)

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, imagery_hard_max_dimension=tile, imagery_tiled_max_dimension=5000
    )
    return AnalysisService(
        imagery_service=ImageryService(
            settings, stac_item_fetcher=lambda sid, _: copy.deepcopy(items[sid]), band_reader=read
        )
    )


def _pair_run(service: AnalysisService, later_id: str):
    first = window(client_scene(SCENE_ID), label="baseline", start="2025-01-01", end="2025-01-10")
    second = window(
        client_scene(later_id, datetime="2025-01-19T05:15:39Z"),
        label="target", start="2025-01-11", end="2025-01-31",
    )
    exec_result = execution([first, second], temporal=True)
    return asyncio.run(
        service.analyze(AnalysisRequest(execution=exec_result, include_temporal_ndwi=True))
    )


def _png(b64: str) -> np.ndarray:
    import base64
    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def test_tiled_comparison_equals_one_read(scene_files: dict, pair_files: dict) -> None:
    later = pair_files["later_id"]
    tiled = _pair_run(_pair_service(scene_files, pair_files, TILE), later).temporal_comparison
    whole = _pair_run(_pair_service(scene_files, pair_files, 2048), later).temporal_comparison
    assert tiled is not None and whole is not None

    for a, b in ((tiled.first, whole.first), (tiled.second, whole.second)):
        ma = {m.name: m.value for m in a.measurements}
        mb = {m.name: m.value for m in b.measurements}
        assert ma.keys() == mb.keys()
        for name, value in mb.items():
            assert math.isclose(ma[name], value, rel_tol=1e-12, abs_tol=1e-12), name
        assert a.window_pixel_count == b.window_pixel_count
        assert a.pixel_quality.tile_count > 1  # type: ignore[union-attr]

    assert [d.name for d in tiled.differences] == [d.name for d in whole.differences]
    for x, y in zip(tiled.differences, whole.differences, strict=True):
        assert math.isclose(x.value, y.value, rel_tol=1e-12, abs_tol=1e-12)

    assert tiled.change is not None and whole.change is not None
    assert tiled.change.paired_valid_pixel_count == whole.change.paired_valid_pixel_count
    assert math.isclose(tiled.change.change_mean, whole.change.change_mean, rel_tol=1e-12)
    assert (tiled.change.change_min, tiled.change.change_max) == (
        whole.change.change_min, whole.change.change_max,
    )
    # This area is under the overlay cap, so the map is drawn at full
    # resolution and must be the same image pixel for pixel.
    assert tiled.change.overlay is not None and whole.change.overlay is not None
    np.testing.assert_array_equal(
        _png(tiled.change.overlay.image_base64), _png(whole.change.overlay.image_base64)
    )
    assert tiled.pair_grid is not None and tiled.pair_grid.status == "valid"


def test_a_large_change_map_is_drawn_smaller_but_measured_in_full(
    scene_files: dict, pair_files: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.analysis import tiling

    later = pair_files["later_id"]
    whole = _pair_run(_pair_service(scene_files, pair_files, 2048), later).temporal_comparison
    monkeypatch.setattr(tiling, "OVERLAY_MAX_PX", 100)
    tiled = _pair_run(_pair_service(scene_files, pair_files, TILE), later).temporal_comparison
    assert tiled is not None and whole is not None
    assert tiled.change is not None and whole.change is not None
    overlay = tiled.change.overlay
    assert overlay is not None and max(overlay.width, overlay.height) <= 100
    assert _png(overlay.image_base64).shape[:2] == (overlay.height, overlay.width)
    # The picture is smaller; the measurement is not.
    assert tiled.change.paired_valid_pixel_count == whole.change.paired_valid_pixel_count
    assert overlay.valid_pixel_count == whole.change.paired_valid_pixel_count
    assert overlay.corners_wgs84 == whole.change.corners_wgs84
