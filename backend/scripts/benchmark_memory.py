"""Peak memory of the heaviest analysis path, at a chosen window size.

    uv run python scripts/benchmark_memory.py --size 1024 --scenes 2   (from backend/)

Writes synthetic Sentinel-2-shaped rasters (10 m green/nir/red, 20 m swir16
and SCL, EPSG:32644, tiled, deflate) to a temporary directory, then runs the
real read -> SCL alignment -> masking -> co-registration -> statistics path
for NDVI, NDWI and NDBI over a ``size`` x ``size`` window, keeping every
scene's bands alive as a two-date comparison does. Prints the peak resident
memory above the post-import baseline.

Local files, no network: this measures what the process holds, not catalog
latency. Run each size in a fresh process; peak RSS never goes down.

Measured 2026-11-26 (Apple M2, Python 3.12, rasterio 1.5.1):

    window      scenes   peak above baseline
    258 px      1         15 MB
    1032 px     1        118 MB
    1032 px     2        132 MB
    2054 px     1        385 MB
    2054 px     2        454 MB

An idle server (app imported, intent model loaded) holds about 104 MB.
"""

from __future__ import annotations

import argparse
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.analysis.engines import (  # noqa: E402
    compute_index_measurements,
    coregister_to_finer_grid,
)
from app.services.analysis.indices import NDBI, NDVI, NDWI  # noqa: E402
from app.services.analysis.pixel_quality import align_scl_to_grid, mask_with_scl  # noqa: E402
from app.services.geospatial import BoundingBox  # noqa: E402
from app.services.satellite.raster import read_band_window  # noqa: E402

ORIGIN = (399960.0, 1500000.0)


def _peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes vs KiB


def _write(path: Path, size: int, res: float, dtype: str, data: np.ndarray) -> None:
    profile = {
        "driver": "GTiff", "width": size, "height": size, "count": 1, "dtype": dtype,
        "crs": "EPSG:32644", "transform": from_origin(*ORIGIN, res, res), "nodata": 0,
        "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def make_rasters(directory: Path, size: int) -> None:
    rng = np.random.default_rng(1)
    full = size + 1024
    for band in ("green", "nir", "red"):
        _write(directory / f"{band}.tif", full, 10, "uint16",
               rng.integers(1, 6000, (full, full), dtype=np.uint16))
    half = full // 2
    _write(directory / "swir16.tif", half, 20, "uint16",
           rng.integers(1, 6000, (half, half), dtype=np.uint16))
    classes = np.array([4, 5, 6, 8, 9], dtype=np.uint8)
    _write(directory / "scl.tif", half, 20, "uint8", rng.choice(classes, (half, half)))


def run(directory: Path, size: int, scenes: int) -> str:
    x0, y0 = ORIGIN[0] + 5000, ORIGIN[1] - 5000
    west, south, east, north = transform_bounds(
        "EPSG:32644", "EPSG:4326", x0, y0 - size * 10, x0 + size * 10, y0
    )
    bbox = BoundingBox(west=west, south=south, east=east, north=north)
    baseline = _peak_mb()
    started = time.perf_counter()
    held = []
    for _ in range(scenes):
        bands = {
            band: read_band_window(
                str(directory / f"{band}.tif"), bbox,
                max_dimension=size + 64, max_window_pixels=50_000_000,
            )
            for band in ("scl", "green", "nir", "red", "swir16")
        }
        for index in (NDVI, NDWI, NDBI):
            high, low = bands[index.high_band], bands[index.low_band]
            if high.resolution != low.resolution:
                high = coregister_to_finer_grid(high, low)
            scl = align_scl_to_grid(bands["scl"], low)
            masked = mask_with_scl(
                high, low, scl, index=index.key, label=index.label,
                scene_id="benchmark", window_label="benchmark",
            )
            compute_index_measurements(index, masked.high, masked.low)
        held.append(bands)
    window = held[-1]["nir"]
    return (
        f"window={window.width}x{window.height} scenes={scenes} "
        f"peak_above_baseline_mb={_peak_mb() - baseline:.0f} "
        f"baseline_mb={baseline:.0f} seconds={time.perf_counter() - started:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=1024, help="window side in 10 m pixels")
    parser.add_argument("--scenes", type=int, default=1, help="1 = one date, 2 = a comparison")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_rasters(directory, args.size)
        print(run(directory, args.size, args.scenes))


if __name__ == "__main__":
    main()
