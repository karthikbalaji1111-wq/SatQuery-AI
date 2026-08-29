"""Bounded (windowed) RGB reads from remote Cloud-Optimized GeoTIFFs.

Uses rasterio / GDAL ``/vsicurl`` HTTP range requests. Only the pixel window
that corresponds to the requested geographic bbox is read - the full source
raster is never downloaded or loaded into memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from app.core.errors import ImageryError, InvalidInputError, UpstreamServiceError
from app.core.logging import get_logger
from app.services.geospatial.schemas import BoundingBox

logger = get_logger("satellite.raster")

_WGS84 = "EPSG:4326"

# GDAL tuning for efficient remote range reads; no credentials, HTTPS only.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
}


@dataclass(frozen=True)
class RgbWindow:
    """Result of a bounded RGB read."""

    array: np.ndarray  # (height, width, 3), uint8
    width: int
    height: int
    crs: str | None
    resolution: float | None
    bands: list[str]
    window: dict[str, int]
    source_shape: list[int]  # [height, width] of the full source raster
    normalization: str


def _open_raster(href: str):
    """Open a remote raster. Isolated for test injection."""

    return rasterio.open(href)


def _read_dataset_window(
    src,
    indexes: int | tuple[int, ...],
    window: Window,
    out_shape: tuple[int, ...],
) -> np.ndarray:
    """Read exactly ``window`` (decimated to ``out_shape``). Isolated so tests
    can assert the reader is given a window and never a full-dataset read.

    ``indexes`` is ``(1, 2, 3)`` for Sentinel-2 RGB or ``1`` for a single-band
    Sentinel-1 backscatter raster."""

    return src.read(indexes=indexes, window=window, out_shape=out_shape, boundless=False)


# --- Sentinel-1 VV display normalization ------------------------------------ #

_SAR_CLIP_PERCENTILES = (2.0, 98.0)
_SAR_NORMALIZATION = (
    "2nd-98th percentile clip -> min-max to 8-bit grayscale (display only, "
    "not calibrated)"
)
# Deterministic level for a degenerate (constant) window: mid-grey, so a
# uniform-return scene is visibly distinct from a no-data (black) one.
_SAR_DEGENERATE_LEVEL = 128


def _normalize_sar_band(band: np.ndarray, *, nodata: float | None) -> np.ndarray:
    """Deterministic *display* normalization of one SAR band to ``uint8``.

    Percentile-clip (2nd..98th) of the finite, non-nodata pixels, then min-max
    scale to 0..255. NaN, +/-Inf and nodata pixels are excluded from the
    statistics and rendered black (0). A constant/degenerate window (``upper``
    <= ``lower``, e.g. one valid pixel) yields a flat mid-grey rather than
    dividing by zero. This is not a calibrated or quantitative transform.

    Raises :class:`ImageryError` when there is no finite, non-nodata pixel.
    """

    values = np.asarray(band, dtype=np.float64)
    finite = np.isfinite(values)
    if nodata is not None and math.isfinite(nodata):
        finite &= values != nodata

    valid = values[finite]
    if valid.size == 0:
        raise ImageryError(
            "The Sentinel-1 window contains no valid (finite) pixels to display."
        )

    lower, upper = (float(x) for x in np.percentile(valid, _SAR_CLIP_PERCENTILES))

    if not (math.isfinite(lower) and math.isfinite(upper)) or upper <= lower:
        out = np.full(values.shape, _SAR_DEGENERATE_LEVEL, dtype=np.uint8)
        out[~finite] = 0
        return out

    scaled = (np.clip(values, lower, upper) - lower) / (upper - lower)
    # Any NaN/Inf that survived clipping (they cannot reach [0, 1]) is forced
    # into range before the uint8 cast so no invalid value ever propagates.
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.rint(scaled * 255.0).astype(np.uint8)
    out[~finite] = 0  # NaN / Inf / nodata -> black
    return out


def _clamp_window_to_source(raw: Window, src_width: int, src_height: int) -> Window:
    col_off = max(0, math.floor(raw.col_off))
    row_off = max(0, math.floor(raw.row_off))
    col_end = min(src_width, math.ceil(raw.col_off + raw.width))
    row_end = min(src_height, math.ceil(raw.row_off + raw.height))
    if col_end <= col_off or row_end <= row_off:
        raise InvalidInputError(
            "The requested bbox does not intersect the selected scene."
        )
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _extract_window(
    src,
    bbox: BoundingBox,
    *,
    max_dimension: int,
    max_window_pixels: int,
) -> RgbWindow:
    band_count = int(src.count)
    if src.crs is None or src.transform is None:
        raise UpstreamServiceError("The remote raster metadata is invalid.")
    # 3+ bands -> RGB; exactly 1 band -> single-band SAR. Anything else
    # (0 or 2 bands) is an unsupported structure.
    if band_count < 3 and band_count != 1:
        raise UpstreamServiceError("The remote raster metadata is invalid.")

    try:
        proj_bounds = transform_bounds(
            _WGS84, src.crs, bbox.west, bbox.south, bbox.east, bbox.north, densify_pts=21
        )
        raw_window = from_bounds(*proj_bounds, transform=src.transform)
    except (ValueError, RasterioIOError) as exc:
        raise ImageryError("Could not map the bbox onto the raster grid.") from exc

    window = _clamp_window_to_source(raw_window, src.width, src.height)

    native_pixels = int(window.width) * int(window.height)
    if native_pixels > max_window_pixels:
        raise InvalidInputError(
            "The requested window exceeds the maximum supported size "
            f"({native_pixels} px > {max_window_pixels} px). Use a smaller bbox."
        )

    scale = min(1.0, max_dimension / max(int(window.width), int(window.height)))
    out_w = max(1, round(int(window.width) * scale))
    out_h = max(1, round(int(window.height) * scale))

    if band_count >= 3:
        try:
            data = _read_dataset_window(src, (1, 2, 3), window, (3, out_h, out_w))
        except (RasterioIOError, MemoryError) as exc:
            raise UpstreamServiceError(
                "Failed to read the requested raster window."
            ) from exc
        if data.dtype != np.uint8:
            # `visual` is uint8 true-colour; anything else is out of scope here
            # and we refuse rather than invent a normalization.
            raise UpstreamServiceError(
                f"Unsupported raster datatype {data.dtype!r}; expected 8-bit RGB."
            )
        rgb = np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
        bands = ["red", "green", "blue"]
        normalization = "none (source is 8-bit RGB)"
    else:
        try:
            band = _read_dataset_window(src, 1, window, (out_h, out_w))
        except (RasterioIOError, MemoryError) as exc:
            raise UpstreamServiceError(
                "Failed to read the requested raster window."
            ) from exc
        gray = _normalize_sar_band(band, nodata=src.nodata)
        rgb = np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))
        bands = ["vv", "vv", "vv"]
        normalization = _SAR_NORMALIZATION

    resolution = abs(float(src.transform.a)) if src.transform.a else None

    return RgbWindow(
        array=rgb,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        crs=src.crs.to_string() if src.crs else None,
        resolution=resolution,
        bands=bands,
        window={
            "col_off": int(window.col_off),
            "row_off": int(window.row_off),
            "width": int(window.width),
            "height": int(window.height),
        },
        source_shape=[int(src.height), int(src.width)],
        normalization=normalization,
    )


def read_rgb_window(
    href: str,
    bbox: BoundingBox,
    *,
    max_dimension: int,
    max_window_pixels: int,
) -> RgbWindow:
    """Open ``href`` remotely and read only the window covering ``bbox``."""

    try:
        with rasterio.Env(**_GDAL_ENV), _open_raster(href) as src:
            result = _extract_window(
                src,
                bbox,
                max_dimension=max_dimension,
                max_window_pixels=max_window_pixels,
            )
    except RasterioIOError as exc:
        logger.warning("Raster open failed for %s: %s", href, exc)
        raise UpstreamServiceError("Could not open the remote raster.") from exc

    logger.info(
        "Read %sx%s window (native %sx%s of %s) from %s",
        result.width,
        result.height,
        result.window["width"],
        result.window["height"],
        result.source_shape,
        href,
    )
    return result
