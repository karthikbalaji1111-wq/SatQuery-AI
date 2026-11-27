"""Measuring an area larger than one read: native-resolution tiles, merged exactly.

One read of a large area at 10 m would hold every band of every pixel at once.
Measured (``scripts/benchmark_memory.py``): about 132 MB for a 1024 px window,
454 MB for 2048 px - so on a small instance the area had to stay small. This
module lets the analyses cover a much larger area without ever downsampling:

* :func:`plan_tiles` splits the area's native window into tiles of at most
  ``tile_size`` px, from the catalog's own grid metadata, before any read. The
  tiles PARTITION the window: adjacent tiles share exact pixel edges, so every
  pixel is read once and none is skipped (the raster layer snaps tile bounds
  to pixel edges - ``raster._window_from_native_bounds``).
* Each tile is read, validated, masked and measured by the SAME code that
  measures an area read at once. Nothing here computes an index.
* The tile results are merged EXACTLY: pixel counts add; a mean is the
  pixel-weighted mean of the tile means (the same value as one pass, up to
  floating-point rounding); a minimum or maximum is the extreme of the
  extremes. A decibel mean is merged in linear power, never as decibels.

What the merge does not do: invent a value for a pixel, resample, or treat a
partially measured area as complete. If any tile of an analysis fails, that
analysis is reported as not computed - a statistic over part of the area must
not be presented as one over all of it.

Memory is that of one tile; time grows with the number of tiles.
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image
from rasterio.crs import CRSError
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine

from app.core.errors import ImageryError, InvalidInputError
from app.services.analysis import geometry
from app.services.analysis.engines import change_rgba, ndwi_change_grid
from app.services.analysis.sar import DECIBEL_UNIT, to_decibels
from app.services.analysis.schemas import (
    GridInput,
    GridState,
    Measurement,
    NdwiOverlay,
    NdwiTemporalChange,
    SarPolarizationDifference,
    SarPolarizationStatistics,
)
from app.services.geospatial.schemas import BoundingBox
from app.services.satellite.raster import BandWindow, aoi_window, image_corners_wgs84

#: The change map of a tiled comparison is drawn at most this many pixels on
#: its long side. It is a DISPLAY: the statistics always use every pixel.
OVERLAY_MAX_PX = 1024


@dataclass(frozen=True)
class Tile:
    """One tile: native-CRS bounds, and its place within the area's window."""

    bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    col_off: int
    row_off: int
    width: int
    height: int


@dataclass(frozen=True)
class TilePlan:
    """The area's native window on one raster grid, and its tiles."""

    crs: str
    #: The affine of the WHOLE area window (not of the source raster).
    transform: Affine
    width: int
    height: int
    tiles: tuple[Tile, ...]
    tile_size: int

    @property
    def tiled(self) -> bool:
        return len(self.tiles) > 1

    @property
    def grid(self) -> geometry.Grid:
        t = self.transform
        return geometry.Grid(
            crs=self.crs,
            transform=(float(t.a), float(t.b), float(t.c), float(t.d), float(t.e), float(t.f)),
            width=self.width,
            height=self.height,
        )

    def corners_wgs84(self) -> list[list[float]]:
        return image_corners_wgs84(
            self.transform, width=self.width, height=self.height, crs=self.crs
        )


def plan_tiles(
    *,
    bbox: BoundingBox,
    epsg: int | None,
    transform: Sequence[float] | None,
    shape: Sequence[int] | None,
    tile_size: int,
) -> TilePlan | None:
    """Split the area's native window into tiles, or ``None`` if it cannot be.

    ``epsg``/``transform``/``shape`` are the asset's ``proj:*`` metadata from
    the catalog (``shape`` is rows, columns). Without all three the window
    cannot be computed before a read, and ``None`` sends the caller down the
    one-read path, whose own size limit then applies.
    """

    if epsg is None or transform is None or shape is None or len(transform) < 6:
        return None
    if tile_size < 2:
        raise ValueError("tile_size must be at least 2")
    crs = f"EPSG:{int(epsg)}"
    source = Affine(*[float(v) for v in list(transform)[:6]])
    if source.b != 0.0 or source.d != 0.0 or source.a <= 0.0 or source.e >= 0.0:
        return None  # not north-up: the reader refuses it anyway
    try:
        window = aoi_window(
            bbox, crs=crs, transform=source, width=int(shape[1]), height=int(shape[0])
        )
    except (ValueError, RasterioIOError, CRSError, ImageryError, InvalidInputError):
        # An area that cannot be mapped onto this grid is the reader's to
        # report, with its own message, on the one-read path.
        return None

    col0, row0 = int(window.col_off), int(window.row_off)
    width, height = int(window.width), int(window.height)
    area = source * Affine.translation(col0, row0)

    tiles: list[Tile] = []
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            left = area.c + col * area.a
            right = area.c + (col + w) * area.a
            top = area.f + row * area.e
            bottom = area.f + (row + h) * area.e
            tiles.append(
                Tile(
                    bounds=(left, bottom, right, top),
                    col_off=col,
                    row_off=row,
                    width=w,
                    height=h,
                )
            )
    return TilePlan(
        crs=crs, transform=area, width=width, height=height, tiles=tuple(tiles),
        tile_size=tile_size,
    )


def require_tile_read(plan: TilePlan, tile: Tile, band: BandWindow) -> None:
    """Refuse a tile whose read is not exactly the planned pixels.

    The merge is exact only if the tiles partition the area. A finer-grid band
    must come back as exactly the planned tile; anything else is refused
    rather than merged.
    """

    t = band.transform
    expected = plan.transform * Affine.translation(tile.col_off, tile.row_off)
    same_origin = (
        math.isclose(t.c, expected.c, abs_tol=1e-6 * abs(t.a or 1.0))
        and math.isclose(t.f, expected.f, abs_tol=1e-6 * abs(t.e or 1.0))
    )
    if (band.width, band.height) != (tile.width, tile.height) or not same_origin:
        raise ImageryError(
            "A tile read did not return the planned pixels "
            f"({band.width}x{band.height} at ({t.c}, {t.f}); planned "
            f"{tile.width}x{tile.height} at ({expected.c}, {expected.f})), so the "
            "area cannot be assembled exactly."
        )


# --------------------------------------------------------------------------- #
# Merging statistics
# --------------------------------------------------------------------------- #


def merge_index_measurements(
    key: str, parts: Sequence[Sequence[Measurement]]
) -> list[Measurement]:
    """``<key>_valid_pixel_count/_mean/_min/_max`` (and any percentage) merged.

    A mean or a percentage of valid pixels is weighted by each tile's valid
    count; a tile with no valid pixel contributes its count and nothing else,
    exactly as one read reports no statistic for an empty set.
    """

    if len(parts) == 1:
        return list(parts[0])
    count_name = f"{key}_valid_pixel_count"
    order: list[str] = []
    units: dict[str, str] = {}
    for part in parts:
        for m in part:
            if m.name not in units:
                order.append(m.name)
                units[m.name] = m.unit
    counts = [
        next((m.value for m in part if m.name == count_name), 0.0) for part in parts
    ]
    total = float(sum(counts))
    merged: list[Measurement] = [Measurement(name=count_name, value=total, unit="pixels")]
    if total == 0:
        return merged
    for name in order:
        if name == count_name:
            continue
        pairs = [
            (m.value, n)
            for part, n in zip(parts, counts, strict=True)
            for m in part
            if m.name == name
        ]
        if name.endswith("_min"):
            value = min(v for v, _ in pairs)
        elif name.endswith("_max"):
            value = max(v for v, _ in pairs)
        else:  # a mean, or a percentage of valid pixels: weight by pixels
            value = sum(v * n for v, n in pairs) / sum(n for _, n in pairs)
        merged.append(Measurement(name=name, value=float(value), unit=units[name]))
    return merged


def merge_grid_state(
    parts: Sequence[GridState], plan: TilePlan, *, analysis: str, scene_id: str | None
) -> GridState:
    """The grid record of the whole area, built from its tiles' records.

    The analysis grid is the area's own window. Each input keeps the
    relationship every tile established for it (identical, or nested with its
    ratio) and states its extent over the whole area.
    """

    if len(parts) == 1 and not plan.tiled:
        return parts[0]
    first = parts[0]
    grid = plan.grid
    inputs: list[GridInput] = []
    for item in first.inputs:
        if item.relationship == "nested" and item.ratio_x and item.ratio_y:
            width = math.ceil(grid.width / item.ratio_x)
            height = math.ceil(grid.height / item.ratio_y)
        else:
            width, height = grid.width, grid.height
        inputs.append(item.model_copy(update={"width": width, "height": height}))
    return geometry.grid_state(
        analysis=analysis,
        scene_id=scene_id,
        grid=grid,
        inputs=inputs,
        notes=[
            *first.notes,
            f"Measured in {len(plan.tiles)} native-resolution tiles of at most "
            f"{plan.tile_size} px; every tile was read and checked on its own "
            "grid, and every pixel of the area was used exactly once.",
        ],
    )


def merge_polarization(
    parts: Sequence[SarPolarizationStatistics], plan: TilePlan
) -> SarPolarizationStatistics:
    """One polarization's statistics over the whole area. Means in LINEAR power."""

    first = parts[0]
    pol = first.polarization
    valid = sum(p.valid_pixel_count for p in parts)
    measurements = [
        Measurement(name=f"{pol}_valid_pixel_count", value=float(valid), unit="pixels")
    ]
    if valid:

        def value(part: SarPolarizationStatistics, suffix: str) -> float | None:
            return next(
                (m.value for m in part.measurements if m.name == f"{pol}_{suffix}"), None
            )

        weighted = [
            (10.0 ** (value(p, "mean_db") / 10.0), p.valid_pixel_count)  # type: ignore[operator]
            for p in parts
            if p.valid_pixel_count
        ]
        linear_mean = sum(v * n for v, n in weighted) / sum(n for _, n in weighted)
        mins = [value(p, "min_db") for p in parts if p.valid_pixel_count]
        maxs = [value(p, "max_db") for p in parts if p.valid_pixel_count]
        measurements.extend(
            [
                Measurement(
                    name=f"{pol}_mean_db",
                    value=float(to_decibels(linear_mean)),
                    unit=DECIBEL_UNIT,
                ),
                Measurement(name=f"{pol}_min_db", value=float(min(mins)), unit=DECIBEL_UNIT),  # type: ignore[type-var]
                Measurement(name=f"{pol}_max_db", value=float(max(maxs)), unit=DECIBEL_UNIT),  # type: ignore[type-var]
            ]
        )
    return first.model_copy(
        update={
            "measurements": measurements,
            "valid_pixel_count": valid,
            "nonpositive_pixel_count": sum(p.nonpositive_pixel_count for p in parts),
            "window_pixel_count": plan.width * plan.height,
            "transform": list(plan.transform)[:6],
        }
    )


def merge_difference(
    parts: Sequence[SarPolarizationDifference], plan: TilePlan
) -> SarPolarizationDifference | None:
    """VV - VH over every paired pixel of the area, means in linear power."""

    usable = [p for p in parts if p.paired_valid_pixel_count]
    if not usable:
        return None
    total = sum(p.paired_valid_pixel_count for p in usable)

    def linear(db: float) -> float:
        return 10.0 ** (db / 10.0)

    vv = sum(linear(p.vv_mean_db) * p.paired_valid_pixel_count for p in usable) / total
    vh = sum(linear(p.vh_mean_db) * p.paired_valid_pixel_count for p in usable) / total
    vv_db, vh_db = float(to_decibels(vv)), float(to_decibels(vh))
    return usable[0].model_copy(
        update={
            "vv_mean_db": vv_db,
            "vh_mean_db": vh_db,
            "vv_minus_vh_mean_db": vv_db - vh_db,
            "paired_valid_pixel_count": total,
            "transform": list(plan.transform)[:6],
        }
    )


# --------------------------------------------------------------------------- #
# The paired NDWI change of a tiled comparison
# --------------------------------------------------------------------------- #


@dataclass
class ChangeAccumulator:
    """``second - first`` NDWI over a tiled area, and its reduced-size map.

    Statistics use every paired pixel of every tile. The map samples one
    pixel in ``stride`` along each axis, at fixed positions of the whole area,
    so neighbouring tiles join without seams; it is a picture, not a measure.
    """

    plan: TilePlan
    index: str = "ndwi"
    label: str = "NDWI"
    stride: int = 1
    count: int = 0
    total: float = 0.0
    low: float = math.inf
    high: float = -math.inf
    canvas: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.stride = max(1, math.ceil(max(self.plan.width, self.plan.height) / OVERLAY_MAX_PX))
        rows = math.ceil(self.plan.height / self.stride)
        cols = math.ceil(self.plan.width / self.stride)
        self.canvas = np.zeros((rows, cols, 4), dtype=np.uint8)

    def add(
        self,
        tile: Tile,
        *,
        first_green: BandWindow,
        first_nir: BandWindow,
        second_green: BandWindow,
        second_nir: BandWindow,
    ) -> None:
        change, paired = ndwi_change_grid(
            first_green=first_green,
            first_nir=first_nir,
            second_green=second_green,
            second_nir=second_nir,
            label=self.label,
        )
        measured = change[paired]
        if measured.size:
            self.count += int(measured.size)
            self.total += float(measured.sum())
            self.low = min(self.low, float(measured.min()))
            self.high = max(self.high, float(measured.max()))
        s = self.stride
        r0 = (-tile.row_off) % s
        c0 = (-tile.col_off) % s
        sampled = change_rgba(change[r0::s, c0::s], paired[r0::s, c0::s])
        top, left = (tile.row_off + r0) // s, (tile.col_off + c0) // s
        assert self.canvas is not None
        h, w = sampled.shape[:2]
        self.canvas[top : top + h, left : left + w] = sampled

    def result(
        self,
        *,
        first_scene_id: str,
        second_scene_id: str,
        first_acquired_at,
        second_acquired_at,
        window_label: str,
    ) -> NdwiTemporalChange | None:
        if self.count == 0:
            return None
        corners = self.plan.corners_wgs84()
        buffer = io.BytesIO()
        assert self.canvas is not None
        Image.fromarray(self.canvas, mode="RGBA").save(buffer, format="PNG")
        overlay_transform = self.plan.transform * Affine.scale(self.stride)
        overlay = NdwiOverlay(
            scene_id=second_scene_id,
            window_label=window_label,
            image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
            width=int(self.canvas.shape[1]),
            height=int(self.canvas.shape[0]),
            crs=self.plan.crs,
            transform=list(overlay_transform)[:6],
            corners_wgs84=corners,
            value_min=self.low,
            value_max=self.high,
            valid_pixel_count=self.count,
        )
        return NdwiTemporalChange(
            index=self.index,  # type: ignore[arg-type]
            first_scene_id=first_scene_id,
            second_scene_id=second_scene_id,
            first_acquired_at=first_acquired_at,
            second_acquired_at=second_acquired_at,
            window_label=window_label,
            paired_valid_pixel_count=self.count,
            change_mean=self.total / self.count,
            change_min=self.low,
            change_max=self.high,
            crs=self.plan.crs,
            transform=list(self.plan.transform)[:6],
            corners_wgs84=corners,
            overlay=overlay,
        )


def same_area_grid(first: TilePlan, second: TilePlan) -> bool:
    """Whether two observations' area windows are the same pixels on one grid."""

    return geometry.same_grid_problem(first.grid, second.grid) is None


__all__ = [
    "OVERLAY_MAX_PX",
    "ChangeAccumulator",
    "Tile",
    "TilePlan",
    "merge_difference",
    "merge_grid_state",
    "merge_index_measurements",
    "merge_polarization",
    "plan_tiles",
    "require_tile_read",
    "same_area_grid",
]
