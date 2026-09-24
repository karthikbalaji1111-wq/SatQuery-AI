"""Stage 5 of the scientific pipeline: may these grids be combined?

VALID REQUEST -> VALID SCENE -> VALID PIXELS -> VALID RADIOMETRY -> VALID
GEOMETRY -> ONLY THEN ARITHMETIC.

The one authoritative statement of when two rasters describe the same ground
pixel by pixel. Every place that combines rasters asks here:

* a band PAIR (NDVI red/NIR, NDWI green/NIR) must sit on IDENTICAL grids;
* a coarser raster placed on a finer grid (NDBI's 20 m SWIR, the 20 m SCL onto
  the 10 m analysis grid) must be EXACTLY NESTED - same CRS, north-up, an
  integer resolution ratio on each axis, and cell edges that coincide with
  fine-pixel edges (no sub-cell phase shift);
* a paired temporal change, or a VV-VH difference, needs IDENTICAL grids.

A relationship that is not one of these is REFUSED - never shifted, snapped,
cropped, padded, resampled or reprojected into agreement. This module decides;
it moves no pixel. The one transformation the pipeline performs (whole-cell
assignment in ``engines.coregister_to_finer_grid``) is permitted only after
:func:`nesting` has proved the relationship it assumes.

Floating-point tolerance
------------------------
Grid coordinates are float64 read from GeoTIFF headers and advanced by integer
window offsets, so two representations of the same grid can differ in the last
bits. Positions and sizes are therefore compared to ``TOLERANCE_PIXELS`` of a
pixel (1e-6 px: a hundred-thousandth of a millimetre on a 10 m grid). It exists
only to absorb representation noise; a genuine shift - half a pixel is five
hundred thousand times larger - is always a mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rasterio.crs import CRS
from rasterio.errors import CRSError

from app.core.errors import ImageryError
from app.services.analysis.schemas import GridInput, GridState
from app.services.satellite.raster import BandWindow

#: Largest difference, in PIXELS of the relevant grid, treated as float noise.
TOLERANCE_PIXELS = 1e-6

GeometryCode = Literal[
    "grid_crs_unknown",
    "grid_crs_mismatch",
    "grid_orientation_unsupported",
    "grid_resolution_mismatch",
    "grid_ratio_invalid",
    "grid_misaligned",
    "grid_shape_mismatch",
    "grid_no_overlap",
]


class GeometryError(ImageryError):
    """Two grids may not be combined the way the computation needs."""

    def __init__(
        self,
        code: GeometryCode,
        message: str,
        *,
        stage: Literal["pre_read", "post_read"] = "post_read",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason = message
        self.stage = stage


# --------------------------------------------------------------------------- #
# A grid, independent of any pixels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Grid:
    """CRS + north-up affine + dimensions. Immutable; carries no pixels."""

    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int

    @classmethod
    def of(cls, band: BandWindow) -> Grid:
        t = band.transform
        return cls(
            crs=band.crs,
            transform=(float(t.a), float(t.b), float(t.c), float(t.d), float(t.e), float(t.f)),
            width=int(band.width),
            height=int(band.height),
        )

    @property
    def res_x(self) -> float:
        return self.transform[0]

    @property
    def res_y(self) -> float:
        """Positive: the magnitude of the (negative, north-up) row step."""

        return -self.transform[4]

    @property
    def origin(self) -> tuple[float, float]:
        return self.transform[2], self.transform[5]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` in the grid's CRS (north-up only)."""

        x0, y0 = self.origin
        return (x0, y0 - self.height * self.res_y, x0 + self.width * self.res_x, y0)

    @property
    def identity(self) -> str:
        """A deterministic name for exactly this grid."""

        a, b, c, d, e, f = self.transform
        return f"{self.crs}|{a:g},{b:g},{c:.6f},{d:g},{e:g},{f:.6f}|{self.width}x{self.height}"


def _close(a: float, b: float, pixel: float) -> bool:
    return abs(a - b) <= TOLERANCE_PIXELS * abs(pixel)


def _whole(value: float) -> int | None:
    nearest = round(value)
    return int(nearest) if abs(value - nearest) <= TOLERANCE_PIXELS else None


# --------------------------------------------------------------------------- #
# The checks - each returns (code, reason) or None
# --------------------------------------------------------------------------- #

Problem = tuple[GeometryCode, str] | None


def _same_crs(a: str, b: str) -> bool:
    if a.strip().upper() == b.strip().upper():
        return True
    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except CRSError:
        return False


def crs_problem(a: Grid, b: Grid) -> Problem:
    if a.crs is None or b.crs is None:
        return ("grid_crs_unknown",
                "the CRS of at least one grid could not be established")
    if not _same_crs(a.crs, b.crs):
        return ("grid_crs_mismatch",
                f"they are in different coordinate reference systems ({a.crs} vs "
                f"{b.crs}) - different CRSs are never reprojected into agreement")
    return None


def orientation_problem(grid: Grid) -> Problem:
    a, b, _, d, e, _ = grid.transform
    if b != 0.0 or d != 0.0:
        return ("grid_orientation_unsupported",
                "a grid is rotated or sheared; only north-up grids are supported")
    if not (a > 0.0 and e < 0.0):
        return ("grid_orientation_unsupported",
                "a grid is not north-up (positive column step, negative row step)")
    return None


def _overlap(a: Grid, b: Grid) -> bool:
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def same_grid_problem(a: Grid, b: Grid) -> Problem:
    """Why ``a`` and ``b`` are not the same grid, or ``None``.

    The same pixel index must be the same ground: same CRS, north-up, equal
    resolution on each axis, equal dimensions and equal origin.
    """

    problem = crs_problem(a, b) or orientation_problem(a) or orientation_problem(b)
    if problem:
        return problem
    if (a.width, a.height) != (b.width, b.height):
        return ("grid_shape_mismatch",
                f"they cover different pixel dimensions ({a.width}x{a.height} vs "
                f"{b.width}x{b.height})")
    if not (_close(a.res_x, b.res_x, a.res_x) and _close(a.res_y, b.res_y, a.res_y)):
        return ("grid_resolution_mismatch",
                f"they sit on different pixel grids: resolutions differ "
                f"({a.res_x}x{a.res_y} vs {b.res_x}x{b.res_y})")
    if not (_close(a.origin[0], b.origin[0], a.res_x)
            and _close(a.origin[1], b.origin[1], a.res_y)):
        return ("grid_misaligned",
                "they sit on different pixel grids: their affine transforms differ "
                f"(origins {a.origin} vs {b.origin}), so the same pixel index is not "
                "the same ground")
    return None


@dataclass(frozen=True)
class Nesting:
    """How a coarse grid sits on a fine one: whole cells, whole-pixel offset."""

    ratio_x: int
    ratio_y: int
    #: Offset of the coarse origin from the fine origin, in FINE pixels.
    offset_x: int
    offset_y: int


def nesting(coarse: Grid, fine: Grid) -> tuple[Nesting | None, Problem]:
    """Whether every ``coarse`` cell covers a whole block of ``fine`` pixels.

    Requires the same CRS, both north-up, an integer ratio >= 1 on EACH axis
    (checked separately - a wrong y ratio is not hidden by a right x one), and
    coarse cell edges that fall on fine pixel edges: a 20 m grid shifted by
    5 m has the right ratio and the wrong ground, and is refused. The two must
    also overlap. Equal grids are the ratio-1 case.
    """

    problem = crs_problem(coarse, fine) or orientation_problem(coarse) \
        or orientation_problem(fine)
    if problem:
        return None, problem

    ratio_x = _whole(coarse.res_x / fine.res_x)
    ratio_y = _whole(coarse.res_y / fine.res_y)
    if ratio_x is None or ratio_y is None or ratio_x < 1 or ratio_y < 1:
        return None, ("grid_ratio_invalid",
                      "the coarser band's resolution is not a whole multiple of the "
                      f"finer one's on both axes ({coarse.res_x}x{coarse.res_y} m vs "
                      f"{fine.res_x}x{fine.res_y} m); whole-cell assignment is only "
                      "exact on a nested grid")

    offset_x = _whole((coarse.origin[0] - fine.origin[0]) / fine.res_x)
    offset_y = _whole((fine.origin[1] - coarse.origin[1]) / fine.res_y)
    if offset_x is None or offset_y is None:
        return None, ("grid_misaligned",
                      "the coarser grid's cell edges do not fall on the finer grid's "
                      f"pixel edges (origins {coarse.origin} vs {fine.origin}); a "
                      "sub-pixel shift is a different ground position, not a nesting")

    if not _overlap(coarse, fine):
        return None, ("grid_no_overlap",
                      "the coarser band does not cover the finer band's window at all")

    return Nesting(ratio_x, ratio_y, offset_x, offset_y), None


def require(problem: Problem, prefix: str) -> None:
    """Raise :class:`GeometryError` for ``problem``; do nothing for ``None``."""

    if problem is not None:
        code, reason = problem
        raise GeometryError(code, f"{prefix}: {reason}.")


# --------------------------------------------------------------------------- #
# Pre-read: the SOURCE rasters' grids, from catalog metadata (no pixel read)
# --------------------------------------------------------------------------- #


def source_grid(asset: Any, epsg: int | None) -> Grid | None:
    """A source raster's full grid from its catalog ``proj:*`` fields, if published."""

    transform = getattr(asset, "proj_transform", None)
    shape = getattr(asset, "proj_shape", None)
    code = getattr(asset, "proj_epsg", None) or epsg
    if transform is None or shape is None:
        return None
    return Grid(
        crs=f"EPSG:{code}" if code else None,
        transform=tuple(float(v) for v in transform[:6]),  # type: ignore[arg-type]
        width=int(shape[1]),
        height=int(shape[0]),
    )


def source_problem(
    assets: Mapping[str, Any], epsg: int | None, *, scientific: Sequence[str]
) -> tuple[Problem, list[str]]:
    """Whether the source rasters an operation reads can be combined as it will.

    ``scientific`` are the measured bands; every other asset (the categorical
    SCL) is placed onto them. The finest scientific source defines the grid;
    each other source must be identical to it or exactly nested in it.
    Returns ``(problem, notes)``. Unpublished grid metadata is not a problem -
    the same rules are applied to the actual windows after the read.
    """

    grids = {key: source_grid(asset, epsg) for key, asset in assets.items()}
    missing = sorted(k for k, g in grids.items() if g is None)
    notes = (
        [f"The catalog publishes no grid for {', '.join(missing)}; geometry is "
         "validated on the windows actually read."]
        if missing else []
    )
    known = {k: g for k, g in grids.items() if g is not None}
    bases = [known[k] for k in scientific if k in known]
    if not bases:
        return None, notes
    base = min(bases, key=lambda g: g.res_x)
    for key, grid in known.items():
        _, problem = nesting(grid, base)
        if problem is not None:
            code, reason = problem
            return (code, f"the source raster {key!r} cannot be placed on the "
                          f"analysis grid: {reason}"), notes
    return None, notes


# --------------------------------------------------------------------------- #
# Recording the grid a computation used
# --------------------------------------------------------------------------- #


def grid_input(role: str, band: BandWindow, analysis: Grid) -> GridInput:
    grid = Grid.of(band)
    relationship: Literal["analysis_grid", "identical", "nested"]
    placed, _ = nesting(grid, analysis)
    if same_grid_problem(grid, analysis) is None:
        relationship = "identical"
    elif placed is not None:
        relationship = "nested"
    else:  # pragma: no cover - callers record only inputs that passed
        relationship = "identical"
    return GridInput(
        role=role,
        crs=grid.crs,
        width=grid.width,
        height=grid.height,
        resolution_x=grid.res_x,
        resolution_y=grid.res_y,
        relationship=relationship,
        ratio_x=placed.ratio_x if placed else None,
        ratio_y=placed.ratio_y if placed else None,
        offset_x=placed.offset_x if placed else None,
        offset_y=placed.offset_y if placed else None,
    )


def grid_state(
    *,
    analysis: str,
    scene_id: str | None,
    grid: Grid | None,
    inputs: Sequence[GridInput] = (),
    refusal: str | None = None,
    stage: Literal["pre_read", "post_read"] = "post_read",
    notes: Sequence[str] = (),
) -> GridState:
    """The typed record of the grid a computation used, or was refused on."""

    return GridState(
        status="refused" if refusal is not None else "valid",
        analysis=analysis,
        scene_id=scene_id,
        stage=stage,
        crs=grid.crs if grid else None,
        width=grid.width if grid else None,
        height=grid.height if grid else None,
        resolution_x=grid.res_x if grid else None,
        resolution_y=grid.res_y if grid else None,
        transform=list(grid.transform) if grid else None,
        origin=list(grid.origin) if grid else None,
        bounds=list(grid.bounds) if grid else None,
        grid_id=grid.identity if grid else None,
        inputs=list(inputs),
        refusal=refusal,
        notes=list(notes),
    )


def summary(label: str, state: GridState) -> str:
    """One citable line for evidence."""

    if state.status == "refused":
        return f"{label} grid: refused ({state.stage}) - {state.refusal}"
    parts = [
        f"{i.role} {i.relationship}"
        + (f" {i.ratio_x}:1" if i.relationship == "nested" and i.ratio_x else "")
        for i in state.inputs
    ]
    res = (
        f"{state.resolution_x:g} x {state.resolution_y:g} m"
        if state.resolution_x is not None and state.resolution_y is not None
        else "unknown resolution"
    )
    return (
        f"{label} grid: valid - {state.crs}, {res}, "
        f"{state.width}x{state.height} px; {', '.join(parts)}."
    )
