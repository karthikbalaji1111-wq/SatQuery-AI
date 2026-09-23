"""Structural invariants of 20 m -> 10 m coregistration.

`coregister_to_finer_grid` places SWIR (20 m) onto the NIR/red/green grid
(10 m) by whole-cell assignment, so NDBI can be computed at all. Shape equality
is NOT sufficient evidence that it did so correctly: a transposed, flipped or
translated array has exactly the right shape and entirely the wrong geography,
and the resulting index would be plausible, self-consistent and wrong.

These tests therefore assert the mapping by COORDINATE. Each coarse cell is
given a unique value, so after coregistration every fine pixel can be checked
against the coarse cell that geographically contains its centre - computed here
from the affines directly, not from the function under test.

The mutations these are written to catch, all verified to fail them:
affine translation, x/y swap, transpose, wrong resolution, wrong band mapping.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from app.core.errors import ImageryError
from app.services.analysis.engines import coregister_to_finer_grid
from app.services.satellite.raster import BandWindow

ORIGIN_X, ORIGIN_Y = 399960.0, 1500000.0
CRS = "EPSG:32644"


def _window(
    values: np.ndarray, resolution: float, *, origin: tuple[float, float] | None = None
) -> BandWindow:
    ox, oy = origin or (ORIGIN_X, ORIGIN_Y)
    height, width = values.shape
    return BandWindow(
        values=values,
        valid=np.ones(values.shape, dtype=bool),
        width=width,
        height=height,
        crs=CRS,
        transform=Affine(resolution, 0.0, ox, 0.0, -resolution, oy),
        resolution=resolution,
        nodata=None,
        window={"col_off": 0, "row_off": 0, "width": width, "height": height},
        source_shape=[height, width],
    )


def _distinct(height: int, width: int) -> np.ndarray:
    """Every cell unique, so a misplacement cannot coincide with the truth."""

    return (np.arange(height * width, dtype=np.float64) + 1).reshape(height, width)


def _coarse_and_fine(factor: int = 2, coarse_shape: tuple[int, int] = (3, 4)):
    ch, cw = coarse_shape
    coarse = _window(_distinct(ch, cw), 20.0)
    fine = _window(np.zeros((ch * factor, cw * factor)), 20.0 / factor)
    return coarse, fine


# --------------------------------------------------------------------------- #
# The mapping is geographic, not positional.
# --------------------------------------------------------------------------- #


def test_each_fine_pixel_takes_the_coarse_cell_containing_its_centre() -> None:
    coarse, fine = _coarse_and_fine()

    out = coregister_to_finer_grid(coarse, fine)

    assert out.values.shape == fine.values.shape
    for row in range(fine.height):
        for col in range(fine.width):
            # Expected parent computed from the affines here, independently.
            x = fine.transform.c + (col + 0.5) * fine.transform.a
            y = fine.transform.f + (row + 0.5) * fine.transform.e
            inv = ~coarse.transform
            pcol = int(np.floor(inv.a * x + inv.b * y + inv.c))
            prow = int(np.floor(inv.d * x + inv.e * y + inv.f))
            assert out.values[row, col] == coarse.values[prow, pcol], (row, col)


def test_no_value_is_invented_only_repeated() -> None:
    """Whole-cell assignment: every output value existed in the input."""

    coarse, fine = _coarse_and_fine()

    out = coregister_to_finer_grid(coarse, fine)

    assert set(np.unique(out.values)).issubset(set(np.unique(coarse.values)))


def test_a_2x_factor_repeats_each_cell_exactly_four_times() -> None:
    coarse, fine = _coarse_and_fine(factor=2)

    out = coregister_to_finer_grid(coarse, fine)

    counts = np.unique(out.values, return_counts=True)[1]
    assert set(counts.tolist()) == {4}


def test_the_result_carries_the_fine_grid_geometry() -> None:
    coarse, fine = _coarse_and_fine()

    out = coregister_to_finer_grid(coarse, fine)

    assert out.crs == fine.crs
    assert out.transform == fine.transform
    assert out.resolution == fine.resolution
    assert (out.height, out.width) == (fine.height, fine.width)


def test_orientation_is_preserved_top_left_stays_top_left() -> None:
    """The single cheapest check that catches a transpose or a flip."""

    coarse, fine = _coarse_and_fine()

    out = coregister_to_finer_grid(coarse, fine)

    assert out.values[0, 0] == coarse.values[0, 0]
    assert out.values[-1, -1] == coarse.values[-1, -1]
    assert out.values[0, -1] == coarse.values[0, -1]
    assert out.values[-1, 0] == coarse.values[-1, 0]


def test_a_non_square_window_is_not_transposed() -> None:
    """A square fixture would let a transpose pass unnoticed."""

    coarse, fine = _coarse_and_fine(coarse_shape=(2, 5))

    out = coregister_to_finer_grid(coarse, fine)

    assert (out.height, out.width) == (4, 10)
    assert out.values[0, 0] == coarse.values[0, 0]
    assert out.values[0, -1] == coarse.values[0, -1]


def test_a_translated_coarse_grid_changes_which_parent_each_pixel_gets() -> None:
    """Pins that the mapping uses the affines rather than the array indices.

    Shift the coarse origin by one whole coarse cell and every fine pixel must
    take a different parent. An implementation that divided indices by the ratio
    would return the identical array and pass every shape-based assertion.
    """

    coarse, fine = _coarse_and_fine()
    shifted = _window(
        coarse.values, 20.0, origin=(ORIGIN_X + 20.0, ORIGIN_Y)
    )

    base = coregister_to_finer_grid(coarse, fine)
    moved = coregister_to_finer_grid(shifted, fine)

    assert not np.array_equal(base.values, moved.values)


# --------------------------------------------------------------------------- #
# What it refuses, rather than approximates.
# --------------------------------------------------------------------------- #


def test_a_different_crs_is_refused() -> None:
    coarse, fine = _coarse_and_fine()
    other = BandWindow(**{**coarse.__dict__, "crs": "EPSG:32643"})

    with pytest.raises(ImageryError, match="coordinate"):
        coregister_to_finer_grid(other, fine)


def test_a_non_integer_resolution_ratio_is_refused() -> None:
    """15 m onto 10 m is 1.5 cells: no whole-cell assignment exists."""

    coarse = _window(_distinct(3, 4), 15.0)
    fine = _window(np.zeros((6, 8)), 10.0)

    with pytest.raises(ImageryError, match="whole multiple"):
        coregister_to_finer_grid(coarse, fine)


def test_an_unknown_resolution_is_refused() -> None:
    coarse, fine = _coarse_and_fine()
    unknown = BandWindow(**{**coarse.__dict__, "resolution": None})

    with pytest.raises(ImageryError, match="resolution"):
        coregister_to_finer_grid(unknown, fine)


def test_a_non_overlapping_coarse_window_is_refused() -> None:
    coarse = _window(_distinct(3, 4), 20.0, origin=(900000.0, 1500000.0))
    fine = _window(np.zeros((6, 8)), 10.0)

    with pytest.raises(ImageryError, match="does not cover"):
        coregister_to_finer_grid(coarse, fine)


def test_partially_covered_pixels_are_invalid_never_filled() -> None:
    """Half-covered window: the uncovered half is masked, not substituted."""

    coarse = _window(_distinct(3, 2), 20.0)  # 40 m wide
    fine = _window(np.zeros((6, 8)), 10.0)  # 80 m wide

    out = coregister_to_finer_grid(coarse, fine)

    assert out.valid[:, :4].all()
    assert not out.valid[:, 4:].any()
