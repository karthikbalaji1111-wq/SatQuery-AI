"""Pure numerical analysis engines.

Everything here is a deterministic function over data that was already read by
the imagery layer - raster arrays, or (since the temporal comparison below)
``Measurement``s those arrays produced. No network, no STAC, no COG access, no
filesystem, no FastAPI, no orchestration - only arithmetic and the decisions
that arithmetic implies. Engines are what
:class:`~app.services.analysis.service.AnalysisService` dispatches to; the
service itself never performs pixel arithmetic.
"""

from __future__ import annotations

import base64
import dataclasses
import io
from collections.abc import Callable
from datetime import datetime

import numpy as np
from PIL import Image

from app.core.errors import ImageryError
from app.services.analysis.indices import SpectralIndex
from app.services.analysis.schemas import (
    Measurement,
    NdwiOverlay,
    NdwiTemporalChange,
    ObservationIndexResult,
    SpatialMeasurement,
)
from app.services.query.compatibility import CompatibilityReport
from app.services.query.schemas import NdwiThreshold
from app.services.satellite.raster import BandWindow, image_corners_wgs84

# --------------------------------------------------------------------------- #
# THE RAW-DN DECISION - the single documented place this choice is made.
#
# Sentinel-2 L2A STAC metadata advertises ``raster:bands`` ``scale = 0.0001``
# and ``offset = -0.1`` for every spectral band. Live inspection of the actual
# COG pixel values (two scenes, two tiles, two processing baselines) showed the
# stored values do NOT follow that convention: applying the advertised offset
# drove "reflectance" negative for 55% / 100% of the probed windows and produced
# NDVI of +1.300 over SCL-classified vegetation - mathematically impossible for
# non-negative operands. The raw stored values gave the textbook +0.637.
#
# Therefore this phase computes normalized-difference indices on the RAW stored
# DN values and applies NEITHER the advertised scale NOR the advertised offset.
#
# This is exact rather than a shortcut: for a shared multiplicative scale ``s``
# and no offset, (g*s - n*s) / (g*s + n*s) == (g - n) / (g + n) - the scale
# cancels identically in a normalized difference. An additive offset would NOT
# cancel, which is precisely why applying it corrupted the result.
#
# This does NOT establish that the raw values are absolute surface reflectance.
# Any future need for absolute reflectance (rather than a ratio) must resolve
# the scale/offset question first. Revisit this constant if the catalog is
# regenerated or its metadata changes.
# --------------------------------------------------------------------------- #
STAC_SCALE_OFFSET_APPLIED = False

#: Index cut-off used for the reported "percent above" statistic. This is an
#: INDEX THRESHOLD only - it is not a validated water or flood classifier.
NDWI_INDEX_THRESHOLD = 0.3

_INDEX_UNIT = "index"


# --------------------------------------------------------------------------- #
# The shared normalised-difference core
# --------------------------------------------------------------------------- #
#
# NDWI, NDVI and NDBI are the same arithmetic over different band pairs:
#
#     NDWI = (green  - nir) / (green  + nir)
#     NDVI = (nir    - red) / (nir    + red)
#     NDBI = (swir16 - nir) / (swir16 + nir)
#
# So they share one implementation and therefore one set of numerical
# guarantees. The NDWI-named helpers below are kept as thin wrappers rather
# than rewritten call sites: NDWI is the validated path, and it must keep
# behaving exactly as it did.


def _require_matching_band_grids(high: BandWindow, low: BandWindow, label: str) -> None:
    """Equal array shapes alone do not establish the same ground pixels."""

    if high.crs != low.crs or tuple(high.transform)[:6] != tuple(low.transform)[:6]:
        raise ImageryError(
            f"Cannot compute {label}: the band CRS or affine grids differ. "
            "Both bands must describe the same ground pixels."
        )


def _normalised_difference_values(
    high: BandWindow, low: BandWindow, label: str
) -> np.ndarray:
    """``(high - low) / (high + low)`` over pixels valid in both. Finite, 1-D."""

    if high.values.shape != low.values.shape:
        raise ImageryError(
            f"Cannot compute {label}: the two band windows have different "
            f"shapes ({high.values.shape} vs {low.values.shape}). Both bands "
            "must describe the same pixels."
        )

    # Promote to float BEFORE any arithmetic. Subtracting/adding uint16 arrays
    # directly would wrap around (e.g. 3 - 4 -> 65535, 65535 + 1 -> 0).
    _require_matching_band_grids(high, low, label)
    a = high.values.astype(np.float64)
    b = low.values.astype(np.float64)

    # Validity comes from the explicit source masks, never from the derived
    # index: a valid pixel whose numerator happens to be zero is data.
    valid = high.valid & low.valid & np.isfinite(a) & np.isfinite(b)

    denominator = a + b
    valid &= denominator != 0.0

    index = (a[valid] - b[valid]) / denominator[valid]
    return index[np.isfinite(index)]


def _normalised_difference_grid(
    high: BandWindow, low: BandWindow, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """The 2-D sibling of :func:`_normalised_difference_values`."""

    if high.values.shape != low.values.shape:
        raise ImageryError(
            f"Cannot compute {label}: the two band windows have different "
            f"shapes ({high.values.shape} vs {low.values.shape}). Both bands "
            "must describe the same pixels."
        )

    _require_matching_band_grids(high, low, label)
    a = high.values.astype(np.float64)
    b = low.values.astype(np.float64)
    valid = high.valid & low.valid & np.isfinite(a) & np.isfinite(b)

    denominator = a + b
    valid &= denominator != 0.0

    index = np.zeros(a.shape, dtype=np.float64)
    np.divide(a - b, denominator, out=index, where=valid)
    valid &= np.isfinite(index)
    return index, valid


def compute_index_measurements(
    index: SpectralIndex, high: BandWindow, low: BandWindow
) -> list[Measurement]:
    """Scalar statistics for one index on one scene. No arrays or geometry.

    ``<key>_valid_pixel_count`` is always present so a caller can tell computed
    statistics apart from "no valid pixels"; when it is zero, no other
    statistic is reported rather than a fabricated one.
    """

    values = _normalised_difference_values(high, low, index.label)
    count = int(values.size)

    measurements = [
        Measurement(
            name=f"{index.key}_valid_pixel_count",
            value=float(count),
            unit="pixels",
        )
    ]
    if count == 0:
        return measurements

    measurements.extend(
        [
            Measurement(
                name=f"{index.key}_mean", value=float(values.mean()), unit=_INDEX_UNIT
            ),
            Measurement(
                name=f"{index.key}_min", value=float(values.min()), unit=_INDEX_UNIT
            ),
            Measurement(
                name=f"{index.key}_max", value=float(values.max()), unit=_INDEX_UNIT
            ),
        ]
    )
    return measurements


def _ndwi_values(green: BandWindow, nir: BandWindow) -> np.ndarray:
    """NDWI over the pixels valid in both bands. Returns a finite 1-D array.

    NDWI = (green - nir) / (green + nir), on raw DN. A thin wrapper over the
    shared normalised-difference core, so NDWI, NDVI and NDBI cannot drift
    apart in their handling of nodata, zero denominators or non-finite samples.
    """

    return _normalised_difference_values(green, nir, "NDWI")


#: The index range the colour ramp spans. NDWI is bounded to [-1, 1] by
#: construction, so the ramp is fixed rather than stretched per scene: two
#: overlays are then directly comparable, and a legend means the same thing
#: every time.
NDWI_DISPLAY_RANGE = (-1.0, 1.0)


def _ndwi_grid(green: BandWindow, nir: BandWindow) -> tuple[np.ndarray, np.ndarray]:
    """NDWI as a 2-D grid plus its validity mask, on the bands' own pixels.

    The grid sibling of :func:`_ndwi_values`, which flattens and drops invalid
    pixels because statistics have no geometry. A picture does: every pixel
    needs a position, so the shape is preserved and validity travels beside it.
    """

    return _normalised_difference_grid(green, nir, "NDWI")


def _ndwi_rgba(ndwi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Colour the index. Dry (low) -> warm sand, wet (high) -> deep blue.

    A single continuous ramp over the fixed [-1, 1] range, so the colour IS the
    value and no threshold is implied - this renders NDWI, it does not classify
    water. Invalid pixels get alpha 0 rather than a colour, because "not
    measured" must not look like a measurement.
    """

    low, high = NDWI_DISPLAY_RANGE
    t = np.clip((ndwi - low) / (high - low), 0.0, 1.0)

    rgba = np.zeros((*ndwi.shape, 4), dtype=np.uint8)
    rgba[..., 0] = ((1.0 - t) * 214).astype(np.uint8)  # red falls away
    rgba[..., 1] = (60 + (1.0 - t) * 120).astype(np.uint8)  # green dips mid-ramp
    rgba[..., 2] = (40 + t * 215).astype(np.uint8)  # blue rises with the index
    rgba[..., 3] = np.where(valid, 200, 0).astype(np.uint8)
    return rgba


#: Exact comparisons, chosen by the operator alone. No tolerance is applied:
#: a pixel whose index equals the threshold matches ``gte``/``lte`` and not
#: ``gt``/``lt``, which is what "above 0.3" means in plain reading.
_COMPARISONS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "gt": lambda values, threshold: values > threshold,
    "gte": lambda values, threshold: values >= threshold,
    "lt": lambda values, threshold: values < threshold,
    "lte": lambda values, threshold: values <= threshold,
}


# --------------------------------------------------------------------------- #
# Cross-resolution co-registration
# --------------------------------------------------------------------------- #
#
# NDBI needs SWIR against NIR, and Sentinel-2 delivers SWIR at 20 m against
# NIR's 10 m. The two cannot be subtracted as they stand, and the project's
# standing rule is that nothing may be silently resampled to make an invalid
# comparison look valid.
#
# What makes this case tractable is that the grids are not merely similar, they
# are NESTED. Verified from the live COG headers (2026-09, tile 44PMV):
#
#     nir     10980 x 10980   res 10 m   origin (399960, 1500000)  EPSG:32644
#     swir16   5490 x 5490    res 20 m   origin (399960, 1500000)  EPSG:32644
#
# Identical CRS, identical origin, and 10980 = 2 x 5490 exactly - so every 20 m
# cell covers exactly four 10 m cells with no sub-pixel phase offset.
#
# On such a grid, assigning each fine pixel the value of the coarse cell that
# CONTAINS ITS CENTRE invents nothing: no averaging, no interpolation, no new
# values: every number in the output already existed in the source. That is the
# whole reason this direction was chosen over the alternative of averaging NIR
# down to 20 m, which would synthesise values that were never measured.
#
# It does NOT create detail. The SWIR contribution is still 20 m, so NDBI is
# sampled at 10 m and RESOLVED at 20 m. That is reported, not glossed
# (`SpectralIndex.limiting_resolution_m`).
#
# The mapping is computed through the two affine transforms rather than by
# assuming a factor, so a pair that is not actually nested produces
# out-of-bounds parents and is refused instead of quietly mismatched.


def coregister_to_finer_grid(
    coarse: BandWindow, fine: BandWindow
) -> BandWindow:
    """Place ``coarse`` on ``fine``'s grid by whole-cell assignment.

    Each fine pixel takes the value of the coarse cell containing its centre.
    No value is averaged, interpolated or invented, and the result carries
    ``fine``'s geometry because that is the grid it now sits on.

    Raises :class:`ImageryError` when the two windows cannot be related exactly
    - different CRS, a non-integer resolution ratio, or a fine pixel with no
    coarse parent. Refusing is the correct outcome: a resampled comparison that
    looks fine is worse than one that is declined.
    """

    if coarse.crs != fine.crs:
        raise ImageryError(
            "Cannot align the bands: they are in different coordinate "
            f"reference systems ({coarse.crs} vs {fine.crs})."
        )
    if coarse.resolution is None or fine.resolution is None:
        raise ImageryError(
            "Cannot align the bands: the resolution of at least one is unknown."
        )
    ratio = coarse.resolution / fine.resolution
    if ratio < 1.0 or abs(ratio - round(ratio)) > 1e-9:
        raise ImageryError(
            "Cannot align the bands: the coarser band's resolution is not a "
            f"whole multiple of the finer one's ({coarse.resolution} m vs "
            f"{fine.resolution} m). Whole-cell assignment is only exact on a "
            "nested grid."
        )

    fine_t = fine.transform
    coarse_t = coarse.transform

    # Centre of every fine pixel, in the shared CRS.
    rows = np.arange(fine.height, dtype=np.float64)[:, None] + 0.5
    cols = np.arange(fine.width, dtype=np.float64)[None, :] + 0.5
    xs = fine_t.c + cols * fine_t.a + rows * fine_t.b
    ys = fine_t.f + cols * fine_t.d + rows * fine_t.e

    # The coarse cell containing each of those points. Inverting the transform
    # rather than dividing by the ratio keeps this correct even when the two
    # windows were clamped to different offsets on their own sources.
    inverse = ~coarse_t
    parent_cols = np.floor(inverse.a * xs + inverse.b * ys + inverse.c)
    parent_rows = np.floor(inverse.d * xs + inverse.e * ys + inverse.f)

    inside = (
        (parent_cols >= 0)
        & (parent_cols < coarse.width)
        & (parent_rows >= 0)
        & (parent_rows < coarse.height)
    )
    if not inside.any():
        raise ImageryError(
            "Cannot align the bands: the coarser band does not cover the "
            "finer band's window at all."
        )

    safe_rows = np.where(inside, parent_rows, 0).astype(np.int64)
    safe_cols = np.where(inside, parent_cols, 0).astype(np.int64)

    values = coarse.values[safe_rows, safe_cols]
    # A fine pixel with no parent is not measured, so it is not valid - it is
    # never filled with a substitute.
    valid = coarse.valid[safe_rows, safe_cols] & inside

    return dataclasses.replace(
        fine,
        values=values,
        valid=valid,
        # The geometry is now the fine grid's, but the numbers are the coarse
        # band's. `resolution` stays the grid's spacing; what the index can
        # actually resolve is reported separately by the index definition.
    )


def compute_ndwi_threshold_measurement(
    green: BandWindow,
    nir: BandWindow,
    *,
    threshold: NdwiThreshold,
    scene_id: str,
    window_label: str,
    acquired_at: datetime | None,
) -> SpatialMeasurement | None:
    """Count the analysed pixels that satisfy ``threshold``.

    Computed from the same two band windows the statistics and the overlay come
    from, so all three describe identical pixels. The arithmetic is entirely
    here: nothing about this number originates in a language model, which may
    only have supplied the threshold being asked about.

    The denominator is the VALID pixel count - nodata, non-finite and
    zero-denominator pixels are excluded from both the numerator and the
    denominator, because a pixel that was never measured cannot be evidence
    either way.

    Returns ``None`` when there is no valid pixel at all: no denominator means
    no percentage, and 0% would assert that the area was measured and found
    empty. A grid with no usable CRS still returns the count, without a
    footprint - the number is real even when its position is not establishable.
    """

    ndwi, valid = _ndwi_grid(green, nir)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return None

    compare = _COMPARISONS[threshold.operator]
    matching = int(np.count_nonzero(compare(ndwi, threshold.value) & valid))

    try:
        corners = image_corners_wgs84(
            green.transform,
            width=green.width,
            height=green.height,
            crs=green.crs,
        )
        crs = green.crs
    except ImageryError:
        corners, crs = None, None

    return SpatialMeasurement(
        operator=threshold.operator,
        threshold=threshold.value,
        matching_pixel_count=matching,
        valid_pixel_count=valid_count,
        percentage=matching / valid_count * 100.0,
        scene_id=scene_id,
        acquired_at=acquired_at,
        window_label=window_label,
        crs=crs,
        corners_wgs84=corners,
    )


def render_ndwi_overlay(
    green: BandWindow,
    nir: BandWindow,
    *,
    scene_id: str,
    window_label: str,
) -> NdwiOverlay | None:
    """The NDWI grid as a georeferenced PNG, or ``None`` when it cannot be one.

    Computed from the SAME band windows the statistics come from, so the
    picture and the numbers describe identical pixels, and positioned by those
    windows' own affine - never by the requested bbox.

    Returns ``None`` rather than raising when there is nothing honest to draw:
    no valid pixel, or a grid whose footprint cannot be established (no CRS, or
    a rotated transform). A missing overlay costs a picture; a misplaced one
    would assert measurements about ground that was never read.
    """

    ndwi, valid = _ndwi_grid(green, nir)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return None

    try:
        corners = image_corners_wgs84(
            green.transform,
            width=green.width,
            height=green.height,
            crs=green.crs,
        )
    except ImageryError:
        return None
    if green.crs is None:  # pragma: no cover - image_corners_wgs84 already refuses
        return None

    rgba = _ndwi_rgba(ndwi, valid)
    try:
        buffer = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    except (ValueError, TypeError, OSError) as exc:
        raise ImageryError("Failed to encode the NDWI overlay as PNG.") from exc

    measured = ndwi[valid]
    return NdwiOverlay(
        scene_id=scene_id,
        window_label=window_label,
        image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        width=int(green.width),
        height=int(green.height),
        crs=green.crs,
        transform=list(green.transform)[:6],
        corners_wgs84=corners,
        value_min=float(measured.min()),
        value_max=float(measured.max()),
        valid_pixel_count=count,
    )


def compute_ndwi_measurements(
    green: BandWindow, nir: BandWindow
) -> list[Measurement]:
    """Scalar NDWI statistics for one scene. No arrays, masks or geometry.

    ``ndwi_valid_pixel_count`` is always present so a caller can tell computed
    statistics apart from "no valid pixels"; when it is zero, no other
    statistic is reported rather than a fabricated one.
    """

    ndwi = _ndwi_values(green, nir)
    count = int(ndwi.size)

    measurements = [
        Measurement(
            name="ndwi_valid_pixel_count", value=float(count), unit="pixels"
        )
    ]
    if count == 0:
        return measurements

    above = float(np.count_nonzero(ndwi > NDWI_INDEX_THRESHOLD)) / count * 100.0
    measurements.extend(
        [
            Measurement(name="ndwi_mean", value=float(ndwi.mean()), unit=_INDEX_UNIT),
            Measurement(name="ndwi_min", value=float(ndwi.min()), unit=_INDEX_UNIT),
            Measurement(name="ndwi_max", value=float(ndwi.max()), unit=_INDEX_UNIT),
            Measurement(
                # An index threshold - deliberately not named as water or flood.
                name=(
                    "ndwi_percent_above_index_threshold_"
                    f"{NDWI_INDEX_THRESHOLD}"
                ),
                value=above,
                unit="%",
            ),
        ]
    )
    return measurements


# --------------------------------------------------------------------------- #
# Temporal NDWI Statistics
#
# Two observations, each already indexed on its OWN pixels, summarised side by
# side. The single derived value is
#
#     mean_ndwi_difference = second.ndwi_mean - first.ndwi_mean
#
# a difference between two aggregate statistics. No pixel is compared against
# another pixel; no grid is aligned; nothing is resampled. Where that framing
# would mislead - the footprints do not overlap, one side has no valid pixels,
# or both sides resolved to the same scene - the value is suppressed rather
# than reported with a caveat, because a number a reader can see is a number a
# reader will use.
# --------------------------------------------------------------------------- #

#: The change range the diverging ramp spans. NDWI is bounded to [-1, 1], so a
#: difference is bounded to [-2, 2]; the ramp is clipped to a symmetric [-1, 1]
#: because that is where real change lives and a symmetric range is what keeps
#: zero at the exact centre. Fixed rather than stretched per scene, so the same
#: colour means the same change in every overlay.
NDWI_CHANGE_DISPLAY_RANGE = (-1.0, 1.0)


def _grids_are_comparable(baseline: BandWindow, target: BandWindow) -> str | None:
    """Why these two grids may NOT be subtracted, or ``None`` if they may.

    A shared requested bbox proves nothing: each observation was reprojected
    from its own scene and floor/ceil clamped onto its own source grid, so two
    reads over the same AOI routinely differ in size or origin. Subtracting
    them anyway would paint change wherever the grids disagree - a border of
    pure artefact around every image.

    The affine is compared exactly. A ten-metre shift is a different pixel over
    different ground, and no tolerance would make it the same one.
    """

    if baseline.crs is None or target.crs is None:
        return "the CRS of at least one observation could not be established"
    if baseline.crs != target.crs:
        return f"the observations are in different CRSs ({baseline.crs} vs {target.crs})"
    if (baseline.width, baseline.height) != (target.width, target.height):
        return (
            "the observations cover different pixel dimensions "
            f"({baseline.width}x{baseline.height} vs {target.width}x{target.height})"
        )
    if tuple(baseline.transform)[:6] != tuple(target.transform)[:6]:
        return (
            "the observations sit on different pixel grids: their affine "
            "transforms differ, so the same pixel index is not the same ground"
        )
    return None


def _change_rgba(change: np.ndarray, paired: np.ndarray) -> np.ndarray:
    """Colour the difference. Falling index -> red, no change -> neutral grey,
    rising index -> blue.

    Diverging and centred on zero, so "no change" is visually inert and the two
    directions are distinguishable at a glance. No threshold is encoded: the
    ramp is continuous, and nothing here marks a value as water gained or lost.
    Unpaired pixels are alpha 0 - "not comparable" must not look like "no
    change", which is exactly what a neutral colour would say.
    """

    low, high = NDWI_CHANGE_DISPLAY_RANGE
    t = np.clip((change - low) / (high - low), 0.0, 1.0)  # 0 -> fall, 0.5 -> none

    rgba = np.zeros((*change.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (215 - t * 175).astype(np.uint8)  # red fades as the index rises
    rgba[..., 1] = (120 - np.abs(t - 0.5) * 120).astype(np.uint8)  # grey at centre
    rgba[..., 2] = (40 + t * 175).astype(np.uint8)  # blue grows as the index rises
    rgba[..., 3] = np.where(paired, 200, 0).astype(np.uint8)
    return rgba


def compute_ndwi_temporal_change(
    *,
    first_green: BandWindow,
    first_nir: BandWindow,
    second_green: BandWindow,
    second_nir: BandWindow,
    first_scene_id: str,
    second_scene_id: str,
    first_acquired_at: datetime | None,
    second_acquired_at: datetime | None,
    window_label: str,
) -> NdwiTemporalChange | None:
    """``second_NDWI - first_NDWI`` over pixels valid in both, or ``None``.

    *first* is the earlier acquisition and *second* the later one, matching
    the pairing order - not the requested baseline/target roles, which may
    be inverted relative to time.

    Computed from the four band windows the temporal statistics already read -
    no second retrieval path, and no extra read.

    Returns ``None``, never a partial or approximated answer, when the two
    grids are not verifiably identical, when no pixel is valid in both, or when
    the shared grid has no derivable footprint. Nothing is resampled to force a
    comparison: co-registration is out of scope, so an incompatible pair is
    reported as incomparable rather than compared badly.
    """

    if _grids_are_comparable(first_green, second_green) is not None:
        return None

    first_ndwi, first_valid = _ndwi_grid(first_green, first_nir)
    second_ndwi, second_valid = _ndwi_grid(second_green, second_nir)

    paired = first_valid & second_valid
    paired_count = int(np.count_nonzero(paired))
    if paired_count == 0:
        return None

    try:
        corners = image_corners_wgs84(
            second_green.transform,
            width=second_green.width,
            height=second_green.height,
            crs=second_green.crs,
        )
    except ImageryError:
        return None
    if second_green.crs is None:  # pragma: no cover - refused above
        return None

    change = np.zeros(first_ndwi.shape, dtype=np.float64)
    np.subtract(second_ndwi, first_ndwi, out=change, where=paired)
    measured = change[paired]

    overlay: NdwiOverlay | None = None
    try:
        buffer = io.BytesIO()
        Image.fromarray(_change_rgba(change, paired), mode="RGBA").save(
            buffer, format="PNG"
        )
        overlay = NdwiOverlay(
            scene_id=second_scene_id,
            window_label=window_label,
            image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
            width=int(second_green.width),
            height=int(second_green.height),
            crs=second_green.crs,
            transform=list(second_green.transform)[:6],
            corners_wgs84=corners,
            value_min=float(measured.min()),
            value_max=float(measured.max()),
            valid_pixel_count=paired_count,
        )
    except (ValueError, TypeError, OSError) as exc:  # pragma: no cover
        raise ImageryError("Failed to encode the NDWI change overlay.") from exc

    return NdwiTemporalChange(
        first_scene_id=first_scene_id,
        second_scene_id=second_scene_id,
        first_acquired_at=first_acquired_at,
        second_acquired_at=second_acquired_at,
        window_label=window_label,
        paired_valid_pixel_count=paired_count,
        change_mean=float(measured.mean()),
        change_min=float(measured.min()),
        change_max=float(measured.max()),
        crs=second_green.crs,
        transform=list(second_green.transform)[:6],
        corners_wgs84=corners,
        overlay=overlay,
    )


#: The one derived measurement this engine may emit.
MEAN_NDWI_DIFFERENCE = "mean_ndwi_difference"

#: Above this reported scene cloud cover the statistics get an explicit warning.
#: The index is never cloud-masked, so this is context, not a correction.
HIGH_CLOUD_COVER_PERCENT = 30.0

_MEAN_NAME = "ndwi_mean"
_COUNT_NAME = "ndwi_valid_pixel_count"

_WARN_AGGREGATE = (
    "mean_ndwi_difference is the difference between two independently computed "
    "aggregate statistics, each summarising its own set of pixels. No pixels "
    "were compared against one another, and the value describes the statistics "
    "only."
)
_WARN_NO_MEAN = (
    "At least one observation has no valid pixels and therefore no mean NDWI, "
    "so no difference was computed."
)
_WARN_NO_OVERLAP = (
    "The two scene footprints do not overlap, so the two sets of statistics "
    "describe separate areas and no difference was computed."
)
_WARN_PARTIAL_OVERLAP = (
    "The two scene footprints overlap only partially, so the two sets of "
    "statistics summarise different sets of pixels."
)


def _coverage_warning(
    first: ObservationIndexResult, second: ObservationIndexResult
) -> list[str]:
    """State how much of the AOI each observation actually contributed.

    A scene footprint is not AOI coverage. Each quantitative read is clamped to
    its own scene and masked by its own nodata, so two observations over the
    SAME requested bbox can analyse very different numbers of pixels - even when
    the footprints report ``bbox_overlap == "full"``.

    This states the measured coverage and stops there. No threshold is applied
    and no suppression is triggered: the repository has no scientifically
    defensible basis for a "materially different coverage" cut-off, and
    inventing one would replace an honest report with a fabricated judgement.
    """

    parts = []
    for observation in (first, second):
        window = observation.window_pixel_count
        valid = _named(observation.measurements).get(_COUNT_NAME)
        if window is None or valid is None or window <= 0:
            return []
        parts.append(
            f"{observation.window_label!r} analysed {int(valid)} valid of "
            f"{window} AOI pixels ({valid / window * 100.0:.1f}%)"
        )

    return [
        "Equal AOI coverage is NOT established: "
        + "; ".join(parts)
        + ". Scene footprint overlap does not establish equal coverage, and "
        "the two means summarise different samples."
    ]


def _grid_warning(
    first: ObservationIndexResult, second: ObservationIndexResult
) -> list[str]:
    """Report the CRS and resolution the reads ACTUALLY used, when known.

    Independent of the metadata-only compatibility report, which sees only
    ``ImageryResponse`` and therefore reports ``"unknown"`` whenever bounded
    display imagery was not retrieved.
    """

    if first.crs is None or second.crs is None:
        return []
    if first.crs == second.crs and first.resolution == second.resolution:
        return [
            f"Both observations were read in {first.crs} at "
            f"{first.resolution} m/px."
        ]
    return [
        "The two observations were read on different grids "
        f"({first.crs} at {first.resolution} m/px versus {second.crs} at "
        f"{second.resolution} m/px); nothing was reprojected or aligned, so "
        "each statistic summarises its own grid."
    ]


def _named(measurements: list[Measurement]) -> dict[str, float]:
    return {m.name: m.value for m in measurements}


def _cloud_warnings(observation: ObservationIndexResult) -> list[str]:
    cover = observation.cloud_cover
    label = observation.window_label
    if cover is None:
        return [
            f"Observation {label!r} reports no cloud cover metadata, so cloud "
            "contamination is unknown; the index is not cloud-masked."
        ]
    if cover > HIGH_CLOUD_COVER_PERCENT:
        return [
            f"Observation {label!r} reports {cover}% cloud cover; the index is "
            "not cloud-masked, so its statistics may reflect cloud rather than "
            "ground."
        ]
    return []


def _sample_warnings(observation: ObservationIndexResult) -> list[str]:
    count = _named(observation.measurements).get(_COUNT_NAME)
    if count is not None and 0 < count <= 1:
        return [
            f"Observation {observation.window_label!r} has only {int(count)} "
            "valid pixel(s), so its statistics are not meaningful."
        ]
    return []


def compare_ndwi_observations(
    *,
    first: ObservationIndexResult,
    second: ObservationIndexResult,
    compatibility: CompatibilityReport,
) -> tuple[list[Measurement], list[str]]:
    """Difference of two aggregate NDWI means, plus what qualifies it.

    Returns ``(differences, warnings)``. ``differences`` holds at most one
    measurement and is empty whenever the comparison would misinform. Inputs are
    never mutated, and the output is deterministic: warnings are emitted in a
    fixed order.
    """

    warnings = [_WARN_AGGREGATE]

    first_mean = _named(first.measurements).get(_MEAN_NAME)
    second_mean = _named(second.measurements).get(_MEAN_NAME)

    # Suppression - at most one reason, checked in a fixed order.
    suppressed: str | None = None
    if first_mean is None or second_mean is None:
        suppressed = _WARN_NO_MEAN
    elif compatibility.bbox_overlap == "none":
        suppressed = _WARN_NO_OVERLAP
    elif first.scene_id == second.scene_id:
        suppressed = (
            f"Both observations resolved to the same scene {first.scene_id!r}, "
            "so a difference would be trivially zero and none was computed."
        )

    if suppressed is not None:
        warnings.append(suppressed)

    if compatibility.bbox_overlap == "partial":
        warnings.append(_WARN_PARTIAL_OVERLAP)

    warnings.extend(_coverage_warning(first, second))
    warnings.extend(_grid_warning(first, second))

    for observation in (first, second):
        warnings.extend(_cloud_warnings(observation))
    for observation in (first, second):
        warnings.extend(_sample_warnings(observation))

    if suppressed is not None:
        return [], warnings

    # Both means are known here; mypy cannot see it through the branch above.
    assert first_mean is not None and second_mean is not None
    difference = Measurement(
        name=MEAN_NDWI_DIFFERENCE,
        value=float(second_mean - first_mean),
        unit=_INDEX_UNIT,
    )
    return [difference], warnings
