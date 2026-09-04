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
import io
from collections.abc import Callable
from datetime import datetime

import numpy as np
from PIL import Image

from app.core.errors import ImageryError
from app.services.analysis.schemas import (
    Measurement,
    NdwiOverlay,
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


def _ndwi_values(green: BandWindow, nir: BandWindow) -> np.ndarray:
    """NDWI over the pixels valid in both bands. Returns a finite 1-D array.

    NDWI = (green - nir) / (green + nir), on raw DN.
    """

    if green.values.shape != nir.values.shape:
        raise ImageryError(
            "Cannot compute NDWI: the green and nir windows have different "
            f"shapes ({green.values.shape} vs {nir.values.shape}). Both bands "
            "must come from the same scene at the same native resolution."
        )

    # Promote to float BEFORE any arithmetic. Subtracting/adding uint16 arrays
    # directly would wrap around (e.g. 3 - 4 -> 65535, 65535 + 1 -> 0).
    g = green.values.astype(np.float64)
    n = nir.values.astype(np.float64)

    # Validity comes from the explicit source masks (nodata == 0 for Sentinel-2
    # spectral bands), never from the derived index: a valid pixel whose
    # numerator happens to be zero is data, not nodata.
    valid = green.valid & nir.valid & np.isfinite(g) & np.isfinite(n)

    denominator = g + n
    valid &= denominator != 0.0

    ndwi = (g[valid] - n[valid]) / denominator[valid]
    return ndwi[np.isfinite(ndwi)]


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

    if green.values.shape != nir.values.shape:
        raise ImageryError(
            "Cannot compute NDWI: the green and nir windows have different "
            f"shapes ({green.values.shape} vs {nir.values.shape}). Both bands "
            "must come from the same scene at the same native resolution."
        )

    g = green.values.astype(np.float64)
    n = nir.values.astype(np.float64)
    valid = green.valid & nir.valid & np.isfinite(g) & np.isfinite(n)

    denominator = g + n
    valid &= denominator != 0.0

    ndwi = np.zeros(g.shape, dtype=np.float64)
    np.divide(g - n, denominator, out=ndwi, where=valid)
    valid &= np.isfinite(ndwi)
    return ndwi, valid


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
