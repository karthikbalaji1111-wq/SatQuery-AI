"""Pure numerical analysis engines.

Everything here is a deterministic function over arrays that were already read
by the imagery layer. No network, no STAC, no COG access, no filesystem, no
FastAPI, no orchestration - only arithmetic. Engines are what
:class:`~app.services.analysis.service.AnalysisService` dispatches to; the
service itself never performs pixel arithmetic.
"""

from __future__ import annotations

import numpy as np

from app.core.errors import ImageryError
from app.services.analysis.schemas import Measurement
from app.services.satellite.raster import BandWindow

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
