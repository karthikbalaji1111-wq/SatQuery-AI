"""Quantitative Sentinel-1 RTC backscatter statistics. Pure arithmetic.

A sibling of :mod:`app.services.analysis.engines` and bound by exactly the same
rules: deterministic functions over arrays the imagery layer already read, no
network, no STAC, no COG access, no filesystem, no orchestration. It lives in
its own module because SAR backscatter is a different physical quantity from a
normalised difference over optical reflectance - different validity rules,
different unit, different failure modes - and mixing the two would let the
decibel conventions below drift into the index code, or the reverse.

WHAT THE PIXELS ARE
-------------------
Collection ``sentinel-1-rtc`` (Microsoft Planetary Computer) publishes ``vv``
and ``vh`` as **linear gamma-naught power**, ``float32``, ``nodata = -32768``,
10 m, radiometrically terrain-corrected BY THE PROVIDER. The provider identifies
its product as gamma naught and uses a 10*log10 conversion in its display
recipe. The scalar computation here follows that product convention. Positive
samples or plausible decibel values alone do not establish power versus
amplitude; neither is used to infer the product's units.

The COG scale/offset and the STAC raster scale/offset are validated at the read
boundary; unsupported encoded units are refused instead of silently converted.

The only conversion is therefore ``dB = 10 * log10(linear power)``. There is no
amplitude-squaring step, because the product is already power, and no amplitude
path exists in this module - adding one would be dead code claiming to handle a
product this system does not read.

THE AVERAGING RULE - the single most important property here
------------------------------------------------------------
Backscatter is averaged in **linear power**, and only the result is converted::

    mean_db = 10 * log10(mean(valid linear power))          # CORRECT
    mean_db = mean(10 * log10(valid linear power))          # WRONG

The second is the *geometric* mean of the power, expressed in decibels. It is
biased low - measured on the window above it gives -12.39 dB where the true
mean backscatter is -4.39 dB, an 8 dB error - and it is not mean backscatter at
all. ``min`` and ``max`` may be converted directly, because ``log10`` is
monotonic and therefore preserves the ordering that selected them.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from app.services.analysis.engines import _grids_are_comparable
from app.services.analysis.schemas import (
    Measurement,
    SarBackscatterResult,
    SarPolarization,
    SarPolarizationDifference,
    SarPolarizationStatistics,
)
from app.services.satellite.raster import BandWindow

#: The unit every decibel value carries. Never "index" - that belongs to the
#: normalised-difference engines and means something else - and never "db" or
#: "decibels": a caller matching on the unit string needs exactly one spelling.
DECIBEL_UNIT = "dB"
_COUNT_UNIT = "pixels"

#: The provider's nodata for both polarizations. Recorded for documentation;
#: validity is taken from ``BandWindow.valid``, which the raster layer already
#: derived from the source's own nodata rather than from a constant here.
RTC_NODATA = -32768.0

#: The two polarizations this engine measures, in report order.
SAR_POLARIZATIONS: tuple[SarPolarization, ...] = ("vv", "vh")

_AVERAGING_NOTE = (
    "Backscatter is averaged in linear gamma-naught power and the mean is then "
    "converted with 10*log10; averaging per-pixel decibel values would give a "
    "geometric mean, which is biased low and is not mean backscatter."
)
_PROVIDER_NOTE = (
    "Values are the provider's radiometrically terrain-corrected gamma naught "
    "(Sentinel-1 RTC); SatQuery performs no calibration, speckle filtering or "
    "terrain correction of its own."
)
_NO_QUALITY_MASK_NOTE = (
    "The Sentinel-1 RTC product exposes no per-pixel quality, coherence or "
    "validity mask beyond nodata, so no quality score is reported. Optical "
    "cloud concepts do not apply to SAR and none were used."
)
_NOT_A_CLASSIFICATION_NOTE = (
    "These are backscatter statistics in decibels, not a land-cover, water, "
    "flood or object classification; no threshold was applied and no class "
    "label was derived."
)


def to_decibels(power: np.ndarray | float) -> np.ndarray:
    """``10 * log10(power)`` - the ONLY conversion this module applies.

    Defined for strictly positive input only. Callers must exclude
    non-positive samples first (see :func:`valid_linear_power`) rather than
    clamping them to a floor, which would fabricate a value that was never
    measured.
    """

    return 10.0 * np.log10(np.asarray(power, dtype=np.float64))


def valid_linear_power(band: BandWindow) -> tuple[np.ndarray, int]:
    """The strictly positive, finite, non-nodata samples, and how many were not.

    Returns ``(values, nonpositive_count)`` where ``values`` is a flat float64
    array of the samples the statistics may be computed from, and
    ``nonpositive_count`` is how many otherwise-valid samples were <= 0 and
    therefore excluded because ``log10`` is undefined on them.

    Three exclusions, all reported rather than repaired:

    * ``nodata`` (-32768 for RTC) and non-finite samples, already marked by the
      raster layer's own validity mask;
    * NaN / +-Inf, re-checked here because a caller may hand over an array whose
      mask was built elsewhere;
    * zero and negative power. A zero-power pixel has no decibel value, and a
      negative one is not power at all. This check is not a unit detector:
      decibel samples may be positive too. Product metadata, rather than
      sample signs, establishes the linear-power convention.
    """

    values = np.asarray(band.values, dtype=np.float64)
    valid = np.asarray(band.valid, dtype=bool) & np.isfinite(values)
    positive = valid & (values > 0.0)
    return values[positive], int(np.count_nonzero(valid & ~positive))


def compute_backscatter_measurements(
    band: BandWindow, *, polarization: SarPolarization
) -> list[Measurement]:
    """Scalar gamma-naught statistics for one polarization, in decibels.

    ``<pol>_valid_pixel_count`` is always present, so "no valid pixel" is
    distinguishable from a computed statistic; when it is zero no other value is
    reported rather than a fabricated one.
    """

    values, _ = valid_linear_power(band)
    count = int(values.size)

    measurements = [
        Measurement(
            name=f"{polarization}_valid_pixel_count",
            value=float(count),
            unit=_COUNT_UNIT,
        )
    ]
    if count == 0:
        return measurements

    # THE AVERAGING RULE. Mean in LINEAR power, convert once, afterwards.
    # Rewriting this as float(to_decibels(values).mean()) would report the
    # geometric mean and would be wrong by several decibels - see the module
    # docstring. min/max convert directly because log10 is monotonic.
    mean_db = float(to_decibels(values.mean()))
    measurements.extend(
        [
            Measurement(name=f"{polarization}_mean_db", value=mean_db, unit=DECIBEL_UNIT),
            Measurement(
                name=f"{polarization}_min_db",
                value=float(to_decibels(values.min())),
                unit=DECIBEL_UNIT,
            ),
            Measurement(
                name=f"{polarization}_max_db",
                value=float(to_decibels(values.max())),
                unit=DECIBEL_UNIT,
            ),
        ]
    )
    return measurements


def compute_polarization_statistics(
    band: BandWindow, *, polarization: SarPolarization
) -> SarPolarizationStatistics:
    """One polarization's measurements plus the provenance of its pixels."""

    values, nonpositive = valid_linear_power(band)
    return SarPolarizationStatistics(
        polarization=polarization,
        measurements=compute_backscatter_measurements(band, polarization=polarization),
        valid_pixel_count=int(values.size),
        nonpositive_pixel_count=nonpositive,
        window_pixel_count=int(band.width) * int(band.height),
        crs=band.crs,
        resolution=band.resolution,
        transform=list(band.transform)[:6],
    )


def compare_polarizations(
    vv: BandWindow, vh: BandWindow
) -> tuple[SarPolarizationDifference | None, list[str]]:
    """VV minus VH in decibels, or ``None`` with the reason it was not computed.

    Grid identity is required and is checked with the SAME function the temporal
    NDWI path uses - reused rather than reimplemented, so the two cannot drift
    apart in what they consider the same ground. Two polarizations of one
    acquisition normally share a grid exactly; when they do not, nothing is
    resampled to force the comparison.

    Statistics are computed over pixels valid in BOTH polarizations, so the
    difference is exactly the difference of the two means reported beside it. A
    pixel measured in only one polarization says nothing about their ratio.
    """

    incomparable = _grids_are_comparable(vv, vh)
    if incomparable is not None:
        return None, [
            "VV and VH were not compared because "
            f"{incomparable.replace('observations', 'polarizations')}. Nothing "
            "was resampled onto a shared grid, so the per-polarization "
            "statistics above remain the only result."
        ]

    vv_values = np.asarray(vv.values, dtype=np.float64)
    vh_values = np.asarray(vh.values, dtype=np.float64)
    paired = (
        np.asarray(vv.valid, dtype=bool)
        & np.asarray(vh.valid, dtype=bool)
        & np.isfinite(vv_values)
        & np.isfinite(vh_values)
        & (vv_values > 0.0)
        & (vh_values > 0.0)
    )
    count = int(np.count_nonzero(paired))
    if count == 0:
        return None, [
            "VV and VH were not compared because no pixel carried a valid, "
            "strictly positive sample in both polarizations."
        ]

    # Linear mean per polarization over the SAME pixels, then one conversion
    # each - the averaging rule again, and the reason the difference below is
    # a ratio of mean backscatter rather than a mean of per-pixel ratios.
    vv_mean_db = float(to_decibels(vv_values[paired].mean()))
    vh_mean_db = float(to_decibels(vh_values[paired].mean()))

    return (
        SarPolarizationDifference(
            vv_mean_db=vv_mean_db,
            vh_mean_db=vh_mean_db,
            vv_minus_vh_mean_db=vv_mean_db - vh_mean_db,
            paired_valid_pixel_count=count,
            crs=vv.crs,
            transform=list(vv.transform)[:6],
        ),
        [],
    )


def _exclusion_warnings(statistics: SarPolarizationStatistics) -> list[str]:
    """State what was left out, in the polarization's own terms."""

    pol = statistics.polarization.upper()
    warnings: list[str] = []
    if statistics.valid_pixel_count == 0:
        warnings.append(
            f"{pol} produced no statistics: no pixel in the window carried a "
            "finite, non-nodata, strictly positive sample. Sentinel-1 RTC "
            "values are linear gamma-naught power, so a band of non-positive "
            "values is not power (data already in decibels would look like "
            "this) and was not converted."
        )
        return warnings
    if statistics.nonpositive_pixel_count:
        warnings.append(
            f"{statistics.nonpositive_pixel_count} {pol} sample(s) were zero or "
            "negative and were excluded, because a decibel value is undefined "
            "for them; they were not clamped or replaced."
        )
    excluded = statistics.window_pixel_count - statistics.valid_pixel_count
    if excluded > 0:
        warnings.append(
            f"{pol} statistics cover {statistics.valid_pixel_count} of the "
            f"{statistics.window_pixel_count} pixel(s) in the requested window; "
            f"{excluded} carried no usable sample."
        )
    return warnings


def compute_sar_backscatter(
    *,
    vv: BandWindow | None,
    vh: BandWindow | None,
    scene_id: str,
    window_label: str,
    acquired_at: datetime | None = None,
    collection: str | None = None,
) -> SarBackscatterResult:
    """Gamma-naught statistics for the polarizations that were read.

    Either polarization may be absent - a read can fail on its own - and what
    was read is still reported. The VV/VH difference needs both, on one grid,
    and is omitted with a reason otherwise.
    """

    statistics: list[SarPolarizationStatistics] = []
    warnings: list[str] = []
    for polarization, band in (("vv", vv), ("vh", vh)):
        if band is None:
            continue
        result = compute_polarization_statistics(band, polarization=polarization)
        statistics.append(result)
        warnings.extend(_exclusion_warnings(result))

    difference: SarPolarizationDifference | None = None
    if vv is not None and vh is not None:
        difference, difference_warnings = compare_polarizations(vv, vh)
        warnings.extend(difference_warnings)
    else:
        missing = "VH" if vv is not None else "VV"
        warnings.append(
            f"No VV/VH difference was computed: {missing} was not read for this "
            "scene."
        )

    measurements: list[Measurement] = []
    seen: set[str] = set()
    for result in statistics:
        for measurement in result.measurements:
            if measurement.name not in seen:
                seen.add(measurement.name)
                measurements.append(measurement)
    if difference is not None:
        measurements.append(
            Measurement(
                name="vv_minus_vh_mean_db",
                value=difference.vv_minus_vh_mean_db,
                unit=DECIBEL_UNIT,
            )
        )

    warnings.extend(
        [_AVERAGING_NOTE, _PROVIDER_NOTE, _NO_QUALITY_MASK_NOTE, _NOT_A_CLASSIFICATION_NOTE]
    )

    return SarBackscatterResult(
        scene_id=scene_id,
        window_label=window_label,
        acquired_at=acquired_at,
        collection=collection,
        polarizations=statistics,
        difference=difference,
        measurements=measurements,
        warnings=warnings,
    )
