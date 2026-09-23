"""Stage 3 of the scientific pipeline: which pixels may be measured?

SCENE VALID IS NOT PIXEL VALID. Stage 2 establishes that a scene and its assets
are fit to read; a fit scene can still be clouded, shadowed or snow-covered
pixel by pixel. This module places the Sentinel-2 Scene Classification Layer
(SCL) on the final analysis grid, counts every pixel into exactly one quality
category, and returns the band pair with only the usable pixels left valid -
BEFORE any mean, minimum, maximum, count, threshold, overlay or temporal
difference is computed. Nothing downstream can see a masked pixel, because it
is masked in the same ``BandWindow.valid`` every statistic already honours.

It MEASURES quality. It does not decide whether a result is acceptable: no
threshold is applied, and a scene with one clear pixel is reported as exactly
that. Pure: no I/O, no network.

The SCL encoding - and where it comes from
------------------------------------------
Neither catalog publishes the class table (Earth Search ``sentinel-2-l2a`` and
``sentinel-2-c1-l2a``, Planetary Computer ``sentinel-2-l2a``: no
``classification:classes``, checked 2026-09-23). The table below is ESA's, from
the Sentinel-2 processing documentation (SentiWiki, "S2 Processing", Level-2A
Scene Classification, read 2026-09-23). Processing baseline 05.11 renamed class
2 from DARK_FEATURES to CAST_SHADOWS; its value and its treatment here are
unchanged. The catalog confirms the rest of the encoding: ``uint8``, nodata 0,
20 m, on the same origin as the 10 m bands.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from app.core.errors import ImageryError
from app.services.analysis.engines import coregister_to_finer_grid, pair_validity
from app.services.analysis.schemas import Measurement, PixelQuality, QualityCategory
from app.services.satellite.raster import BandWindow

SCL_SOURCE = (
    "ESA SentiWiki, Sentinel-2 Processing, Level-2A Scene Classification "
    "(https://sentiwiki.copernicus.eu/web/s2-processing), read 2026-09-23"
)


class SclClass(IntEnum):
    """The documented Sentinel-2 L2A Scene Classification values."""

    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    CAST_SHADOWS = 2  # DARK_FEATURES before processing baseline 05.11
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW_OR_ICE = 11


#: The ONE mapping from SCL class to quality category. ``None`` means usable.
#:
#: Usable are the three classes that assert a clear surface observation:
#: vegetation, not-vegetated and water. Cast shadows (terrain shadow, formerly
#: "dark features") and unclassified pixels are excluded as ``other_masked``:
#: Sen2Cor assigned them no surface class, so nothing establishes that their
#: radiometry describes the ground. Thin cirrus is cloud.
SCL_CATEGORY: Mapping[int, QualityCategory | None] = {
    SclClass.NO_DATA: "nodata",
    SclClass.SATURATED_OR_DEFECTIVE: "saturated_or_defective",
    SclClass.CAST_SHADOWS: "other_masked",
    SclClass.CLOUD_SHADOWS: "cloud_shadow",
    SclClass.VEGETATION: None,
    SclClass.NOT_VEGETATED: None,
    SclClass.WATER: None,
    SclClass.UNCLASSIFIED: "other_masked",
    SclClass.CLOUD_MEDIUM_PROBABILITY: "cloud",
    SclClass.CLOUD_HIGH_PROBABILITY: "cloud",
    SclClass.THIN_CIRRUS: "cloud",
    SclClass.SNOW_OR_ICE: "snow",
}

USABLE_SCL_CLASSES: tuple[int, ...] = tuple(
    int(value) for value, category in SCL_CATEGORY.items() if category is None
)

#: The order in which a masked pixel is attributed to ONE category. Only the
#: first applies, so no pixel is counted twice. ``nodata`` comes first because
#: a pixel with no band data, or no classification, was never observed at all -
#: whatever else might be said of it. After that the SCL class decides, and each
#: class maps to exactly one category, so the remaining order only fixes the
#: order of reporting.
QUALITY_PRECEDENCE: tuple[QualityCategory, ...] = (
    "nodata",
    "saturated_or_defective",
    "cloud",
    "cloud_shadow",
    "snow",
    "unknown_class",
    "other_masked",
)

_LABELS: Mapping[QualityCategory, str] = {
    "nodata": "no data",
    "saturated_or_defective": "saturated or defective",
    "cloud": "cloud",
    "cloud_shadow": "cloud shadow",
    "snow": "snow or ice",
    "unknown_class": "undocumented SCL class",
    "other_masked": "other (cast shadow or unclassified)",
}


# --------------------------------------------------------------------------- #
# The SCL on the analysis grid
# --------------------------------------------------------------------------- #


def align_scl_to_grid(scl: BandWindow, grid: BandWindow) -> BandWindow:
    """Place the SCL on ``grid`` without ever blending two class labels.

    Reuses the band co-registration NDBI's SWIR already goes through: each grid
    pixel takes the value of the SCL cell containing its centre - nearest
    neighbour on a nested grid, which is the only resampling a categorical
    layer admits. Averaging classes 4 and 8 would yield 6 - "water" - out of
    vegetation and cloud. A grid pixel with no SCL cell is left without a
    class, never given one.
    """

    if not np.issubdtype(scl.values.dtype, np.integer):
        raise ImageryError(
            "The scene classification layer is not integer-valued, so it is not "
            "a class map; no pixel quality can be established from it."
        )
    if (
        scl.crs == grid.crs
        and scl.values.shape == grid.values.shape
        and tuple(scl.transform)[:6] == tuple(grid.transform)[:6]
    ):
        # Already on this grid (aligned once, reused across indices).
        return scl
    return coregister_to_finer_grid(scl, grid)


# --------------------------------------------------------------------------- #
# Assessment + masking
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MaskedPair:
    """A band pair with only usable pixels left valid, and the accounting for it."""

    high: BandWindow
    low: BandWindow
    quality: PixelQuality


def _fraction(part: int, total: int) -> float | None:
    return part / total if total > 0 else None


def _notes(
    counts: Mapping[QualityCategory, int],
    *,
    total: int,
    valid: int,
    unknown_values: list[int],
    unclassified_by_scl: int,
    metadata_status: str,
) -> list[str]:
    """Plain statements of what was measured, in a fixed order. No judgement."""

    notes: list[str] = []
    masked = total - valid
    if total == 0:
        notes.append("The analysis grid has no pixels, so no fraction is reported.")
        return notes
    if valid == 0:
        notes.append(
            f"No pixel was usable: all {total} pixels of the analysis grid were "
            "masked, so no index statistic was computed."
        )
    if masked > 0:
        parts = ", ".join(
            f"{_LABELS[c]} {counts[c]}" for c in QUALITY_PRECEDENCE if counts[c]
        )
        notes.append(
            f"{masked} of {total} pixels were excluded before any statistic "
            f"was computed ({parts})."
        )
    if unclassified_by_scl:
        notes.append(
            f"{unclassified_by_scl} pixels had no value in the scene "
            "classification layer and were excluded as no data."
        )
    if unknown_values:
        notes.append(
            "The scene classification layer contains values outside the "
            f"documented SCL classes 0-11 ({', '.join(map(str, unknown_values))}); "
            f"those {counts['unknown_class']} pixels were excluded, never treated "
            "as clear."
        )
    if metadata_status == "unknown":
        notes.append(
            "The catalog publishes no raster metadata for the SCL asset, so its "
            "encoding was taken from the documented SCL classes rather than "
            "confirmed from the scene."
        )
    elif metadata_status == "not_validated":
        notes.append(
            "The SCL asset was not validated against its catalog item, so its "
            "encoding was taken from the documented SCL classes."
        )
    return notes


def mask_with_scl(
    high: BandWindow,
    low: BandWindow,
    scl: BandWindow,
    *,
    index: str,
    label: str,
    scene_id: str,
    window_label: str,
    scl_metadata_status: str = "not_validated",
) -> MaskedPair:
    """Assess pixel quality on the pair's grid and mask the pair with it.

    ``high`` and ``low`` must already share one grid - the final analysis grid,
    after any SWIR co-registration. The SCL is aligned onto that grid here. A
    pixel stays valid only if both bands carry defined data there AND its SCL
    class is a usable surface class. Masked pixels are not zeroed and not set
    to NaN: they are marked invalid in ``BandWindow.valid``, the mask every
    engine already reads, and their values are left as read.
    """

    band_ok = pair_validity(high, low, label)
    aligned = align_scl_to_grid(scl, high)
    # Kept in the SCL's own integer dtype (uint8 as published): a wider copy of
    # a full window would cost memory and add nothing.
    classes = aligned.values
    has_class = aligned.valid

    total = int(band_ok.size)
    remaining = np.ones(band_ok.shape, dtype=bool)
    counts: dict[QualityCategory, int] = {}

    # 1. nodata: no band data, no SCL value at the pixel, or SCL NO_DATA.
    nodata = ~band_ok | ~has_class | (classes == SclClass.NO_DATA)
    counts["nodata"] = int(np.count_nonzero(nodata))
    remaining &= ~nodata

    documented = np.isin(classes, np.fromiter(SCL_CATEGORY, dtype=np.int64))
    by_category: dict[QualityCategory, np.ndarray] = {
        category: np.isin(
            classes,
            np.array([v for v, c in SCL_CATEGORY.items() if c == category], dtype=np.int64),
        )
        for category in QUALITY_PRECEDENCE
        if category not in ("nodata", "unknown_class")
    }
    by_category["unknown_class"] = ~documented

    # 2..7: in precedence order, each pixel attributed once.
    for category in QUALITY_PRECEDENCE[1:]:
        hit = remaining & by_category[category]
        counts[category] = int(np.count_nonzero(hit))
        remaining &= ~hit

    usable = remaining & np.isin(classes, np.array(USABLE_SCL_CLASSES, dtype=np.int64))
    valid = int(np.count_nonzero(usable))
    masked = total - valid

    present = classes[has_class]
    values, frequencies = np.unique(present, return_counts=True)
    class_counts = {str(int(v)): int(n) for v, n in zip(values, frequencies, strict=True)}
    unknown_values = sorted(int(v) for v in values if int(v) not in SCL_CATEGORY)
    unclassified_by_scl = int(np.count_nonzero(band_ok & ~has_class))

    quality = PixelQuality(
        index=index,
        scene_id=scene_id,
        window_label=window_label,
        usable_scl_classes=list(USABLE_SCL_CLASSES),
        scl_metadata_status=scl_metadata_status,  # type: ignore[arg-type]
        grid_width=int(high.width),
        grid_height=int(high.height),
        grid_crs=high.crs,
        grid_resolution=high.resolution,
        total_pixels=total,
        valid_pixels=valid,
        masked_pixels=masked,
        nodata_pixels=counts["nodata"],
        saturated_or_defective_pixels=counts["saturated_or_defective"],
        cloud_pixels=counts["cloud"],
        cloud_shadow_pixels=counts["cloud_shadow"],
        snow_pixels=counts["snow"],
        unknown_class_pixels=counts["unknown_class"],
        other_masked_pixels=counts["other_masked"],
        valid_fraction=_fraction(valid, total),
        contamination_fraction=_fraction(masked, total),
        scl_class_counts=class_counts,
        unknown_scl_values=unknown_values,
        quality_notes=_notes(
            counts,
            total=total,
            valid=valid,
            unknown_values=unknown_values,
            unclassified_by_scl=unclassified_by_scl,
            metadata_status=scl_metadata_status,
        ),
    )
    return MaskedPair(
        high=dataclasses.replace(high, valid=high.valid & usable),
        low=dataclasses.replace(low, valid=low.valid & usable),
        quality=quality,
    )


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def quality_measurements(quality: PixelQuality) -> list[Measurement]:
    """The quality record as citable measurements, named under its index.

    ``<index>_valid_pixel_count`` is already the engine's own count of the
    pixels the statistic used, and equals ``valid_pixels``; it is not repeated.
    Percentages are omitted, not zeroed, when the grid has no pixels.
    """

    key = quality.index
    counts = [
        ("quality_total_pixel_count", quality.total_pixels),
        ("quality_masked_pixel_count", quality.masked_pixels),
        ("quality_nodata_pixel_count", quality.nodata_pixels),
        ("quality_cloud_pixel_count", quality.cloud_pixels),
        ("quality_cloud_shadow_pixel_count", quality.cloud_shadow_pixels),
        ("quality_snow_pixel_count", quality.snow_pixels),
        ("quality_saturated_or_defective_pixel_count",
         quality.saturated_or_defective_pixels),
        ("quality_unknown_class_pixel_count", quality.unknown_class_pixels),
        ("quality_other_masked_pixel_count", quality.other_masked_pixels),
    ]
    measurements = [
        Measurement(name=f"{key}_{name}", value=float(value), unit="pixels")
        for name, value in counts
    ]
    if quality.valid_fraction is not None and quality.contamination_fraction is not None:
        measurements.extend([
            Measurement(
                name=f"{key}_quality_valid_percent",
                value=quality.valid_fraction * 100.0,
                unit="%",
            ),
            Measurement(
                name=f"{key}_quality_contamination_percent",
                value=quality.contamination_fraction * 100.0,
                unit="%",
            ),
        ])
    return measurements


def temporal_quality_note(
    first: PixelQuality, second: PixelQuality, paired_valid: int | None
) -> str:
    """Each observation's own usable pixels, and those usable in both.

    Each side is masked by ITS OWN classification; the paired change uses only
    pixels usable in both, which is the pairing rule, not one side's mask
    applied to the other.
    """

    paired = (
        f"{paired_valid} pixels are usable on both dates and are the only pixels "
        "the paired NDWI change uses"
        if paired_valid is not None
        else "no paired NDWI change was computed"
    )
    return (
        f"Pixel quality: {first.window_label!r} has {first.valid_pixels} usable "
        f"of {first.total_pixels} pixels and {second.window_label!r} has "
        f"{second.valid_pixels} usable of {second.total_pixels}, each masked by "
        f"its own scene classification; {paired}."
    )
