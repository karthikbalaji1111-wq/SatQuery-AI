"""Stage 4 of the scientific pipeline: are these numbers what the formulas assume?

A scene can be catalog-valid (Stage 2) and its pixels clear (Stage 3) and still
hold numbers on the wrong radiometric representation for the formula that will
consume them. This module decides, from catalog metadata Stage 2 already
fetched and parsed, whether the values are SAFE TO USE AS-IS. It never changes
a value: no scale is multiplied, no offset is added, nothing is normalised. A
representation it cannot establish is refused, not corrected.

What the engines assume
-----------------------
* Optical normalised differences run on raw ``uint16`` digital numbers. A
  multiplicative scale SHARED by both bands cancels exactly; an additive offset
  does NOT cancel (``analysis/indices.py``). So the pixels must carry NO
  additive offset, and the two bands must share one scale.
* SAR statistics average provider RTC gamma naught in LINEAR POWER and convert
  to decibels themselves (``analysis/sar.py``). So the pixels must be linear
  power: not decibels, not scaled, not offset.

The Sentinel-2 offset - what the evidence says (2026-09-23)
-----------------------------------------------------------
Processing baseline 04.00 (25 January 2022) introduced an additive radiometric
offset (Element 84's Earth Search README; ESA SentiWiki documents the matching
L1C ``RADIO_ADD_OFFSET`` from the same baseline). Earth Search publishes two
signals about it, and they disagree:

* ``earthsearch:boa_offset_applied`` - the provider's statement that the offset
  was already removed from the COG pixels.
* ``raster:bands[].offset`` - ``-0.1`` on nearly every item, including items
  whose flag is ``true``.

Pixel measurement settles the ``true`` case: over SCL-classified vegetation an
applied ``-0.1`` gives NDVI 1.30, which is impossible, while raw values give
0.637 (CLAUDE.md section 7), and 2023 open water reads a median NIR DN of 338
(offset-free). So for ``true`` the flag is right and the declared offset is
stale. The ``false`` case is NOT settled by metadata: baseline-04.00 items
flagged ``false`` also read offset-free over water (3 of 3 sampled, median NIR
DN 256-433), contradicting both of their signals. Earth Search does not link
ESA's product XML, which carries the authoritative ``BOA_ADD_OFFSET``. So:

    offset established absent  <=>  flag is true (baseline >= 04.00)
                                or  baseline < 04.00, flag not true, and no
                                    used band declares a non-zero offset

Everything else is UNDETERMINED - not "incompatible", because measurement did
not find the offset the metadata claims, and not "verified", because metadata
cannot establish it. Undetermined is refused exactly as incompatible is: no
number is produced from a representation nobody can state.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import InvalidInputError

if TYPE_CHECKING:  # pragma: no cover - typing only; no runtime import cycle
    from app.services.satellite.scene_validation import ValidatedAsset, ValidatedScene

RadiometricStatus = Literal[
    "verified",
    "verified_with_unknown_metadata",
    "incompatible",
    "undetermined",
]

OffsetState = Literal[
    "removed_by_provider",  # A: flag true - the provider removed it
    "not_removed",  # B: flag false on a baseline that has the offset
    "not_published",  # C: flag absent on a baseline that has the offset
    "unparseable",  # D: flag present but not a boolean
    "not_introduced",  # baseline predates the offset
    "contradictory",  # flag true on a baseline that never had the offset
    "not_applicable",  # SAR
]

#: The first processing baseline with the additive reflectance offset. The ONE
#: place the baseline rule lives; nothing else compares baselines.
OFFSET_INTRODUCED_BASELINE: tuple[int, int] = (4, 0)

#: Categorical layers read beside the bands. Not radiometric: never assessed.
CATEGORICAL_ASSETS = frozenset({"scl"})

_BASELINE = re.compile(r"(\d{2})\.(\d{2})")
_DECIBEL_UNITS = frozenset({"db", "decibel", "decibels"})
_LINEAR_POWER_UNITS = frozenset({"1", "linear", "power"})

AUTHORITY = (
    "STAC item properties (s2:processing_baseline, earthsearch:boa_offset_applied) "
    "and raster:bands, as fetched and parsed by scene validation; the offset rule "
    "is satellite.radiometry.OFFSET_INTRODUCED_BASELINE"
)


class RadiometricValidationError(InvalidInputError):
    """The values are not on a representation the engine can consume as-is."""

    def __init__(
        self,
        code: Literal["radiometric_incompatible", "radiometric_undetermined"],
        message: str,
    ) -> None:
        super().__init__(f"Radiometric validation failed ({code}): {message}", code=code)


class AssetEncoding(BaseModel):
    """What the catalog DECLARES about one asset's numbers. ``None`` = not declared.

    Never filled in: an absent scale is not 1 and an absent offset is not 0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    data_type: str | None = None
    scale: float | None = None
    offset: float | None = None
    unit: str | None = None
    nodata: float | None = None
    bits_per_sample: int | None = None


class RadiometricState(BaseModel):
    """The radiometric representation of the values ONE analysis will consume.

    ``status``:
    * ``verified`` - representation established and every relevant field is
      declared and consistent with it;
    * ``verified_with_unknown_metadata`` - representation established, but some
      fields are undeclared or contradicted (listed in ``unknown_fields`` and
      ``metadata_conflicts``);
    * ``incompatible`` - the metadata establishes a representation the engine
      cannot consume as-is;
    * ``undetermined`` - the metadata cannot establish the representation.

    The last two are refused before any read. Nothing here says the scene is
    "good"; it says only whether its numbers mean what the formula assumes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: RadiometricStatus
    modality: str
    scene_id: str
    collection: str
    assets: list[str]
    #: The representation the engine will consume, when established.
    representation: str | None = None
    processing_baseline: str | None = None
    product_type: str | None = None
    offset_state: OffsetState
    #: As published; ``None`` when the item does not publish it.
    boa_offset_applied: bool | None = None
    encodings: list[AssetEncoding] = Field(default_factory=list)
    #: No valid range is published by either catalog; ``None`` until one is.
    valid_range: list[float] | None = None
    #: Where saturated/defective pixels are identified. Optical: the SCL class 1
    #: mask of Stage 3 - no second mask is built here.
    saturation_source: Literal["scl_class_1", "not_published"]
    unknown_fields: list[str] = Field(default_factory=list)
    metadata_conflicts: list[str] = Field(default_factory=list)
    authority: str = AUTHORITY
    notes: list[str] = Field(default_factory=list)
    validation_stage: Literal["radiometric"] = "radiometric"

    @property
    def usable(self) -> bool:
        return self.status in ("verified", "verified_with_unknown_metadata")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def parse_baseline(value: str | None) -> tuple[int, int] | None:
    """``"05.11"`` -> ``(5, 11)``; anything else -> ``None``."""

    if not isinstance(value, str):
        return None
    match = _BASELINE.fullmatch(value.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def _encoding(asset: ValidatedAsset) -> AssetEncoding:
    return AssetEncoding(
        key=asset.key,
        data_type=asset.data_type,
        scale=asset.scale,
        offset=asset.offset,
        unit=asset.unit,
        nodata=asset.nodata,
        bits_per_sample=asset.bits_per_sample,
    )


def _radiometric_assets(
    scene: ValidatedScene, assets: Sequence[str]
) -> list[ValidatedAsset]:
    keys = [k for k in dict.fromkeys(assets) if k not in CATEGORICAL_ASSETS]
    found = [scene.asset(k) for k in keys]
    return [a for a in found if a is not None]


def _unknown(encodings: Sequence[AssetEncoding], fields: Sequence[str]) -> list[str]:
    return [
        f"{e.key}.{name}" for e in encodings for name in fields if getattr(e, name) is None
    ]


def _shared(encodings: Sequence[AssetEncoding], field: str) -> str | None:
    """Why the assets' declared ``field`` disagree, or ``None``. Undeclared != different."""

    declared = {(e.key, getattr(e, field)) for e in encodings if getattr(e, field) is not None}
    if len({value for _, value in declared}) > 1:
        listing = ", ".join(f"{k}={v}" for k, v in sorted(declared))
        return f"the bands declare different {field}s ({listing})"
    return None


# --------------------------------------------------------------------------- #
# Sentinel-2
# --------------------------------------------------------------------------- #


def _offset_state(baseline: tuple[int, int] | None, flag: str) -> OffsetState:
    """The central rule. Only the flag (as published) and the baseline decide it."""

    has_offset_baseline = baseline is not None and baseline >= OFFSET_INTRODUCED_BASELINE
    if flag == "true":
        return "removed_by_provider" if has_offset_baseline else "contradictory"
    if not has_offset_baseline:
        return "not_introduced"
    if flag == "false":
        return "not_removed"
    if flag == "absent":
        return "not_published"
    return "unparseable"


def _offset_decision(
    processing_baseline: str | None, flag: str, encodings: Sequence[AssetEncoding]
) -> tuple[OffsetState, str | None, list[str], list[str]]:
    """``(state, undetermined_reason, notes, conflicts)`` for the additive offset.

    ``undetermined_reason is None`` means the values are established
    offset-free. The single place that decides it.
    """

    baseline = parse_baseline(processing_baseline)
    state = _offset_state(baseline, flag)
    nonzero = {e.key: e.offset for e in encodings if e.offset not in (None, 0.0)}
    listing = ", ".join(f"{k}={v}" for k, v in sorted(nonzero.items()))

    if baseline is None:
        return state, (
            "the processing baseline is missing or unreadable, so whether the "
            "pixels carry the baseline-04.00 offset cannot be placed"
        ), [], []
    if state == "removed_by_provider":
        notes = [
            f"Baseline {processing_baseline}: the provider states the reflectance "
            "offset was already removed from the pixels "
            "(earthsearch:boa_offset_applied=true)."
        ]
        conflicts = (
            [
                f"declared raster:bands offset {listing} contradicts the provider "
                "flag; it is NOT applied - pixel measurements agree with the flag"
            ]
            if nonzero
            else []
        )
        return state, None, notes, conflicts
    if state == "not_introduced":
        notes = [f"Baseline {processing_baseline} predates the baseline-04.00 offset."]
        if nonzero:
            return state, (
                f"the bands declare a non-zero offset ({listing}) on a baseline that "
                "predates the offset, and no provider statement says it was removed"
            ), notes, []
        return state, None, notes, []
    if state == "contradictory":
        return state, (
            f"the provider says an offset was removed from a baseline "
            f"({processing_baseline}) that never carried one"
        ), [], []
    if state == "not_removed":
        return state, (
            f"on baseline {processing_baseline} the provider states the reflectance "
            "offset was NOT removed (earthsearch:boa_offset_applied=false), so the "
            "metadata does not establish offset-free values. It is not corrected "
            "here; bounded pixel checks of such scenes did not find the offset "
            "either, so the representation is undetermined rather than known to "
            "carry it"
        ), [], []
    if state == "not_published":
        return state, (
            f"baseline {processing_baseline} carries the reflectance offset and the "
            "item does not say whether it was removed"
        ), [], []
    return state, "earthsearch:boa_offset_applied is published but is not a boolean", [], []


def _assess_optical(scene: ValidatedScene, assets: Sequence[str]) -> RadiometricState:
    encodings = [_encoding(a) for a in _radiometric_assets(scene, assets)]
    processing = scene.processing
    state, undetermined, notes, conflicts = _offset_decision(
        processing.processing_baseline, processing.boa_offset_flag, encodings
    )

    incompatible: list[str] = []
    for field in ("scale", "unit"):
        problem = _shared(encodings, field)
        if problem is not None:
            incompatible.append(
                f"{problem}; a normalised difference is independent of {field} "
                "only when both bands share it"
            )

    unknown = _unknown(encodings, ("data_type", "scale", "offset", "unit"))
    if incompatible:
        status: RadiometricStatus = "incompatible"
    elif undetermined is not None:
        status = "undetermined"
    elif unknown or conflicts:
        status = "verified_with_unknown_metadata"
    else:
        status = "verified"

    refused = [*incompatible, *([undetermined] if undetermined else [])]
    return RadiometricState(
        status=status,
        modality=scene.modality,
        scene_id=scene.scene_id,
        collection=scene.collection,
        assets=[e.key for e in encodings],
        representation=(
            "Sentinel-2 L2A digital numbers proportional to surface reflectance, "
            "with no additive offset, read as-is"
            if status in ("verified", "verified_with_unknown_metadata")
            else None
        ),
        processing_baseline=processing.processing_baseline,
        product_type=processing.product_type,
        offset_state=state,
        boa_offset_applied=processing.boa_offset_applied,
        encodings=encodings,
        saturation_source="scl_class_1",
        unknown_fields=unknown,
        metadata_conflicts=conflicts,
        notes=notes + [f"Refused: {reason}." for reason in refused],
    )


# --------------------------------------------------------------------------- #
# Sentinel-1 RTC
# --------------------------------------------------------------------------- #


def _assess_sar(scene: ValidatedScene, assets: Sequence[str]) -> RadiometricState:
    encodings = [_encoding(a) for a in _radiometric_assets(scene, assets)]
    incompatible: list[str] = []
    for e in encodings:
        unit = e.unit.strip().lower() if e.unit is not None else None
        if unit is not None and unit in _DECIBEL_UNITS:
            incompatible.append(
                f"{e.key} is declared in decibels; the engine averages linear "
                "power and converts to decibels itself"
            )
        elif unit is not None and unit not in _LINEAR_POWER_UNITS:
            incompatible.append(f"{e.key} declares unit {e.unit!r}, not linear power")
        if e.scale is not None and not math.isclose(e.scale, 1.0):
            incompatible.append(f"{e.key} declares scale {e.scale}; values would be encoded")
        if e.offset is not None and e.offset != 0.0:
            incompatible.append(f"{e.key} declares offset {e.offset}; values would be encoded")

    unknown = _unknown(encodings, ("data_type", "scale", "offset", "unit"))
    notes = [
        "Provider RTC gamma naught (terrain-corrected by the provider, not by "
        "SatQuery). Linear power is established from the product itself - "
        "strictly positive values, GDAL scale 1 / offset 0 - and the header is "
        "re-checked after the read."
    ]
    if incompatible:
        status: RadiometricStatus = "incompatible"
    elif unknown:
        status = "verified_with_unknown_metadata"
    else:
        status = "verified"
    return RadiometricState(
        status=status,
        modality=scene.modality,
        scene_id=scene.scene_id,
        collection=scene.collection,
        assets=[e.key for e in encodings],
        representation=(
            "provider RTC gamma naught, linear power, read as-is"
            if status != "incompatible"
            else None
        ),
        processing_baseline=scene.processing.processing_baseline,
        product_type=scene.processing.product_type,
        offset_state="not_applicable",
        encodings=encodings,
        saturation_source="not_published",
        unknown_fields=unknown,
        notes=notes + [f"Refused: {reason}." for reason in incompatible],
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def assess_radiometry(scene: ValidatedScene, assets: Sequence[str]) -> RadiometricState:
    """The radiometric state of ``assets`` of a validated scene. Pure; no I/O."""

    if scene.modality == "sentinel-1-sar":
        return _assess_sar(scene, assets)
    return _assess_optical(scene, assets)


def require_usable(state: RadiometricState) -> RadiometricState:
    """Return ``state`` if the engine may consume the values as-is, else raise."""

    if state.usable:
        return state
    reasons = "; ".join(n.removeprefix("Refused: ").rstrip(".") for n in state.notes
                        if n.startswith("Refused: "))
    code = (
        "radiometric_incompatible" if state.status == "incompatible"
        else "radiometric_undetermined"
    )
    raise RadiometricValidationError(code, f"scene {state.scene_id}: {reasons}.")


def offset_problem(scene: ValidatedScene) -> str | None:
    """Why the scene's optical values are NOT established offset-free, or ``None``.

    Used by the scene-pair rule so two scenes are compared by the
    REPRESENTATION their pixels carry, not by their flags: a baseline-03.01
    scene (flag false - no offset ever existed) and a baseline-05.09 scene
    (flag true - offset removed) are both offset-free.
    """

    encodings = [_encoding(a) for a in _radiometric_assets(scene, [a.key for a in scene.assets])]
    _, undetermined, _, _ = _offset_decision(
        scene.processing.processing_baseline, scene.processing.boa_offset_flag, encodings
    )
    return undetermined


def radiometric_pair_problem(first: RadiometricState, second: RadiometricState) -> str | None:
    """Why two usable states may NOT be compared, or ``None``.

    Both must already be usable. Their representations must match, and a scale
    or unit declared on both dates must agree - a cross-date difference of
    indices is only meaningful on one representation. Nothing is corrected to
    make them agree.
    """

    if not (first.usable and second.usable):
        return "at least one observation's radiometric representation is not established"
    if first.representation != second.representation:
        return (
            "the two observations are on different radiometric representations "
            f"({first.representation!r} vs {second.representation!r})"
        )
    for field in ("scale", "unit"):
        problem = _shared([*first.encodings, *second.encodings], field)
        if problem is not None:
            return f"across the two dates {problem}"
    return None
