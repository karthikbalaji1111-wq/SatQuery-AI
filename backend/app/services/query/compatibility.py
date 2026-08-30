"""Observation compatibility reporting - metadata only.

    (Observation, Observation) -> CompatibilityReport

This module answers one narrow question: *given two observations, what can be
honestly established from the metadata the system already holds?* It is
deliberately a **boundary**, not a mechanism.

Nothing here opens a raster, reads a pixel, reprojects, resamples, or inspects a
pixel grid. It performs no co-registration and no analysis. It reports what is
knowable and, just as importantly, names what is not.

**Anti-inference is the rule of this module.** None of the following, in any
combination, implies that two observations are co-registered:

    same modality    != co-registration
    same CRS         != co-registration
    same resolution  != co-registration
    same footprint   != co-registration

Co-registration requires pixel-grid alignment, which belongs to a later phase.
:attr:`CompatibilityReport.co_registration_status` is therefore assigned
*structurally* - from modality alone - and no match result can ever upgrade it.

What the repository's metadata actually supports:

- ``Scene`` carries no CRS and no ground sample distance. ``_normalize_scene``
  keeps only ``datetime``, ``bbox``, ``geometry``, ``eo:cloud_cover``,
  ``collection``, ``platform`` and ``processing:level``; the STAC item's
  ``proj:epsg`` and ``gsd`` are dropped. The **only** in-repo source of a CRS or
  a resolution is ``ImageryResponse``, which exists solely when bounded imagery
  was retrieved. So with ``include_imagery=False`` - the common case -
  ``crs_match`` and ``resolution_match`` are always ``"unknown"``. That is the
  correct report, not a gap to paper over.
- ``processing_level`` may be *derived from the collection name* rather than
  read from the item (see ``satellite/service.py::_processing_level``), so a
  ``"same"`` verdict says nothing about the processing baseline.

Dependency direction: this module belongs to the query domain, alongside the
``Observation``/``ObservationSet`` models it reports over. It imports nothing
from the analysis layer; future analysis code consumes this layer rather than
owning it.

Note on geometry: ``BoundingBox`` requires ``west < east``, so it cannot
represent an antimeridian-crossing footprint. Overlap here is a plain
axis-aligned WGS84 relation and inherits that limitation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.services.geospatial.schemas import BoundingBox
from app.services.query.schemas import Modality, Observation, ObservationSet

#: ``same``      - both values were determinable and agree.
#: ``different`` - both values were determinable and disagree.
#: ``unknown``   - at least one value could not be determined.
#: Unknown metadata must NEVER be reported as ``different``.
MatchStatus = Literal["same", "different", "unknown"]

#: Coarse whole-footprint relation. ``full`` means one footprint contains the
#: other (identical footprints included); ``partial`` means they intersect with
#: positive area but neither contains the other; ``none`` means no positive-area
#: intersection; ``unknown`` means a footprint was absent. No area, fraction or
#: intersection geometry is reported - degrees are not an equal-area unit.
BboxOverlapStatus = Literal["none", "partial", "full", "unknown"]

#: ``not_evaluated``             - same modality; alignment was never assessed.
#: ``not_supported_cross_modal`` - Sentinel-1 vs Sentinel-2.
#: There is deliberately no value meaning "co-registered": this layer cannot
#: establish that, so it cannot report it.
CoRegistrationStatus = Literal["not_evaluated", "not_supported_cross_modal"]

#: Relative tolerance for comparing two ground sample distances. Both values
#: originate from the same ``abs(transform.a)`` extraction, so they agree
#: exactly in practice; the tolerance guards float representation only.
_RESOLUTION_REL_TOL = 1e-6

_LIMIT_METADATA_ONLY = (
    "This report is derived from metadata only: no raster was opened, no pixel "
    "grid was inspected, and no co-registration was performed."
)
_LIMIT_NOT_EVALUATED = (
    "co_registration_status is 'not_evaluated': matching CRS, resolution, "
    "footprint or modality do not establish co-registration, which requires "
    "pixel-grid alignment that this phase does not perform."
)
_LIMIT_CROSS_MODAL = (
    "co_registration_status is 'not_supported_cross_modal': Sentinel-1 and "
    "Sentinel-2 observations are not geometrically comparable without SAR "
    "terrain correction, which this system does not perform."
)
_LIMIT_CRS_UNKNOWN = (
    "CRS could not be determined: a Scene carries no CRS, and bounded imagery "
    "was not retrieved for both observations."
)
_LIMIT_CRS_KNOWN = (
    "CRS was compared lexically between the strings reported by retrieved "
    "display imagery; that is not a geometric equivalence proof, and the value "
    "describes the display asset actually retrieved rather than any band a "
    "later analysis would read."
)
_LIMIT_RESOLUTION_UNKNOWN = (
    "Resolution could not be determined: a Scene carries no ground sample "
    "distance, and bounded imagery was not retrieved for both observations."
)
_LIMIT_RESOLUTION_KNOWN = (
    "Resolution describes the display asset actually retrieved, not any band a "
    "later analysis would read, and is the source ground sample distance rather "
    "than the delivered pixel size."
)
_LIMIT_PROCESSING_KNOWN = (
    "processing_level may be derived from the STAC collection name rather than "
    "read from the item, so it does not describe the processing baseline; "
    "scenes built from different baselines can report the same level."
)
_LIMIT_PROCESSING_UNKNOWN = (
    "processing_level was unavailable for at least one observation."
)
_LIMIT_TEMPORAL_UNKNOWN = (
    "At least one acquisition time is absent or unparseable, so the temporal "
    "separation was not computed."
)
_LIMIT_TEMPORAL_MIXED = (
    "The acquisition times mix time zone-aware and naive values, so the "
    "temporal separation was not computed rather than assume a time zone."
)
_LIMIT_BBOX_UNKNOWN = (
    "At least one scene footprint is absent, so bounding-box overlap could not "
    "be determined."
)
_LIMIT_BBOX_KNOWN = (
    "Bounding-box overlap is a coarse WGS84 relation between whole-scene "
    "footprints, not the analysed AOI, and is not an area measurement."
)


class CompatibilityReport(BaseModel):
    """What two observations can honestly be said to share.

    Every field is a *report*, never a decision: nothing here authorises a
    comparison, and no combination of values implies co-registration.
    """

    same_modality: bool
    #: Absolute separation between the two acquisition times, in days. ``None``
    #: when either time is absent, unparseable, or when the two mix time
    #: zone-aware and naive values.
    temporal_separation_days: float | None
    bbox_overlap: BboxOverlapStatus
    crs_match: MatchStatus
    resolution_match: MatchStatus
    processing_level_match: MatchStatus
    #: Explicit, deterministic statements of what this report does NOT
    #: establish. Never empty.
    limitations: list[str] = Field(default_factory=list)
    co_registration_status: CoRegistrationStatus


class ObservationPair(BaseModel):
    """Two observations proposed for comparison.

    ``first`` and ``second`` are deliberately neutral names: when an acquisition
    time is unknown, calling them "earlier"/"later" would be a claim the data
    does not support. Ordering is documented on :func:`pair_observations`.
    """

    first: Observation
    second: Observation


class PairingFailure(BaseModel):
    """Why a set of observations produced no pair.

    ``modality`` is ``None`` when the failure is not attributable to one
    modality (an empty observation set).
    """

    modality: Modality | None
    reason: str


# --------------------------------------------------------------------------- #
# Metadata accessors - each returns None rather than guessing
# --------------------------------------------------------------------------- #


def _imagery_crs(observation: Observation) -> str | None:
    imagery = observation.imagery
    return None if imagery is None else imagery.crs


def _imagery_resolution(observation: Observation) -> float | None:
    imagery = observation.imagery
    return None if imagery is None else imagery.resolution


def _normalized_text(value: str | None) -> str | None:
    """Strip and casefold; blank becomes ``None`` so it reports as unknown."""

    if value is None:
        return None
    cleaned = value.strip()
    return cleaned.casefold() if cleaned else None


def _match_text(first: str | None, second: str | None) -> MatchStatus:
    left, right = _normalized_text(first), _normalized_text(second)
    if left is None or right is None:
        return "unknown"
    return "same" if left == right else "different"


def _match_resolution(first: float | None, second: float | None) -> MatchStatus:
    if first is None or second is None:
        return "unknown"
    if not (math.isfinite(first) and math.isfinite(second)):
        return "unknown"
    return (
        "same"
        if math.isclose(first, second, rel_tol=_RESOLUTION_REL_TOL)
        else "different"
    )


def _is_aware(moment: datetime) -> bool:
    return moment.utcoffset() is not None


def _temporal_separation_days(
    first: Observation, second: Observation
) -> tuple[float | None, bool]:
    """Absolute separation in days, plus whether awareness was mixed.

    Returns ``(None, True)`` for a mixed aware/naive pair: subtracting them
    raises, and assuming a time zone would be inventing data.
    """

    left, right = first.acquired_at, second.acquired_at
    if left is None or right is None:
        return None, False
    if _is_aware(left) != _is_aware(right):
        return None, True
    return abs((left - right).total_seconds()) / 86400.0, False


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (
        outer.west <= inner.west
        and outer.east >= inner.east
        and outer.south <= inner.south
        and outer.north >= inner.north
    )


def _bbox_overlap(
    first: BoundingBox | None, second: BoundingBox | None
) -> BboxOverlapStatus:
    """Coarse footprint relation. A shared edge has zero area and is ``none``."""

    if first is None or second is None:
        return "unknown"

    west = max(first.west, second.west)
    east = min(first.east, second.east)
    south = max(first.south, second.south)
    north = min(first.north, second.north)
    if west >= east or south >= north:
        return "none"

    if _contains(first, second) or _contains(second, first):
        return "full"
    return "partial"


def _limitations(
    *,
    co_registration_status: CoRegistrationStatus,
    crs_match: MatchStatus,
    resolution_match: MatchStatus,
    processing_level_match: MatchStatus,
    temporal_days: float | None,
    mixed_awareness: bool,
    bbox_overlap: BboxOverlapStatus,
) -> list[str]:
    """Deterministic, fixed-order statements of what is not established."""

    notes = [_LIMIT_METADATA_ONLY]

    if co_registration_status == "not_supported_cross_modal":
        notes.append(_LIMIT_CROSS_MODAL)
    else:
        notes.append(_LIMIT_NOT_EVALUATED)

    notes.append(_LIMIT_CRS_UNKNOWN if crs_match == "unknown" else _LIMIT_CRS_KNOWN)
    notes.append(
        _LIMIT_RESOLUTION_UNKNOWN
        if resolution_match == "unknown"
        else _LIMIT_RESOLUTION_KNOWN
    )
    notes.append(
        _LIMIT_PROCESSING_UNKNOWN
        if processing_level_match == "unknown"
        else _LIMIT_PROCESSING_KNOWN
    )

    if mixed_awareness:
        notes.append(_LIMIT_TEMPORAL_MIXED)
    elif temporal_days is None:
        notes.append(_LIMIT_TEMPORAL_UNKNOWN)

    notes.append(
        _LIMIT_BBOX_UNKNOWN if bbox_overlap == "unknown" else _LIMIT_BBOX_KNOWN
    )
    return notes


def compute_compatibility(
    first: Observation, second: Observation
) -> CompatibilityReport:
    """Report what two observations demonstrably share, and what they do not.

    Pure and symmetric in its match fields: swapping the arguments changes no
    verdict. Cross-modal pairs (Sentinel-1 with Sentinel-2) are fully supported
    here - they simply report ``not_supported_cross_modal`` for co-registration,
    which is itself the useful answer.
    """

    same_modality = first.modality == second.modality
    co_registration_status: CoRegistrationStatus = (
        "not_evaluated" if same_modality else "not_supported_cross_modal"
    )

    crs_match = _match_text(_imagery_crs(first), _imagery_crs(second))
    resolution_match = _match_resolution(
        _imagery_resolution(first), _imagery_resolution(second)
    )
    processing_level_match = _match_text(
        first.scene.processing_level, second.scene.processing_level
    )
    temporal_days, mixed_awareness = _temporal_separation_days(first, second)
    bbox_overlap = _bbox_overlap(first.scene.bbox, second.scene.bbox)

    return CompatibilityReport(
        same_modality=same_modality,
        temporal_separation_days=temporal_days,
        bbox_overlap=bbox_overlap,
        crs_match=crs_match,
        resolution_match=resolution_match,
        processing_level_match=processing_level_match,
        limitations=_limitations(
            co_registration_status=co_registration_status,
            crs_match=crs_match,
            resolution_match=resolution_match,
            processing_level_match=processing_level_match,
            temporal_days=temporal_days,
            mixed_awareness=mixed_awareness,
            bbox_overlap=bbox_overlap,
        ),
        co_registration_status=co_registration_status,
    )


# --------------------------------------------------------------------------- #
# Pairing - same-modality only
#
# Cross-modal pairing is deliberately NOT a pairing mode. Sentinel-1 and
# Sentinel-2 observations are not geometrically comparable without SAR terrain
# correction, so proposing such a pair automatically would imply a comparison
# the system cannot make. Cross-modal compatibility remains directly reachable
# by calling ``compute_compatibility`` with the two observations.
# --------------------------------------------------------------------------- #


def _ordering_seconds(moment: datetime | None) -> float:
    """Total-order key for acquisition time.

    A naive datetime is treated as UTC **for ordering only**, so that a mixed
    aware/naive group sorts deterministically instead of raising. This is not a
    claim about the value's time zone - and note the deliberate asymmetry with
    :func:`_temporal_separation_days`, which refuses to make that assumption
    because a *reported measurement* must not invent a time zone.
    """

    if moment is None:
        return 0.0
    if _is_aware(moment):
        return moment.timestamp()
    # Anchored to UTC explicitly: a naive ``.timestamp()`` would resolve against
    # the host's local time zone and make the ordering machine-dependent.
    return moment.replace(tzinfo=UTC).timestamp()


def _sort_key(observation: Observation) -> tuple[int, float, str, str]:
    moment = observation.acquired_at
    return (
        1 if moment is None else 0,  # unknown acquisition times sort last
        _ordering_seconds(moment),
        observation.window_label,
        observation.scene_id,
    )


def pair_observations(
    observations: ObservationSet,
) -> tuple[list[ObservationPair], list[PairingFailure]]:
    """Propose deterministic same-modality pairs, and explain what could not pair.

    Within each modality, observations are ordered by **acquisition time**
    ascending - by ``Observation.acquired_at``, the time the data was actually
    acquired, not the window that was requested. Observations whose acquisition
    time is absent or unparseable **sort last**, with ties broken by window
    label and then scene id so the order is always total and deterministic.
    Ordered observations are paired consecutively, so *n* observations yield
    *n - 1* pairs.

    ``first`` and ``second`` therefore mean *acquired earlier* and *acquired
    later*. They do **not** mean "baseline" and "target": ``TemporalComparison``
    does not require the baseline window to precede the target window, so a
    two-window comparison whose baseline was acquired after its target pairs as
    ``first=target, second=baseline``. No baseline-to-target ordering is
    guaranteed or implied. The requested roles are never lost - they remain
    readable on each observation as ``Observation.window_label`` (``"baseline"``,
    ``"target"``, ``"series[0]"``, ...) - so a caller needing role semantics
    reads the label rather than the pair position.

    Modalities are visited in order of first appearance, so both returned lists
    are stable. A modality with fewer than two observations yields a
    :class:`PairingFailure` rather than a pair; an empty set yields a single
    failure with ``modality=None``.

    Pairing proposes *candidates for comparison*. It does not assert that a pair
    is comparable - that is what :func:`compute_compatibility` reports on.
    """

    pairs: list[ObservationPair] = []
    failures: list[PairingFailure] = []

    if not observations.observations:
        return pairs, [
            PairingFailure(
                modality=None,
                reason=(
                    "The observation set is empty; there is nothing to pair."
                ),
            )
        ]

    grouped: dict[Modality, list[Observation]] = {}
    for observation in observations.observations:
        grouped.setdefault(observation.modality, []).append(observation)

    for modality, group in grouped.items():
        if len(group) < 2:
            failures.append(
                PairingFailure(
                    modality=modality,
                    reason=(
                        f"Only {len(group)} {modality} observation is available; "
                        "at least two are needed to form a pair."
                    ),
                )
            )
            continue

        ordered = sorted(group, key=_sort_key)
        pairs.extend(
            ObservationPair(first=earlier, second=later)
            for earlier, later in zip(ordered, ordered[1:], strict=False)
        )

    return pairs, failures
