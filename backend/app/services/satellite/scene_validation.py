"""Stage 2 of the scientific pipeline: is this scene fit to be measured?

Stage 1 (``analysis/validation.py``) decides whether a REQUEST may start. This
module decides whether the SCENE a request selected, and the assets it would
read, are structurally and scientifically eligible to be measured - before a
single pixel is read.

The catalog is authoritative
----------------------------
Everything here is read from the STAC item the catalog returns for a scene id,
never from what a client said about that scene. ``/query/analyze`` accepts a
client-supplied execution result, whose ``Scene`` objects carry a date, cloud
cover, footprint and collection; none of those reach a :class:`ValidatedScene`.
The client's id and collection only say WHICH item to fetch.

What passing establishes, and what it does not
----------------------------------------------
A :class:`ValidatedScene` says: the catalog knows this scene; it is the sensor
and product the operation needs; its footprint reaches the requested area (and
how much of it); every asset the operation reads exists, is a readable
cloud-optimised GeoTIFF of the expected data type; and the processing metadata
the next stages need is recorded. It says nothing about clouds over the area,
nodata inside the window, whether the radiometry means what the engine assumes,
or whether the result will be meaningful. Those are later stages.

Metadata that is merely ABSENT is recorded as unknown, not rejected, unless a
rule below needs it. Metadata that is PRESENT and contradicts what the reader
assumes is rejected: a declared ``float32`` where ``uint16`` is expected means
the pixels are not what the engine thinks they are.

Pure: no network, no raster, no clock. The fetch lives in
:meth:`ImageryService.validate_scene`, which calls into this module.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from app.core.errors import AppError, InvalidInputError
from app.services.geospatial.schemas import BoundingBox
from app.services.satellite.radiometry import offset_problem
from app.services.satellite.rtc import RTC_COLLECTION

Modality = Literal["sentinel-2-optical", "sentinel-1-sar"]

SceneRejectionCode = Literal[
    "scene_not_found",
    "scene_does_not_cover_aoi",
    "insufficient_aoi_coverage",
    "scene_coverage_unknown",
    "required_asset_missing",
    "unsupported_asset_type",
    "unsupported_asset_encoding",
    "unknown_processing_baseline",
    "incompatible_sensor",
    "incompatible_collection",
    "temporal_scene_incompatible",
]


class SceneValidationError(InvalidInputError):
    """A scene, or one of its assets, may not be measured.

    The existing 422 :class:`InvalidInputError` with a specific ``code``. The
    code is also carried in the message, because in ``/query/analyze`` a scene
    failure degrades one operation to a warning rather than failing the whole
    request, and the warning is where a reader will see it.
    """

    def __init__(self, code: SceneRejectionCode, message: str) -> None:
        super().__init__(f"Scene validation failed ({code}): {message}", code=code)


# --------------------------------------------------------------------------- #
# Collection profiles: what each served collection is expected to publish
# --------------------------------------------------------------------------- #

_COG_MEDIA_HINT = "geotiff"
_BASELINE = re.compile(r"\d{2}\.\d{2}")


@dataclass(frozen=True)
class CollectionProfile:
    """What a collection's items must say for their assets to be measured.

    Every expectation here was read from the live catalogs (2026-09-23):
    Earth Search ``sentinel-2-l2a`` item ``S2B_44PMV_20250104_0_L2A`` and a
    Planetary Computer ``sentinel-1-rtc`` item over the same area.
    """

    collection: str
    modality: Modality
    #: Lower-cased ``constellation`` (or ``platform`` prefix) the item must name.
    constellation: str
    #: ``raster:bands[0].data_type`` the reader assumes for a measured band.
    data_type: str
    #: The property holding the processing baseline, or ``None`` when the
    #: collection does not publish one. Planetary Computer's RTC items carry no
    #: baseline - and ``sar:product_type`` reads "GRD", the product the RTC was
    #: derived from - so neither is required of them.
    baseline_property: str | None
    #: Assets whose data type differs from ``data_type``. Sentinel-2's ``scl``
    #: is a ``uint8`` class map beside ``uint16`` reflectance bands.
    asset_data_types: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def data_type_for(self, key: str) -> str:
        return self.asset_data_types.get(key, self.data_type)


PROFILES: Mapping[str, CollectionProfile] = {
    profile.collection: profile
    for profile in (
        CollectionProfile(
            collection="sentinel-2-l2a",
            modality="sentinel-2-optical",
            constellation="sentinel-2",
            data_type="uint16",
            baseline_property="s2:processing_baseline",
            asset_data_types={"scl": "uint8"},
        ),
        CollectionProfile(
            collection=RTC_COLLECTION,
            modality="sentinel-1-sar",
            constellation="sentinel-1",
            data_type="float32",
            baseline_property=None,
        ),
    )
}


# --------------------------------------------------------------------------- #
# Validated structures
# --------------------------------------------------------------------------- #

MetadataStatus = Literal["known", "unknown"]


class ValidatedAsset(BaseModel):
    """One asset the operation will read, as the catalog describes it.

    ``metadata_status`` is ``"known"`` when the item publishes
    ``raster:bands`` for the asset. The encoding fields are RECORDED for the
    radiometric stage; none is applied here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    href: str
    media_type: str
    roles: tuple[str, ...] | None = None
    metadata_status: MetadataStatus
    data_type: str | None = None
    nodata: float | None = None
    spatial_resolution: float | None = None
    scale: float | None = None
    offset: float | None = None
    unit: str | None = None
    #: ``raster:bands[0].bits_per_sample`` (Earth Search declares 15 for the
    #: Sentinel-2 bands). Recorded, never used to build a mask.
    bits_per_sample: int | None = None


class AoiCoverage(BaseModel):
    """How much of the requested area the scene's footprint reaches.

    ``basis`` says which footprint was used: the item's ``geometry`` when it
    has one, else its ``bbox`` (coarser - a bbox overstates a tilted swath).
    The fraction is an area ratio inside a small WGS84 box, with longitude
    scaled by the cosine of the box's mid-latitude; it is a coverage estimate
    for eligibility, not a measured quantity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    basis: Literal["geometry", "bbox"]
    fraction: float
    status: Literal["full", "partial"]


class ProcessingMetadata(BaseModel):
    """What the catalog says about how the product was made. Recorded, not applied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``s2:processing_baseline`` for Sentinel-2; ``None`` where the collection
    #: publishes none (Sentinel-1 RTC).
    processing_baseline: str | None = None
    product_type: str | None = None
    processing_level: str | None = None
    #: Earth Search's statement that the baseline-04.00 reflectance offset was
    #: already removed from the pixels. ``None`` when not published.
    boa_offset_applied: bool | None = None
    #: How that flag was published, so "absent" and "published but not a
    #: boolean" stay distinguishable (Stage 4 treats both as undetermined, and
    #: reports which).
    boa_offset_flag: Literal["true", "false", "absent", "unparseable"] = "absent"
    software: str | None = None


class ValidatedScene(BaseModel):
    """A scene the catalog vouches for, fit to have its assets read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_id: str
    collection: str
    modality: Modality
    platform: str | None = None
    constellation: str | None = None
    #: The catalog's acquisition time, verbatim.
    acquired_at: str | None = None
    geometry: dict[str, Any] | None = None
    aoi_coverage: AoiCoverage
    processing: ProcessingMetadata
    assets: tuple[ValidatedAsset, ...]
    #: Assets that were ASKED FOR but could not be validated, when the caller
    #: accepts partial availability (e.g. NDBI's SWIR missing beside a good
    #: NDVI). Each value is the rejection message.
    unavailable_assets: dict[str, str] = {}
    validation_stage: Literal["scene"] = "scene"

    def asset(self, key: str) -> ValidatedAsset | None:
        return next((a for a in self.assets if a.key == key), None)


class ValidatedScenePair(BaseModel):
    """Two independently validated scenes that may be compared, earlier first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    earlier: ValidatedScene
    later: ValidatedScene
    status: Literal["compatible"] = "compatible"
    #: Differences that do not block a comparison but a reader should see.
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# AOI coverage
# --------------------------------------------------------------------------- #

Point = tuple[float, float]


def _clip_to_box(ring: Sequence[Point], box: BoundingBox) -> list[Point]:
    """Sutherland-Hodgman: ``ring`` clipped to the axis-aligned ``box``.

    Exact for a convex clip window, which a box always is, whatever the shape
    of the ring being clipped.
    """

    edges: tuple[tuple[Callable[[Point], bool], Callable[[Point, Point], Point]], ...] = (
        (lambda p: p[0] >= box.west, lambda a, b: _at_x(a, b, box.west)),
        (lambda p: p[0] <= box.east, lambda a, b: _at_x(a, b, box.east)),
        (lambda p: p[1] >= box.south, lambda a, b: _at_y(a, b, box.south)),
        (lambda p: p[1] <= box.north, lambda a, b: _at_y(a, b, box.north)),
    )
    output = list(ring)
    for inside, cross in edges:
        points, output = output, []
        if not points:
            break
        previous = points[-1]
        for current in points:
            if inside(current):
                if not inside(previous):
                    output.append(cross(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(cross(previous, current))
            previous = current
    return output


def _at_x(a: Point, b: Point, x: float) -> Point:
    t = (x - a[0]) / (b[0] - a[0])
    return x, a[1] + t * (b[1] - a[1])


def _at_y(a: Point, b: Point, y: float) -> Point:
    t = (y - a[1]) / (b[1] - a[1])
    return a[0] + t * (b[0] - a[0]), y


def _area(ring: Sequence[Point], x_scale: float) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring, [*ring[1:], *ring[:1]], strict=True):
        total += x1 * x_scale * y2 - x2 * x_scale * y1
    return abs(total) / 2.0


def _polygons(geometry: Mapping[str, Any]) -> list[list[list[Point]]] | None:
    """A Polygon or MultiPolygon as a list of polygons of rings, or ``None``."""

    kind, coordinates = geometry.get("type"), geometry.get("coordinates")
    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = coordinates
    else:
        return None
    if not isinstance(polygons, list):
        return None
    parsed: list[list[list[Point]]] = []
    for polygon in polygons:
        if not isinstance(polygon, list):
            return None
        rings: list[list[Point]] = []
        for ring in polygon:
            if not isinstance(ring, list):
                return None
            points: list[Point] = []
            for position in ring:
                if (
                    not isinstance(position, list | tuple)
                    or len(position) < 2
                    or not all(isinstance(v, int | float) for v in position[:2])
                    or not all(math.isfinite(v) for v in position[:2])
                ):
                    return None
                points.append((float(position[0]), float(position[1])))
            if points and points[0] == points[-1]:
                points = points[:-1]
            rings.append(points)
        parsed.append(rings)
    return parsed


def aoi_coverage_fraction(
    geometry: Mapping[str, Any], aoi: BoundingBox
) -> float | None:
    """The fraction of ``aoi`` inside a GeoJSON (Multi)Polygon, or ``None``.

    ``None`` means the geometry could not be read, which is not the same as
    "does not cover": unknown must never become "no".
    """

    polygons = _polygons(geometry)
    if polygons is None:
        return None
    x_scale = math.cos(math.radians((aoi.south + aoi.north) / 2.0))
    box_ring: list[Point] = [
        (aoi.west, aoi.south), (aoi.east, aoi.south),
        (aoi.east, aoi.north), (aoi.west, aoi.north),
    ]
    box_area = _area(box_ring, x_scale)
    covered = 0.0
    for rings in polygons:
        if not rings:
            continue
        exterior, holes = rings[0], rings[1:]
        covered += _area(_clip_to_box(exterior, aoi), x_scale)
        for hole in holes:
            covered -= _area(_clip_to_box(hole, aoi), x_scale)
    # Floating-point clipping can land a hair outside [0, 1].
    return min(1.0, max(0.0, covered / box_area))


def _bbox_geometry(bbox: object) -> dict[str, Any] | None:
    if (
        not isinstance(bbox, list | tuple)
        or len(bbox) != 4
        or not all(isinstance(v, int | float) and math.isfinite(v) for v in bbox)
    ):
        return None
    west, south, east, north = (float(v) for v in bbox)
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north],
                         [west, south]]],
    }


def _coverage(item: Mapping[str, Any], aoi: BoundingBox, min_fraction: float) -> AoiCoverage:
    geometry = item.get("geometry")
    basis: Literal["geometry", "bbox"] = "geometry"
    fraction = (
        aoi_coverage_fraction(geometry, aoi) if isinstance(geometry, Mapping) else None
    )
    if fraction is None:
        box = _bbox_geometry(item.get("bbox"))
        basis = "bbox"
        fraction = aoi_coverage_fraction(box, aoi) if box is not None else None
    if fraction is None:
        raise SceneValidationError(
            "scene_coverage_unknown",
            "the catalog publishes no readable footprint for this scene, so "
            "whether it reaches the requested area cannot be established.",
        )
    if fraction <= 0.0:
        raise SceneValidationError(
            "scene_does_not_cover_aoi",
            f"the scene's {basis} footprint does not reach the requested area.",
        )
    if fraction < min_fraction:
        raise SceneValidationError(
            "insufficient_aoi_coverage",
            f"the scene's {basis} footprint covers {fraction:.1%} of the requested "
            f"area; this deployment requires at least {min_fraction:.1%}.",
        )
    return AoiCoverage(
        basis=basis, fraction=fraction, status="full" if fraction >= 1.0 else "partial"
    )


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


HrefCheck = Callable[[str, str], None]


def _validate_asset(
    item: Mapping[str, Any],
    key: str,
    profile: CollectionProfile,
    check_href: HrefCheck,
) -> ValidatedAsset:
    assets = item.get("assets")
    asset = assets.get(key) if isinstance(assets, Mapping) else None
    if not isinstance(asset, Mapping):
        raise SceneValidationError(
            "required_asset_missing", f"the scene has no {key!r} asset."
        )
    href = asset.get("href")
    if not isinstance(href, str) or not href:
        raise SceneValidationError(
            "required_asset_missing", f"the {key!r} asset has no href."
        )
    media_type = str(asset.get("type") or "")
    if _COG_MEDIA_HINT not in media_type.lower():
        raise SceneValidationError(
            "unsupported_asset_type",
            f"the {key!r} asset is {media_type or 'of unknown type'}, not a "
            "cloud-optimised GeoTIFF that can be read by window.",
        )
    roles_raw = asset.get("roles")
    roles = (
        tuple(str(r) for r in roles_raw) if isinstance(roles_raw, list | tuple) else None
    )
    if roles is not None and "data" not in roles:
        raise SceneValidationError(
            "unsupported_asset_type",
            f"the {key!r} asset's roles {list(roles)} do not include 'data'; it is "
            "not published as measurement data.",
        )
    try:
        check_href(key, href)
    except AppError as exc:
        raise SceneValidationError("unsupported_asset_type", exc.message) from None

    bands = asset.get("raster:bands")
    band = bands[0] if isinstance(bands, list) and bands and isinstance(
        bands[0], Mapping
    ) else None
    if band is None:
        return ValidatedAsset(
            key=key, href=href, media_type=media_type, roles=roles,
            metadata_status="unknown",
        )
    data_type = band.get("data_type")
    expected = profile.data_type_for(key)
    if data_type is not None and data_type != expected:
        raise SceneValidationError(
            "unsupported_asset_encoding",
            f"the {key!r} asset is published as {data_type!r}; the "
            f"{profile.collection} reader expects {expected!r}.",
        )
    unit = band.get("unit")
    return ValidatedAsset(
        key=key,
        href=href,
        media_type=media_type,
        roles=roles,
        metadata_status="known",
        data_type=data_type if isinstance(data_type, str) else None,
        nodata=_number(band.get("nodata")),
        spatial_resolution=_number(band.get("spatial_resolution")),
        scale=_number(band.get("scale")),
        offset=_number(band.get("offset")),
        unit=unit if isinstance(unit, str) else None,
        bits_per_sample=(
            bits if isinstance(bits := band.get("bits_per_sample"), int)
            and not isinstance(bits, bool) else None
        ),
    )


def shared_scale_problem(first: ValidatedAsset, second: ValidatedAsset) -> str | None:
    """Why a normalised difference of these two assets would not be scale-free.

    The raw-DN indices rest on both bands sharing ONE multiplicative scale, so
    it cancels (``indices.py``). Two declared scales that differ break that
    identity; an undeclared scale is unknown, not different.
    """

    if first.scale is None or second.scale is None or first.scale == second.scale:
        return None
    return (
        f"the {first.key!r} and {second.key!r} assets declare different scales "
        f"({first.scale} and {second.scale}), so their normalised difference "
        "would not be independent of scale."
    )


# --------------------------------------------------------------------------- #
# The scene
# --------------------------------------------------------------------------- #


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _processing(
    properties: Mapping[str, Any], profile: CollectionProfile
) -> ProcessingMetadata:
    baseline: str | None = None
    if profile.baseline_property is not None:
        raw = properties.get(profile.baseline_property)
        if not isinstance(raw, str) or _BASELINE.fullmatch(raw.strip()) is None:
            raise SceneValidationError(
                "unknown_processing_baseline",
                f"{profile.collection} scenes must state "
                f"{profile.baseline_property} as NN.NN; this one "
                f"{'does not' if raw is None else 'states an unreadable value'}. "
                "Without it the pixel encoding cannot be placed.",
            )
        baseline = raw.strip()
    boa = properties.get("earthsearch:boa_offset_applied")
    boa_flag: Literal["true", "false", "absent", "unparseable"] = (
        "absent" if "earthsearch:boa_offset_applied" not in properties
        else "true" if boa is True
        else "false" if boa is False
        else "unparseable"
    )
    software = properties.get("processing:software")
    level = properties.get("processing:level", properties.get("s1:processing_level"))
    return ProcessingMetadata(
        processing_baseline=baseline,
        product_type=_text(properties.get("s2:product_type"))
        or _text(properties.get("sar:product_type")),
        processing_level=str(level) if isinstance(level, str | int) else None,
        boa_offset_applied=boa if isinstance(boa, bool) else None,
        boa_offset_flag=boa_flag,
        software=(
            ", ".join(f"{k} {v}" for k, v in sorted(software.items()))
            if isinstance(software, Mapping)
            else _text(software)
        ),
    )


def validate_scene_item(
    item: Mapping[str, Any],
    *,
    scene_id: str,
    collection: str,
    modality: Modality,
    assets: Sequence[str],
    aoi: BoundingBox,
    check_href: HrefCheck,
    require_all_assets: bool = True,
    min_coverage: float = 0.0,
) -> ValidatedScene:
    """Validate one catalog item for measuring ``assets`` over ``aoi``.

    ``collection`` and ``scene_id`` are what the item was FETCHED by; the item
    must agree with both. With ``require_all_assets=False`` an asset that fails
    is recorded in ``unavailable_assets`` rather than failing the scene - but
    at least one must pass. The first failing rule is raised, in a fixed order:
    identity, sensor, footprint, processing, assets.
    """

    profile = PROFILES.get(collection)
    if profile is None:
        raise SceneValidationError(
            "incompatible_collection",
            f"{collection!r} has no validation profile; its assets cannot be "
            "checked, so they are not measured.",
        )
    if profile.modality != modality:
        raise SceneValidationError(
            "incompatible_collection",
            f"{collection!r} holds {profile.modality} data; the operation needs "
            f"{modality}.",
        )
    if item.get("id") != scene_id:
        raise SceneValidationError(
            "scene_not_found",
            f"the catalog answered for {scene_id!r} with a different item.",
        )
    item_collection = item.get("collection")
    if item_collection is not None and item_collection != collection:
        raise SceneValidationError(
            "incompatible_collection",
            f"the catalog places this scene in {item_collection!r}, not {collection!r}.",
        )

    properties = item.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    constellation = _text(properties.get("constellation"))
    platform = _text(properties.get("platform"))
    stated = (constellation or platform or "").lower()
    if stated and not stated.startswith(profile.constellation):
        raise SceneValidationError(
            "incompatible_sensor",
            f"the catalog says this scene is from {constellation or platform!r}; "
            f"the operation needs {profile.constellation}.",
        )

    coverage = _coverage(item, aoi, min_coverage)
    processing = _processing(properties, profile)

    validated: list[ValidatedAsset] = []
    unavailable: dict[str, str] = {}
    for key in dict.fromkeys(assets):
        try:
            validated.append(_validate_asset(item, key, profile, check_href))
        except SceneValidationError as exc:
            if require_all_assets:
                raise
            unavailable[key] = exc.message
    if assets and not validated:
        raise SceneValidationError(
            "required_asset_missing",
            "none of the requested assets can be read: "
            + "; ".join(unavailable.values()),
        )

    geometry = item.get("geometry")
    return ValidatedScene(
        scene_id=scene_id,
        collection=collection,
        modality=modality,
        platform=platform,
        constellation=constellation,
        acquired_at=_text(properties.get("datetime")),
        geometry=dict(geometry) if isinstance(geometry, Mapping) else None,
        aoi_coverage=coverage,
        processing=processing,
        assets=tuple(validated),
        unavailable_assets=unavailable,
    )


# --------------------------------------------------------------------------- #
# The pair
# --------------------------------------------------------------------------- #


def _instant(value: str | None) -> datetime | None:
    """A catalog timestamp as an aware datetime; a naive one is not guessed at."""

    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def validate_scene_pair(
    earlier: ValidatedScene, later: ValidatedScene
) -> ValidatedScenePair:
    """Whether two independently validated scenes may be compared.

    ``earlier``/``later`` are the caller's order; the CATALOG's acquisition
    times must agree with it. Pair eligibility only: no pixel is compared.
    """

    def refuse(reason: str) -> SceneValidationError:
        return SceneValidationError("temporal_scene_incompatible", reason)

    if earlier.scene_id == later.scene_id:
        raise refuse("both sides are the same scene.")
    if earlier.collection != later.collection:
        raise refuse(
            f"the scenes come from different collections ({earlier.collection} "
            f"and {later.collection})."
        )
    if earlier.modality != later.modality:
        raise refuse("the scenes come from different sensors.")
    first_time, second_time = _instant(earlier.acquired_at), _instant(later.acquired_at)
    if first_time is None or second_time is None:
        raise refuse(
            "the catalog does not state, in a readable time-zone-aware form, when "
            "both scenes were acquired."
        )
    if not first_time < second_time:
        # Equal is refused: "earlier" would then be decided by nothing.
        raise refuse(
            f"the catalog dates the earlier scene {earlier.acquired_at} and the "
            f"later one {later.acquired_at}; they are not in acquisition order."
        )
    # Compared by the REPRESENTATION the pixels carry, not by the flags: a
    # baseline-03.01 scene (flag false - no offset ever existed) and a
    # baseline-05.09 scene (flag true - offset removed) are both offset-free,
    # and comparing the flags refused that valid pair. The rule itself lives in
    # ONE place, ``satellite.radiometry``.
    if earlier.modality == "sentinel-2-optical":
        for side, scene in (("earlier", earlier), ("later", later)):
            problem = offset_problem(scene)
            if problem is not None:
                raise refuse(
                    f"the {side} scene's reflectance offset state is not "
                    f"established ({problem}), so the two scenes cannot be shown "
                    "to share one pixel representation."
                )

    notes: list[str] = []
    if earlier.processing.processing_baseline != later.processing.processing_baseline:
        notes.append(
            "The scenes were processed with different baselines "
            f"({earlier.processing.processing_baseline} and "
            f"{later.processing.processing_baseline})."
        )
    if earlier.aoi_coverage.fraction != later.aoi_coverage.fraction:
        notes.append(
            "The scenes' footprints cover different fractions of the requested "
            f"area ({earlier.aoi_coverage.fraction:.1%} and "
            f"{later.aoi_coverage.fraction:.1%})."
        )
    return ValidatedScenePair(earlier=earlier, later=later, notes=tuple(notes))


@runtime_checkable
class SceneValidator(Protocol):
    """A reader that validates a scene against its catalog item before reading.

    The real :class:`~app.services.satellite.imagery.ImageryService` does. A
    reader that does not declare it gets no scene validation - the rule lives
    with the reader that owns the catalog lookup, as the read limits do.
    """

    def validate_scene(
        self,
        *,
        scene_id: str,
        collection: str | None,
        modality: Modality,
        assets: Sequence[str],
        bbox: BoundingBox,
        require_all_assets: bool = True,
    ) -> ValidatedScene: ...
