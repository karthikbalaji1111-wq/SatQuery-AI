"""Stage 1 of the scientific pipeline: is this request allowed to start?

Everything after this module costs something real - a catalog search, a signed
remote open, a band read - and until now the only size check on an analysis
area lived INSIDE the band read, after all of that had already been paid for.
This module is the gate in front of it. It decides, with no network, no clock
beyond "today" and no raster, whether a request is structurally and
semantically eligible to enter the pipeline, and it returns a normalised object
holding only values that passed.

What passing establishes, and what it does not
----------------------------------------------
A :class:`ValidatedAnalysisRequest` says the request is WELL-FORMED: a known
operation on a dataset that can serve it, a finite WGS84 area small enough to
read at native resolution, dates in a coherent order that could contain an
observation, and only parameters the engines actually support. It says nothing
about whether a suitable scene exists, whether its pixels are clear, whether
the radiometry is what the engine assumes, or whether the result will be
meaningful. Those are later stages. A valid request is not a valid
measurement, and nothing here may be reported as if it were.

Two entry points, one set of rules
----------------------------------
* :func:`validate_analysis_request` - the direct contract (area, dataset,
  Date 1 / Date 2, optional scene ids, parameters) that later stages will
  execute. It has no route yet; it is the pipeline's input type.
* :func:`validate_execution_analysis` - the same rules applied to the live
  ``/query/analyze`` contract, called by :class:`AnalysisService` before its
  first read.

Rejections raise :class:`AnalysisRequestRejectedError`: HTTP 422 with a specific,
stable ``code`` in the existing error envelope, and a message that names the
field. The offending value is never echoed beyond a short, truncated form.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from app.core.config import Settings, get_settings
from app.core.errors import InvalidInputError
from app.services.analysis.schemas import AnalysisRequest
from app.services.geospatial.schemas import BoundingBox
from app.services.query.schemas import Modality, TimeRange, observation_period_problem
from app.services.satellite.imagery import QuantitativeReadLimits, is_valid_stac_identifier
from app.services.satellite.rtc import RTC_COLLECTION

_OPTICAL: Modality = "sentinel-2-optical"
_SAR: Modality = "sentinel-1-sar"

# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #

RejectionCode = Literal[
    "field_unknown",
    "analysis_missing",
    "analysis_unknown",
    "analysis_not_implemented",
    "dataset_unknown",
    "dataset_incompatible",
    "aoi_missing",
    "aoi_invalid",
    "aoi_empty",
    "aoi_geometry_unsupported",
    "aoi_crs_unsupported",
    "aoi_too_large",
    "date_missing",
    "date_invalid",
    "date_order_invalid",
    "date_out_of_range",
    "temporal_structure_invalid",
    "temporal_windows_overlap",
    "scene_invalid",
    "parameter_unsupported",
    "parameter_not_applicable",
]


class AnalysisRequestRejectedError(InvalidInputError):
    """A request that may not enter the scientific pipeline.

    A 422 like every :class:`InvalidInputError`, but with a specific ``code`` so
    a client can act on WHICH rule failed without parsing prose. ``field`` is
    kept on the exception for callers and tests; the HTTP envelope is unchanged
    and names the field in its message instead.
    """

    def __init__(self, code: RejectionCode, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}", code=code)
        self.field = field


def _shown(value: object) -> str:
    """A short, safe rendering of client input for a message. Never the whole."""

    text = repr(value)
    return text if len(text) <= 40 else f"{text[:37]}..."


# --------------------------------------------------------------------------- #
# The operation allowlist
# --------------------------------------------------------------------------- #


class AnalysisOperation(StrEnum):
    """Every operation the pipeline RECOGNISES.

    Recognised is not the same as implemented: temporal NDVI and NDBI are named
    here so a request for them is refused as "not implemented" rather than as
    a string nobody has heard of. The difference tells a client whether to fix
    its request or wait for an engine.
    """

    NDVI = "ndvi"
    NDWI = "ndwi"
    NDBI = "ndbi"
    SAR_BACKSCATTER = "sar_backscatter"
    TEMPORAL_NDWI = "temporal_ndwi"
    TEMPORAL_NDVI = "temporal_ndvi"
    TEMPORAL_NDBI = "temporal_ndbi"


@dataclass(frozen=True)
class OperationSpec:
    """What an operation needs, stated once."""

    operation: AnalysisOperation
    modality: Modality
    #: Two observations (Date 1 and Date 2) rather than one.
    temporal: bool
    #: Whether an engine exists. Unimplemented operations are refused.
    implemented: bool
    #: Pixel spacing of the finest grid the operation reads. The reader sizes
    #: its native window on this grid, so it is what the area limit is judged
    #: against. Sentinel-2 10 m bands: live COG headers (CLAUDE.md section 6).
    #: Sentinel-1 RTC: the collection's ``raster:bands.spatial_resolution`` for
    #: ``vv``/``vh`` is 10 (read from Planetary Computer, 2026-09-22) - the SAR
    #: RESOLUTION is 20 m, but the window is counted in grid pixels.
    grid_spacing_m: float


OPERATIONS: Mapping[AnalysisOperation, OperationSpec] = {
    spec.operation: spec
    for spec in (
        OperationSpec(AnalysisOperation.NDVI, _OPTICAL, False, True, 10.0),
        OperationSpec(AnalysisOperation.NDWI, _OPTICAL, False, True, 10.0),
        # NDBI's SWIR band is 20 m, but its NIR band is 10 m and is read over
        # the same area, so the 10 m read is the one that bounds the window.
        OperationSpec(AnalysisOperation.NDBI, _OPTICAL, False, True, 10.0),
        OperationSpec(AnalysisOperation.SAR_BACKSCATTER, _SAR, False, True, 10.0),
        OperationSpec(AnalysisOperation.TEMPORAL_NDWI, _OPTICAL, True, True, 10.0),
        OperationSpec(AnalysisOperation.TEMPORAL_NDVI, _OPTICAL, True, False, 10.0),
        OperationSpec(AnalysisOperation.TEMPORAL_NDBI, _OPTICAL, True, False, 10.0),
    )
}


def _parse_operation(raw: object) -> OperationSpec:
    if raw is None:
        raise AnalysisRequestRejectedError(
            "analysis_missing", "operation", "an analysis operation is required."
        )
    supported = ", ".join(op.value for op, s in OPERATIONS.items() if s.implemented)
    if not isinstance(raw, str):
        raise AnalysisRequestRejectedError(
            "analysis_unknown", "operation", f"must be one of: {supported}."
        )
    try:
        operation = AnalysisOperation(raw.strip().lower())
    except ValueError:
        raise AnalysisRequestRejectedError(
            "analysis_unknown",
            "operation",
            f"{_shown(raw)} is not a supported analysis. Supported: {supported}.",
        ) from None
    spec = OPERATIONS[operation]
    if not spec.implemented:
        raise AnalysisRequestRejectedError(
            "analysis_not_implemented",
            "operation",
            f"{operation.value} is recognised but has no engine yet. "
            f"Supported: {supported}.",
        )
    return spec


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def _known_datasets(settings: Settings) -> dict[str, Modality]:
    """Every collection this deployment knows, and the sensor family it holds."""

    return {
        settings.stac_collection: _OPTICAL,
        settings.stac_s1_collection: _SAR,
        RTC_COLLECTION: _SAR,
    }


def _analysable_dataset(spec: OperationSpec, settings: Settings) -> str:
    """The one collection each family is MEASURED from.

    SAR is RTC regardless of the configured discovery collection: backscatter
    is only a physical quantity in the provider's terrain-corrected product,
    and the reader refuses a polarization from anywhere else.
    """

    return settings.stac_collection if spec.modality == _OPTICAL else RTC_COLLECTION


def _parse_dataset(raw: object, spec: OperationSpec, settings: Settings) -> str:
    expected = _analysable_dataset(spec, settings)
    if raw is None:
        return expected
    known = _known_datasets(settings)
    if not isinstance(raw, str) or raw not in known:
        raise AnalysisRequestRejectedError(
            "dataset_unknown",
            "dataset",
            f"{_shown(raw)} is not a dataset this deployment serves. "
            f"Known: {', '.join(sorted(known))}.",
        )
    if known[raw] != spec.modality:
        family = "an optical" if spec.modality == _OPTICAL else "a SAR"
        raise AnalysisRequestRejectedError(
            "dataset_incompatible",
            "dataset",
            f"{spec.operation.value} needs {family} dataset; {raw} holds "
            f"{'SAR' if known[raw] == _SAR else 'optical'} data.",
        )
    if raw != expected:
        raise AnalysisRequestRejectedError(
            "dataset_incompatible",
            "dataset",
            f"{spec.operation.value} is measured only from {expected}; {raw} "
            "cannot be read quantitatively by this pipeline.",
        )
    return raw


# --------------------------------------------------------------------------- #
# The area of interest
# --------------------------------------------------------------------------- #

#: Spellings accepted for the one CRS the pipeline takes. Coordinates are
#: ALWAYS read as longitude, latitude - the GeoJSON convention - whichever of
#: these names the request uses. Nothing is transformed.
_WGS84_NAMES = frozenset({
    "EPSG:4326",
    "OGC:CRS84",
    "urn:ogc:def:crs:OGC:1.3:CRS84",
    "urn:ogc:def:crs:EPSG::4326",
})

_GEOJSON_TYPES = frozenset({
    "Point", "MultiPoint", "LineString", "MultiLineString",
    "Polygon", "MultiPolygon", "GeometryCollection",
})


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _coordinate_pair(value: object, field: str) -> tuple[float, float]:
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or not all(_is_number(v) for v in value)
    ):
        raise AnalysisRequestRejectedError(
            "aoi_invalid", field, "each position must be [longitude, latitude]."
        )
    lon, lat = float(value[0]), float(value[1])
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise AnalysisRequestRejectedError("aoi_invalid", field, "coordinates must be finite.")
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise AnalysisRequestRejectedError(
            "aoi_invalid",
            field,
            "longitude must lie in [-180, 180] and latitude in [-90, 90].",
        )
    return lon, lat


def _bbox_from_values(values: object, field: str) -> BoundingBox:
    if not isinstance(values, list | tuple) or len(values) != 4:
        raise AnalysisRequestRejectedError(
            "aoi_invalid", field, "must be [west, south, east, north]."
        )
    if not all(_is_number(v) for v in values):
        raise AnalysisRequestRejectedError("aoi_invalid", field, "values must be numbers.")
    west, south, east, north = (float(v) for v in values)
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise AnalysisRequestRejectedError("aoi_invalid", field, "values must be finite.")
    _coordinate_pair([west, south], field)
    _coordinate_pair([east, north], field)
    if west == east or south == north:
        raise AnalysisRequestRejectedError(
            "aoi_empty", field, "the box has zero width or height."
        )
    if west > east:
        # Not repaired: "west > east" is either a swapped box or one crossing the
        # antimeridian, and guessing which would measure the wrong place.
        raise AnalysisRequestRejectedError(
            "aoi_invalid",
            field,
            "west is greater than east. Boxes crossing the antimeridian are not "
            "supported; a swapped box is not corrected.",
        )
    if south > north:
        raise AnalysisRequestRejectedError(
            "aoi_invalid", field, "south is greater than north; it is not corrected."
        )
    return BoundingBox(west=west, south=south, east=east, north=north)


def _bbox_from_polygon(geometry: Mapping[str, Any], field: str) -> BoundingBox:
    """An exact axis-aligned rectangle, as a box. Anything else is refused.

    A rectangle drawn on the map arrives as a Polygon, and it IS its bounding
    box - the conversion loses nothing, which is the only reason it is made.
    Any other polygon is refused rather than enveloped: the measurement window
    is rectangular, and measuring a polygon's envelope would count pixels
    outside the area the reader drew.
    """

    rings = geometry.get("coordinates")
    if not isinstance(rings, list):
        raise AnalysisRequestRejectedError(
            "aoi_invalid", f"{field}.coordinates", "must be a list of rings."
        )
    if not rings:
        raise AnalysisRequestRejectedError(
            "aoi_empty", f"{field}.coordinates", "the polygon has no rings."
        )
    if len(rings) > 1:
        raise AnalysisRequestRejectedError(
            "aoi_geometry_unsupported",
            f"{field}.coordinates",
            "polygons with holes are not supported.",
        )
    ring = rings[0]
    if not isinstance(ring, list):
        raise AnalysisRequestRejectedError(
            "aoi_invalid", f"{field}.coordinates[0]", "must be a list of positions."
        )
    if not ring:
        raise AnalysisRequestRejectedError(
            "aoi_empty", f"{field}.coordinates[0]", "the ring has no positions."
        )
    points = [
        _coordinate_pair(position, f"{field}.coordinates[0][{i}]")
        for i, position in enumerate(ring)
    ]
    if len(points) < 4:
        raise AnalysisRequestRejectedError(
            "aoi_invalid",
            f"{field}.coordinates[0]",
            "a ring needs at least four positions.",
        )
    if points[0] != points[-1]:
        raise AnalysisRequestRejectedError(
            "aoi_invalid",
            f"{field}.coordinates[0]",
            "the ring is not closed; its first and last positions differ.",
        )
    vertices = points[:-1]

    # Classified BEFORE the area test: a bow-tie through a rectangle's corners
    # has zero signed area (its two triangles cancel), so testing area first
    # reported a self-intersecting ring as an empty one.
    xs = sorted({x for x, _ in vertices})
    ys = sorted({y for _, y in vertices})
    corners = {(x, y) for x in xs for y in ys}
    is_corner_set = len(vertices) == 4 and len(xs) == 2 and len(ys) == 2 and set(
        vertices
    ) == corners
    if is_corner_set:
        # Walking a rectangle's boundary changes one coordinate per edge. The
        # same four corners in bow-tie order change both - a self-intersecting
        # ring, which is malformed rather than merely unsupported.
        edges = zip(vertices, vertices[1:] + vertices[:1], strict=True)
        if all((a[0] == b[0]) != (a[1] == b[1]) for a, b in edges):
            return BoundingBox(west=xs[0], south=ys[0], east=xs[1], north=ys[1])
        raise AnalysisRequestRejectedError(
            "aoi_invalid",
            f"{field}.coordinates[0]",
            "the ring crosses itself.",
        )

    # Shoelace area: zero means every vertex lies on one line (or one point).
    area = 0.0
    for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        area += x1 * y2 - x2 * y1
    if area == 0.0:
        raise AnalysisRequestRejectedError(
            "aoi_empty", f"{field}.coordinates[0]", "the polygon has zero area."
        )
    raise AnalysisRequestRejectedError(
        "aoi_geometry_unsupported",
        field,
        "only axis-aligned rectangles are supported. A measurement window is "
        "rectangular, and measuring this polygon's envelope would include "
        "pixels outside the drawn area.",
    )


@dataclass(frozen=True)
class _ParsedAoi:
    bbox: BoundingBox
    source: Literal["bbox", "polygon"]


def _parse_aoi(raw: object) -> _ParsedAoi:
    if raw is None:
        raise AnalysisRequestRejectedError("aoi_missing", "aoi", "an area of interest is required.")
    if not isinstance(raw, Mapping):
        raise AnalysisRequestRejectedError(
            "aoi_invalid", "aoi", "must be an object with 'bbox' or 'geometry'."
        )
    unknown = set(raw) - {"bbox", "geometry", "crs"}
    if unknown:
        name = sorted(map(str, unknown))[0]
        raise AnalysisRequestRejectedError(
            "field_unknown", f"aoi.{_shown(name)}", "is not a field of the area."
        )

    crs = raw.get("crs")
    if crs is not None and (not isinstance(crs, str) or crs.strip() not in _WGS84_NAMES):
        raise AnalysisRequestRejectedError(
            "aoi_crs_unsupported",
            "aoi.crs",
            f"{_shown(crs)} is not supported. Supply WGS84 longitude/latitude "
            "(EPSG:4326 or OGC:CRS84); coordinates are never reprojected here.",
        )

    bbox_raw, geometry = raw.get("bbox"), raw.get("geometry")
    if bbox_raw is None and geometry is None:
        raise AnalysisRequestRejectedError(
            "aoi_missing", "aoi", "provide exactly one of 'bbox' or 'geometry'."
        )
    if bbox_raw is not None and geometry is not None:
        raise AnalysisRequestRejectedError(
            "aoi_invalid",
            "aoi",
            "provide exactly one of 'bbox' or 'geometry', not both - two "
            "descriptions of one area can disagree.",
        )
    if bbox_raw is not None:
        return _ParsedAoi(_bbox_from_values(bbox_raw, "aoi.bbox"), "bbox")

    if not isinstance(geometry, Mapping) or not isinstance(geometry.get("type"), str):
        raise AnalysisRequestRejectedError(
            "aoi_invalid", "aoi.geometry", "must be a GeoJSON geometry object."
        )
    kind = geometry["type"]
    if kind not in _GEOJSON_TYPES:
        raise AnalysisRequestRejectedError(
            "aoi_invalid", "aoi.geometry.type", f"{_shown(kind)} is not a GeoJSON type."
        )
    if kind != "Polygon":
        raise AnalysisRequestRejectedError(
            "aoi_geometry_unsupported",
            "aoi.geometry.type",
            f"{kind} is not supported; an analysis area is a rectangle.",
        )
    return _ParsedAoi(_bbox_from_polygon(geometry, "aoi.geometry"), "polygon")


# -- size ------------------------------------------------------------------ #

_WGS84_A = 6_378_137.0
_WGS84_E2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)

#: Converts a true ground extent into a LOWER BOUND on the reader's window.
#:
#: The reader projects the box into the scene's UTM zone and takes the pixel
#: envelope. UTM distorts ground distance by a scale factor no smaller than
#: 0.9996 (at the central meridian), and grid convergence tilts a parallel by
#: a few degrees at most within a zone (cos 4 deg = 0.9976). 0.99 sits below
#: their product, so an area this module rejects is one the reader would
#: reject too - the early check can only ever be the stricter-sounding copy of
#: the real one, never a new, stricter rule. Verified against rasterio's own
#: transform in the tests.
_WINDOW_LOWER_BOUND = 0.99


def _parallel_radius(lat_deg: float) -> float:
    phi = math.radians(lat_deg)
    s = math.sin(phi)
    return _WGS84_A * math.cos(phi) / math.sqrt(1.0 - _WGS84_E2 * s * s)


def _meridian_radius(lat_deg: float) -> float:
    s = math.sin(math.radians(lat_deg))
    return _WGS84_A * (1.0 - _WGS84_E2) / (1.0 - _WGS84_E2 * s * s) ** 1.5


def aoi_extent_m(bbox: BoundingBox) -> tuple[float, float]:
    """Ground width and height of ``bbox`` in metres, on the WGS84 ellipsoid.

    Width is taken along the widest parallel in the box - the one nearest the
    equator - because that is the parallel that sets the projected envelope.
    Height is the meridional arc, integrated with Simpson's rule.
    """

    if bbox.south <= 0.0 <= bbox.north:
        widest = 0.0
    else:
        widest = bbox.south if abs(bbox.south) < abs(bbox.north) else bbox.north
    width = math.radians(bbox.east - bbox.west) * _parallel_radius(widest)

    panels = 16
    step = (bbox.north - bbox.south) / panels
    total = _meridian_radius(bbox.south) + _meridian_radius(bbox.north)
    for i in range(1, panels):
        total += (4 if i % 2 else 2) * _meridian_radius(bbox.south + i * step)
    height = math.radians(step) * total / 3.0
    return width, height


def estimated_window_pixels(bbox: BoundingBox, grid_spacing_m: float) -> tuple[float, float]:
    """A lower bound on the native window the reader would open over ``bbox``."""

    width, height = aoi_extent_m(bbox)
    return (
        width * _WINDOW_LOWER_BOUND / grid_spacing_m,
        height * _WINDOW_LOWER_BOUND / grid_spacing_m,
    )


def _check_aoi_size(
    bbox: BoundingBox,
    grid_spacing_m: float,
    limits: QuantitativeReadLimits | None,
    field: str,
) -> None:
    """Refuse an area the reader would refuse, before anything is fetched.

    ``limits`` is whatever the reader DECLARES. ``None`` - a reader that
    declares none - means no early check: the rule belongs to the reader, and
    this module only moves it earlier.
    """

    if limits is None:
        return
    columns, rows = estimated_window_pixels(bbox, grid_spacing_m)
    if max(columns, rows) <= limits.max_dimension and (
        columns * rows <= limits.max_window_pixels
    ):
        return
    width_m, height_m = aoi_extent_m(bbox)
    limit_km = limits.max_dimension * grid_spacing_m / 1000.0
    raise AnalysisRequestRejectedError(
        "aoi_too_large",
        field,
        f"this area is about {width_m / 1000:.1f} x {height_m / 1000:.1f} km. "
        f"Measurements are read on the sensor's native {grid_spacing_m:.0f} m "
        "grid and are never downsampled, which limits an analysis area to "
        f"roughly {limit_km:.0f} km across. Choose a smaller area.",
    )


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _parse_date(value: object, field: str) -> date:
    if value is None:
        raise AnalysisRequestRejectedError("date_missing", field, "a date is required.")
    if isinstance(value, datetime):
        raise AnalysisRequestRejectedError(
            "date_invalid",
            field,
            "must be a calendar date (YYYY-MM-DD); time of day is not used to "
            "select observations and is not silently discarded.",
        )
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or _ISO_DATE.fullmatch(value.strip()) is None:
        raise AnalysisRequestRejectedError(
            "date_invalid",
            field,
            f"{_shown(value)} is not a calendar date in YYYY-MM-DD form.",
        )
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise AnalysisRequestRejectedError(
            "date_invalid", field, f"{_shown(value)} is not a real calendar date."
        ) from None


Role = Literal["single", "date1", "date2"]


class ValidatedObservation(BaseModel):
    """One side of an analysis: a period, and optionally a named scene."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    period: TimeRange
    #: A scene the CLIENT named. Its identifier has the right form; that is all.
    #: It has not been looked up, and nothing the client said about it - date,
    #: cloud cover, footprint - is accepted here. Scene selection (the next
    #: stage) reads it from the catalog, which stays authoritative.
    requested_scene_id: str | None = None


def _parse_observation(raw: object, field: str, role: Role, today: date) -> ValidatedObservation:
    if raw is None:
        raise AnalysisRequestRejectedError("date_missing", field, "is required.")
    if not isinstance(raw, Mapping):
        raise AnalysisRequestRejectedError(
            "date_invalid", field, "must be an object with start_date and end_date."
        )
    unknown = set(raw) - {"start_date", "end_date", "scene_id"}
    if unknown:
        name = sorted(map(str, unknown))[0]
        raise AnalysisRequestRejectedError(
            "field_unknown",
            f"{field}.{_shown(name)}",
            "is not accepted; scene metadata is read from the catalog, not the request.",
        )
    start = _parse_date(raw.get("start_date"), f"{field}.start_date")
    end = _parse_date(raw.get("end_date"), f"{field}.end_date")
    if end < start:
        raise AnalysisRequestRejectedError(
            "date_order_invalid", f"{field}.end_date", "is before start_date."
        )
    period = TimeRange(start_date=start, end_date=end)
    problem = observation_period_problem(period, today=today)
    if problem is not None:
        raise AnalysisRequestRejectedError("date_out_of_range", field, f"{problem}.")

    scene_id = raw.get("scene_id")
    if scene_id is not None and not is_valid_stac_identifier(scene_id):
        raise AnalysisRequestRejectedError(
            "scene_invalid",
            f"{field}.scene_id",
            "is not a valid scene identifier (1-200 characters from A-Z, a-z, "
            "0-9, '.', '_' or '-').",
        )
    return ValidatedObservation(role=role, period=period, requested_scene_id=scene_id)


def _check_temporal_order(first: ValidatedObservation, second: ValidatedObservation) -> None:
    """Date 1 is the EARLIER observation, by the request's own dates.

    Order is never inferred from position, and overlapping periods are refused
    rather than accepted: if Date 1 and Date 2 can share a day, which one is
    earlier would be decided by whichever scenes happen to be found, not by the
    request - and a comparison whose direction can flip is not a comparison.
    """

    a, b = first.period, second.period
    if a.start_date > b.end_date:
        raise AnalysisRequestRejectedError(
            "date_order_invalid",
            "date1",
            f"Date 1 ({a.start_date.isoformat()}..{a.end_date.isoformat()}) must be "
            f"the EARLIER observation, but it falls after Date 2 "
            f"({b.start_date.isoformat()}..{b.end_date.isoformat()}). It is not "
            "swapped for you.",
        )
    if not a.end_date < b.start_date:
        raise AnalysisRequestRejectedError(
            "temporal_windows_overlap",
            "date2",
            "Date 1 and Date 2 overlap, so which observation is earlier would be "
            "decided by the scenes found rather than by the request. Date 1 must "
            "end before Date 2 begins.",
        )
    if (
        first.requested_scene_id is not None
        and first.requested_scene_id == second.requested_scene_id
    ):
        raise AnalysisRequestRejectedError(
            "scene_invalid",
            "date2.scene_id",
            "is the same scene as Date 1; one acquisition cannot be both sides of "
            "a comparison.",
        )


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

_PARAMETER_KEYS = frozenset({"resolution", "resampling", "polarizations", "max_cloud_cover"})
_SAR_POLARIZATIONS = ("vv", "vh")


class ValidatedParameters(BaseModel):
    """Processing parameters, restricted to what the engines actually do."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The only resolution there is: quantitative reads are never resampled.
    resolution: Literal["native"] = "native"
    #: SAR only. The engine measures VV and VH together, so this is both.
    polarizations: tuple[Literal["vv", "vh"], ...] | None = None
    #: Optical only: an upper bound on scene cloud cover for scene selection.
    max_cloud_cover: float | None = None


def _parse_parameters(raw: object, spec: OperationSpec) -> ValidatedParameters:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise AnalysisRequestRejectedError(
            "parameter_unsupported", "parameters", "must be an object."
        )
    unknown = set(raw) - _PARAMETER_KEYS
    if unknown:
        name = sorted(map(str, unknown))[0]
        raise AnalysisRequestRejectedError(
            "parameter_unsupported",
            f"parameters.{_shown(name)}",
            f"is not a processing parameter. Supported: {', '.join(sorted(_PARAMETER_KEYS))}.",
        )

    resolution = raw.get("resolution")
    if resolution is not None and resolution != "native":
        raise AnalysisRequestRejectedError(
            "parameter_unsupported",
            "parameters.resolution",
            f"{_shown(resolution)} is not supported. Measurements are made on the "
            "sensor's native grid only; the only accepted value is 'native'.",
        )

    if raw.get("resampling") is not None:
        raise AnalysisRequestRejectedError(
            "parameter_unsupported",
            "parameters.resampling",
            "cannot be chosen per request. Resampling is fixed by the engine: "
            "single-grid indices are not resampled at all, and NDBI's 20 m band "
            "is placed on the 10 m grid by whole-cell assignment.",
        )

    polarizations = raw.get("polarizations")
    parsed_polarizations: tuple[Literal["vv", "vh"], ...] | None = None
    if spec.modality == _SAR:
        if polarizations is None:
            parsed_polarizations = _SAR_POLARIZATIONS
        else:
            if not isinstance(polarizations, list | tuple) or not all(
                isinstance(p, str) for p in polarizations
            ):
                raise AnalysisRequestRejectedError(
                    "parameter_unsupported",
                    "parameters.polarizations",
                    "must be a list of polarization names.",
                )
            names = [p.strip().lower() for p in polarizations]
            if not set(names) <= set(_SAR_POLARIZATIONS) or len(set(names)) != len(names):
                raise AnalysisRequestRejectedError(
                    "parameter_unsupported",
                    "parameters.polarizations",
                    "only 'vv' and 'vh' are measured, each at most once.",
                )
            if set(names) != set(_SAR_POLARIZATIONS):
                raise AnalysisRequestRejectedError(
                    "parameter_unsupported",
                    "parameters.polarizations",
                    "the SAR engine measures VV and VH together; one polarization "
                    "cannot be requested on its own.",
                )
            parsed_polarizations = _SAR_POLARIZATIONS
    elif polarizations is not None:
        raise AnalysisRequestRejectedError(
            "parameter_not_applicable",
            "parameters.polarizations",
            f"applies to SAR only; {spec.operation.value} is optical.",
        )

    cloud = raw.get("max_cloud_cover")
    parsed_cloud: float | None = None
    if cloud is not None:
        if spec.modality == _SAR:
            raise AnalysisRequestRejectedError(
                "parameter_not_applicable",
                "parameters.max_cloud_cover",
                "applies to optical data only; radar is not obscured by cloud.",
            )
        if not _is_number(cloud) or not math.isfinite(float(cloud)) or not (
            0.0 <= float(cloud) <= 100.0
        ):
            raise AnalysisRequestRejectedError(
                "parameter_unsupported",
                "parameters.max_cloud_cover",
                "must be a number from 0 to 100.",
            )
        parsed_cloud = float(cloud)

    return ValidatedParameters(
        polarizations=parsed_polarizations, max_cloud_cover=parsed_cloud
    )


# --------------------------------------------------------------------------- #
# The direct contract
# --------------------------------------------------------------------------- #


class ValidatedAoi(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bbox: BoundingBox
    crs: Literal["EPSG:4326"] = "EPSG:4326"
    #: How the client described it. "polygon" means an exact rectangle drawn as
    #: a polygon and converted losslessly; nothing was repaired or enveloped.
    source: Literal["bbox", "polygon"]
    width_m: float
    height_m: float


class ValidatedAnalysisRequest(BaseModel):
    """A request the scientific pipeline may start. Holds only validated values.

    ``validation_stage`` names what was established so it travels with the
    object: the request is eligible - not that a scene exists, that its pixels
    are usable, or that any measurement will be valid.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: AnalysisOperation
    modality: Modality
    dataset: str
    aoi: ValidatedAoi
    #: ``(single,)`` or ``(date1, date2)``, Date 1 strictly earlier.
    observations: tuple[ValidatedObservation, ...]
    parameters: ValidatedParameters
    validation_stage: Literal["request"] = "request"

    @property
    def is_temporal(self) -> bool:
        return len(self.observations) == 2


_REQUEST_KEYS = frozenset(
    {"operation", "dataset", "aoi", "time_window", "date1", "date2", "parameters"}
)


def validate_analysis_request(
    payload: object,
    *,
    limits: QuantitativeReadLimits | None = None,
    settings: Settings | None = None,
    today: date | None = None,
) -> ValidatedAnalysisRequest:
    """Validate a direct analysis request, or raise :class:`AnalysisRequestRejectedError`.

    Pure: no network, no raster, no catalog. ``limits`` is what the reader
    declares; ``None`` skips only the size check. The first failing rule is
    reported, in a fixed order - what to do before where, where before when -
    so the same bad request always produces the same code.
    """

    settings = settings or get_settings()
    today = today or datetime.now(UTC).date()

    if not isinstance(payload, Mapping):
        raise AnalysisRequestRejectedError("field_unknown", "body", "must be a JSON object.")
    unknown = set(payload) - _REQUEST_KEYS
    if unknown:
        name = sorted(map(str, unknown))[0]
        raise AnalysisRequestRejectedError(
            "field_unknown", _shown(name), "is not a field of an analysis request."
        )

    spec = _parse_operation(payload.get("operation"))
    dataset = _parse_dataset(payload.get("dataset"), spec, settings)

    parsed = _parse_aoi(payload.get("aoi"))
    _check_aoi_size(parsed.bbox, spec.grid_spacing_m, limits, "aoi")
    width_m, height_m = aoi_extent_m(parsed.bbox)
    aoi = ValidatedAoi(
        bbox=parsed.bbox, source=parsed.source, width_m=width_m, height_m=height_m
    )

    window, date1, date2 = (payload.get(k) for k in ("time_window", "date1", "date2"))
    observations: tuple[ValidatedObservation, ...]
    if spec.temporal:
        if window is not None:
            raise AnalysisRequestRejectedError(
                "temporal_structure_invalid",
                "time_window",
                f"{spec.operation.value} compares two observations; give date1 "
                "and date2 instead.",
            )
        first = _parse_observation(date1, "date1", "date1", today)
        second = _parse_observation(date2, "date2", "date2", today)
        _check_temporal_order(first, second)
        observations = (first, second)
    else:
        if date1 is not None or date2 is not None:
            raise AnalysisRequestRejectedError(
                "temporal_structure_invalid",
                "date1" if date1 is not None else "date2",
                f"{spec.operation.value} measures one observation; give "
                "time_window instead.",
            )
        observations = (_parse_observation(window, "time_window", "single", today),)

    return ValidatedAnalysisRequest(
        operation=spec.operation,
        modality=spec.modality,
        dataset=dataset,
        aoi=aoi,
        observations=observations,
        parameters=_parse_parameters(payload.get("parameters"), spec),
    )


# --------------------------------------------------------------------------- #
# The live contract: /query/analyze
# --------------------------------------------------------------------------- #


@runtime_checkable
class DeclaresReadLimits(Protocol):
    """A band reader that states the window it refuses beyond."""

    def quantitative_read_limits(self) -> QuantitativeReadLimits: ...


@dataclass(frozen=True)
class ExecutionAnalysisGate:
    """What the gate established about a live analysis request."""

    operations: tuple[AnalysisOperation, ...]
    bbox: BoundingBox
    #: False when the reader declared no limits, so no early size check ran.
    size_checked: bool


def operations_for_flags(
    indices: Sequence[str] = (),
    *,
    include_ndwi: bool = False,
    include_temporal_ndwi: bool = False,
    include_sar_backscatter: bool = False,
) -> tuple[AnalysisOperation, ...]:
    """The live contract's flags, in the allowlist's vocabulary, de-duplicated.

    Takes the flags themselves rather than an :class:`AnalysisRequest`, so the
    agent can state what it WILL ask for before any execution result exists -
    which is what lets the area rule run before discovery.
    """

    ordered: list[AnalysisOperation] = [AnalysisOperation(key) for key in indices]
    if include_ndwi:
        ordered.append(AnalysisOperation.NDWI)
    if include_temporal_ndwi:
        ordered.append(AnalysisOperation.TEMPORAL_NDWI)
    if include_sar_backscatter:
        ordered.append(AnalysisOperation.SAR_BACKSCATTER)
    return tuple(dict.fromkeys(ordered))


def requested_operations(request: AnalysisRequest) -> tuple[AnalysisOperation, ...]:
    """The live request's flags, in the allowlist's vocabulary, de-duplicated."""

    return operations_for_flags(
        request.indices,
        include_ndwi=request.include_ndwi,
        include_temporal_ndwi=request.include_temporal_ndwi,
        include_sar_backscatter=request.include_sar_backscatter,
    )


def check_operations_area(
    bbox: BoundingBox,
    operations: Sequence[AnalysisOperation],
    limits: QuantitativeReadLimits | None,
    field: str,
) -> None:
    """The area rule for a set of operations, on its own.

    The one statement of it that both the ``/query/analyze`` gate and the
    agent's pre-discovery check apply, so the two cannot disagree about which
    area is too large. No operations means nothing will be read: no check.
    """

    if not operations:
        return
    spacing = min(OPERATIONS[op].grid_spacing_m for op in operations)
    _check_aoi_size(bbox, spacing, limits, field)


def validate_execution_analysis(
    request: AnalysisRequest,
    *,
    limits: QuantitativeReadLimits | None,
    settings: Settings | None = None,
) -> ExecutionAnalysisGate:
    """Apply the request-level rules to a live ``/query/analyze`` request.

    Runs before the first read. It refuses the WHOLE request only for defects
    that spoil every operation in it:

    * an area too large to read at native resolution;
    * a window that will be read whose scene comes from a collection outside
      the window's own sensor family - a Sentinel-1 scene in an optical
      window. ``/query/execute`` cannot produce that; only a fabricated result
      can, and reading it would fetch from the wrong catalog before failing.

    It deliberately does NOT refuse an operation for which the execution simply
    holds no suitable scene. That is data availability, not a malformed
    request: this contract coalesces several operations into one call, and a
    missing Sentinel-1 scene is no reason to discard a valid NDVI beside it.
    Those operations already stop before any read and are reported per
    operation in ``analysis_outcomes``.
    """

    settings = settings or get_settings()
    operations = requested_operations(request)
    execution = request.execution
    bbox = execution.plan.bbox
    if not operations:
        # Nothing quantitative is read, so there is nothing to protect.
        return ExecutionAnalysisGate(operations=(), bbox=bbox, size_checked=False)

    specs = [OPERATIONS[op] for op in operations]
    check_operations_area(bbox, operations, limits, "execution.plan.bbox")

    known = _known_datasets(settings)
    families = {s.modality for s in specs}
    for index, window in enumerate(execution.windows):
        if window.modality not in families or window.selected_scene_id is None:
            continue
        collection = next(
            (s.collection for s in window.scenes if s.id == window.selected_scene_id),
            None,
        )
        if not collection:
            # The reader falls back to its configured default, which for an
            # optical window is the optical collection and for a SAR window is
            # refused before any fetch. Neither reaches the wrong catalog.
            continue
        field = f"execution.windows[{index}]"
        family = known.get(collection)
        if family is None:
            raise AnalysisRequestRejectedError(
                "dataset_unknown",
                field,
                f"selected a scene from {_shown(collection)}, which this deployment "
                "does not serve.",
            )
        if family != window.modality:
            raise AnalysisRequestRejectedError(
                "dataset_incompatible",
                field,
                f"is a {window.modality} window but selected a scene from "
                f"{collection}, a {family} collection.",
            )

    return ExecutionAnalysisGate(
        operations=operations, bbox=bbox, size_checked=limits is not None
    )
