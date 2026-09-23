"""Request/response models for Sentinel-2 scene discovery.

The geospatial :class:`BoundingBox` is reused verbatim - this phase does not
define a competing geometry schema.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.services.geospatial.schemas import BoundingBox

DEFAULT_LIMIT = 10
MAX_LIMIT = 100

# Display-only assets. RTC VV/VH are provider terrain-corrected gamma naught;
# visualization does not perform calibration or expose quantitative SAR analysis.
DEFAULT_IMAGERY_ASSET = "visual"
SAR_IMAGERY_ASSET = "vv"
SUPPORTED_IMAGERY_ASSETS = (DEFAULT_IMAGERY_ASSET, SAR_IMAGERY_ASSET, "vh")

# Assets readable QUANTITATIVELY (raw values, native resolution) for analysis.
# Deliberately a separate allowlist from the display whitelist above: display
# and analysis are different concerns and must not be merged. These are Earth
# Search *STAC asset keys* (common names), not Sentinel-2 band identifiers -
# "green" is the key for band B03, "nir" for B08, "red" for B04. 20 m assets
# ("swir16", "scl") are excluded: mixing them with 10 m bands would require
# resampling, which this phase does not do.
#: ``swir16`` (B11) is 20 m where the others are 10 m. It is readable because
#: NDBI needs it, and mixing resolutions is safe here ONLY through the explicit
#: whole-cell co-registration in ``analysis.engines.coregister_to_finer_grid``,
#: which is guarded and refuses anything it cannot relate exactly. Nothing
#: resamples implicitly.
ANALYSIS_BAND_ASSETS = ("green", "nir", "red", "swir16")

#: Sentinel-2 assets read for PIXEL QUALITY, not measured. ``scl`` is the L2A
#: Scene Classification Layer: 20 m, ``uint8``, nodata 0, CATEGORICAL (read
#: from the live Earth Search item, 2026-09-23). Its own allowlist so a class
#: label can never be read as if it were a reflectance band, and it is placed
#: on the analysis grid only by whole-cell assignment - a class label is never
#: averaged or interpolated (``analysis.pixel_quality``).
QUALITY_BAND_ASSETS = ("scl",)

# Sentinel-1 assets readable QUANTITATIVELY. Deliberately a THIRD allowlist:
# these are not Sentinel-2 spectral bands and they are not display assets, and
# collapsing any two of the three lists would let a SAR asset be read as an
# optical band (or vice versa) without anyone noticing.
#
# WHAT THESE PIXELS ARE - established from the live catalog and the pixels
# themselves (collection ``sentinel-1-rtc``, Microsoft Planetary Computer):
#   * ``raster:bands`` declares ``float32``, ``nodata = -32768``, 10 m, and
#     carries NO ``scale``, NO ``offset`` and NO ``unit``; the COG's own GDAL
#     scale/offset are 1.0/0.0 and its band description reads "Sentinel-1
#     Calibrated and Terrain Corrected VV"/"... VH".
#   * The stored values are LINEAR GAMMA-NAUGHT POWER - not decibels and not
#     amplitude. Over an 844,296 px Chennai window every valid sample was
#     strictly positive (VV median 0.0272, VH median 0.0083), which rules out
#     decibels outright, and the provider's own rendering expression takes a
#     logarithm of the values, which nobody does to data already in dB.
#
# The conversion to decibels therefore belongs to the analysis engine and is
# exactly ``10 * log10(power)``; this layer returns the provider's numbers
# untouched. Only the ``sentinel-1-rtc`` collection publishes these assets -
# Earth Search's Sentinel-1 GRD measurement assets are uncalibrated DN on a
# requester-pays ``s3://`` bucket and remain unreadable here.
SAR_ANALYSIS_BAND_ASSETS = ("vv", "vh")


class SceneSearchRequest(BaseModel):
    """Validated input for a satellite scene search.

    ``collection`` optionally overrides the configured default STAC collection
    (e.g. to target Sentinel-1 rather than Sentinel-2). ``None`` preserves the
    existing Sentinel-2 behaviour.
    """

    bbox: BoundingBox
    start_date: date
    end_date: date
    collection: str | None = None
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)

    @model_validator(mode="after")
    def _check_dates(self) -> Self:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        return self


class SceneAsset(BaseModel):
    """Minimal pointer to a scene asset - metadata only, never downloaded."""

    key: str
    href: str
    type: str | None = None
    title: str | None = None
    roles: list[str] | None = None


class Scene(BaseModel):
    """A normalised Sentinel-2 scene. Never the raw STAC item."""

    id: str
    datetime: str | None
    bbox: BoundingBox | None
    geometry: dict[str, Any] | None
    cloud_cover: float | None
    collection: str | None
    platform: str | None
    processing_level: str | None
    thumbnail_url: str | None
    assets: list[SceneAsset]
    sar_polarizations: list[str] | None = None
    sar_instrument_mode: str | None = None
    orbit_state: str | None = None


class QueryEcho(BaseModel):
    """The exact parameters sent to the STAC API."""

    collections: list[str]
    bbox: list[float]
    datetime: str
    max_cloud_cover: float | None
    limit: int
    filter: dict[str, Any] | None


class SceneSearchResponse(BaseModel):
    """Normalised search result returned to the client."""

    query: QueryEcho
    scene_count: int
    scenes: list[Scene]
    catalog: str
    #: How many scenes the query matched IN THE CATALOG, when it reported that.
    #:
    #: ``scene_count`` is how many are carried here - one bounded page, capped
    #: by ``limit``. Deterministic selection then picks from THIS page, so a
    #: result is the best of what was returned, not the best of what exists.
    #: Reporting only ``scene_count`` made those two indistinguishable.
    #: ``None`` when the catalog does not say: unknown is not "all of them".
    scenes_matched: int | None = None


class ImageryRequest(BaseModel):
    """Bounded imagery request for an already-discovered scene.

    No natural-language search happens here - the scene must already be known
    from the discovery phase.

    ``collection`` names the STAC collection the scene belongs to (e.g.
    ``sentinel-1-grd``); ``None`` preserves the existing Sentinel-2 default.
    """

    scene_id: str = Field(min_length=1, max_length=200)
    bbox: BoundingBox
    asset: str = Field(default=DEFAULT_IMAGERY_ASSET, min_length=1, max_length=50)
    collection: str | None = None
    max_dimension: int | None = Field(default=None, ge=16, le=4096)


class WindowInfo(BaseModel):
    """The pixel window actually read from the source raster."""

    col_off: int
    row_off: int
    width: int
    height: int


class ImageryResponse(BaseModel):
    """A bounded RGB representation of a scene, suitable for a later VLM phase.

    The raw STAC item is never exposed.
    """

    scene_id: str
    bbox: BoundingBox
    asset: str
    asset_href: str
    width: int
    height: int
    format: Literal["png"]
    media_type: Literal["image/png"]
    bands: list[str]
    crs: str | None
    resolution: float | None
    normalization: str
    window: WindowInfo
    source_shape: list[int]
    #: Affine coefficients ``[a, b, c, d, e, f]`` of the window actually read,
    #: in ``crs``. Optional for backward compatibility. This - never ``bbox`` -
    #: is the georeferencing source: ``bbox`` echoes the REQUEST, while the read
    #: window is floor/ceil clamped onto the source grid and so covers more.
    transform: list[float] | None = Field(
        default=None, min_length=6, max_length=6
    )
    #: The image's footprint as exactly four ``[lon, lat]`` pairs in EPSG:4326,
    #: ordered ``[NW, NE, SE, SW]`` - the order a MapLibre image source expects.
    #:
    #: Derived from ``transform`` at the RETURNED image's size, never from
    #: ``bbox``: ``bbox`` echoes the request, while the read window is clamped
    #: onto the source grid. Four corners rather than a rectangle because a
    #: reprojected UTM window is a quadrilateral in WGS84, not an axis-aligned
    #: box.
    #:
    #: Optional for backward compatibility, and ``None`` when the corners could
    #: not be established honestly - a rotated/sheared transform or an unusable
    #: CRS - so a consumer never receives a misleading footprint.
    corners_wgs84: list[list[float]] | None = Field(
        default=None, min_length=4, max_length=4
    )
    image_base64: str
