"""Bounded imagery retrieval.

scene_id + bbox + asset -> STAC item lookup (metadata only) -> (Sentinel-1 RTC
only: bounded SAS signing) -> windowed COG read -> standardized RGB PNG. No
AI/VLM logic lives here.

Two sources, one contract. Sentinel-2 assets on Earth Search are anonymously
readable. Sentinel-1 RTC assets on the Planetary Computer are NOT: the storage
account refuses public access (HTTP 409 PublicAccessNotPermitted), so each read
is preceded by a call to the provider's anonymous signing endpoint. The
resulting token is used to open the raster and never leaves the server -
``ImageryResponse.asset_href`` carries the unsigned href.
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, get_settings
from app.core.errors import (
    ImageryError,
    InvalidInputError,
    NotFoundError,
    UpstreamServiceError,
)
from app.core.logging import get_logger
from app.services.base import DomainService
from app.services.geospatial.schemas import BoundingBox
from app.services.satellite.raster import (
    BandWindow,
    RgbWindow,
    image_corners_wgs84,
    read_band_window,
    read_rgb_window,
)
from app.services.satellite.rtc import (
    RTC_COLLECTION,
    catalog_for,
    require_linear_power_encoding,
    sign_rtc_asset,
)
from app.services.satellite.scene_validation import (
    Modality,
    SceneValidationError,
    ValidatedScene,
    validate_scene_item,
)
from app.services.satellite.schemas import (
    ANALYSIS_BAND_ASSETS,
    QUALITY_BAND_ASSETS,
    SAR_ANALYSIS_BAND_ASSETS,
    SUPPORTED_IMAGERY_ASSETS,
    ImageryRequest,
    ImageryResponse,
    WindowInfo,
)

logger = get_logger("satellite.imagery")

StacItemFetcher = Callable[[str, str], dict[str, Any]]
RasterReader = Callable[..., RgbWindow]
BandReader = Callable[..., BandWindow]

_COG_TYPE_HINT = "geotiff"

# --------------------------------------------------------------------------- #
# STAC identifier validation
#
# ``scene_id`` and ``collection`` are interpolated into the STAC item URL by
# ``_default_fetch_item``. They arrive from a client-supplied
# ``QueryExecutionResult`` through /query/analyze, so at this boundary they are
# untrusted input.
#
# The HOST is always ``settings.stac_base_url`` and is never taken from the
# request, so this is NOT arbitrary-host SSRF. What is reachable without
# validation is:
#
#   * fixed-host path manipulation - ``collection="../../../search"`` rewrites
#     the request path from /collections/{c}/items/{id} to /search/items/{id};
#     a ``?`` or ``#`` likewise splits off a query or fragment;
#   * remote-read resource abuse - unbounded, arbitrary identifiers driving
#     outbound requests on the server's behalf.
#
# Real Earth Search identifiers ("S2B_44PLV_20241026_0_L2A", "sentinel-2-l2a")
# are covered by this allowlist, so the check refuses malformed input without
# constraining legitimate use.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class QuantitativeReadLimits:
    """How large a native-resolution window :meth:`ImageryService.read_band` reads.

    Quantitative reads are never decimated, so these are REJECTION bounds: a
    window beyond either is refused, not resampled to fit.
    """

    #: Longest side of the native window, in pixels.
    max_dimension: int
    #: Width x height of the native window, in pixels.
    max_window_pixels: int


_STAC_IDENTIFIER = re.compile(r"[A-Za-z0-9._-]{1,200}")
#: Path segments that would traverse even though their characters are allowed.
_RESERVED_SEGMENTS = frozenset({".", ".."})


def is_valid_stac_identifier(value: object) -> bool:
    """Whether ``value`` is a safe single STAC URL path segment.

    The predicate behind :func:`_validate_stac_identifier`, public so the
    analysis request gate can apply the SAME rule before any catalog call
    instead of restating it.
    """

    return (
        isinstance(value, str)
        and _STAC_IDENTIFIER.fullmatch(value) is not None
        and value not in _RESERVED_SEGMENTS
    )


def _validate_stac_identifier(value: str, field: str) -> str:
    """Return ``value`` if it is a safe single URL path segment, else raise."""

    if not is_valid_stac_identifier(value):
        shown = value[:80] if isinstance(value, str) else value
        raise InvalidInputError(
            f"{field} {shown!r} is not a valid STAC identifier. Expected 1-200 "
            "characters from A-Z, a-z, 0-9, '.', '_' or '-'."
        )
    return value


def _bbox_intersects(a: BoundingBox, b: list[float]) -> bool:
    if len(b) < 4:
        return True  # unknown footprint - defer to the raster-level check
    bw, bs, be, bn = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return not (a.east < bw or a.west > be or a.north < bs or a.south > bn)


def _image_corners(window: RgbWindow) -> list[list[float]] | None:
    """The image's WGS84 footprint, or ``None`` when it cannot be established.

    A raster whose corners cannot be derived honestly - a rotated grid or an
    unusable CRS - still has a perfectly good PNG, so the image is returned
    without a footprint rather than failing the whole request. Omitting the
    quad costs a map overlay; emitting a wrong one would put the imagery in the
    wrong place, which is the failure this phase exists to prevent.
    """

    try:
        return image_corners_wgs84(
            window.transform,
            width=window.width,
            height=window.height,
            crs=window.crs,
        )
    except ImageryError as exc:
        logger.warning("No WGS84 footprint for this window: %s", exc)
        return None


#: Schemes this service will open. The raster layer is configured for
#: anonymous HTTPS range reads and holds no credentials, so anything else -
#: most importantly ``s3://`` - cannot be read and must be refused here rather
#: than handed to GDAL, which fails with an opaque credentials error.
#:
#: This is also the URL boundary: an href arrives from an EXTERNAL catalog, so
#: restricting the scheme keeps a catalog entry from steering the server at a
#: protocol it was never meant to speak.
_READABLE_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _require_readable_scheme(asset_key: str, href: str) -> None:
    """Refuse an asset this deployment provably cannot read.

    Earth Search publishes Sentinel-1 GRD measurement assets as ``s3://``
    URIs on a requester-pays bucket. Without this check the href reaches
    rasterio, which reports a credentials failure - a message that describes
    the symptom and hides the cause.
    """

    # A control character has no place in a URL and is how one string becomes
    # two: a NUL truncates the path for any C consumer (GDAL and curl are C),
    # and CR/LF are the separators of an HTTP request, so either can make the
    # request sent differ from the URL that was checked. Refuse before the
    # scheme test, so nothing downstream ever sees a href this layer only
    # partly validated.
    if any(character in href for character in "\x00\r\n\t"):
        raise InvalidInputError(
            f"Asset {asset_key!r} has a href containing a control character, "
            "which cannot be part of a valid URL."
        )

    scheme = href.split("://", 1)[0].lower() if "://" in href else ""
    if scheme not in _READABLE_SCHEMES:
        raise InvalidInputError(
            f"Asset {asset_key!r} is published as {scheme or 'an unknown'}:// "
            "which this deployment cannot read; only HTTPS assets are "
            "supported. Earth Search publishes Sentinel-1 GRD measurement "
            "assets on a requester-pays s3:// bucket, and this deployment "
            "reads only anonymous HTTPS. Sentinel-1 imagery is served instead "
            "from the 'sentinel-1-rtc' collection (the configured default), "
            "whose provider terrain-corrected rasters are readable over HTTPS."
        )


def _require_permitted_host(
    asset_key: str, href: str, trusted: list[str]
) -> None:
    """Refuse an href pointing somewhere this server must not reach.

    The scheme check above establishes that the URL is HTTP(S). It does not say
    WHERE. The href comes from an external catalog's response, and whatever it
    names is fetched by this process, from inside whatever network it runs in -
    which is the shape of a server-side request forgery whether or not the
    catalog is hostile today.

    Two rules, deliberately different in strength:

    * **Always**: a host that is a private, loopback, link-local, reserved,
      multicast or unspecified IP ADDRESS is refused. That is the dangerous
      class - ``169.254.169.254`` is a cloud metadata service, ``127.0.0.1`` is
      whatever else this host runs - and refusing it costs nothing legitimate,
      because no public satellite catalog publishes rasters there.
    * **When configured**: ``trusted_asset_hosts`` restricts reads to named
      hosts. Left empty, any public host is allowed; a deployment that knows
      its catalogs should say so, and the production profile does.

    KNOWN LIMITATION: this checks the NAME. It does not pin the address that
    name resolves to, so a DNS rebind between this check and the read is not
    prevented - the raster stack does not expose the connection for that.
    """

    host = (urlsplit(href).hostname or "").strip()
    if not host:
        raise InvalidInputError(
            f"Asset {asset_key!r} has a href with no host, which cannot be read."
        )

    try:
        address = ip_address(host)
    except ValueError:
        address = None
    # Two clauses, because neither alone is correct - measured on this Python
    # rather than assumed:
    #
    #   100.64.0.1  (carrier-grade NAT)  is_private=False  is_global=False
    #   224.0.0.1   (IPv4 multicast)     is_private=False  is_global=True
    #
    # So a private-address check misses shared address space, and a global-only
    # check admits multicast. A satellite catalog publishes its rasters on the
    # public unicast internet; anything else here is this server being pointed
    # inwards at whatever it can reach.
    if address is not None and (not address.is_global or address.is_multicast):
        # Named rather than echoed back in full: the refusal says what KIND of
        # address it was, which is what an operator needs.
        raise InvalidInputError(
            f"Asset {asset_key!r} points at a non-public address, which this "
            "deployment refuses to read on the catalog's behalf."
        )

    if not trusted:
        return
    lowered = host.lower()
    for entry in trusted:
        candidate = entry.strip().lower().lstrip("*")
        if not candidate:
            continue
        if lowered == candidate.lstrip(".") or lowered.endswith(
            candidate if candidate.startswith(".") else f".{candidate}"
        ):
            return
    raise InvalidInputError(
        f"Asset {asset_key!r} is hosted at {lowered!r}, which is not among the "
        "asset hosts this deployment is configured to read."
    )


class ImageryService(DomainService):
    """Windowed reads for an already-selected scene.

    Sentinel-2 ``visual`` is returned as true colour. Sentinel-1 RTC ``vv``/
    ``vh`` are single-band provider gamma-naught rasters, display-stretched to
    grayscale by the raster layer - a rendering, never a calibration.
    """

    name = "satellite.imagery"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        stac_item_fetcher: StacItemFetcher | None = None,
        raster_reader: RasterReader | None = None,
        band_reader: BandReader | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport
        self._fetch_item = stac_item_fetcher or self._default_fetch_item
        self._read_window = raster_reader or read_rgb_window
        self._read_band = band_reader or read_band_window

    def quantitative_read_limits(self) -> QuantitativeReadLimits:
        """The bounds :meth:`read_band` refuses a window beyond.

        Declared rather than private so a caller can apply them BEFORE the
        catalog fetch and the remote open that :meth:`read_band` performs
        first. :meth:`read_band` reads its own bounds from here, so the early
        check and the authoritative one cannot come to disagree.
        """

        return QuantitativeReadLimits(
            max_dimension=self._settings.imagery_hard_max_dimension,
            max_window_pixels=self._settings.imagery_max_window_pixels,
        )

    def validate_scene(
        self,
        *,
        scene_id: str,
        collection: str | None,
        modality: Modality,
        assets: Sequence[str],
        bbox: BoundingBox,
        require_all_assets: bool = True,
    ) -> ValidatedScene:
        """Blocking: fetch ``scene_id``'s catalog item and validate it for measurement.

        The catalog item is the authority - nothing a client said about the
        scene is consulted. It is the same item lookup :meth:`read_band` uses,
        and the same href scheme and host rules, applied BEFORE any raster is
        opened. The rules themselves live in
        :mod:`app.services.satellite.scene_validation`.
        """

        resolved = _validate_stac_identifier(
            collection or self._settings.stac_collection, "collection"
        )
        scene_id = _validate_stac_identifier(scene_id, "scene_id")
        try:
            item = self._fetch_item(scene_id, resolved)
        except NotFoundError:
            raise SceneValidationError(
                "scene_not_found",
                f"the catalog has no scene {scene_id!r} in {resolved!r}.",
            ) from None

        def check_href(key: str, href: str) -> None:
            _require_readable_scheme(key, href)
            _require_permitted_host(key, href, self._settings.trusted_asset_hosts)

        return validate_scene_item(
            item,
            scene_id=scene_id,
            collection=resolved,
            modality=modality,
            assets=assets,
            aoi=bbox,
            check_href=check_href,
            require_all_assets=require_all_assets,
            min_coverage=self._settings.scene_min_aoi_coverage,
        )

    def describe(self) -> str:
        return (
            "Bounded Sentinel-2 RGB and Sentinel-1 RTC grayscale imagery "
            "retrieval via windowed COG reads."
        )

    # -- STAC item lookup (metadata only) ---------------------------------- #

    def _default_fetch_item(self, scene_id: str, collection: str) -> dict[str, Any]:
        url = (
            f"{catalog_for(collection, self._settings)}/collections/"
            f"{collection}/items/{scene_id}"
        )
        try:
            with httpx.Client(
                timeout=self._settings.http_timeout_seconds,
                headers={"Accept": "application/geo+json"},
                transport=self._transport,
            ) as client:
                response = client.get(url)  # STAC metadata only - never imagery
        except httpx.TimeoutException as exc:
            raise UpstreamServiceError("The satellite catalog timed out.") from exc
        except httpx.HTTPError as exc:
            raise UpstreamServiceError("The satellite catalog is unavailable.") from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            raise NotFoundError(f"Scene {scene_id!r} was not found in the catalog.")
        if response.status_code != httpx.codes.OK:
            raise UpstreamServiceError(
                f"The satellite catalog responded with status {response.status_code}."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamServiceError(
                "The satellite catalog returned malformed data."
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("assets"), dict):
            raise UpstreamServiceError("The STAC item is malformed.")
        return payload

    # -- orchestration --------------------------------------------------- #

    def _resolve_asset_href(self, item: dict[str, Any], asset_key: str) -> str:
        assets = item.get("assets")
        if not isinstance(assets, dict):
            raise UpstreamServiceError("The STAC item is malformed.")
        asset = assets.get(asset_key)
        if not isinstance(asset, dict):
            raise NotFoundError(
                f"Asset {asset_key!r} is not available on this scene."
            )
        href = asset.get("href")
        if not isinstance(href, str) or not href:
            raise UpstreamServiceError(f"Asset {asset_key!r} has no usable href.")
        media_type = str(asset.get("type") or "").lower()
        if _COG_TYPE_HINT not in media_type:
            raise InvalidInputError(
                f"Asset {asset_key!r} ({media_type or 'unknown type'}) is not a "
                "windowed-readable GeoTIFF; bounded retrieval is not supported."
            )
        _require_readable_scheme(asset_key, href)
        _require_permitted_host(asset_key, href, self._settings.trusted_asset_hosts)
        return href

    def retrieve(self, request: ImageryRequest) -> ImageryResponse:
        """Blocking: performs a STAC item GET and a windowed COG read.

        The route handler is a sync ``def`` so Starlette runs this in a
        threadpool rather than blocking the event loop.
        """

        if request.asset not in SUPPORTED_IMAGERY_ASSETS:
            raise InvalidInputError(
                f"Asset {request.asset!r} is not supported for bounded RGB "
                f"retrieval. Supported: {', '.join(SUPPORTED_IMAGERY_ASSETS)}."
            )

        collection = _validate_stac_identifier(
            request.collection or self._settings.stac_collection, "collection"
        )
        if request.asset == "vh" and collection != RTC_COLLECTION:
            raise InvalidInputError("VH display is supported only for Sentinel-1 RTC assets.")
        scene_id = _validate_stac_identifier(request.scene_id, "scene_id")
        item = self._fetch_item(scene_id, collection)
        scene_bbox = item.get("bbox")
        if isinstance(scene_bbox, list) and not _bbox_intersects(request.bbox, scene_bbox):
            raise InvalidInputError(
                "The requested bbox does not intersect the selected scene."
            )

        href = self._resolve_asset_href(item, request.asset)

        max_dimension = min(
            request.max_dimension or self._settings.imagery_max_dimension,
            self._settings.imagery_hard_max_dimension,
        )

        read_href = (
            sign_rtc_asset(href, settings=self._settings, transport=self._transport)
            if collection == RTC_COLLECTION else href
        )
        window = self._read_window(
            read_href,
            request.bbox,
            max_dimension=max_dimension,
            max_window_pixels=self._settings.imagery_max_window_pixels,
        )

        image_b64 = _encode_png(window.array)
        logger.info(
            "Imagery for %s: %sx%s px, %d B64 chars",
            request.scene_id,
            window.width,
            window.height,
            len(image_b64),
        )

        return ImageryResponse(
            scene_id=request.scene_id,
            bbox=request.bbox,
            asset=request.asset,
            asset_href=href,
            width=window.width,
            height=window.height,
            format="png",
            media_type="image/png",
            bands=[request.asset] * 3 if request.asset in {"vv", "vh"} else window.bands,
            crs=window.crs,
            resolution=window.resolution,
            normalization=(
                # The provider terrain-corrected these values; SatQuery did
                # not. Everything after the semicolon is a display transform
                # over the provider's numbers, which are themselves untouched.
                "Provider RTC gamma naught (terrain-corrected by the data "
                "provider, not by SatQuery); "
                + window.normalization.replace(
                    "display only, not calibrated", "display only"
                )
                + "; the stretch is for display only and is applied to this "
                "PNG alone - the quantitative statistics, when requested, are "
                "computed from the provider's linear power values, never from "
                "these display bytes. SatQuery performs no radiometric "
                "calibration, speckle filtering or terrain correction of its "
                "own"
                if collection == RTC_COLLECTION else window.normalization
            ),
            window=WindowInfo(**window.window),
            source_shape=window.source_shape,
            # Passed through, never recomputed: this is the affine of the
            # window actually read, not of the requested bbox.
            transform=list(window.transform)[:6],
            corners_wgs84=_image_corners(window),
            image_base64=image_b64,
        )


    # -- quantitative band access (analysis path) -------------------------- #

    def read_band(
        self,
        *,
        scene_id: str,
        bbox: BoundingBox,
        asset: str,
        collection: str | None = None,
    ) -> BandWindow:
        """Blocking: read one band's raw values over ``bbox`` at native GSD.

        The quantitative sibling of :meth:`retrieve`, and the ONLY quantitative
        imagery-access boundary. It reuses the same collection-aware STAC item
        lookup and asset resolution (so non-GeoTIFF assets such as the ``-jp2``
        variants are refused), but reads through
        :func:`~app.services.satellite.raster.read_band_window` - never through
        the display path, and never decimated.

        ``asset`` is a STAC asset key and must be in ``ANALYSIS_BAND_ASSETS``
        (Sentinel-2 spectral bands) or, for the Sentinel-1 RTC collection only,
        ``SAR_ANALYSIS_BAND_ASSETS`` (``vv``/``vh``); the display whitelist does
        not apply here. The STAC-advertised ``scale``/``offset`` are
        deliberately NOT read or applied - see
        ``app.services.analysis.engines``. Sentinel-1 RTC values are the
        provider's linear gamma-naught power, returned untouched; the
        conversion to decibels is the analysis engine's.

        Sentinel-1 RTC rasters are not anonymously readable, so the resolved
        href is exchanged for a short-lived read URL exactly as
        :meth:`retrieve` does - the same bounded signing path, not a second
        one. The token never leaves this method.
        """

        resolved_collection = _validate_stac_identifier(
            collection or self._settings.stac_collection, "collection"
        )
        if asset in SAR_ANALYSIS_BAND_ASSETS:
            # A polarization is only a physical quantity in the collection that
            # publishes it as terrain-corrected gamma naught. Refusing here
            # keeps a SAR asset key from being resolved against an optical
            # collection, where it would either not exist or not mean this.
            if resolved_collection != RTC_COLLECTION:
                raise InvalidInputError(
                    f"Asset {asset!r} is a Sentinel-1 polarization and is "
                    f"readable quantitatively only from the "
                    f"{RTC_COLLECTION!r} collection, not "
                    f"{resolved_collection!r}."
                )
        elif asset in QUALITY_BAND_ASSETS:
            # Sentinel-2's scene classification: meaningless on a SAR product.
            if resolved_collection == RTC_COLLECTION:
                raise InvalidInputError(
                    f"Asset {asset!r} is a Sentinel-2 quality layer and cannot be "
                    f"read from {RTC_COLLECTION!r}."
                )
        elif asset not in ANALYSIS_BAND_ASSETS:
            readable = (*ANALYSIS_BAND_ASSETS, *QUALITY_BAND_ASSETS, *SAR_ANALYSIS_BAND_ASSETS)
            raise InvalidInputError(
                f"Asset {asset!r} is not supported for quantitative band "
                f"reads. Supported: {', '.join(readable)}."
            )
        scene_id = _validate_stac_identifier(scene_id, "scene_id")
        item = self._fetch_item(scene_id, resolved_collection)
        scene_bbox = item.get("bbox")
        if isinstance(scene_bbox, list) and not _bbox_intersects(bbox, scene_bbox):
            raise InvalidInputError(
                "The requested bbox does not intersect the selected scene."
            )

        href = self._resolve_asset_href(item, asset)
        if asset in SAR_ANALYSIS_BAND_ASSETS:
            descriptions = item["assets"][asset].get("raster:bands", [])
            if descriptions:
                if not isinstance(descriptions, list) or not isinstance(descriptions[0], dict):
                    raise InvalidInputError("The Sentinel-1 RTC band metadata is malformed.")
                encoding = descriptions[0]
                require_linear_power_encoding(
                    scale=encoding.get("scale", 1.0),
                    offset=encoding.get("offset", 0.0), unit=encoding.get("unit"),
                )
        read_href = (
            sign_rtc_asset(href, settings=self._settings, transport=self._transport)
            if resolved_collection == RTC_COLLECTION else href
        )

        limits = self.quantitative_read_limits()
        band = self._read_band(
            read_href,
            bbox,
            # Rejection bounds, not a decimation target: quantitative reads stay
            # at native resolution, so an oversized window is refused.
            max_dimension=limits.max_dimension,
            max_window_pixels=limits.max_window_pixels,
        )
        if asset in SAR_ANALYSIS_BAND_ASSETS:
            require_linear_power_encoding(
                scale=band.source_scale, offset=band.source_offset, unit=band.source_unit,
            )
        logger.info(
            "Band %s for %s (%s): %sx%s px at %s m/px",
            asset,
            scene_id,
            resolved_collection,
            band.width,
            band.height,
            band.resolution,
        )
        return band


def _encode_png(array: Any) -> str:
    try:
        image = Image.fromarray(array)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    except (ValueError, TypeError, OSError, UnidentifiedImageError) as exc:
        raise ImageryError("Failed to encode the bounded image as PNG.") from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")
