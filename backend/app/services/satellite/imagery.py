"""Bounded Sentinel-2 imagery retrieval.

scene_id + bbox + asset -> STAC item lookup (metadata only) -> windowed COG
read -> standardized RGB PNG. No AI/VLM logic lives here.
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Callable
from typing import Any

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
    read_band_window,
    read_rgb_window,
)
from app.services.satellite.schemas import (
    ANALYSIS_BAND_ASSETS,
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

_STAC_IDENTIFIER = re.compile(r"[A-Za-z0-9._-]{1,200}")
#: Path segments that would traverse even though their characters are allowed.
_RESERVED_SEGMENTS = frozenset({".", ".."})


def _validate_stac_identifier(value: str, field: str) -> str:
    """Return ``value`` if it is a safe single URL path segment, else raise."""

    if (
        not isinstance(value, str)
        or _STAC_IDENTIFIER.fullmatch(value) is None
        or value in _RESERVED_SEGMENTS
    ):
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


class ImageryService(DomainService):
    """Windowed RGB reads for an already-selected Sentinel-2 scene."""

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

    def describe(self) -> str:
        return "Bounded Sentinel-2 RGB imagery retrieval via windowed COG reads."

    # -- STAC item lookup (metadata only) ---------------------------------- #

    def _default_fetch_item(self, scene_id: str, collection: str) -> dict[str, Any]:
        url = (
            f"{self._settings.stac_base_url}/collections/"
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

        window = self._read_window(
            href,
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
            bands=window.bands,
            crs=window.crs,
            resolution=window.resolution,
            normalization=window.normalization,
            window=WindowInfo(**window.window),
            source_shape=window.source_shape,
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

        ``asset`` is an Earth Search STAC asset key and must be in
        ``ANALYSIS_BAND_ASSETS``; the display whitelist does not apply here.
        The STAC-advertised ``scale``/``offset`` are deliberately NOT read or
        applied - see ``app.services.analysis.engines``.
        """

        if asset not in ANALYSIS_BAND_ASSETS:
            raise InvalidInputError(
                f"Asset {asset!r} is not supported for quantitative band "
                f"reads. Supported: {', '.join(ANALYSIS_BAND_ASSETS)}."
            )

        resolved_collection = _validate_stac_identifier(
            collection or self._settings.stac_collection, "collection"
        )
        scene_id = _validate_stac_identifier(scene_id, "scene_id")
        item = self._fetch_item(scene_id, resolved_collection)
        scene_bbox = item.get("bbox")
        if isinstance(scene_bbox, list) and not _bbox_intersects(bbox, scene_bbox):
            raise InvalidInputError(
                "The requested bbox does not intersect the selected scene."
            )

        href = self._resolve_asset_href(item, asset)

        band = self._read_band(
            href,
            bbox,
            # Rejection bounds, not a decimation target: quantitative reads stay
            # at native resolution, so an oversized window is refused.
            max_dimension=self._settings.imagery_hard_max_dimension,
            max_window_pixels=self._settings.imagery_max_window_pixels,
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
