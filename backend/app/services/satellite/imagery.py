"""Bounded Sentinel-2 imagery retrieval.

scene_id + bbox + asset -> STAC item lookup (metadata only) -> windowed COG
read -> standardized RGB PNG. No AI/VLM logic lives here.
"""

from __future__ import annotations

import base64
import io
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
from app.services.satellite.raster import RgbWindow, read_rgb_window
from app.services.satellite.schemas import (
    SUPPORTED_IMAGERY_ASSETS,
    ImageryRequest,
    ImageryResponse,
    WindowInfo,
)

logger = get_logger("satellite.imagery")

StacItemFetcher = Callable[[str, str], dict[str, Any]]
RasterReader = Callable[..., RgbWindow]

_COG_TYPE_HINT = "geotiff"


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
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport
        self._fetch_item = stac_item_fetcher or self._default_fetch_item
        self._read_window = raster_reader or read_rgb_window

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

        collection = request.collection or self._settings.stac_collection
        item = self._fetch_item(request.scene_id, collection)
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
