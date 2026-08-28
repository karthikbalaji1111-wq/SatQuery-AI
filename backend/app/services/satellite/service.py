"""Sentinel-2 scene-discovery service.

place name -> geospatial resolve -> validated bbox -> (this) STAC scene search.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import UpstreamServiceError
from app.core.logging import get_logger
from app.services.base import DomainService
from app.services.geospatial.schemas import BoundingBox
from app.services.satellite.schemas import (
    QueryEcho,
    Scene,
    SceneAsset,
    SceneSearchRequest,
    SceneSearchResponse,
)
from app.services.satellite.stac import search_items

logger = get_logger("satellite")

# Curated, discovery-stage assets. Band rasters (B01..B12) are deliberately
# excluded - this phase never downloads or references pixel data.
_USEFUL_ASSET_KEYS = (
    "thumbnail",
    "overview",
    "visual",
    "granule_metadata",
    "tileinfo_metadata",
)
_THUMBNAIL_KEYS = ("thumbnail", "overview", "rendered_preview", "preview")


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _str_list_or_none(value: object) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value or None
    return None


def _normalize_bbox(raw: object) -> BoundingBox | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        west, south, east, north = (float(raw[i]) for i in range(4))
        return BoundingBox(west=west, south=south, east=east, north=north)
    except (TypeError, ValueError):
        return None


def _pick_thumbnail(assets: dict[str, Any]) -> str | None:
    for key in _THUMBNAIL_KEYS:
        entry = assets.get(key)
        if isinstance(entry, dict):
            href = _str_or_none(entry.get("href"))
            if href:
                return href
    return None


def _minimal_assets(assets: dict[str, Any]) -> list[SceneAsset]:
    result: list[SceneAsset] = []
    for key in _USEFUL_ASSET_KEYS:
        entry = assets.get(key)
        if not isinstance(entry, dict):
            continue
        href = _str_or_none(entry.get("href"))
        if not href:
            continue
        result.append(
            SceneAsset(
                key=key,
                href=href,
                type=_str_or_none(entry.get("type")),
                title=_str_or_none(entry.get("title")),
                roles=_str_list_or_none(entry.get("roles")),
            )
        )
    return result


def _processing_level(props: dict[str, Any], collection: str | None) -> str | None:
    level = _str_or_none(props.get("processing:level"))
    if level:
        return level
    if collection == "sentinel-2-l2a":
        return "L2A"
    if collection == "sentinel-2-l1c":
        return "L1C"
    return None


def _normalize_scene(feature: object) -> Scene:
    if not isinstance(feature, dict):
        raise UpstreamServiceError("The satellite catalog returned a malformed scene.")

    scene_id = feature.get("id")
    if not isinstance(scene_id, str) or not scene_id:
        raise UpstreamServiceError(
            "The satellite catalog returned a scene without an id."
        )

    props = feature.get("properties")
    if not isinstance(props, dict):
        props = {}

    collection = _str_or_none(feature.get("collection"))
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        geometry = None

    cloud = props.get("eo:cloud_cover")
    cloud_cover = float(cloud) if isinstance(cloud, (int, float)) else None

    assets_raw = feature.get("assets")
    thumbnail_url: str | None = None
    assets: list[SceneAsset] = []
    if isinstance(assets_raw, dict):
        thumbnail_url = _pick_thumbnail(assets_raw)
        assets = _minimal_assets(assets_raw)

    return Scene(
        id=scene_id,
        datetime=_str_or_none(props.get("datetime")),
        bbox=_normalize_bbox(feature.get("bbox")),
        geometry=geometry,
        cloud_cover=cloud_cover,
        collection=collection,
        platform=_str_or_none(props.get("platform")),
        processing_level=_processing_level(props, collection),
        thumbnail_url=thumbnail_url,
        assets=assets,
    )


class SatelliteService(DomainService):
    """Sentinel-2 optical scene discovery via the Earth Search STAC API.

    The generic :meth:`run` hook stays unimplemented; :meth:`search` is the
    typed entry point for this phase. Sentinel-1 / SAR is out of scope here.
    """

    name = "satellite"

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    def describe(self) -> str:
        return "Sentinel-1 SAR and Sentinel-2 optical imagery retrieval."

    async def search(self, request: SceneSearchRequest) -> SceneSearchResponse:
        """Discover Sentinel-2 L2A scenes for a bounding box and date range."""

        collections = [self._settings.stac_collection]
        bbox = [
            request.bbox.west,
            request.bbox.south,
            request.bbox.east,
            request.bbox.north,
        ]
        datetime_interval = (
            f"{request.start_date.isoformat()}T00:00:00Z"
            f"/{request.end_date.isoformat()}T23:59:59Z"
        )

        stac_filter: dict[str, Any] | None = None
        if request.max_cloud_cover is not None:
            # Standard STAC "query" extension - supported by Earth Search.
            stac_filter = {"eo:cloud_cover": {"lte": request.max_cloud_cover}}

        body: dict[str, Any] = {
            "collections": collections,
            "bbox": bbox,
            "datetime": datetime_interval,
            "limit": request.limit,
        }
        if stac_filter is not None:
            body["query"] = stac_filter

        features = await search_items(
            settings=self._settings,
            body=body,
            transport=self._transport,
        )

        scenes = [_normalize_scene(feature) for feature in features][: request.limit]
        logger.info(
            "STAC search (%s, %s) returned %d scene(s)",
            self._settings.stac_collection,
            datetime_interval,
            len(scenes),
        )

        return SceneSearchResponse(
            query=QueryEcho(
                collections=collections,
                bbox=bbox,
                datetime=datetime_interval,
                max_cloud_cover=request.max_cloud_cover,
                limit=request.limit,
                filter=stac_filter,
            ),
            scene_count=len(scenes),
            scenes=scenes,
            catalog=self._settings.stac_base_url,
        )
