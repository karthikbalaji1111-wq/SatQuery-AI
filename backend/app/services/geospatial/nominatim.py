"""Thin async client for the OpenStreetMap Nominatim geocoding API.

Only the ``/search`` endpoint is used. Network, status, and payload failures are
normalised into :class:`app.core.errors.AppError` subclasses so the API layer can
return consistent JSON errors.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import NotFoundError, UpstreamServiceError
from app.core.logging import get_logger
from app.services.geospatial.schemas import BoundingBox, Coordinate

logger = get_logger("geospatial.nominatim")


class NominatimPlace(BaseModel):
    """Parsed first result from a Nominatim search."""

    display_name: str
    center: Coordinate
    bbox: BoundingBox


def _parse_first_result(payload: object) -> NominatimPlace:
    if not isinstance(payload, list) or not payload:
        raise NotFoundError("No matching location was found.")

    item = payload[0]
    try:
        lat = float(item["lat"])
        lon = float(item["lon"])
        # Nominatim `boundingbox` is [south, north, west, east] as strings.
        south, north, west, east = (float(value) for value in item["boundingbox"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UpstreamServiceError(
            "The geocoding service returned an unexpected payload."
        ) from exc

    try:
        bbox = BoundingBox(west=west, south=south, east=east, north=north)
    except ValueError as exc:
        raise UpstreamServiceError(
            "The geocoding service returned an invalid bounding box."
        ) from exc

    return NominatimPlace(
        display_name=str(item.get("display_name") or ""),
        center=Coordinate(lat=lat, lon=lon),
        bbox=bbox,
    )


async def geocode(
    query: str,
    *,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> NominatimPlace:
    """Resolve a free-text place name to coordinates and a bounding box.

    ``transport`` is injectable so tests can stub the HTTP call without touching
    the live service.
    """

    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": settings.nominatim_user_agent}

    try:
        async with httpx.AsyncClient(
            base_url=settings.nominatim_base_url,
            timeout=settings.http_timeout_seconds,
            headers=headers,
            transport=transport,
        ) as client:
            response = await client.get("/search", params=params)
    except httpx.TimeoutException as exc:
        raise UpstreamServiceError("The geocoding service timed out.") from exc
    except httpx.HTTPError as exc:
        raise UpstreamServiceError("The geocoding service is unavailable.") from exc

    if response.status_code != httpx.codes.OK:
        logger.warning(
            "Nominatim responded with HTTP %s for query %r",
            response.status_code,
            query,
        )
        raise UpstreamServiceError(
            f"The geocoding service responded with status {response.status_code}."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise UpstreamServiceError(
            "The geocoding service returned malformed data."
        ) from exc

    return _parse_first_result(payload)
