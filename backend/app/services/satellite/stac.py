"""Thin async client for the Earth Search STAC API (Element 84).

Only ``POST /search`` is used. No credentials, no asset/imagery requests - the
client fetches STAC metadata and nothing else. Network, status, and payload
failures are normalised into :class:`app.core.errors.UpstreamServiceError`.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import UpstreamServiceError
from app.core.logging import get_logger

logger = get_logger("satellite.stac")


async def search_items(
    *,
    settings: Settings,
    body: dict[str, Any],
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Any]:
    """POST a STAC search and return the raw ``features`` list.

    ``transport`` is injectable so tests can stub the HTTP call without touching
    the live catalog.
    """

    try:
        async with httpx.AsyncClient(
            base_url=settings.stac_base_url,
            timeout=settings.http_timeout_seconds,
            headers={
                "Accept": "application/geo+json",
                "Content-Type": "application/json",
            },
            transport=transport,
        ) as client:
            response = await client.post("/search", json=body)
    except httpx.TimeoutException as exc:
        raise UpstreamServiceError("The satellite catalog timed out.") from exc
    except httpx.HTTPError as exc:
        raise UpstreamServiceError("The satellite catalog is unavailable.") from exc

    if response.status_code != httpx.codes.OK:
        logger.warning("Earth Search responded with HTTP %s", response.status_code)
        raise UpstreamServiceError(
            f"The satellite catalog responded with status {response.status_code}."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise UpstreamServiceError(
            "The satellite catalog returned malformed data."
        ) from exc

    if not isinstance(payload, dict):
        raise UpstreamServiceError(
            "The satellite catalog returned an unexpected payload."
        )

    features = payload.get("features")
    if not isinstance(features, list):
        raise UpstreamServiceError(
            "The satellite catalog response is missing a 'features' list."
        )

    return features
