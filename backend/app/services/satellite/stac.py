"""Thin async client for the Earth Search STAC API (Element 84).

Only ``POST /search`` is used. No credentials, no asset/imagery requests - the
client fetches STAC metadata and nothing else. Network, status, and payload
failures are normalised into :class:`app.core.errors.UpstreamServiceError`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import UpstreamServiceError
from app.core.logging import get_logger
from app.services.satellite.rtc import catalog_for

logger = get_logger("satellite.stac")

#: Attempts for one catalog search, retries included. A search is an idempotent
#: read, so asking again cannot change anything. Observed live: one Sentinel-1
#: search failed with a transport error and the same request succeeded seconds
#: later - a whole agent run was lost to a single dropped connection.
_SEARCH_ATTEMPTS = 2

#: Pause before the second attempt.
_SEARCH_BACKOFF_SECONDS = 1.0

#: Statuses that mean "busy", not "wrong". A 4xx other than 429 is the
#: request's own fault and is never repeated.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class SearchPage:
    """One page of a STAC search, and how much it is a page OF.

    ``matched`` is the catalog's own ``numberMatched`` - how many scenes the
    query matched in total, which is usually far more than one page returns.
    It was read off the response and thrown away, so a selection made from ten
    returned scenes was indistinguishable from a selection made from every
    scene that matched. ``None`` when the catalog does not report it: unknown
    is not zero, and it is not "all of them" either.
    """

    features: list[Any]
    matched: int | None


async def search_items(
    *,
    settings: Settings,
    body: dict[str, Any],
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Any]:
    """POST a STAC search and return the raw ``features`` list.

    Kept for callers that only want the features; :func:`search_page` carries
    the match count as well.
    """

    return (
        await search_page(settings=settings, body=body, transport=transport)
    ).features


async def search_page(
    *,
    settings: Settings,
    body: dict[str, Any],
    transport: httpx.AsyncBaseTransport | None = None,
) -> SearchPage:
    """POST a STAC search and return its features with the total match count.

    ``transport`` is injectable so tests can stub the HTTP call without touching
    the live catalog.
    """

    response = await _post_search(settings=settings, body=body, transport=transport)

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

    # `numberMatched` is the OGC API - Features name; older STAC deployments
    # report it under `context.matched`. Both are read, and neither is invented:
    # a catalog that reports nothing leaves this None.
    matched = payload.get("numberMatched")
    if not isinstance(matched, int):
        context = payload.get("context")
        matched = context.get("matched") if isinstance(context, dict) else None
    if not isinstance(matched, int) or matched < 0:
        matched = None

    return SearchPage(features=features, matched=matched)


async def _post_search(
    *,
    settings: Settings,
    body: dict[str, Any],
    transport: httpx.AsyncBaseTransport | None,
) -> httpx.Response:
    """POST ``/search``, re-issuing it once after a transient failure.

    Only a transport failure or a busy status is retried. The final response is
    returned whatever its status, so the caller's existing status handling still
    decides what a non-200 means; the final transport failure is raised with the
    same messages as before.
    """

    for attempt in range(1, _SEARCH_ATTEMPTS + 1):
        final = attempt == _SEARCH_ATTEMPTS
        try:
            async with httpx.AsyncClient(
                base_url=catalog_for(body["collections"][0], settings),
                timeout=settings.http_timeout_seconds,
                headers={
                    "Accept": "application/geo+json",
                    "Content-Type": "application/json",
                },
                transport=transport,
            ) as client:
                response = await client.post("/search", json=body)
        except httpx.TimeoutException as exc:
            if final:
                raise UpstreamServiceError("The satellite catalog timed out.") from exc
            logger.warning("Catalog search timed out (attempt %d/%d)", attempt, _SEARCH_ATTEMPTS)
        except httpx.HTTPError as exc:
            if final:
                raise UpstreamServiceError("The satellite catalog is unavailable.") from exc
            logger.warning(
                "Catalog search transport error %s (attempt %d/%d)",
                type(exc).__name__,
                attempt,
                _SEARCH_ATTEMPTS,
            )
        else:
            if final or response.status_code not in _RETRYABLE_STATUS:
                return response
            logger.warning(
                "Catalog search answered HTTP %s (attempt %d/%d)",
                response.status_code,
                attempt,
                _SEARCH_ATTEMPTS,
            )
        await asyncio.sleep(_SEARCH_BACKOFF_SECONDS)
    raise AssertionError("unreachable: the final attempt returns or raises")
