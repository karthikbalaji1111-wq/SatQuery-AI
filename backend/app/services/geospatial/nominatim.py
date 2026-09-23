"""Thin async client for the OpenStreetMap Nominatim geocoding API.

Only the ``/search`` endpoint is used. Network, status, and payload failures are
normalised into :class:`app.core.errors.AppError` subclasses so the API layer can
return consistent JSON errors.

**This module owns how the application treats Nominatim.** Its operator asks
users of the public instance for at most one request per second from an
application, a genuine identifying ``User-Agent``, and caching of repeated
queries. A ``User-Agent`` was already sent; the other two were not, and nothing
stopped this process issuing as many concurrent geocodes as it had requests -
one per window per modality on the execution path alone.

The budget is therefore held HERE, once, at module scope:

* one geocode in flight at a time, and at least
  ``geocoder_min_interval_seconds`` between the starts of two of them;
* a small TTL cache, so a repeated place name costs nothing upstream;
* one retry for a transient failure, spaced like any other request.

It is deliberately application-wide rather than per request or per client: "one
request per second per user" would let ten users send ten per second under this
application's single User-Agent, which is exactly the budget the policy is
about.

**SCOPE: ONE PROCESS.** Like everything in ``app/core/limits.py``, this holds
for a single replica. Two replicas make two requests per second, so a
multi-instance deployment MUST NOT claim policy compliance on the strength of
this module alone; see the deployment documentation.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

import httpx
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import NotFoundError, UpstreamServiceError
from app.core.logging import get_logger
from app.services.geospatial.schemas import BoundingBox, Coordinate

logger = get_logger("geospatial.nominatim")

#: Statuses worth one more attempt: a rate limit or a server-side fault. A 4xx
#: other than 429 is the request's own problem and repeating it is rude.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Total attempts, including the first. Bounded on purpose: an unbounded retry
#: loop against a rate-limited service is how an application gets blocked.
_MAX_ATTEMPTS = 2

#: Serialises geocoding for this process; also what makes the spacing below
#: meaningful, since two concurrent callers would otherwise both see the same
#: "last request" timestamp and both proceed.
_GEOCODE_LOCK = asyncio.Lock()

#: When the last upstream request was STARTED, on the monotonic clock.
_LAST_REQUEST_AT: float | None = None

#: query -> (stored_at, place). Ordered so the oldest entry is evicted first.
_CACHE: OrderedDict[str, tuple[float, NominatimPlace]] = OrderedDict()


class NominatimPlace(BaseModel):
    """Parsed first result from a Nominatim search.

    ``place_class`` and ``place_type`` are the geocoder's OWN classification
    (e.g. ``natural``/``beach``, ``place``/``city``, ``shop``/``supermarket``).
    They are recorded rather than judged: this repository invents no confidence
    score, and a class is not one. What they give a reader is the ability to see
    WHAT was matched - a search that landed on a shop rather than the coastline
    is otherwise indistinguishable from the right answer, because both come back
    with a plausible display name and a valid bounding box.
    """

    display_name: str
    center: Coordinate
    bbox: BoundingBox
    place_class: str | None = None
    place_type: str | None = None


class _TransientError(Exception):
    """A failure worth one more attempt.

    Carries the message the caller will see if the retry fails too, so a
    retried failure reports the same thing it always did rather than a vaguer
    summary of several attempts.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def reset_geocoder_state() -> None:
    """Forget the throttle and the cache.

    For tests. Module-level state shared across a test session would otherwise
    let one test's cached answer satisfy another test's request - and one
    test's timestamp delay another's.
    """

    global _LAST_REQUEST_AT
    _LAST_REQUEST_AT = None
    _CACHE.clear()


def _cache_key(query: str, settings: Settings) -> str:
    """Case- and whitespace-insensitive, and scoped to the service asked.

    The base URL is part of the key because it identifies WHICH geocoder
    answered; a cached answer from one deployment must never satisfy a request
    aimed at another.
    """

    return f"{settings.nominatim_base_url}|{' '.join(query.lower().split())}"


def _cached(key: str, settings: Settings) -> NominatimPlace | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    stored_at, place = entry
    if time.monotonic() - stored_at > settings.geocoder_cache_ttl_seconds:
        del _CACHE[key]
        return None
    _CACHE.move_to_end(key)
    return place


def _store(key: str, place: NominatimPlace, settings: Settings) -> None:
    """Cache a SUCCESSFUL resolution only.

    A failure is never cached: an upstream outage would otherwise be repeated
    back to every caller for the whole TTL, long after the service recovered.
    """

    _CACHE[key] = (time.monotonic(), place)
    _CACHE.move_to_end(key)
    while len(_CACHE) > settings.geocoder_cache_entries:
        _CACHE.popitem(last=False)


async def _wait_for_turn(settings: Settings) -> None:
    """Hold until this process may issue another request."""

    global _LAST_REQUEST_AT
    interval = settings.geocoder_min_interval_seconds
    if interval > 0 and _LAST_REQUEST_AT is not None:
        delay = interval - (time.monotonic() - _LAST_REQUEST_AT)
        if delay > 0:
            logger.info("Geocoder throttle: waiting %.2fs", delay)
            await asyncio.sleep(delay)
    _LAST_REQUEST_AT = time.monotonic()


def _optional_text(value: object) -> str | None:
    """A short text field, or None. Never an empty string masquerading as data."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:60] or None


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
        # Absent on some results; recorded as unknown rather than guessed.
        place_class=_optional_text(item.get("class")),
        place_type=_optional_text(item.get("type")),
    )


async def _fetch_once(
    query: str,
    *,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None,
) -> object:
    """One upstream request. Raises :class:`_TransientError` if worth retrying."""

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
        raise _TransientError("The geocoding service timed out.") from exc
    except httpx.HTTPError as exc:
        raise _TransientError("The geocoding service is unavailable.") from exc

    if response.status_code in _RETRYABLE_STATUS:
        logger.warning(
            "Nominatim responded with HTTP %s for query %r",
            response.status_code,
            query,
        )
        raise _TransientError(
            f"The geocoding service responded with status {response.status_code}."
        )

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
        return response.json()
    except ValueError as exc:
        raise UpstreamServiceError(
            "The geocoding service returned malformed data."
        ) from exc


async def geocode(
    query: str,
    *,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> NominatimPlace:
    """Resolve a free-text place name to coordinates and a bounding box.

    ``transport`` is injectable so tests can stub the HTTP call without touching
    the live service.

    A cached answer is returned without contacting the service at all. Otherwise
    the caller waits its turn - one request at a time, spaced by the configured
    interval - and a transient failure is retried once, spaced the same way.
    """

    key = _cache_key(query, settings)
    cached = _cached(key, settings)
    if cached is not None:
        logger.info("Geocoder cache hit for %r", query)
        return cached

    async with _GEOCODE_LOCK:
        # Re-checked after waiting: while this caller held the queue, another
        # may have fetched exactly this place. Without it, N concurrent
        # requests for one place cost N upstream requests instead of one.
        cached = _cached(key, settings)
        if cached is not None:
            logger.info("Geocoder cache hit for %r (filled while waiting)", query)
            return cached

        last: _TransientError | None = None
        payload: object = None
        for attempt in range(_MAX_ATTEMPTS):
            await _wait_for_turn(settings)
            try:
                payload = await _fetch_once(
                    query, settings=settings, transport=transport
                )
                break
            except _TransientError as exc:
                last = exc
                logger.info(
                    "Geocoder attempt %d/%d failed: %s",
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    exc.message,
                )
        else:
            # Every attempt was transient. The message is the last failure's
            # own, so the caller reads the same sentence as before retries.
            raise UpstreamServiceError(last.message if last else "unavailable")

    place = _parse_first_result(payload)
    _store(key, place, settings)
    return place
