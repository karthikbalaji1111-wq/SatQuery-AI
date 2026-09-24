"""Thin async client for the OpenStreetMap Nominatim geocoding API.

Only the ``/search`` endpoint is used. Network, status, and payload failures are
normalised into :class:`app.core.errors.AppError` subclasses so the API layer can
return consistent JSON errors.

**This module owns how the application treats Nominatim.** Its operator asks
users of the public instance for at most one request per second from an
application, a genuine identifying ``User-Agent``, and caching of results.

**Why the budget has to be this careful.** A cloud host's outbound IP is shared:
Render states that its outbound ranges are "shared across all services in the
same region", and Nominatim limits per IP. So this application spends a budget
it shares with strangers, and an HTTP 429 says the SHARED budget is spent - not
that this process sent too much. Observed in production (2026-09-24): for about
forty minutes most geocodes from Render were refused with 429 while the same
User-Agent from another network got 200, at a rate from this application of a
few requests per MINUTE. What this process can control is how little it asks
and how it behaves when refused:

* **cache** - successful answers only, bounded (LRU) and expiring (TTL, a day
  by default), keyed by the normalised query AND the service asked; each entry
  records its source and when it was resolved;
* **coalescing** - concurrent callers for one place share ONE upstream sequence
  and its outcome, success or failure;
* **pacing** - one request in flight, at least
  ``geocoder_min_interval_seconds`` between the starts of two;
* **cooldown** - after a 429 NO request leaves this process until Retry-After
  (or an exponential backoff when absent) has passed. A caller that would wait
  longer than its budget is told so at once (``geocoding_unavailable``, 503,
  with Retry-After) without contacting the service;
* **bounded retry** - at most ``geocoder_max_attempts`` requests per geocode,
  with backoff between them, and never more than
  ``geocoder_retry_budget_seconds`` of waiting. When a 5xx, a timeout or an
  unreachable service outlasts them, that too is ``geocoding_unavailable`` (no
  wait claimed); a malformed answer or a 4xx stays ``upstream_error``.

Nothing here invents a location: a failure is an error, never a fallback
coordinate, a nearby city or a stale guess.

**SCOPE: ONE PROCESS.** Like everything in ``app/core/limits.py``, this holds
for a single replica. Two replicas make two requests per second, so a
multi-instance deployment MUST NOT claim policy compliance on the strength of
this module alone; see the deployment documentation.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import (
    GeocodingUnavailableError,
    NotFoundError,
    UpstreamServiceError,
)
from app.core.logging import get_logger
from app.services.geospatial.schemas import BoundingBox, Coordinate

logger = get_logger("geospatial.nominatim")

#: The service asked this process to slow down. Starts the process-wide
#: cooldown, not merely a retry of this one request.
_THROTTLE_STATUS = frozenset({429})

#: Worth another attempt after a backoff: a server-side fault. A 4xx other
#: than 429 is the request's own problem and repeating it is rude.
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})

#: Upper bound on an honoured Retry-After, so an absurd header cannot silence
#: geocoding for days. An hour is far beyond anything observed.
_MAX_RETRY_AFTER_SECONDS = 3_600.0


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


@dataclass(frozen=True)
class _CacheEntry:
    """One successful resolution, with where and when it came from."""

    query: str
    source: str
    place: NominatimPlace
    #: Monotonic clock, for expiry.
    stored_at: float
    #: Wall clock, for a reader.
    resolved_at: datetime


@dataclass
class _Counters:
    upstream_requests: int = 0
    throttled: int = 0
    cache_hits: int = 0
    coalesced: int = 0
    refused_during_cooldown: int = 0


@dataclass(frozen=True)
class GeocoderStatus:
    """What this process has done with the geocoder since it started."""

    upstream_requests: int
    throttled: int
    cache_hits: int
    coalesced: int
    refused_during_cooldown: int
    cache_entries: int
    cooldown_remaining_seconds: float


class _ThrottledError(Exception):
    """HTTP 429. ``retry_after`` is the upstream's own ask, when it gave one."""

    def __init__(self, retry_after: float | None) -> None:
        super().__init__("throttled")
        self.retry_after = retry_after


class _TransientError(Exception):
    """A failure worth another attempt after a backoff.

    Carries the message the caller will see if every attempt fails, so a
    retried failure reports the same thing it always did rather than a vaguer
    summary of several attempts.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


# --------------------------------------------------------------------------- #
# Process-wide state
# --------------------------------------------------------------------------- #

#: Serialises upstream requests; also what makes the spacing meaningful, since
#: two concurrent callers would otherwise both see the same "last request"
#: timestamp and both proceed. Held for ONE request at a time, never across a
#: backoff, so a caller retrying one place does not stall every other place.
_GEOCODE_LOCK = asyncio.Lock()

#: When the last upstream request was STARTED, on the monotonic clock.
_LAST_REQUEST_AT: float | None = None

#: Normalised key -> entry. Ordered so the least recently used is evicted first.
_CACHE: OrderedDict[str, _CacheEntry] = OrderedDict()

#: Key -> the future of the upstream sequence already running for it.
_IN_FLIGHT: dict[str, asyncio.Future[NominatimPlace]] = {}

#: No request leaves this process before this monotonic instant.
_COOLDOWN_UNTIL: float | None = None

#: 429s since the last success; drives the backoff when Retry-After is absent.
_CONSECUTIVE_THROTTLES = 0

_COUNTERS = _Counters()


def _now() -> float:
    """The monotonic clock. A function so a test can substitute a fake one."""

    return time.monotonic()


async def _pause(seconds: float) -> None:
    """Wait. A function so a test can advance a fake clock instead of sleeping."""

    await asyncio.sleep(seconds)


def reset_geocoder_state() -> None:
    """Forget the throttle, the cooldown, the cache and the counters.

    For tests. Module-level state shared across a test session would otherwise
    let one test's cached answer satisfy another test's request - and one
    test's timestamp or cooldown delay another's.
    """

    global _LAST_REQUEST_AT, _COOLDOWN_UNTIL, _CONSECUTIVE_THROTTLES, _COUNTERS
    global _GEOCODE_LOCK
    _LAST_REQUEST_AT = None
    _COOLDOWN_UNTIL = None
    _CONSECUTIVE_THROTTLES = 0
    _COUNTERS = _Counters()
    _CACHE.clear()
    _IN_FLIGHT.clear()
    # A lock is bound to the loop that first waited on it; each test runs its
    # own loop.
    _GEOCODE_LOCK = asyncio.Lock()


def geocoder_status() -> GeocoderStatus:
    """A snapshot for the readiness report. Reads state; contacts nothing."""

    return GeocoderStatus(
        upstream_requests=_COUNTERS.upstream_requests,
        throttled=_COUNTERS.throttled,
        cache_hits=_COUNTERS.cache_hits,
        coalesced=_COUNTERS.coalesced,
        refused_during_cooldown=_COUNTERS.refused_during_cooldown,
        cache_entries=len(_CACHE),
        cooldown_remaining_seconds=_cooldown_remaining(),
    )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def _normalise(query: str) -> str:
    return " ".join(query.lower().split())


def _cache_key(query: str, settings: Settings) -> str:
    """Case- and whitespace-insensitive, and scoped to the service asked.

    The base URL is part of the key because it identifies WHICH geocoder
    answered; a cached answer from one deployment must never satisfy a request
    aimed at another.
    """

    return f"{settings.nominatim_base_url}|{_normalise(query)}"


def _cached(key: str, settings: Settings) -> NominatimPlace | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if _now() - entry.stored_at > settings.geocoder_cache_ttl_seconds:
        del _CACHE[key]
        return None
    _CACHE.move_to_end(key)
    return entry.place


def _store(key: str, query: str, place: NominatimPlace, settings: Settings) -> None:
    """Cache a SUCCESSFUL resolution only.

    A failure is never cached: an upstream outage would otherwise be repeated
    back to every caller for the whole TTL, long after the service recovered.
    """

    _CACHE[key] = _CacheEntry(
        query=_normalise(query),
        source=f"nominatim {settings.nominatim_base_url}",
        place=place,
        stored_at=_now(),
        resolved_at=datetime.now(UTC),
    )
    _CACHE.move_to_end(key)
    while len(_CACHE) > settings.geocoder_cache_entries:
        _CACHE.popitem(last=False)


# --------------------------------------------------------------------------- #
# Pacing and cooldown
# --------------------------------------------------------------------------- #


def _cooldown_remaining() -> float:
    if _COOLDOWN_UNTIL is None:
        return 0.0
    return max(0.0, _COOLDOWN_UNTIL - _now())


def _backoff(failures: int, settings: Settings) -> float:
    """``base * 2**(failures - 1)``, capped. Zero when the base is zero."""

    base = settings.geocoder_backoff_base_seconds
    return min(settings.geocoder_max_cooldown_seconds, base * 2 ** max(0, failures - 1))


def _register_throttle(retry_after: float | None, settings: Settings) -> float:
    """Start (or extend) the process-wide cooldown; return its length."""

    global _COOLDOWN_UNTIL, _CONSECUTIVE_THROTTLES
    _CONSECUTIVE_THROTTLES += 1
    _COUNTERS.throttled += 1
    wait = (
        retry_after
        if retry_after is not None
        else _backoff(_CONSECUTIVE_THROTTLES, settings)
    )
    until = _now() + wait
    _COOLDOWN_UNTIL = until if _COOLDOWN_UNTIL is None else max(_COOLDOWN_UNTIL, until)
    logger.warning(
        "Geocoder throttled (HTTP 429, %d in a row): no upstream request for %.1fs%s",
        _CONSECUTIVE_THROTTLES,
        wait,
        " (Retry-After)" if retry_after is not None else " (backoff)",
    )
    return wait


def _register_success() -> None:
    global _COOLDOWN_UNTIL, _CONSECUTIVE_THROTTLES
    _CONSECUTIVE_THROTTLES = 0
    _COOLDOWN_UNTIL = None


async def _wait_for_turn(settings: Settings) -> None:
    """Hold until this process may issue another request."""

    global _LAST_REQUEST_AT
    interval = settings.geocoder_min_interval_seconds
    if interval > 0 and _LAST_REQUEST_AT is not None:
        delay = interval - (_now() - _LAST_REQUEST_AT)
        if delay > 0:
            logger.info("Geocoder throttle: waiting %.2fs", delay)
            await _pause(delay)
    _LAST_REQUEST_AT = _now()


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds from a ``Retry-After`` header (delta or HTTP-date), or None."""

    value = response.headers.get("retry-after")
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()
    if seconds != seconds:  # NaN
        return None
    return min(_MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))


def _unavailable(wait: float) -> GeocodingUnavailableError:
    seconds = max(1, round(wait))
    return GeocodingUnavailableError(
        "Location lookup is temporarily unavailable: the public OpenStreetMap "
        "geocoder is limiting requests from this server (HTTP 429). Nothing was "
        f"searched. Try again in about {seconds} seconds.",
        retry_after_seconds=wait,
    )


# --------------------------------------------------------------------------- #
# One request
# --------------------------------------------------------------------------- #


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
    """One upstream request.

    Raises :class:`_ThrottledError` on 429 and :class:`_TransientError` on anything
    else worth another attempt.
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
        raise _TransientError("The geocoding service timed out.") from exc
    except httpx.HTTPError as exc:
        raise _TransientError("The geocoding service is unavailable.") from exc

    if response.status_code in _THROTTLE_STATUS:
        raise _ThrottledError(_retry_after(response))

    if response.status_code in _RETRYABLE_STATUS:
        logger.warning(
            "Nominatim responded with HTTP %s for query %r",
            response.status_code,
            query,
        )
        raise _TransientError(
            f"The geocoding service responded with status {response.status_code}.",
            retry_after=_retry_after(response),
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


async def _resolve_upstream(
    query: str,
    key: str,
    *,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None,
) -> NominatimPlace:
    """The bounded upstream sequence for one place.

    Every wait - a cooldown, a Retry-After, a backoff - is charged to one
    budget, counted as PLANNED waiting rather than read off the wall clock, so
    the loop is bounded however the clock behaves.
    """

    budget = settings.geocoder_retry_budget_seconds
    waited = 0.0
    attempts = 0
    last: _TransientError | None = None
    # A cooldown can be re-armed by another caller between our wait and our
    # turn; this caps how often that can send us back to wait.
    for _ in range(settings.geocoder_max_attempts * 4 + 4):
        cooling = _cooldown_remaining()
        if cooling > 0:
            if waited + cooling > budget:
                if attempts == 0:
                    # Answered without sending anything at all.
                    _COUNTERS.refused_during_cooldown += 1
                raise _unavailable(cooling)
            waited += cooling
            await _pause(cooling)
            continue

        async with _GEOCODE_LOCK:
            if _cooldown_remaining() > 0:
                continue  # started by another caller while this one queued
            cached = _cached(key, settings)
            if cached is not None:
                logger.info("Geocoder cache hit for %r (filled while waiting)", query)
                _COUNTERS.cache_hits += 1
                return cached
            await _wait_for_turn(settings)
            attempts += 1
            _COUNTERS.upstream_requests += 1
            try:
                payload = await _fetch_once(query, settings=settings, transport=transport)
            except _ThrottledError as exc:
                wait = _register_throttle(exc.retry_after, settings)
                if attempts >= settings.geocoder_max_attempts:
                    raise _unavailable(wait) from None
                # The wait happens at the top of the loop, charged to the
                # budget - outside the lock, so other places are not stalled.
                continue
            except _TransientError as exc:
                last = exc
                logger.info(
                    "Geocoder attempt %d/%d failed: %s",
                    attempts,
                    settings.geocoder_max_attempts,
                    exc.message,
                )
            else:
                _register_success()
                place = _parse_first_result(payload)
                _store(key, query, place, settings)
                return place

        # A transient failure: back off (outside the lock), within budget.
        if attempts >= settings.geocoder_max_attempts:
            break
        delay = (
            last.retry_after
            if last is not None and last.retry_after is not None
            else _backoff(attempts, settings)
        )
        if waited + delay > budget:
            break
        waited += delay
        if delay > 0:
            await _pause(delay)

    # Every attempt failed transiently (5xx, timeout, unreachable): the
    # location service is unavailable. The last failure's own words are kept
    # for diagnostics; no wait is claimed, because none is known.
    raise GeocodingUnavailableError(
        "Location lookup is temporarily unavailable: "
        + (last.message if last else "The geocoding service is unavailable.")
        + " Nothing was searched."
    )


def _retrieve(future: asyncio.Future[NominatimPlace]) -> None:
    """Mark a finished future's exception as seen, so nobody-waiting is quiet."""

    if not future.cancelled():
        future.exception()


async def geocode(
    query: str,
    *,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> NominatimPlace:
    """Resolve a free-text place name to coordinates and a bounding box.

    ``transport`` is injectable so tests can stub the HTTP call without touching
    the live service.

    A cached answer is returned without contacting the service at all. A
    caller asking for a place another caller is already resolving waits for
    THAT resolution and shares its outcome. Otherwise the bounded upstream
    sequence runs (see :func:`_resolve_upstream`).
    """

    key = _cache_key(query, settings)
    cached = _cached(key, settings)
    if cached is not None:
        logger.info("Geocoder cache hit for %r", query)
        _COUNTERS.cache_hits += 1
        return cached

    loop = asyncio.get_running_loop()
    pending = _IN_FLIGHT.get(key)
    if pending is not None and not pending.done() and pending.get_loop() is loop:
        _COUNTERS.coalesced += 1
        logger.info("Geocoder request for %r joined one already in flight", query)
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if pending.cancelled() and not (task is not None and task.cancelling()):
                # The caller we were waiting on was cancelled, not us: resolve
                # it ourselves rather than inherit someone else's cancellation.
                return await geocode(query, settings=settings, transport=transport)
            raise

    future: asyncio.Future[NominatimPlace] = loop.create_future()
    future.add_done_callback(_retrieve)
    _IN_FLIGHT[key] = future
    try:
        place = await _resolve_upstream(
            query, key, settings=settings, transport=transport
        )
    except asyncio.CancelledError:
        future.cancel()
        raise
    except BaseException as exc:
        future.set_exception(exc)
        raise
    else:
        future.set_result(place)
        return place
    finally:
        if _IN_FLIGHT.get(key) is future:
            del _IN_FLIGHT[key]


# --------------------------------------------------------------------------- #
# Warming the cache
# --------------------------------------------------------------------------- #


def warm_places(settings: Settings) -> list[str]:
    """The configured places to warm, in order, without blanks or repeats."""

    seen: dict[str, str] = {}
    for raw in settings.geocoder_warm_places.split(";"):
        place = " ".join(raw.split())
        if place and _normalise(place) not in seen:
            seen[_normalise(place)] = place[:300]
    return list(seen.values())[:20]


async def warm_geocoder_cache(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    rounds: int = 4,
) -> list[str]:
    """Geocode the configured places through the normal path; return the failed.

    Runs in the background after startup. Each place goes through
    :func:`geocode` - the same cache, pacing, coalescing and cooldown as a
    user's request, so warming can never outpace the policy. A place the
    service refuses for now is tried again after the cooldown, for at most
    ``rounds`` rounds; a place the service cannot find is not retried. Real
    answers only: a place that never resolves is simply not cached.
    """

    remaining = warm_places(settings)
    for round_number in range(rounds):
        failed: list[str] = []
        for place in remaining:
            try:
                await geocode(place, settings=settings, transport=transport)
            except GeocodingUnavailableError:
                failed.append(place)
            except (NotFoundError, UpstreamServiceError) as exc:
                logger.warning("Geocoder warm-up skipped %r: %s", place, exc.message)
        if not failed:
            logger.info("Geocoder warm-up complete (%d places)", len(remaining))
            return []
        remaining = failed
        wait = max(_cooldown_remaining(), settings.geocoder_backoff_base_seconds)
        if round_number + 1 < rounds:
            logger.info(
                "Geocoder warm-up: %d place(s) refused for now; next round in %.0fs",
                len(failed),
                wait,
            )
            await _pause(wait)
    logger.warning("Geocoder warm-up gave up on %d place(s)", len(remaining))
    return remaining
