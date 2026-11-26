"""Admission control for expensive, unauthenticated endpoints.

**What this bounds.** One anonymous request can cost this deployment a great
deal: a catalog search per (modality x window), a windowed COG read per band,
a base64 PNG held in memory, and one or more provider calls that may be
metered. Nothing bounded how MANY such requests could be in flight, or how
often one client could ask. The per-request contracts already cap the shape of
a single request (``MAX_TIME_WINDOWS``, ``imagery_max_dimension``,
``imagery_max_window_pixels``); this module caps the RATE and the CONCURRENCY,
which no schema can express.

**Three separate questions, three separate answers:**

===========================  ===========================================
Is this client asking too     :class:`FixedWindowRateLimiter` -> 429 with
often?                        ``Retry-After``
Is this process already       :class:`ConcurrencyGate` -> 503, after a brief
doing all it can?             wait, so a passing burst queues rather than fails
Is this one request costing   the workflow budget -> 504
more than we allow?
===========================  ===========================================

Rejections are distinguishable on purpose: 429 is about the caller, 503 is
about this process, 504 is about this request. Collapsing them would tell a
client to back off when the honest answer is "try again now".

**SCOPE: ONE PROCESS. This is not a distributed rate limiter.** Counters and
semaphores live in this process's memory, so N replicas admit N times as much,
and a restart forgets every counter. That is a deliberate, documented limit
rather than an oversight: the deployment this ships with is a single container,
and a shared-storage limiter would add a hard dependency (Redis) that nothing
else here needs. The boundary is drawn so that changing it later means
substituting the two classes below - both are constructed once, in
:func:`install_request_limits`, and used only through the dependencies at the
bottom of this module. Multi-replica deployments MUST NOT claim these limits
hold globally; see the deployment documentation.

Nothing here is a security control. It is a cost and stability control: it
makes a burst expensive for the burster rather than for everyone else.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from ipaddress import ip_address

from fastapi import FastAPI, Request

from app.core.config import Settings
from app.core.errors import (
    PayloadTooLargeError,
    RateLimitedError,
    ServiceOverloadedError,
    WorkflowTimeoutError,
    error_response,
)
from app.core.logging import get_logger

logger = get_logger("limits")

#: How many distinct clients the limiter will track before it sweeps. An
#: unauthenticated endpoint is addressed by anyone, so the bookkeeping itself
#: has to be bounded - otherwise the defence against unbounded memory is an
#: unbounded dictionary.
_MAX_TRACKED_CLIENTS = 4096



@dataclass
class FixedWindowRateLimiter:
    """Requests per client within a sliding window, counted in memory.

    A sliding window rather than a fixed calendar window: the latter admits
    twice the limit across a boundary, which is exactly the burst this exists
    to stop. The cost is one timestamp per admitted request per client, bounded
    by ``limit`` entries and by :data:`_MAX_TRACKED_CLIENTS` clients.
    """

    limit: int
    window_seconds: float
    _hits: dict[str, deque[float]] = field(default_factory=dict, repr=False)

    def admit(self, client: str, *, now: float | None = None) -> None:
        """Record one request, or raise :class:`RateLimitedError`.

        ``now`` is injectable so a test can prove the window's behaviour
        without sleeping through it.
        """

        moment = time.monotonic() if now is None else now
        hits = self._hits.get(client)
        if hits is None:
            if len(self._hits) >= _MAX_TRACKED_CLIENTS:
                self._sweep(moment)
            hits = self._hits.setdefault(client, deque())

        cutoff = moment - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.limit:
            # The oldest hit is the one that has to age out before this client
            # may proceed, so it is exactly how long to wait.
            retry_after = hits[0] + self.window_seconds - moment
            raise RateLimitedError(
                f"Too many requests: at most {self.limit} are allowed every "
                f"{self.window_seconds:.0f} seconds.",
                retry_after_seconds=retry_after,
            )

        hits.append(moment)

    def _sweep(self, moment: float) -> None:
        """Drop clients whose window has fully aged out."""

        cutoff = moment - self.window_seconds
        stale = [
            client
            for client, hits in self._hits.items()
            if not hits or hits[-1] <= cutoff
        ]
        for client in stale:
            del self._hits[client]
        if not stale:
            # Every tracked client is currently active. Rather than grow without
            # bound, forget the least recently seen one; it is re-admitted on
            # its next request, which is the safe direction to fail.
            oldest = min(self._hits, key=lambda c: self._hits[c][-1])
            del self._hits[oldest]
        logger.info("Rate-limiter swept; tracking %d clients", len(self._hits))


class ConcurrencyGate:
    """A bounded number of simultaneous holders, with a brief queue.

    The wait matters. Rejecting the instant the last slot is taken turns a
    perfectly serviceable burst of two into a failure; waiting forever turns a
    slow upstream into an unbounded queue of hanging requests. A short wait
    absorbs the first and refuses the second.
    """

    def __init__(self, limit: int, *, name: str) -> None:
        self._semaphore = asyncio.Semaphore(limit)
        self._name = name
        self._limit = limit

    async def __aenter__(self) -> None:  # pragma: no cover - see acquire()
        raise NotImplementedError("use `async with gate.slot(wait=...)`")

    async def __aexit__(self, *_: object) -> None:  # pragma: no cover
        raise NotImplementedError

    def slot(self, *, wait: float) -> _GateSlot:
        return _GateSlot(self, wait)

    async def _acquire(self, wait: float) -> None:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=wait)
        except TimeoutError as exc:
            logger.info("%s at capacity (%d in use)", self._name, self._limit)
            raise ServiceOverloadedError(
                f"The service is at capacity for {self._name}. "
                "Please retry in a few seconds."
            ) from exc

    def _release(self) -> None:
        self._semaphore.release()


class _GateSlot:
    """One holder of a :class:`ConcurrencyGate`, as an async context manager."""

    def __init__(self, gate: ConcurrencyGate, wait: float) -> None:
        self._gate = gate
        self._wait = wait

    async def __aenter__(self) -> None:
        await self._gate._acquire(self._wait)

    async def __aexit__(self, *_: object) -> None:
        self._gate._release()


# --------------------------------------------------------------------------- #
# Process-wide gates
#
# These guard resources that belong to the PROCESS rather than to a request, so
# they are module-level: a geocoder politeness budget and a raster-read budget
# mean nothing if each request brings its own. They are used from the service
# layer, which has no access to the application object.
# --------------------------------------------------------------------------- #

#: Windowed COG reads are CPU- and memory-bound and run in a threadpool. Without
#: a cap, concurrent analyses multiply resident raster memory on a machine that
#: has already been observed being OOM-killed.
#:
#: A THREADING semaphore, not an asyncio one: the reads run inside
#: ``run_in_threadpool`` workers (and the sync imagery route), where there is no
#: event loop to await on. An earlier asyncio ``RASTER_GATE`` was declared here
#: and documented as enforced, but nothing ever entered it - so the setting
#: bounded nothing. This one is entered by ``ImageryService`` around every
#: remote raster read.
#:
#: One semaphore per configured limit, created on first use: the limit is read
#: from the settings the service was built with, never fixed at import time.
#:
#: The geocoder's own budget is NOT here: OpenStreetMap's policy is about
#: request SPACING as well as concurrency, and both live together in
#: ``services/geospatial/nominatim.py`` so there is one place that decides how
#: this application treats that service, rather than two that must agree.
_RASTER_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_RASTER_SEMAPHORES_LOCK = threading.Lock()


def _raster_semaphore(limit: int) -> threading.BoundedSemaphore:
    with _RASTER_SEMAPHORES_LOCK:
        semaphore = _RASTER_SEMAPHORES.get(limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max(1, limit))
            _RASTER_SEMAPHORES[limit] = semaphore
        return semaphore


@contextmanager
def raster_read_slot(settings: Settings) -> Iterator[None]:
    """Hold one of the process's raster-read slots for the duration of a read.

    Blocks the calling worker thread for up to
    ``raster_admission_wait_seconds``; past that the read is refused with 503
    rather than queueing without bound behind a slow upstream.
    """

    semaphore = _raster_semaphore(settings.max_concurrent_raster_reads)
    wait = settings.raster_admission_wait_seconds
    if not semaphore.acquire(timeout=wait):
        logger.info(
            "raster reads at capacity (%d in use, waited %.0fs)",
            settings.max_concurrent_raster_reads,
            wait,
        )
        raise ServiceOverloadedError(
            "The service is at capacity for raster reads. "
            "Please retry in a few seconds."
        )
    try:
        yield
    finally:
        semaphore.release()


async def within_budget[T](request: Request, awaitable: Awaitable[T]) -> T:
    """Await one workflow, ending it with 504 once its budget is spent.

    ``workflow_budget_seconds`` was configured and documented as enforced, and
    nothing read it. This is the one place it is applied; the expensive routes
    await their service call through it.

    Only THIS timeout becomes a 504: a ``TimeoutError`` raised by the work
    itself (an upstream that timed out) propagates unchanged, because
    ``asyncio.timeout`` reports separately whether it was the one that expired.

    KNOWN LIMITATION: cancelling the awaiting coroutine cannot stop a raster
    read already running in a worker thread. That read finishes (bounded by
    the HTTP timeouts below it) and its result is discarded; the request's
    workflow slot is released at once.
    """

    limits: RequestLimits | None = getattr(request.app.state, "limits", None)
    if limits is None:
        return await awaitable
    budget = limits.settings.workflow_budget_seconds
    deadline = asyncio.timeout(budget)
    try:
        async with deadline:
            return await awaitable
    except TimeoutError as exc:
        if not deadline.expired():
            raise
        logger.warning("workflow exceeded its %.0fs budget", budget)
        raise WorkflowTimeoutError(
            f"The request did not finish within {budget:.0f} seconds and was "
            "stopped. Try a smaller area or a single period."
        ) from exc


@dataclass
class RequestLimits:
    """Per-application admission state.

    Held on ``app.state`` rather than at module level so each application - and
    therefore each test - starts with its own counters. A limiter shared across
    every test in a process would make one test's requests another's failures.
    """

    limiter: FixedWindowRateLimiter
    workflows: ConcurrencyGate
    settings: Settings


def install_request_limits(app: FastAPI, settings: Settings) -> None:
    """Attach admission state and the payload-size guard to ``app``."""

    app.state.limits = RequestLimits(
        limiter=FixedWindowRateLimiter(
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        ),
        workflows=ConcurrencyGate(
            settings.max_concurrent_workflows, name="workflows"
        ),
        settings=settings,
    )

    max_bytes = settings.max_request_bytes

    @app.middleware("http")
    async def _limit_request_size(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Refuse an oversized body before it is read into memory.

        ``Content-Length`` is checked rather than the body itself: the point is
        to answer before the bytes arrive. A chunked request without that header
        is not bounded here - it remains bounded by the field limits on every
        contract it must validate against, which is stated in the deployment
        documentation rather than quietly assumed.
        """

        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            logger.info("Refused a %s byte body (limit %d)", declared, max_bytes)
            # Returned, not raised: an exception here would travel past the
            # registered handlers and surface as a bare 500 in a different
            # shape. Same envelope, same vocabulary.
            return error_response(
                PayloadTooLargeError.code,
                f"The request body exceeds the {max_bytes} byte limit.",
                status_code=PayloadTooLargeError.status_code,
            )
        return await call_next(request)


_FORWARDED_CHAIN_REPORTED = False


def _peer(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _client_of(request: Request, trusted_proxy_hops: int = 0) -> str:
    """Who this request is attributed to.

    With ``trusted_proxy_hops == 0`` (the default) it is the TCP peer, never a
    header, because ``X-Forwarded-For`` is written by whoever sends the request.

    Behind proxies, the peer is the proxy, so every caller would share ONE
    allowance and one busy client would lock everyone out. Each proxy APPENDS
    the address it received the connection from, so the entry ``hops`` places
    from the right was written by the outermost trusted proxy and cannot be
    forged by the client; anything to its left can be. That entry is used.

    The header is read here rather than taken from ``request.client`` after
    uvicorn's proxy-header rewriting: told to trust every proxy
    (``FORWARDED_ALLOW_IPS=*``), uvicorn takes the LEFTMOST entry, which the
    client chose. A chain shorter than the configured hop count, or an entry
    that is not an IP address, falls back to the peer rather than to anything
    the client supplied.
    """

    if trusted_proxy_hops <= 0:
        return _peer(request)

    header = ",".join(request.headers.getlist("x-forwarded-for"))
    chain = [entry.strip() for entry in header.split(",") if entry.strip()]
    _report_chain_length(len(chain))
    if len(chain) < trusted_proxy_hops:
        return _peer(request)
    candidate = chain[-trusted_proxy_hops]
    try:
        return str(ip_address(candidate))
    except ValueError:
        return _peer(request)


def _report_chain_length(length: int) -> None:
    """Log, once per process, how many forwarding entries arrive.

    Hosting platforms rarely document how many proxies sit in front of a
    service. For a request the client sent WITHOUT its own X-Forwarded-For,
    this count is exactly the value ``SATQUERY_TRUSTED_PROXY_HOPS`` should
    have. Only the count is logged - never an address.
    """

    global _FORWARDED_CHAIN_REPORTED
    if not _FORWARDED_CHAIN_REPORTED:
        _FORWARDED_CHAIN_REPORTED = True
        logger.info("first request carried %d X-Forwarded-For entries", length)


async def rate_limited(request: Request) -> None:
    """Dependency: count this request against its client's allowance."""

    limits: RequestLimits | None = getattr(request.app.state, "limits", None)
    if limits is None:  # an app built without limits installed
        return
    limits.limiter.admit(_client_of(request, limits.settings.trusted_proxy_hops))


async def workflow_slot(request: Request) -> AsyncIterator[None]:
    """Dependency: hold one workflow slot for the lifetime of the request.

    Applied to the endpoints that do real work - discovery, analysis and the
    agent - and deliberately NOT to ``/health`` or the model catalog, which must
    keep answering while the process is busy. A readiness probe that queues
    behind four analyses reports an outage that is not happening.
    """

    limits: RequestLimits | None = getattr(request.app.state, "limits", None)
    if limits is None:
        yield
        return
    async with limits.workflows.slot(wait=limits.settings.admission_wait_seconds):
        yield
