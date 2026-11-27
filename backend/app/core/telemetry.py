"""Request ids, access logs and in-process metrics.

Before this module only the agent route had a run id, nothing was counted,
and an operator could not tell how many requests failed, which routes were
slow, or which log lines belonged to a request a user reported. Three things
are added and nothing else:

* every response carries ``X-Request-ID``, and every log line written while
  the request runs is stamped with the same id (``core.observability``), so a
  user can quote it and an operator can find the run;
* one access line per request: method, route TEMPLATE, status, milliseconds.
  Never a query string, a body, a place name or a client address;
* Prometheus counters and a latency histogram per route template, served at
  ``GET /metrics`` only when ``SATQUERY_METRICS_TOKEN`` is set, and only to a
  caller presenting it. No dependency: the text format is written here.

Metrics are per PROCESS, like the rate limits; each replica reports its own.
Label values are bounded: a path that matched no route is reported as
``unmatched``, so a scanner probing random URLs cannot grow the series set.
"""

from __future__ import annotations

import hmac
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from fastapi import APIRouter, Request, Response

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import RUN_ID

logger = get_logger("http")

#: Latency buckets in seconds. Wide on purpose: a geocode answers in under a
#: second, a local-model run can take minutes.
BUCKETS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

#: Probes and the metrics scrape itself are logged at DEBUG: a health check
#: every 30 seconds would otherwise drown the lines that matter.
QUIET_ROUTES = frozenset({"/health", "/ready", "/metrics"})

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class Metrics:
    """Thread-safe request counters and a latency histogram per route."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._buckets: dict[str, list[int]] = defaultdict(lambda: [0] * len(BUCKETS))
        self._count: dict[str, int] = defaultdict(int)
        self._sum: dict[str, float] = defaultdict(float)
        self._in_flight = 0

    def started(self) -> None:
        with self._lock:
            self._in_flight += 1

    def finished(self, route: str, method: str, status: int, seconds: float) -> None:
        with self._lock:
            self._in_flight -= 1
            self._requests[(route, method, str(status))] += 1
            self._count[route] += 1
            self._sum[route] += seconds
            buckets = self._buckets[route]
            for index, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    buckets[index] += 1

    def render(self) -> str:
        """The Prometheus text exposition format, version 0.0.4."""

        with self._lock:
            lines = [
                "# HELP satquery_http_requests_total Completed HTTP requests.",
                "# TYPE satquery_http_requests_total counter",
            ]
            for (route, method, status), value in sorted(self._requests.items()):
                lines.append(
                    f'satquery_http_requests_total{{route="{_label(route)}",'
                    f'method="{method}",status="{status}"}} {value}'
                )
            lines += [
                "# HELP satquery_http_request_duration_seconds Request latency.",
                "# TYPE satquery_http_request_duration_seconds histogram",
            ]
            for route in sorted(self._count):
                label = _label(route)
                for bound, value in zip(BUCKETS, self._buckets[route], strict=True):
                    lines.append(
                        f'satquery_http_request_duration_seconds_bucket{{route="{label}",'
                        f'le="{bound:g}"}} {value}'
                    )
                lines.append(
                    f'satquery_http_request_duration_seconds_bucket{{route="{label}",'
                    f'le="+Inf"}} {self._count[route]}'
                )
                lines.append(
                    f'satquery_http_request_duration_seconds_sum{{route="{label}"}} '
                    f"{self._sum[route]:.6f}"
                )
                lines.append(
                    f'satquery_http_request_duration_seconds_count{{route="{label}"}} '
                    f"{self._count[route]}"
                )
            lines += [
                "# HELP satquery_http_requests_in_flight Requests being served now.",
                "# TYPE satquery_http_requests_in_flight gauge",
                f"satquery_http_requests_in_flight {self._in_flight}",
            ]
        return "\n".join(lines) + "\n"


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class RequestTelemetryMiddleware:
    """Stamp, time, count and log every HTTP request.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so it adds one header to the
    response start message and nothing else: no buffering, no task switch.
    Installed OUTERMOST, so a refusal produced by another middleware (413, a
    CORS preflight) is counted and carries an id too.
    """

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:12]
        token = RUN_ID.set(request_id)
        status = 500
        started = time.perf_counter()
        self.metrics.started()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            seconds = time.perf_counter() - started
            # The full request path once a route matched it. With nested
            # routers the matched route object knows only its own suffix
            # ("/parse"); and no route here has a path parameter, so the path
            # of a MATCHED request is one of a fixed set. A route with path
            # parameters would need its template here instead.
            route = str(scope.get("path", "")) if scope.get("route") is not None else "unmatched"
            method = str(scope.get("method", "-"))
            self.metrics.finished(route, method, status, seconds)
            level = 10 if route in QUIET_ROUTES else 20  # DEBUG / INFO
            logger.log(
                level,
                "request method=%s route=%s status=%d ms=%.0f",
                method,
                route,
                status,
                seconds * 1000,
            )
            RUN_ID.reset(token)


router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Prometheus metrics, for a caller holding ``SATQUERY_METRICS_TOKEN``.

    Without a configured token the route does not exist (404), so a default
    deployment publishes nothing. With one, any other caller gets 401.
    """

    secret = get_settings().metrics_token
    if not secret:
        return Response(status_code=404)
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {secret.get_secret_value()}"
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        return Response(status_code=401, headers={"WWW-Authenticate": "Bearer"})
    store: Metrics = request.app.state.metrics
    return Response(
        content=store.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
