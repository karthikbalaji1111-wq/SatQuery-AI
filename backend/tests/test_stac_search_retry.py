"""A catalog search survives one transient failure, and only a transient one.

Observed live: one Sentinel-1 search failed with a transport error and the
identical request succeeded seconds later, so a whole agent run was lost to a
single dropped connection. A search is an idempotent read, so it is re-issued
once - but a request the catalog rejects, and a response that arrives malformed,
are real outcomes and are reported unchanged.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from app.core.config import Settings
from app.core.errors import UpstreamServiceError
from app.services.satellite import stac as stac_mod
from app.services.satellite.stac import search_items

BODY = {"collections": ["sentinel-2-l2a"], "bbox": [80.27, 13.03, 80.29, 13.07], "limit": 5}
FEATURES = {"type": "FeatureCollection", "features": [{"id": "S2B_TEST"}]}


class Script(httpx.AsyncBaseTransport):
    """Plays one scripted outcome per request: an exception, or (status, body)."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        outcome = self._outcomes[min(len(self.requests), len(self._outcomes) - 1)]
        self.requests.append(request)
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome  # type: ignore[misc]
        return httpx.Response(status, json=body, request=request)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stac_mod, "_SEARCH_BACKOFF_SECONDS", 0.0)


def search(script: Script) -> object:
    async def go() -> object:
        try:
            return await search_items(
                settings=Settings(_env_file=None),  # type: ignore[call-arg]
                body=BODY,
                transport=script,
            )
        except Exception as exc:  # returned so the test can assert on it
            return exc

    return asyncio.run(go())


def test_one_dropped_connection_is_retried() -> None:
    script = Script(httpx.ConnectError("reset"), (200, FEATURES))
    assert search(script) == [{"id": "S2B_TEST"}]
    assert len(script.requests) == 2


def test_one_timeout_is_retried() -> None:
    script = Script(httpx.ReadTimeout("slow"), (200, FEATURES))
    assert search(script) == [{"id": "S2B_TEST"}]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_busy_status_is_retried(status: int) -> None:
    script = Script((status, {}), (200, FEATURES))
    assert search(script) == [{"id": "S2B_TEST"}]
    assert len(script.requests) == 2


def test_the_retry_is_bounded_and_the_real_failure_surfaces() -> None:
    script = Script(httpx.ConnectError("down"))
    result = search(script)
    assert isinstance(result, UpstreamServiceError)
    assert result.message == "The satellite catalog is unavailable."
    assert len(script.requests) == stac_mod._SEARCH_ATTEMPTS


def test_a_persistent_timeout_keeps_its_own_message() -> None:
    result = search(Script(httpx.ReadTimeout("slow")))
    assert isinstance(result, UpstreamServiceError)
    assert result.message == "The satellite catalog timed out."


def test_a_persistent_busy_status_is_reported_with_its_status() -> None:
    script = Script((503, {}))
    result = search(script)
    assert isinstance(result, UpstreamServiceError)
    assert "503" in result.message
    assert len(script.requests) == stac_mod._SEARCH_ATTEMPTS


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_a_rejected_request_is_not_repeated(status: int) -> None:
    script = Script((status, {}))
    assert isinstance(search(script), UpstreamServiceError)
    assert len(script.requests) == 1


def test_a_malformed_answer_is_reported_not_resampled() -> None:
    script = Script((200, {"no": "features"}))
    assert isinstance(search(script), UpstreamServiceError)
    assert len(script.requests) == 1


def test_every_attempt_sends_the_same_search() -> None:
    script = Script(httpx.ConnectError("reset"), (200, FEATURES))
    search(script)
    assert script.requests[0].content == script.requests[1].content
