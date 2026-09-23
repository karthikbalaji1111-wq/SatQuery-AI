"""The local provider never carries a cloud credential to the model or back.

Local inference has no key of its own, which makes it easy to assume there is
nothing to leak. The risk runs the other way: a deployment configured for every
provider holds three cloud keys, and selecting ``local`` must never put one of
them into a request sent to the model, into a failure response, or into a log.
The same distinctive sentinels as the cloud providers' suite are used, so a
single substring search catches a key wherever it surfaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest
from app.core.config import Settings
from app.main import app
from app.services.agent.providers import factory as factory_mod
from app.services.agent.providers.local import (
    LocalAgentPlanner,
    LocalAnswerSynthesizer,
    LocalIntentParser,
    LocalVisualAnalyst,
)
from fastapi.testclient import TestClient

from tests.test_agent_service import make_evidence
from tests.test_provider_credential_isolation import (
    ANTHROPIC_SENTINEL,
    GEMINI_SENTINEL,
    NVIDIA_SENTINEL,
)

SENTINELS = (GEMINI_SENTINEL, NVIDIA_SENTINEL, ANTHROPIC_SENTINEL)
PNG = b"\x89PNG\r\n\x1a\nthe-retrieved-scene"
INTENT = {
    "location_query": "Marina Beach, Chennai",
    "temporal_mode": "single",
    "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
    "modalities": ["sentinel-2-optical"],
    "task": "visualize",
}
PLAN = json.dumps({"discovery": {"tool": "execute_query", "intent": INTENT}, "analysis": []})
DRAFT = json.dumps(
    {"summary": "The mean NDWI was 0.2777 index.", "evidence_refs": ["ndwi.ndwi_mean"]}
)


def settings(base_url: str = "http://ollama.test:11434") -> Settings:
    """Every cloud provider credentialed with a sentinel, local selected."""

    return Settings(  # type: ignore[arg-type]
        _env_file=None,
        AI_PROVIDER="local",
        LOCAL_AI_BASE_URL=base_url,
        GEMINI_API_KEY=GEMINI_SENTINEL,
        NVIDIA_API_KEY=NVIDIA_SENTINEL,
        ANTHROPIC_API_KEY=ANTHROPIC_SENTINEL,
    )


def assert_clean(text: str, where: str) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in text, f"a cloud credential reached {where}"


class Capture(httpx.AsyncBaseTransport):
    """Records the request and answers with a well-formed chat response."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = {"message": {"role": "assistant", "content": self._content}, "done": True}
        return httpx.Response(200, json=body, request=request)


@pytest.mark.parametrize(
    ("role_cls", "content", "call"),
    [
        (LocalAgentPlanner, PLAN, lambda r: r.plan("What is the NDWI?")),
        (
            LocalAnswerSynthesizer,
            DRAFT,
            lambda r: r.synthesize("What is the NDWI?", make_evidence()),
        ),
        (LocalIntentParser, json.dumps(INTENT), lambda r: r.parse_intent("NDWI?")),
        (
            LocalVisualAnalyst,
            "Water is visible.",
            lambda r: r.observe(question="Is water visible?", image=PNG, media_type="image/png"),
        ),
    ],
    ids=["planner", "synthesizer", "intent", "visual"],
)
def test_no_cloud_credential_is_sent_to_the_local_model(
    role_cls: type, content: str, call: Any
) -> None:
    transport = Capture(content)

    async def go() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            await call(role_cls(settings=settings(), client=client))

    asyncio.run(go())

    [request] = transport.requests
    assert_clean(request.content.decode(), "the request body sent to the local model")
    assert_clean(str(request.headers), "the request headers sent to the local model")
    assert "authorization" not in {name.lower() for name in request.headers}


def test_a_local_failure_discloses_no_cloud_credential(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ollama unreachable (the discard port), every cloud key configured."""

    configured = settings(base_url="http://127.0.0.1:9")
    monkeypatch.setattr(factory_mod, "get_settings", lambda: configured)
    satquery = logging.getLogger("satquery")
    satquery.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger="satquery"):
            app.dependency_overrides.clear()
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/query/agent",
                    json={"question": "What is the NDWI of Marina Beach?", "provider": "local"},
                )
    finally:
        satquery.removeHandler(caplog.handler)

    assert response.status_code == 200
    assert response.json()["status"] == "planner_unavailable"
    assert_clean(response.text, "the API response")
    assert_clean(caplog.text, "the logs")
