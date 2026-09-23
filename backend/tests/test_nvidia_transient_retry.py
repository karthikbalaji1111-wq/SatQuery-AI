"""NVIDIA synthesis and observation survive one transient endpoint failure.

Observed live: the planner (which retried) succeeded, the deterministic tools
ran, and then ONE shared-capacity 503 during synthesis ended the run as
``synthesis_unavailable``. The retry added for the synthesis and visual roles is
deliberately narrow, and these tests pin its edges as much as its purpose:
transient statuses are retried a bounded number of times; a request the
endpoint rejects, and a response that arrives but is malformed, are not.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.services.agent.grounding import DraftAnswer
from app.services.agent.providers import nvidia as nvidia_mod
from app.services.agent.providers.nvidia import (
    NvidiaAnswerSynthesizer,
    NvidiaVisualAnalyst,
)
from app.services.agent.schemas import AgentEvidence
from app.services.agent.visual import VisualAnswer

from tests.test_agent_providers import PNG, completion, settings

DRAFT = json.dumps(
    {"summary": "The mean NDWI was 0.278 index.", "evidence_refs": ["ndwi.ndwi_mean"]}
)


class StatusSequence(httpx.AsyncBaseTransport):
    """Replays ``(status, body)`` pairs in order; the last pair repeats."""

    def __init__(self, *responses: tuple[int, object]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        status, body = self._responses[min(len(self.requests), len(self._responses) - 1)]
        self.requests.append(request)
        return httpx.Response(status, json=body, request=request)


BUSY = (503, {"error": {"message": "Worker local total request limit reached"}})


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "_TRANSIENT_BACKOFF_SECONDS", 0.0)


def synthesize(*responses: tuple[int, object]) -> tuple[object, StatusSequence]:
    transport = StatusSequence(*responses)

    async def go() -> object:
        async with httpx.AsyncClient(transport=transport) as client:
            synth = NvidiaAnswerSynthesizer(settings=settings(), client=client)
            try:
                return await synth.synthesize("What is the NDWI?", AgentEvidence())
            except Exception as exc:  # returned so the test can assert on it
                return exc

    return asyncio.run(go()), transport


def observe(*responses: tuple[int, object]) -> tuple[object, StatusSequence]:
    transport = StatusSequence(*responses)

    async def go() -> object:
        async with httpx.AsyncClient(transport=transport) as client:
            analyst = NvidiaVisualAnalyst(settings=settings(), client=client)
            try:
                return await analyst.observe(
                    question="What is visible?", image=PNG, media_type="image/png"
                )
            except Exception as exc:
                return exc

    return asyncio.run(go()), transport


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


def test_synthesis_recovers_from_one_busy_response() -> None:
    result, transport = synthesize(BUSY, (200, completion(DRAFT)))
    assert isinstance(result, DraftAnswer)
    assert len(transport.requests) == 2


def test_synthesis_gives_up_after_the_bounded_attempts() -> None:
    result, transport = synthesize(BUSY)
    assert isinstance(result, UpstreamServiceError)
    assert len(transport.requests) == nvidia_mod._TRANSIENT_ATTEMPTS


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_synthesis_does_not_repeat_a_rejected_request(status: int) -> None:
    result, transport = synthesize((status, {"error": "no"}))
    assert isinstance(result, UpstreamServiceError)
    assert len(transport.requests) == 1


def test_a_malformed_answer_is_reported_not_resampled() -> None:
    result, transport = synthesize((200, completion("not an answer at all")))
    assert isinstance(result, IntentParsingError)
    assert len(transport.requests) == 1


def test_every_synthesis_attempt_sends_the_same_request() -> None:
    _, transport = synthesize(BUSY, BUSY, (200, completion(DRAFT)))
    bodies = {request.content for request in transport.requests}
    assert len(bodies) == 1


# --------------------------------------------------------------------------- #
# Visual observation
# --------------------------------------------------------------------------- #


def test_observation_recovers_from_one_busy_response() -> None:
    result, transport = observe(BUSY, (200, completion("Sandy beach beside the sea.")))
    assert isinstance(result, VisualAnswer)
    assert result.answer == "Sandy beach beside the sea."
    assert len(transport.requests) == 2


def test_observation_does_not_repeat_an_unauthorised_request() -> None:
    result, transport = observe((401, {"error": "bad key"}))
    assert isinstance(result, UpstreamServiceError)
    assert len(transport.requests) == 1
