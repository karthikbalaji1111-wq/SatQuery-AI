"""A location-service outage is reported as one, not as missing evidence.

Before this pass a geocoder outage reached the user as "Insufficient evidence
to answer the question.": ``AgentService`` special-cased only ``not_found``
and ``aoi_too_large``, so every other discovery failure went on to synthesis,
which - correctly, given empty evidence - abstained, and the run came back
``ok`` with the abstention as its answer. No scene was ever looked at, so
"insufficient evidence" described something that never happened.

Four outcomes must stay distinct:

A. clarification      - the question is missing something; ask for it.
B. location outage    - the question is fine; the geocoder is not.
                        ``location_unavailable``, a ``failure`` naming the
                        dependency, the wait when known, NO answer and NO
                        clarification.
C. evidence verdict   - the place resolved and the pipeline ran; the existing
                        abstention wording stays.
D. refusals           - area too large, unsupported: unchanged.

The geocoder here is REAL (``GeospatialService`` -> ``nominatim.geocode``)
behind a scripted transport, so these tests cross the actual error path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from app.api.routes.query import get_agent_service
from app.core.errors import UpstreamServiceError
from app.main import create_app
from app.services.agent.schemas import AgentFailure, AgentQuestionRequest, AgentResult
from app.services.agent.standard import ABSTENTION
from app.services.geospatial import GeospatialService, ResolveRequest, nominatim
from app.services.geospatial.nominatim import reset_geocoder_state
from app.services.query.schemas import QueryExecutionRequest, QueryExecutionResult
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.test_standard_workflow import (
    RecordingQueryExecution,
    execution_for,
    standard_service,
)

QUESTION = "Show vegetation around Cubbon Park, Bengaluru in December 2024"
CUBBON = [
    {
        "lat": "12.9763",
        "lon": "77.5929",
        "boundingbox": ["12.9700", "12.9830", "77.5870", "77.5990"],
        "display_name": "Cubbon Park, Bengaluru, Karnataka, India",
    }
]


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def now(self) -> float:
        return self.t

    async def pause(self, seconds: float) -> None:
        self.t += seconds
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def _clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    reset_geocoder_state()
    clock = FakeClock()
    monkeypatch.setattr(nominatim, "_now", clock.now)
    monkeypatch.setattr(nominatim, "_pause", clock.pause)
    return clock


class Upstream:
    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]

        return httpx.MockTransport(handle)


class GeocodingExecution:
    """Query execution that geocodes through the REAL geospatial service.

    On success it returns the usual one-scene-per-window result, exactly as
    ``RecordingQueryExecution`` does; the geocoder is the only thing added.
    """

    def __init__(self, upstream: Upstream, *, catalog_error: Exception | None = None):
        from app.core.config import Settings

        self.geocoder = GeospatialService(
            settings=Settings(
                geocoder_min_interval_seconds=0.0,
                geocoder_backoff_base_seconds=2.0,
            ),
            transport=upstream.transport(),
        )
        self.calls: list[QueryExecutionRequest] = []
        self.searched = 0
        self._catalog_error = catalog_error

    async def execute(self, request: QueryExecutionRequest, **_: Any) -> QueryExecutionResult:
        self.calls.append(request)
        await self.geocoder.resolve(ResolveRequest(place=request.intent.location_query))
        self.searched += 1  # only ever reached with a location
        if self._catalog_error is not None:
            raise self._catalog_error
        return execution_for(request)


def ask(execution: Any, question: str = QUESTION) -> tuple[AgentResult, Any]:
    service, _, analysis = standard_service(query=execution)
    result = asyncio.run(service.answer(AgentQuestionRequest(question=question)))
    return result, analysis


def throttled(retry_after: str | None = None) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(429, text="Too Many Requests from your IP", headers=headers)


def assert_location_outage(result: AgentResult) -> AgentFailure:
    assert result.status == "location_unavailable"
    assert result.answer is None
    assert result.clarification is None
    failure = result.failure
    assert failure is not None
    assert failure.stage == "location"
    assert failure.code == "geocoding_unavailable"
    assert failure.dependency == "geocoder"
    assert failure.message.startswith("Location service temporarily unavailable.")
    # Diagnostics survive: the failed step and the evidence note.
    [step] = [s for s in result.trace.steps if s.status == "failed"]
    assert step.error_message
    notes = [i for i in result.evidence.items if i.id == "execution.discovery_failure"]
    assert len(notes) == 1
    # Nothing was measured and nothing claims to have been.
    assert not any(i.measurement for i in result.evidence.items)
    return failure


# =========================================================================== #
# B. The outage, by cause
# =========================================================================== #


def test_http_429_is_a_location_outage_with_the_upstreams_wait() -> None:
    upstream = Upstream(throttled("45"))
    execution = GeocodingExecution(upstream)
    result, analysis = ask(execution)

    failure = assert_location_outage(result)
    assert failure.retry_after_seconds == pytest.approx(45.0)
    assert execution.searched == 0 and analysis.calls == []
    assert len(upstream.requests) == 1


def test_a_cooldown_refusal_is_a_location_outage_without_a_request() -> None:
    upstream = Upstream(throttled("60"))
    execution = GeocodingExecution(upstream)
    first, _ = ask(execution)
    second, _ = ask(execution, "Show water around Dal Lake in January 2025")

    assert_location_outage(first)
    failure = assert_location_outage(second)
    # The second was refused from the cooldown: no request, the remaining wait.
    assert len(upstream.requests) == 1
    assert failure.retry_after_seconds is not None
    assert 0 < failure.retry_after_seconds <= 60


def test_a_geocoder_5xx_is_a_location_outage_with_no_invented_wait() -> None:
    upstream = Upstream(httpx.Response(502, text="Bad Gateway from upstream"))
    result, _ = ask(GeocodingExecution(upstream))

    failure = assert_location_outage(result)
    assert failure.retry_after_seconds is None
    assert len(upstream.requests) == 3  # the bounded retries, then the outage


def test_the_outage_is_not_described_as_missing_evidence() -> None:
    result, _ = ask(GeocodingExecution(Upstream(throttled("30"))))

    assert result.answer is None
    assert result.status != "ok"
    assert "Insufficient evidence" not in result.model_dump_json()


def test_an_outage_is_never_a_clarification() -> None:
    for response in (throttled("30"), throttled(), httpx.Response(503)):
        reset_geocoder_state()
        result, _ = ask(GeocodingExecution(Upstream(response)))
        assert result.status != "needs_clarification"
        assert result.clarification is None


def test_the_users_sentence_never_carries_upstream_text() -> None:
    result, _ = ask(GeocodingExecution(Upstream(throttled("30"))))

    assert result.failure is not None
    assert "Too Many Requests" not in result.failure.message
    assert "IP" not in result.failure.message


# =========================================================================== #
# C, D and success: unchanged
# =========================================================================== #


def test_genuine_insufficient_evidence_keeps_its_wording() -> None:
    """The place resolved; the catalog had nothing - that IS an evidence verdict."""

    result, _ = ask(RecordingQueryExecution(selected=False))

    assert result.status == "ok"
    assert result.answer == ABSTENTION
    assert result.failure is None


def test_a_catalog_outage_is_not_a_location_outage() -> None:
    """Only the geocoder earns the new status; other executor failures surface
    through the trace and the evidence, exactly as before."""

    execution = GeocodingExecution(
        Upstream(httpx.Response(200, json=CUBBON)),
        catalog_error=UpstreamServiceError("The satellite catalog is unavailable."),
    )
    result, _ = ask(execution)

    assert result.status == "ok"
    assert result.answer == ABSTENTION
    assert result.failure is None


def test_a_place_the_geocoder_cannot_find_is_still_a_clarification() -> None:
    result, _ = ask(GeocodingExecution(Upstream(httpx.Response(200, json=[]))))

    assert result.status == "needs_clarification"
    assert result.clarification is not None
    assert result.clarification.reason == "location_not_found"
    assert result.failure is None


def test_a_successful_geocode_runs_the_existing_workflow_unchanged() -> None:
    upstream = Upstream(httpx.Response(200, json=CUBBON))
    execution = GeocodingExecution(upstream)
    result, analysis = ask(execution)

    assert result.status == "ok"
    assert result.failure is None and result.clarification is None
    assert result.answer is not None and "The mean NDVI was" in result.answer
    assert execution.searched == 1 and len(analysis.calls) == 1

    # Identical to the same run without a real geocoder in front of it.
    baseline, _ = ask(RecordingQueryExecution())
    assert result.answer == baseline.answer
    assert [i.id for i in result.evidence.items] == [i.id for i in baseline.evidence.items]


# =========================================================================== #
# The contract
# =========================================================================== #


def test_location_unavailable_requires_a_location_failure_and_no_answer() -> None:
    trace = {"plan": None, "steps": [], "evidence_refs": [], "answer_validation": None}
    evidence = {"items": [], "execution": None, "analysis": None}
    failure = {
        "stage": "location", "code": "geocoding_unavailable", "dependency": "geocoder",
        "message": "Location service temporarily unavailable.",
    }

    AgentResult.model_validate(
        {"status": "location_unavailable", "failure": failure, "trace": trace,
         "evidence": evidence}
    )
    for broken in (
        {"status": "location_unavailable", "trace": trace, "evidence": evidence},
        {"status": "location_unavailable", "failure": failure, "answer": "x",
         "trace": trace, "evidence": evidence},
        {"status": "location_unavailable", "failure": {**failure, "stage": "synthesis"},
         "trace": trace, "evidence": evidence},
        {"status": "ok", "answer": "x", "failure": failure, "trace": trace,
         "evidence": evidence},
    ):
        with pytest.raises(ValidationError):
            AgentResult.model_validate(broken)


# =========================================================================== #
# End to end, over HTTP
# =========================================================================== #


def test_the_agent_route_returns_the_structured_location_state() -> None:
    upstream = Upstream(throttled("45"))
    service, _, _ = standard_service(query=GeocodingExecution(upstream))
    app = create_app()
    app.dependency_overrides[get_agent_service] = lambda: service

    response = TestClient(app).post("/api/v1/query/agent", json={"question": QUESTION})

    assert response.status_code == 200  # every agent outcome is a 200
    body = response.json()
    assert body["status"] == "location_unavailable"
    assert body["answer"] is None and body["clarification"] is None
    assert body["failure"] == {
        "stage": "location",
        "code": "geocoding_unavailable",
        "dependency": "geocoder",
        "message": (
            "Location service temporarily unavailable. The place could not be "
            "looked up, so no scene was searched and nothing was measured."
        ),
        "retry_after_seconds": pytest.approx(45.0),
    }
    assert "Too Many Requests" not in response.text
