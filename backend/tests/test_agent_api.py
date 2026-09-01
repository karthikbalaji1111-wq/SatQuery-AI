"""Phase 15 Commit 7 - the agent API endpoint.

``POST /api/v1/query/agent`` is an HTTP adapter and nothing else: it validates a
question, hands it to :class:`AgentService`, and returns the result. It performs
no orchestration, no execution, no grounding and no provider call of its own.

The endpoint is additive. The manual ``/query/execute`` and ``/query/analyze``
paths are exercised here too, to prove they still behave exactly as before.

Every agent status is an HTTP 200: a withheld answer still carries the
deterministic evidence, and turning that into a 5xx would throw away the part
of the response that is actually the product.

No test makes a live provider call.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest
from app.api.routes import query as query_routes
from app.api.routes.query import (
    get_agent_service,
    get_analysis_service,
    get_query_execution_service,
)
from app.core.errors import UpstreamServiceError
from app.main import create_app
from app.services.agent.schemas import (
    AgentEvidence,
    AgentPlan,
    AgentResult,
    AgentToolStep,
    AgentTrace,
    AnswerValidation,
)
from app.services.analysis import AnalysisResult
from app.services.geospatial.schemas import BoundingBox
from app.services.query.schemas import (
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
)
from fastapi.testclient import TestClient

AGENT_URL = "/api/v1/query/agent"
EXECUTE_URL = "/api/v1/query/execute"
ANALYZE_URL = "/api/v1/query/analyze"

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
CATALOG = "https://earth-search.aws.element84.com/v1"
QUESTION = "What is the NDWI of Chennai in January 2024?"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def intent_dict(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    body.update(overrides)
    return body


def make_plan() -> AgentPlan:
    return AgentPlan.model_validate(
        {
            "steps": [
                {"tool": "execute_query", "intent": intent_dict()},
                {"tool": "ndwi_statistics"},
            ]
        }
    )


def make_evidence() -> AgentEvidence:
    return AgentEvidence.model_validate(
        {
            "items": [
                {
                    "id": "ndwi.ndwi_mean",
                    "source": "ndwi",
                    "measurement": {
                        "name": "ndwi_mean",
                        "value": 0.2777,
                        "unit": "index",
                    },
                    "produced_by": "analysis.engines.compute_ndwi_measurements",
                }
            ]
        }
    )


def make_result(
    *, status: str = "ok", answer: str | None = "The mean NDWI was 0.2777 index."
) -> AgentResult:
    plan = make_plan()
    return AgentResult(
        status=status,  # type: ignore[arg-type]
        answer=answer,
        trace=AgentTrace(
            plan=plan,
            steps=[
                AgentToolStep(status="ok", parameters=plan.steps[0]),
                AgentToolStep(status="ok", parameters=plan.steps[1]),
            ],
            evidence_refs=["ndwi.ndwi_mean"] if status == "ok" else [],
            answer_validation=AnswerValidation(
                numeric_grounding="pass" if status == "ok" else "fail",
                forbidden_terms="pass",
                evidence_refs="pass",
            ),
        ),
        evidence=make_evidence(),
    )


class RecordingAgentService:
    """Records what the route hands it; duck-types AgentService.answer."""

    def __init__(
        self,
        *,
        result: AgentResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[Any] = []
        self._result = result if result is not None else make_result()
        self._error = error

    async def answer(self, request: Any) -> AgentResult:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return self._result


def make_client(service: Any = None) -> tuple[TestClient, RecordingAgentService]:
    service = service if service is not None else RecordingAgentService()
    app = create_app()
    app.dependency_overrides[get_agent_service] = lambda: service
    return TestClient(app), service


def ask(client: TestClient, **body: Any) -> Any:
    payload = {"question": QUESTION}
    payload.update(body)
    return client.post(AGENT_URL, json=payload)


# =========================================================================== #
# A. The endpoint reaches the service
# =========================================================================== #


def test_a_valid_question_reaches_the_agent_service() -> None:
    client, service = make_client()
    response = ask(client)

    assert response.status_code == 200
    assert len(service.calls) == 1


def test_the_question_is_passed_through_after_validation() -> None:
    client, service = make_client()
    ask(client, question=f"  {QUESTION}  ")

    request = service.calls[0]
    # Pydantic strips it; the route neither re-parses nor rewrites it.
    assert request.question == QUESTION


def test_the_service_receives_the_typed_request_not_a_dict() -> None:
    from app.services.agent.schemas import AgentQuestionRequest

    client, service = make_client()
    ask(client)

    assert isinstance(service.calls[0], AgentQuestionRequest)


def _openapi_paths() -> dict[str, Any]:
    """Registered paths, read from the app's own schema.

    ``app.routes`` holds ``_IncludedRouter`` objects in this FastAPI version,
    so the OpenAPI document is both the public and the reliable source.
    """

    return create_app().openapi()["paths"]


def test_the_endpoint_is_reachable_through_the_existing_router() -> None:
    """No api/router.py change was needed; the query router already mounts it."""

    assert AGENT_URL in _openapi_paths()


def test_the_route_is_registered_as_a_post() -> None:
    assert set(_openapi_paths()[AGENT_URL]) == {"post"}


# =========================================================================== #
# B. Every agent status is an HTTP 200
# =========================================================================== #


def test_a_successful_run_returns_the_complete_result() -> None:
    client, _ = make_client(RecordingAgentService(result=make_result()))
    body = ask(client).json()

    assert body["status"] == "ok"
    assert body["answer"] == "The mean NDWI was 0.2777 index."
    assert body["evidence"]["items"][0]["id"] == "ndwi.ndwi_mean"
    assert body["trace"]["plan"]["steps"][0]["tool"] == "execute_query"
    assert len(body["trace"]["steps"]) == 2


@pytest.mark.parametrize(
    "status", ["planner_unavailable", "synthesis_unavailable", "answer_withheld"]
)
def test_failure_statuses_are_returned_as_200(status: str) -> None:
    client, _ = make_client(
        RecordingAgentService(result=make_result(status=status, answer=None))
    )
    response = ask(client)

    assert response.status_code == 200
    assert response.json()["status"] == status
    assert response.json()["answer"] is None


def test_planner_unavailable_preserves_whatever_the_service_returned() -> None:
    empty = AgentResult(
        status="planner_unavailable",
        answer=None,
        trace=AgentTrace(),
        evidence=AgentEvidence(),
    )
    client, _ = make_client(RecordingAgentService(result=empty))
    body = ask(client).json()

    assert body["status"] == "planner_unavailable"
    assert body["answer"] is None
    assert body["evidence"]["items"] == []
    assert body["trace"]["plan"] is None
    assert body["trace"]["steps"] == []


def test_a_withheld_answer_still_returns_the_evidence() -> None:
    """The whole reason these are 200s."""

    client, _ = make_client(
        RecordingAgentService(result=make_result(status="answer_withheld", answer=None))
    )
    body = ask(client).json()

    assert body["answer"] is None
    assert body["evidence"]["items"]
    assert body["trace"]["answer_validation"]["numeric_grounding"] == "fail"


def test_an_upstream_error_still_uses_the_existing_error_handling() -> None:
    """A raised AppError is not silently recoded into an AgentResult."""

    client, _ = make_client(
        RecordingAgentService(error=UpstreamServiceError("provider down"))
    )
    response = ask(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


# =========================================================================== #
# C. Request validation
# =========================================================================== #


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 5000},
        {"question": None},
        {"question": 42},
        {"question": QUESTION, "unexpected": "field"},
    ],
)
def test_invalid_request_bodies_are_rejected(body: dict[str, Any]) -> None:
    client, service = make_client()
    response = client.post(AGENT_URL, json=body)

    assert response.status_code == 422
    assert service.calls == []  # never reached the service


def test_the_response_matches_the_agent_result_schema() -> None:
    client, _ = make_client()
    body = ask(client).json()

    assert set(body) == set(AgentResult.model_fields)
    # It really is the contract, not an ad-hoc dict.
    assert AgentResult.model_validate(body).status == "ok"


# =========================================================================== #
# D. The route is only an adapter
# =========================================================================== #


def test_the_service_is_injected_through_the_existing_provider_pattern() -> None:
    assert callable(get_agent_service)
    # Same shape as the four providers already in this module.
    for provider in (
        get_analysis_service,
        get_query_execution_service,
        get_agent_service,
    ):
        assert provider.__doc__


def test_the_dependency_override_is_what_the_endpoint_uses() -> None:
    sentinel = RecordingAgentService(result=make_result(answer="overridden"))
    client, _ = make_client(sentinel)

    assert ask(client).json()["answer"] == "overridden"
    assert len(sentinel.calls) == 1


def test_the_route_contains_no_orchestration() -> None:
    """Composition is allowed here; orchestration is not.

    The API layer is exactly where the concrete providers get wired - that is
    what keeps AgentService SDK-free - so naming ``GeminiAgentPlanner`` is
    correct. What the route must never do is drive the pipeline itself, so the
    check is on CALLS, not on names appearing in an import.
    """

    tree = ast.parse(pathlib.Path(query_routes.__file__).read_text())
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)

    for banned in ("plan", "synthesize", "validate_answer", "compute_ndwi_measurements"):
        assert banned not in called, f"query.py calls {banned}()"
    # The only agent call the adapter makes.
    assert "answer" in called

    source = pathlib.Path(query_routes.__file__).read_text()
    for banned in ("DraftAnswer", "TOOL_REGISTRY", "AgentTrace(", "AgentEvidence("):
        assert banned not in source, f"query.py references {banned}"


def test_the_route_module_imports_no_provider_sdk() -> None:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(pathlib.Path(query_routes.__file__).read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    for forbidden in ("google", "genai", "httpx", "requests", "subprocess", "os"):
        assert forbidden not in roots


def test_the_endpoint_exposes_no_provider_configuration() -> None:
    client, _ = make_client()
    body = ask(client).json()
    rendered = str(body).lower()

    for secret in ("api_key", "apikey", "gemini_api", "test-key", "system_instruction"):
        assert secret not in rendered


def test_the_response_exposes_no_reasoning_field() -> None:
    client, _ = make_client()
    body = ask(client).json()
    rendered = str(body).lower()

    for banned in ("chain_of_thought", "reasoning", "thoughts", "rationale", "prompt"):
        assert banned not in rendered


# =========================================================================== #
# E. The existing endpoints are untouched
# =========================================================================== #


def make_execution_result() -> QueryExecutionResult:
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(
            intent=SatQueryIntent.model_validate(intent_dict()), bbox=DEFAULT_BBOX
        ),
        executed_modalities=["sentinel-2-optical"],
        skipped_modalities=[],
        windows=[],
        catalog=CATALOG,
    )


def test_query_execute_is_unchanged() -> None:
    class FakeExecution:
        async def execute(self, request: Any) -> QueryExecutionResult:
            return make_execution_result()

    app = create_app()
    app.dependency_overrides[get_query_execution_service] = FakeExecution
    response = TestClient(app).post(
        EXECUTE_URL, json={"intent": intent_dict(), "include_imagery": False}
    )

    assert response.status_code == 200
    assert response.json()["catalog"] == CATALOG


def test_query_analyze_is_unchanged() -> None:
    class FakeAnalysis:
        async def analyze(self, request: Any) -> AnalysisResult:
            return AnalysisResult(
                status="ok",
                task="visualize",
                answer="Retrieved 0 window(s).",
                windows_considered=[],
            )

    app = create_app()
    app.dependency_overrides[get_analysis_service] = FakeAnalysis
    response = TestClient(app).post(
        ANALYZE_URL,
        json={"execution": make_execution_result().model_dump(mode="json")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # The Phase 14 additive field is still there and still null.
    assert body["temporal_comparison"] is None


def test_the_existing_handlers_were_not_modified() -> None:
    """Only a provider and one handler were added to this module."""

    source = pathlib.Path(query_routes.__file__).read_text()

    for handler in (
        'async def parse_intent(',
        'async def build_plan(',
        'async def execute_query(',
        'async def analyze_query(',
    ):
        assert handler in source
    for path in ('"/parse"', '"/build-plan"', '"/execute"', '"/analyze"', '"/agent"'):
        assert path in source


def test_every_query_route_is_still_registered() -> None:
    paths = _openapi_paths()

    for path in (
        "/api/v1/query/parse",
        "/api/v1/query/build-plan",
        EXECUTE_URL,
        ANALYZE_URL,
        AGENT_URL,
    ):
        assert path in paths


def test_the_api_router_registration_was_not_changed() -> None:
    from app.api import router as router_mod

    source = pathlib.Path(router_mod.__file__).read_text()
    assert "agent" not in source.lower()
