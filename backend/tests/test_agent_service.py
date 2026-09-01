"""Phase 15 Commit 6 - AgentService orchestration tests.

The service coordinates four components and owns none of their logic. It does
not parse language, choose tools, execute anything, compute anything, or ground
anything - it calls the planner, the executor, the synthesizer and the Commit 3
validator in order, preserves their contracts, and translates their outcomes
into the existing ``AgentResult`` statuses.

The property under most scrutiny here is honest failure: at every stage where
something can go wrong, the deterministic evidence that was already collected
must survive, and no answer may be fabricated to fill the gap.

All collaborators are injected fakes. No network, no provider, no filesystem.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import pathlib
from typing import Any

import pytest
from app.core.errors import (
    IntentParsingError,
    InvalidInputError,
    UpstreamServiceError,
)
from app.services.agent import service as service_mod
from app.services.agent.executor import ExecutionOutcome
from app.services.agent.grounding import DraftAnswer
from app.services.agent.planner import AgentPlanner, MockAgentPlanner
from app.services.agent.schemas import (
    AgentEvidence,
    AgentPlan,
    AgentQuestionRequest,
    AgentResult,
    AgentToolStep,
    AgentTrace,
    AnswerValidation,
)
from app.services.agent.service import AgentService
from app.services.agent.synthesizer import AnswerSynthesizer, MockAnswerSynthesizer

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


def make_evidence(**overrides: Any) -> AgentEvidence:
    payload: dict[str, Any] = {
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
    payload.update(overrides)
    return AgentEvidence.model_validate(payload)


def make_outcome(evidence: AgentEvidence | None = None) -> ExecutionOutcome:
    plan = make_plan()
    return ExecutionOutcome(
        steps=[
            AgentToolStep(status="ok", parameters=plan.steps[0]),
            AgentToolStep(status="ok", parameters=plan.steps[1]),
        ],
        evidence=evidence if evidence is not None else make_evidence(),
    )


GROUNDED_ANSWER = DraftAnswer(
    summary="The mean NDWI was 0.2777 index.", evidence_refs=["ndwi.ndwi_mean"]
)


class RecordingPlanner(AgentPlanner):
    def __init__(
        self, *, plan: AgentPlan | None = None, error: Exception | None = None
    ) -> None:
        self.calls: list[str] = []
        self._plan = plan if plan is not None else make_plan()
        self._error = error

    async def plan(self, question: str) -> AgentPlan:
        self.calls.append(question)
        if self._error is not None:
            raise self._error
        return self._plan


class RecordingExecutor:
    """Duck-types AgentExecutor.execute; records the plan it was handed."""

    def __init__(
        self,
        *,
        outcome: ExecutionOutcome | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[AgentPlan] = []
        self._outcome = outcome if outcome is not None else make_outcome()
        self._error = error

    async def execute(self, plan: AgentPlan) -> ExecutionOutcome:
        self.calls.append(plan)
        if self._error is not None:
            raise self._error
        return self._outcome


class RecordingSynthesizer(AnswerSynthesizer):
    def __init__(
        self,
        *,
        answer: DraftAnswer | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, AgentEvidence]] = []
        self._answer = answer if answer is not None else GROUNDED_ANSWER
        self._error = error

    async def synthesize(
        self, question: str, evidence: AgentEvidence
    ) -> DraftAnswer:
        self.calls.append((question, evidence))
        if self._error is not None:
            raise self._error
        return self._answer


def build(
    *,
    planner: Any = None,
    executor: Any = None,
    synthesizer: Any = None,
) -> tuple[AgentService, Any, Any, Any]:
    planner = planner if planner is not None else RecordingPlanner()
    executor = executor if executor is not None else RecordingExecutor()
    synthesizer = synthesizer if synthesizer is not None else RecordingSynthesizer()
    service = AgentService(
        planner=planner,
        executor=executor,  # type: ignore[arg-type]
        synthesizer=synthesizer,
    )
    return service, planner, executor, synthesizer


def ask(service: AgentService, question: str = QUESTION) -> AgentResult:
    return asyncio.run(service.answer(AgentQuestionRequest(question=question)))


# =========================================================================== #
# A. The happy path
# =========================================================================== #


def test_a_valid_question_flows_through_every_stage() -> None:
    service, planner, executor, synthesizer = build()
    result = ask(service)

    assert planner.calls == [QUESTION]
    assert len(executor.calls) == 1
    assert len(synthesizer.calls) == 1
    assert isinstance(result, AgentResult)
    assert result.status == "ok"
    assert result.answer == GROUNDED_ANSWER.summary


def test_the_planner_receives_only_the_question() -> None:
    service, planner, _, _ = build()
    ask(service)

    assert planner.calls == [QUESTION]
    assert list(inspect.signature(AgentPlanner.plan).parameters) == [
        "self",
        "question",
    ]


def test_the_executor_receives_the_validated_plan_object() -> None:
    plan = make_plan()
    service, _, executor, _ = build(planner=RecordingPlanner(plan=plan))
    ask(service)

    assert executor.calls[0] is plan
    assert isinstance(executor.calls[0], AgentPlan)


def test_the_synthesizer_receives_the_question_and_the_evidence() -> None:
    evidence = make_evidence()
    service, _, _, synthesizer = build(
        executor=RecordingExecutor(outcome=make_outcome(evidence))
    )
    ask(service)

    question, handed = synthesizer.calls[0]
    assert question == QUESTION
    assert handed is evidence
    assert isinstance(handed, AgentEvidence)
    # Not the executor's own return type.
    assert not isinstance(handed, ExecutionOutcome)


def test_the_result_carries_the_executed_steps_and_the_plan() -> None:
    plan = make_plan()
    result = ask(build(planner=RecordingPlanner(plan=plan))[0])

    assert result.trace.plan == plan
    assert [step.tool for step in result.trace.steps] == [
        "execute_query",
        "ndwi_statistics",
    ]


def test_the_result_carries_the_deterministic_evidence_verbatim() -> None:
    evidence = make_evidence()
    result = ask(build(executor=RecordingExecutor(outcome=make_outcome(evidence)))[0])

    assert result.evidence == evidence


def test_grounding_outcomes_are_recorded_on_the_trace() -> None:
    result = ask(build()[0])

    assert isinstance(result.trace.answer_validation, AnswerValidation)
    assert result.trace.answer_validation.numeric_grounding == "pass"
    assert result.trace.answer_validation.forbidden_terms == "pass"
    assert result.trace.answer_validation.evidence_refs == "pass"


def test_resolved_evidence_references_reach_the_trace() -> None:
    result = ask(build()[0])
    assert result.trace.evidence_refs == ["ndwi.ndwi_mean"]


def test_the_service_works_with_the_repository_mocks() -> None:
    """Commit 4 and Commit 5 mocks drive it with no provider at all."""

    service = AgentService(
        planner=MockAgentPlanner(plan=make_plan()),
        executor=RecordingExecutor(),  # type: ignore[arg-type]
        synthesizer=MockAnswerSynthesizer(),
    )
    result = ask(service)

    assert result.status == "ok"
    assert result.answer


# =========================================================================== #
# B. Planner failure
# =========================================================================== #


@pytest.mark.parametrize(
    "error",
    [
        UpstreamServiceError("The language-model service is unavailable."),
        IntentParsingError("Could not extract a valid analysis plan."),
    ],
)
def test_planner_failure_stops_everything(error: Exception) -> None:
    service, _, executor, synthesizer = build(
        planner=RecordingPlanner(error=error)
    )
    result = ask(service)

    assert result.status == "planner_unavailable"
    assert executor.calls == []  # nothing executed
    assert synthesizer.calls == []  # nothing synthesised


def test_planner_failure_fabricates_neither_answer_nor_evidence() -> None:
    # A handled provider failure. The providers already normalise transport
    # errors into UpstreamServiceError; a bare TimeoutError would be a bug and
    # must propagate instead - see test_a_non_app_error_is_not_swallowed.
    result = ask(
        build(planner=RecordingPlanner(error=UpstreamServiceError("timed out")))[0]
    )

    assert result.answer is None
    assert result.evidence.items == []
    assert result.evidence.execution is None
    assert result.evidence.analysis is None


def test_planner_failure_records_no_plan_and_no_steps() -> None:
    result = ask(
        build(planner=RecordingPlanner(error=UpstreamServiceError("down")))[0]
    )

    assert result.trace.plan is None
    assert result.trace.steps == []
    assert result.trace.evidence_refs == []
    # Nothing was checked, so nothing may read as checked.
    assert result.trace.answer_validation is None


# =========================================================================== #
# C. Synthesis failure - evidence must survive
# =========================================================================== #


@pytest.mark.parametrize(
    "error",
    [
        UpstreamServiceError("The language-model service timed out."),
        IntentParsingError("Could not extract a valid answer."),
    ],
)
def test_synthesis_failure_preserves_the_evidence(error: Exception) -> None:
    evidence = make_evidence()
    service, _, _, _ = build(
        executor=RecordingExecutor(outcome=make_outcome(evidence)),
        synthesizer=RecordingSynthesizer(error=error),
    )
    result = ask(service)

    assert result.status == "synthesis_unavailable"
    assert result.answer is None
    assert result.evidence == evidence  # the deterministic product survives
    assert result.evidence.items


def test_synthesis_failure_still_reports_what_executed() -> None:
    result = ask(
        build(synthesizer=RecordingSynthesizer(error=UpstreamServiceError("x")))[0]
    )

    assert result.trace.plan is not None
    assert len(result.trace.steps) == 2
    assert result.trace.answer_validation is None  # nothing was validated


# =========================================================================== #
# D. Grounding failure - the answer is withheld, never returned as ok
# =========================================================================== #


def test_an_ungrounded_number_withholds_the_answer() -> None:
    fabricated = DraftAnswer(
        summary="The mean NDWI was 0.99 index.", evidence_refs=["ndwi.ndwi_mean"]
    )
    service, _, _, _ = build(synthesizer=RecordingSynthesizer(answer=fabricated))
    result = ask(service)

    assert result.status == "answer_withheld"
    assert result.answer is None
    assert result.trace.answer_validation is not None
    assert result.trace.answer_validation.numeric_grounding == "fail"


def test_a_forbidden_phrase_withholds_the_answer() -> None:
    offending = DraftAnswer(
        summary="Change detection shows a difference.", evidence_refs=[]
    )
    result = ask(build(synthesizer=RecordingSynthesizer(answer=offending))[0])

    assert result.status == "answer_withheld"
    assert result.trace.answer_validation.forbidden_terms == "fail"


def test_a_dangling_evidence_reference_withholds_the_answer() -> None:
    dangling = DraftAnswer(summary="An answer.", evidence_refs=["ndwi.nope"])
    result = ask(build(synthesizer=RecordingSynthesizer(answer=dangling))[0])

    assert result.status == "answer_withheld"
    assert result.trace.answer_validation.evidence_refs == "fail"
    # The trace must not repeat a reference the evidence cannot resolve.
    assert result.trace.evidence_refs == []


def test_grounding_failure_preserves_the_evidence() -> None:
    evidence = make_evidence()
    fabricated = DraftAnswer(summary="NDWI was 0.99.", evidence_refs=[])
    service, _, _, _ = build(
        executor=RecordingExecutor(outcome=make_outcome(evidence)),
        synthesizer=RecordingSynthesizer(answer=fabricated),
    )
    result = ask(service)

    assert result.evidence == evidence
    assert result.trace.plan is not None
    assert len(result.trace.steps) == 2


def test_a_fabricated_answer_cannot_bypass_grounding() -> None:
    """Whatever a synthesizer returns, the validator sees it."""

    for summary in (
        "The mean NDWI was 12345 index.",
        "Per-pixel change detection found water.",
        "The area is 42 square kilometres.",
    ):
        result = ask(
            build(
                synthesizer=RecordingSynthesizer(
                    answer=DraftAnswer(summary=summary, evidence_refs=[])
                )
            )[0]
        )
        assert result.status == "answer_withheld", summary
        assert result.answer is None


def test_grounding_uses_the_commit_3_validator_not_a_reimplementation() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()

    assert "validate_answer" in source
    # No second implementation of any grounding concern.
    for banned in ("FORBIDDEN_PHRASES", "re.compile", "toFixed", "_is_grounded"):
        assert banned not in source


# =========================================================================== #
# E. The service owns no domain logic
# =========================================================================== #


def test_the_service_validates_no_domain_rules_of_its_own() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()

    for banned in (
        "max_cloud_cover",
        "temporal_mode",
        "time_windows",
        "modalities",
        "sentinel-2-optical",
        "ndwi",
        "compatibility",
        "SatQueryIntent",
    ):
        assert banned not in source, f"service.py reasons about {banned}"


def test_the_service_selects_no_tools() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()

    for banned in (
        "execute_query",
        "ndwi_statistics",
        "temporal_ndwi_statistics",
        "TOOL_REGISTRY",
        "resolve_tool",
    ):
        assert banned not in source, f"service.py references the tool {banned}"


def test_the_service_uses_no_dynamic_dispatch() -> None:
    forbidden = {
        "eval",
        "exec",
        "setattr",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "open",
    }
    tree = ast.parse(pathlib.Path(service_mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr":
                assert node.args and isinstance(node.args[1], ast.Constant)
                continue
            assert node.func.id not in forbidden


def test_the_service_does_not_mutate_the_evidence() -> None:
    evidence = make_evidence()
    before = evidence.model_dump(mode="json")

    ask(build(executor=RecordingExecutor(outcome=make_outcome(evidence)))[0])

    assert evidence.model_dump(mode="json") == before


def test_only_the_four_existing_statuses_are_produced() -> None:
    permitted = {"ok", "planner_unavailable", "synthesis_unavailable", "answer_withheld"}
    seen = {
        ask(build()[0]).status,
        ask(build(planner=RecordingPlanner(error=UpstreamServiceError("x")))[0]).status,
        ask(build(synthesizer=RecordingSynthesizer(error=UpstreamServiceError("x")))[0]).status,
        ask(
            build(
                synthesizer=RecordingSynthesizer(
                    answer=DraftAnswer(summary="NDWI 0.99.", evidence_refs=[])
                )
            )[0]
        ).status,
    }
    assert seen <= permitted
    assert seen == permitted  # every branch is reachable


def test_a_non_app_error_is_not_swallowed() -> None:
    """A bug must surface, not be recoded as a handled status."""

    class BugError(RuntimeError):
        pass

    with pytest.raises(BugError):
        ask(build(planner=RecordingPlanner(error=BugError("bug")))[0])


# =========================================================================== #
# F. Dependency injection and provider independence
# =========================================================================== #


def test_all_three_collaborators_are_required() -> None:
    with pytest.raises(TypeError):
        AgentService()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        AgentService(planner=MockAgentPlanner())  # type: ignore[call-arg]


def test_the_service_constructs_no_infrastructure() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()

    for banned in (
        "GeminiAgentPlanner",
        "GeminiAnswerSynthesizer",
        "genai.Client",
        "QueryExecutionService(",
        "AnalysisService(",
        "ImageryService(",
        "httpx",
    ):
        assert banned not in source


def test_the_service_holds_only_the_three_collaborators() -> None:
    service, planner, executor, synthesizer = build()

    assert service._planner is planner
    assert service._executor is executor
    assert service._synthesizer is synthesizer
    for banned in ("_query", "_analysis", "_imagery", "_client", "_raster"):
        assert not hasattr(service, banned)


def test_the_service_depends_on_the_abstractions() -> None:
    signature = inspect.signature(AgentService.__init__)
    annotations = {
        name: str(param.annotation)
        for name, param in signature.parameters.items()
        if name != "self"
    }

    assert "AgentPlanner" in annotations["planner"]
    assert "AnswerSynthesizer" in annotations["synthesizer"]
    assert "Gemini" not in "".join(annotations.values())


def test_service_imports_no_provider_sdk_or_transport() -> None:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(pathlib.Path(service_mod.__file__).read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    for forbidden in (
        "google",
        "genai",
        "httpx",
        "requests",
        "subprocess",
        "os",
        "sys",
        "importlib",
        "socket",
        "rasterio",
        "numpy",
        "fastapi",
    ):
        assert forbidden not in roots, f"service.py imports {forbidden!r}"


def test_the_service_does_not_import_the_provider_module() -> None:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(pathlib.Path(service_mod.__file__).read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    assert not any("providers" in module for module in modules)


# =========================================================================== #
# G. Trace integrity - no reasoning, no unearned claims
# =========================================================================== #


def test_the_trace_exposes_no_reasoning_field() -> None:
    for banned in (
        "reasoning",
        "thoughts",
        "thinking",
        "chain_of_thought",
        "rationale",
        "scratchpad",
        "prompt",
        "raw_response",
    ):
        assert banned not in AgentTrace.model_fields
        assert banned not in AgentToolStep.model_fields
        assert banned not in AgentResult.model_fields


def test_the_service_stores_no_prompt_or_provider_internals() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()

    for banned in ("system_instruction", "response_schema", "candidate", "prompt"):
        assert banned not in source


def test_the_trace_reports_execution_from_the_executor_not_the_plan() -> None:
    """A step is reported because it ran, not because it was requested."""

    plan = make_plan()
    # The executor reports only ONE step despite a two-step plan.
    outcome = ExecutionOutcome(
        steps=[AgentToolStep(status="ok", parameters=plan.steps[0])],
        evidence=make_evidence(),
    )
    result = ask(
        build(
            planner=RecordingPlanner(plan=plan),
            executor=RecordingExecutor(outcome=outcome),
        )[0]
    )

    assert len(result.trace.plan.steps) == 2  # what was requested
    assert len(result.trace.steps) == 1  # what actually ran


def test_failed_steps_are_reported_without_claiming_success() -> None:
    plan = make_plan()
    outcome = ExecutionOutcome(
        steps=[
            AgentToolStep(
                status="failed",
                parameters=plan.steps[0],
                error_message="The satellite catalog is unavailable.",
            ),
            AgentToolStep(status="skipped", parameters=plan.steps[1]),
        ],
        evidence=AgentEvidence(),
    )
    result = ask(
        build(
            planner=RecordingPlanner(plan=plan),
            executor=RecordingExecutor(outcome=outcome),
            synthesizer=RecordingSynthesizer(
                answer=DraftAnswer(summary="No imagery was retrieved.", evidence_refs=[])
            ),
        )[0]
    )

    assert [step.status for step in result.trace.steps] == ["failed", "skipped"]
    assert result.trace.steps[0].error_message


# =========================================================================== #
# H. Import boundary
# =========================================================================== #


def test_no_lower_layer_imports_the_agent_package() -> None:
    """Nothing BELOW the agent may import it.

    ``app/api`` is deliberately excluded: the dependency direction is
    ``api -> agent -> {analysis, query} -> satellite``, so the API layer
    importing the agent is correct and is asserted positively below. This test
    previously scanned it too, which only passed because no route existed yet.
    """

    backend = pathlib.Path(__file__).resolve().parents[1]
    for package in ("analysis", "query", "satellite", "geospatial", "ai"):
        for path in (backend / "app" / "services" / package).rglob("*.py"):
            assert "services.agent" not in path.read_text(), (
                f"{path} imports the agent package - dependency direction violated"
            )
    for path in (backend / "app" / "core").rglob("*.py"):
        assert "services.agent" not in path.read_text()


def test_the_api_layer_is_the_one_that_composes_the_agent() -> None:
    """The direction is real: api imports agent, never the reverse."""

    backend = pathlib.Path(__file__).resolve().parents[1]
    routes = (backend / "app" / "api" / "routes" / "query.py").read_text()
    assert "services.agent" in routes

def test_no_api_route_was_added() -> None:
    """Commit 7 has not started."""

    source = pathlib.Path(service_mod.__file__).read_text()
    assert "APIRouter" not in source
    assert "@router" not in source


# =========================================================================== #
# Post-Phase-15 hardening - HIGH-2: executor failure has no status.
#
# None of the four statuses describes "execution raised": planning succeeded,
# synthesis never ran, and no answer was generated to withhold. Rather than
# mislabel it, the service lets it propagate to the existing error handlers.
#
# This is defensible because, since the evidence-id fix, the executor's only
# remaining raise is unreachable through a validated AgentPlan - the
# discriminated union guarantees a registered tool - so a raise here means a
# genuine fault, which should surface as one.
#
# These tests pin that deliberate choice. If a fifth status is ever added, they
# are the ones to revisit.
# =========================================================================== #


@pytest.mark.parametrize(
    "error",
    [
        InvalidInputError("Tool 'x' is not available."),
        UpstreamServiceError("catalog down"),
    ],
)
def test_an_executor_failure_is_not_mislabelled_as_another_status(
    error: Exception,
) -> None:
    service, _, _, synthesizer = build(executor=RecordingExecutor(error=error))

    with pytest.raises(type(error)):
        ask(service)

    # Nothing downstream ran, and no status was invented to cover it.
    assert synthesizer.calls == []


def test_the_status_vocabulary_was_not_widened() -> None:
    """No fifth status was introduced to paper over executor failure."""

    from typing import get_args

    from app.services.agent.schemas import AgentStatus

    assert set(get_args(AgentStatus)) == {
        "ok",
        "planner_unavailable",
        "synthesis_unavailable",
        "answer_withheld",
    }
