"""Phase 15 Commit 1 - agent contract tests (hardened).

Contracts only: no planner, no executor, no provider, no route. Nothing here
touches the network, Gemini, the filesystem, or a dataset.

The contracts exist to make things structurally impossible rather than merely
discouraged:

* an unknown tool name cannot survive validation, so the executor can never be
  handed a tool that is not in the closed set;
* a plan cannot carry code, a command, or a free-form instruction, and an
  unexpected field is REFUSED rather than silently dropped;
* a trace step cannot name one tool while carrying another's parameters,
  because it carries no second copy of the tool name at all;
* the trace cannot carry chain-of-thought, under that name or any synonym;
* an ``ok`` result cannot exist without the answer it claims.

Those are asserted here, before any executor exists to rely on them.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.services.agent.schemas import (
    AgentEvidence,
    AgentPlan,
    AgentQuestionRequest,
    AgentResult,
    AgentToolStep,
    AgentTrace,
    AnswerValidation,
    EvidenceItem,
    ExecuteQueryParams,
    NdwiParams,
    TemporalNdwiParams,
    ToolCall,
)
from app.services.analysis.schemas import AnalysisResult, Measurement
from app.services.geospatial.schemas import BoundingBox
from app.services.query.schemas import (
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
)
from pydantic import TypeAdapter, ValidationError

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
CATALOG = "https://earth-search.aws.element84.com/v1"

_TOOL_CALL = TypeAdapter(ToolCall)


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


def execute_params(**overrides: Any) -> dict[str, Any]:
    """A valid ``execute_query`` tool call. Note: NO ``limit`` - see section A."""

    payload: dict[str, Any] = {"tool": "execute_query", "intent": intent_dict()}
    payload.update(overrides)
    return payload


def make_execution() -> QueryExecutionResult:
    intent = SatQueryIntent.model_validate(intent_dict())
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=intent, bbox=DEFAULT_BBOX),
        executed_modalities=["sentinel-2-optical"],
        skipped_modalities=[],
        windows=[],
        catalog=CATALOG,
    )


def make_analysis() -> AnalysisResult:
    return AnalysisResult(
        status="ok",
        task="visualize",
        answer="Retrieved 0 window(s).",
        windows_considered=[],
    )


def make_plan(*, steps: list[dict[str, Any]] | None = None) -> AgentPlan:
    return AgentPlan.model_validate(
        {"steps": steps if steps is not None else [execute_params()]}
    )


def make_evidence(**overrides: Any) -> AgentEvidence:
    payload: dict[str, Any] = {
        "items": [
            {
                "id": "ndwi.ndwi_mean",
                "source": "ndwi",
                "measurement": {
                    "name": "ndwi_mean",
                    "value": 0.25,
                    "unit": "index",
                },
            }
        ]
    }
    payload.update(overrides)
    return AgentEvidence.model_validate(payload)


def make_trace(*, evidence_refs: list[str] | None = None) -> AgentTrace:
    plan = make_plan()
    return AgentTrace(
        plan=plan,
        steps=[AgentToolStep(status="ok", parameters=plan.steps[0])],
        evidence_refs=(
            evidence_refs if evidence_refs is not None else ["ndwi.ndwi_mean"]
        ),
        answer_validation=AnswerValidation(
            numeric_grounding="pass", forbidden_terms="pass", evidence_refs="pass"
        ),
    )


def make_result(**overrides: Any) -> AgentResult:
    payload: dict[str, Any] = {
        "status": "ok",
        "answer": "NDWI mean was 0.2500 index.",
        "trace": make_trace().model_dump(),
        "evidence": make_evidence().model_dump(),
    }
    payload.update(overrides)
    return AgentResult.model_validate(payload)


# =========================================================================== #
# A. Tool parameter contracts - narrow, model-facing surface
# =========================================================================== #


def test_valid_execute_query_params() -> None:
    params = ExecuteQueryParams.model_validate(execute_params())

    assert params.tool == "execute_query"
    assert params.intent.location_query == "Chennai"
    assert params.include_imagery is False
    assert params.max_cloud_cover is None


def test_execute_query_reuses_the_existing_intent_type() -> None:
    """No duplicated fields or validators - the existing model is embedded."""

    params = ExecuteQueryParams.model_validate(execute_params())
    assert isinstance(params.intent, SatQueryIntent)


def test_execute_query_exposes_only_analytical_decisions() -> None:
    """``limit`` is a server resource budget, not a decision a planner makes."""

    assert set(ExecuteQueryParams.model_fields) == {
        "tool",
        "intent",
        "include_imagery",
        "max_cloud_cover",
    }
    assert "limit" not in ExecuteQueryParams.model_fields


def test_execute_query_refuses_a_planner_supplied_limit() -> None:
    with pytest.raises(ValidationError):
        ExecuteQueryParams.model_validate(execute_params(limit=100))


@pytest.mark.parametrize(
    "bad_intent",
    [
        # end_date before start_date - TimeRange validator
        {"time_windows": [{"start_date": "2024-02-01", "end_date": "2024-01-01"}]},
        # 'single' mode with two windows - SatQueryIntent validator
        {
            "time_windows": [
                {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                {"start_date": "2024-02-01", "end_date": "2024-02-28"},
            ]
        },
        # duplicate modalities - SatQueryIntent validator
        {"modalities": ["sentinel-2-optical", "sentinel-2-optical"]},
        # empty modalities
        {"modalities": []},
        # unknown task
        {"task": "segment_everything"},
        # blank location
        {"location_query": "   "},
    ],
)
def test_existing_satqueryintent_validation_is_preserved(
    bad_intent: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ExecuteQueryParams.model_validate(
            execute_params(intent=intent_dict(**bad_intent))
        )


@pytest.mark.parametrize("bad", [{"max_cloud_cover": 101}, {"max_cloud_cover": -1}])
def test_invalid_execute_query_parameters_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ExecuteQueryParams.model_validate(execute_params(**bad))


def test_analysis_tool_params_carry_no_free_parameters() -> None:
    """The analysis tools are flags; they must not accept tunable knobs."""

    for model in (NdwiParams, TemporalNdwiParams):
        assert set(model.model_fields) == {"tool"}


# =========================================================================== #
# B. Closed tool set - the discriminator, and refusal of smuggled fields
# =========================================================================== #


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"tool": "execute_query", "intent": intent_dict()}, ExecuteQueryParams),
        ({"tool": "ndwi_statistics"}, NdwiParams),
        ({"tool": "temporal_ndwi_statistics"}, TemporalNdwiParams),
    ],
)
def test_discriminator_selects_the_right_parameter_type(
    payload: dict[str, Any], expected: type
) -> None:
    assert isinstance(_TOOL_CALL.validate_python(payload), expected)


@pytest.mark.parametrize(
    "unknown",
    [
        "run_python",
        "shell",
        "read_file",
        "retrieve_imagery",
        "compatibility_report",
        "rs_model_analysis",
        "",
        "EXECUTE_QUERY",
    ],
)
def test_unknown_tool_names_are_rejected(unknown: str) -> None:
    with pytest.raises(ValidationError):
        _TOOL_CALL.validate_python({"tool": unknown})


@pytest.mark.parametrize(
    "malformed",
    [{"tool": {"$ne": "x"}}, {"tool": ["ndwi_statistics"]}, {"tool": None}, {}],
)
def test_malformed_discriminator_tags_are_rejected(
    malformed: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _TOOL_CALL.validate_python(malformed)


@pytest.mark.parametrize(
    "smuggled",
    [
        {"code": "import os; os.system('rm -rf /')"},
        {"script": "curl evil.example.com"},
        {"command": "ls"},
        {"eval": "1+1"},
        {"limit": 9999},
        {"asset_href": "https://evil.example.com/x.tif"},
    ],
)
def test_a_tool_call_refuses_smuggled_fields(smuggled: dict[str, Any]) -> None:
    """Refused outright, not silently dropped - this is a safety boundary."""

    with pytest.raises(ValidationError):
        _TOOL_CALL.validate_python({"tool": "ndwi_statistics", **smuggled})


def test_serialization_round_trip_preserves_the_exact_tool_type() -> None:
    original = _TOOL_CALL.validate_python(execute_params())
    restored = _TOOL_CALL.validate_json(_TOOL_CALL.dump_json(original))

    assert type(restored) is type(original) is ExecuteQueryParams
    assert restored == original


# =========================================================================== #
# C. AgentPlan shape
# =========================================================================== #


def test_plan_with_one_valid_step_is_accepted() -> None:
    plan = make_plan()
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "execute_query"


def test_plan_with_three_valid_steps_is_accepted() -> None:
    plan = make_plan(
        steps=[
            execute_params(),
            {"tool": "ndwi_statistics"},
            {"tool": "temporal_ndwi_statistics"},
        ]
    )
    assert [step.tool for step in plan.steps] == [
        "execute_query",
        "ndwi_statistics",
        "temporal_ndwi_statistics",
    ]


def test_plan_with_zero_steps_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({"steps": []})


def test_plan_with_more_than_three_steps_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate(
            {
                "steps": [
                    execute_params(),
                    {"tool": "ndwi_statistics"},
                    {"tool": "temporal_ndwi_statistics"},
                    {"tool": "ndwi_statistics"},
                ]
            }
        )


def test_plan_must_start_with_execute_query() -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate(
            {"steps": [{"tool": "ndwi_statistics"}, execute_params()]}
        )


def test_plan_rejects_analysis_without_an_execution() -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({"steps": [{"tool": "ndwi_statistics"}]})


def test_plan_rejects_a_second_execution() -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({"steps": [execute_params(), execute_params()]})


def test_plan_rejects_a_repeated_analysis_tool() -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate(
            {
                "steps": [
                    execute_params(),
                    {"tool": "ndwi_statistics"},
                    {"tool": "ndwi_statistics"},
                ]
            }
        )


def test_plan_rejects_an_unknown_tool_among_valid_ones() -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate(
            {"steps": [execute_params(), {"tool": "exfiltrate"}]}
        )


def test_plan_carries_no_executable_field() -> None:
    forbidden = {"code", "script", "command", "python", "exec", "eval", "shell"}
    assert forbidden.isdisjoint(AgentPlan.model_fields)
    for model in (ExecuteQueryParams, NdwiParams, TemporalNdwiParams):
        assert forbidden.isdisjoint(model.model_fields)


# =========================================================================== #
# D. AgentTrace - observable decisions only, never reasoning
# =========================================================================== #

_REASONING_FIELDS = {
    "chain_of_thought",
    "reasoning",
    "thoughts",
    "thought",
    "thinking",
    "model_reasoning",
    "hidden_reasoning",
    "rationale",
    "deliberation",
    "scratchpad",
    "internal",
    "explanation_of_reasoning",
}


@pytest.mark.parametrize(
    "model", [AgentTrace, AgentToolStep, AgentPlan, AgentResult, AnswerValidation]
)
def test_no_contract_exposes_chain_of_thought(model: type) -> None:
    assert _REASONING_FIELDS.isdisjoint(model.model_fields), (
        f"{model.__name__} exposes a reasoning field"
    )


def test_trace_carries_only_observable_execution_facts() -> None:
    assert set(AgentTrace.model_fields) == {
        "plan",
        "steps",
        "evidence_refs",
        "answer_validation",
    }


def test_tool_step_has_no_second_copy_of_the_tool_name() -> None:
    """The mismatch is unrepresentable: there is only one tool name, in params."""

    assert set(AgentToolStep.model_fields) == {
        "status",
        "parameters",
        "rejection_reason",
        "error_message",
    }
    assert "tool" not in AgentToolStep.model_fields


def test_tool_step_tool_is_derived_from_the_validated_parameters() -> None:
    step = AgentToolStep(status="ok", parameters=NdwiParams())
    assert step.tool == "ndwi_statistics"

    plan = make_plan()
    execute_step = AgentToolStep(status="ok", parameters=plan.steps[0])
    assert execute_step.tool == "execute_query"


def test_tool_step_refuses_an_independent_tool_field() -> None:
    """A planner (or a bug) cannot assert a tool that contradicts the params."""

    with pytest.raises(ValidationError):
        AgentToolStep.model_validate(
            {
                "tool": "execute_query",
                "status": "ok",
                "parameters": {"tool": "ndwi_statistics"},
            }
        )


def test_trace_parameters_are_the_validated_tool_call() -> None:
    trace = make_trace()
    assert isinstance(trace.steps[0].parameters, ExecuteQueryParams)


@pytest.mark.parametrize("status", ["ok", "rejected", "failed", "skipped"])
def test_tool_step_statuses_are_accepted(status: str) -> None:
    step = AgentToolStep.model_validate(
        {"status": status, "parameters": {"tool": "ndwi_statistics"}}
    )
    assert step.status == status


def test_invalid_tool_step_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentToolStep.model_validate(
            {"status": "thinking", "parameters": {"tool": "ndwi_statistics"}}
        )


# =========================================================================== #
# E. AgentResult
# =========================================================================== #


@pytest.mark.parametrize(
    "status", ["planner_unavailable", "synthesis_unavailable", "answer_withheld"]
)
def test_non_ok_statuses_may_omit_the_answer(status: str) -> None:
    result = make_result(status=status, answer=None)
    assert result.answer is None
    assert result.status == status


def test_ok_status_requires_an_answer() -> None:
    """An 'ok' result without an answer would claim a success it cannot show."""

    with pytest.raises(ValidationError):
        make_result(status="ok", answer=None)


def test_ok_status_with_an_answer_is_accepted() -> None:
    assert make_result(status="ok", answer="0.2500 index.").answer is not None


@pytest.mark.parametrize("status", ["error", "failed", "OK", "", "pending"])
def test_invalid_agent_result_status_is_rejected(status: str) -> None:
    with pytest.raises(ValidationError):
        make_result(status=status)


def test_evidence_survives_a_withheld_answer() -> None:
    result = make_result(status="answer_withheld", answer=None)
    assert result.evidence.items[0].measurement is not None


def test_result_refuses_evidence_refs_that_name_nothing() -> None:
    """A trace must not cite evidence the result does not carry."""

    with pytest.raises(ValidationError):
        make_result(trace=make_trace(evidence_refs=["ndwi.does_not_exist"]).model_dump())


def test_result_accepts_evidence_refs_that_resolve() -> None:
    result = make_result(trace=make_trace(evidence_refs=["ndwi.ndwi_mean"]).model_dump())
    assert result.trace.evidence_refs == ["ndwi.ndwi_mean"]


# =========================================================================== #
# F. AgentEvidence - one shape for deterministic and future model evidence
# =========================================================================== #


@pytest.mark.parametrize(
    "source", ["execution", "ndwi", "temporal_ndwi", "compatibility", "model"]
)
def test_evidence_sources_include_the_reserved_model_source(source: str) -> None:
    item = EvidenceItem(
        id=f"{source}.example",
        source=source,  # type: ignore[arg-type]
        measurement=Measurement(name="ndwi_mean", value=0.25, unit="index"),
    )
    assert item.source == source


def test_evidence_item_carries_provenance() -> None:
    assert "produced_by" in EvidenceItem.model_fields

    deterministic = EvidenceItem(
        id="ndwi.ndwi_mean",
        source="ndwi",
        measurement=Measurement(name="ndwi_mean", value=0.25, unit="index"),
        produced_by="analysis.engines.compute_ndwi_measurements",
    )
    assert deterministic.produced_by is not None
    # Provenance is optional, so existing deterministic evidence stays valid.
    assert EvidenceItem(
        id="x.y", source="ndwi", text="note"
    ).produced_by is None


def test_evidence_item_accepts_text_instead_of_a_measurement() -> None:
    item = EvidenceItem(
        id="compatibility.limitation.0",
        source="compatibility",
        text="Equal AOI coverage is NOT established.",
    )
    assert item.measurement is None
    assert item.text


def test_evidence_item_must_carry_something() -> None:
    """An item with neither a measurement nor text cites nothing."""

    with pytest.raises(ValidationError):
        EvidenceItem(id="empty.item", source="ndwi")


def test_duplicate_evidence_ids_are_rejected() -> None:
    """Grounding resolves references by id, so ids must be unique keys."""

    with pytest.raises(ValidationError):
        AgentEvidence.model_validate(
            {
                "items": [
                    {"id": "ndwi.ndwi_mean", "source": "ndwi", "text": "a"},
                    {"id": "ndwi.ndwi_mean", "source": "ndwi", "text": "b"},
                ]
            }
        )


def test_distinct_evidence_ids_are_accepted() -> None:
    evidence = AgentEvidence.model_validate(
        {
            "items": [
                {"id": "ndwi.a", "source": "ndwi", "text": "a"},
                {"id": "ndwi.b", "source": "ndwi", "text": "b"},
            ]
        }
    )
    assert len(evidence.items) == 2


def test_evidence_holds_the_deterministic_results_verbatim() -> None:
    evidence = make_evidence(
        execution=make_execution().model_dump(mode="json"),
        analysis=make_analysis().model_dump(mode="json"),
    )
    assert evidence.execution is not None
    assert evidence.analysis is not None
    assert evidence.items[0].measurement is not None


def test_empty_evidence_is_valid_so_a_failed_plan_still_returns_a_shape() -> None:
    evidence = AgentEvidence()
    assert evidence.items == []
    assert evidence.execution is None


# =========================================================================== #
# G. Extra fields are refused across the contract surface
# =========================================================================== #


def test_plan_refuses_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate(
            {"steps": [execute_params()], "reasoning": "because I said so"}
        )


def test_trace_refuses_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AgentTrace.model_validate({"steps": [], "chain_of_thought": "step 1..."})


def test_result_refuses_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_result(thoughts="I considered several options")


def test_evidence_and_items_refuse_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AgentEvidence.model_validate({"items": [], "notes": "x"})
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate(
            {"id": "a.b", "source": "ndwi", "text": "t", "confidence": 0.9}
        )


# =========================================================================== #
# H. JSON round-trips
# =========================================================================== #


def test_agent_question_request_validates_and_bounds_the_question() -> None:
    request = AgentQuestionRequest.model_validate({"question": "  What is here? "})
    assert request.question == "What is here?"

    for bad in ({"question": ""}, {"question": "   "}, {"question": "x" * 5000}):
        with pytest.raises(ValidationError):
            AgentQuestionRequest.model_validate(bad)


def test_plan_round_trips_through_json() -> None:
    plan = make_plan(
        steps=[execute_params(), {"tool": "temporal_ndwi_statistics"}]
    )
    assert AgentPlan.model_validate_json(plan.model_dump_json()) == plan


def test_trace_round_trips_through_json() -> None:
    trace = make_trace()
    assert AgentTrace.model_validate_json(trace.model_dump_json()) == trace


def test_evidence_round_trips_through_json() -> None:
    evidence = make_evidence(
        execution=make_execution().model_dump(mode="json"),
        analysis=make_analysis().model_dump(mode="json"),
    )
    assert AgentEvidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_result_round_trips_through_json() -> None:
    result = make_result()
    assert AgentResult.model_validate_json(result.model_dump_json()) == result


# =========================================================================== #
# I. Architectural boundary
# =========================================================================== #


def test_schemas_import_no_provider_sdk_and_no_lower_layer_internals() -> None:
    import ast
    import pathlib

    from app.services.agent import schemas as schemas_mod

    tree = ast.parse(pathlib.Path(schemas_mod.__file__).read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    for forbidden in ("google", "genai", "httpx", "rasterio", "numpy", "fastapi"):
        assert forbidden not in roots, f"{forbidden!r} must not be imported"
    assert roots <= {"__future__", "typing", "pydantic", "app"}


def test_lower_layers_do_not_import_the_agent_package() -> None:
    import pathlib

    backend = pathlib.Path(__file__).resolve().parents[1]
    for package in ("analysis", "query", "satellite", "geospatial", "ai"):
        for path in (backend / "app" / "services" / package).rglob("*.py"):
            assert "services.agent" not in path.read_text(), (
                f"{path} imports the agent package - dependency direction violated"
            )
    for path in (backend / "app" / "core").rglob("*.py"):
        assert "services.agent" not in path.read_text()
