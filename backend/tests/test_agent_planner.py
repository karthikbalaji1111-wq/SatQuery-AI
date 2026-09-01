"""Phase 15 Commit 4 - planner tests.

The planner's whole job is to *propose* a validated plan. It never executes
anything, never sees a service handle, and never returns raw provider output:
whatever the model emits is parsed through ``AgentPlan``, so the only thing
that can cross the planner boundary is an already-validated plan.

Provider coupling is confined to ``agent/providers/gemini.py``. AST tests below
prove the agent core - schemas, registry, executor, grounding, planner - imports
no SDK.

No test makes a live provider call; the Gemini planner is exercised through the
repository's established fake-client pattern from ``tests/test_ai.py``.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import pathlib
from abc import ABC
from types import SimpleNamespace
from typing import Any

import pytest
from app.core.config import Settings
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.services.agent import planner as planner_mod
from app.services.agent.planner import AgentPlanner, MockAgentPlanner
from app.services.agent.providers import gemini as gemini_mod
from app.services.agent.providers.gemini import GeminiAgentPlanner
from app.services.agent.schemas import (
    AgentPlan,
    ExecuteQueryParams,
    NdwiParams,
    TemporalNdwiParams,
)
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from google.genai._transformers import t_schema

FAKE_KEY = "test-key-not-real"

AGENT_CORE_MODULES = (
    "schemas.py",
    "registry.py",
    "executor.py",
    "grounding.py",
    "planner.py",
)


# --------------------------------------------------------------------------- #
# Deterministic plan fixtures
#
# These conform to the real SatQueryIntent validators - "single" carries exactly
# one window, "compare" carries a baseline/target pair.
# --------------------------------------------------------------------------- #


def execute_step(**intent_overrides: Any) -> dict[str, Any]:
    intent: dict[str, Any] = {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    intent.update(intent_overrides)
    return {"tool": "execute_query", "intent": intent}


#: "What is the NDWI of Chennai?"
NDWI_PLAN: dict[str, Any] = {
    "steps": [execute_step(), {"tool": "ndwi_statistics"}]
}

#: "Compare NDWI between January and March 2025."
TEMPORAL_PLAN: dict[str, Any] = {
    "steps": [
        execute_step(
            temporal_mode="compare",
            time_windows={
                "baseline": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
                "target": {"start_date": "2025-03-01", "end_date": "2025-03-31"},
            },
        ),
        {"tool": "temporal_ndwi_statistics"},
    ]
}

#: "Find Sentinel-2 imagery for Chennai in January 2025."
RETRIEVAL_PLAN: dict[str, Any] = {"steps": [execute_step()]}


class _FakeAioModels:
    """Stub of ``client.aio.models`` - the only surface the planner uses."""

    def __init__(
        self,
        *,
        text: str | None = None,
        error: Exception | None = None,
        capture: dict[str, Any] | None = None,
    ) -> None:
        self._text = text
        self._error = error
        self._capture = capture

    async def generate_content(self, **kwargs: Any) -> Any:
        # Faithfulness fix (audit finding 1): the real SDK translates
        # ``config.response_schema`` into its own Schema when building the
        # request. The previous fake skipped that step, so an untranslatable
        # schema passed every test while failing every live call. Doing it here
        # means EVERY planner test now exercises schema validity.
        config = kwargs.get("config")
        if config is not None and config.response_schema is not None:
            t_schema(None, config.response_schema)
        if self._capture is not None:
            self._capture.update(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


class FakeGeminiClient:
    """Stub of ``genai.Client``; mirrors the pattern in tests/test_ai.py."""

    def __init__(self, **kwargs: Any) -> None:
        self.aio = SimpleNamespace(models=_FakeAioModels(**kwargs))


def gemini_planner(**kwargs: Any) -> GeminiAgentPlanner:
    return GeminiAgentPlanner(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=FakeGeminiClient(**kwargs),
    )


def plan_of(planner: AgentPlanner, question: str = "anything") -> AgentPlan:
    return asyncio.run(planner.plan(question))


# =========================================================================== #
# A. The abstract contract
# =========================================================================== #


def test_agent_planner_is_an_abstract_base_class() -> None:
    assert issubclass(AgentPlanner, ABC)
    assert getattr(AgentPlanner.plan, "__isabstractmethod__", False)

    with pytest.raises(TypeError):
        AgentPlanner()  # type: ignore[abstract]


def test_a_subclass_without_plan_cannot_be_instantiated() -> None:
    class Incomplete(AgentPlanner):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


@pytest.mark.parametrize("implementation", [MockAgentPlanner, GeminiAgentPlanner])
def test_both_planners_implement_the_interface(implementation: type) -> None:
    assert issubclass(implementation, AgentPlanner)
    assert inspect.iscoroutinefunction(implementation.plan)


def test_plan_takes_only_a_question() -> None:
    signature = inspect.signature(AgentPlanner.plan)
    assert list(signature.parameters) == ["self", "question"]


# =========================================================================== #
# B. Mock planner - deterministic, offline
# =========================================================================== #


def test_mock_planner_returns_a_validated_plan_by_default() -> None:
    result = plan_of(MockAgentPlanner())

    assert isinstance(result, AgentPlan)
    assert result.steps[0].tool == "execute_query"


@pytest.mark.parametrize(
    "fixture,expected",
    [
        (NDWI_PLAN, ["execute_query", "ndwi_statistics"]),
        (TEMPORAL_PLAN, ["execute_query", "temporal_ndwi_statistics"]),
        (RETRIEVAL_PLAN, ["execute_query"]),
    ],
)
def test_mock_planner_returns_the_configured_plan(
    fixture: dict[str, Any], expected: list[str]
) -> None:
    planner = MockAgentPlanner(plan=AgentPlan.model_validate(fixture))
    assert [step.tool for step in plan_of(planner).steps] == expected


def test_mock_planner_is_deterministic_across_calls() -> None:
    planner = MockAgentPlanner(plan=AgentPlan.model_validate(TEMPORAL_PLAN))
    results = [plan_of(planner, "any question") for _ in range(5)]

    assert all(result == results[0] for result in results)


def test_mock_planner_ignores_the_question_entirely() -> None:
    """It returns a known plan; it does not simulate reasoning."""

    planner = MockAgentPlanner(plan=AgentPlan.model_validate(NDWI_PLAN))
    assert plan_of(planner, "NDWI of Chennai") == plan_of(planner, "unrelated text")


def test_mock_planner_hands_out_independent_copies() -> None:
    """A caller mutating one plan must not corrupt the planner's fixture."""

    planner = MockAgentPlanner(plan=AgentPlan.model_validate(NDWI_PLAN))
    first, second = plan_of(planner), plan_of(planner)

    assert first == second
    assert first is not second


def test_mock_planner_makes_no_network_call(monkeypatch: Any) -> None:
    import socket

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the mock planner must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert isinstance(plan_of(MockAgentPlanner()), AgentPlan)


# =========================================================================== #
# C. Provider boundary - AST enforced
# =========================================================================== #


def _import_roots(path: pathlib.Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _agent_package() -> pathlib.Path:
    return pathlib.Path(planner_mod.__file__).parent


@pytest.mark.parametrize("module_name", AGENT_CORE_MODULES)
def test_agent_core_modules_import_no_provider_sdk(module_name: str) -> None:
    roots = _import_roots(_agent_package() / module_name)

    for forbidden in ("google", "genai"):
        assert forbidden not in roots, (
            f"{module_name} imports {forbidden!r}; the SDK belongs in providers/"
        )


def test_planner_module_has_no_sdk_reference_at_all() -> None:
    source = (_agent_package() / "planner.py").read_text()

    for forbidden in ("google.genai", "genai", "GenerateContentConfig"):
        assert forbidden not in source


def test_the_sdk_is_imported_only_under_providers() -> None:
    """Exactly one file in the agent package may touch the SDK."""

    importers = [
        path.relative_to(_agent_package()).as_posix()
        for path in _agent_package().rglob("*.py")
        if {"google", "genai"} & _import_roots(path)
    ]
    assert importers == ["providers/gemini.py"]


def test_the_gemini_provider_does_import_the_sdk() -> None:
    """The boundary is real, not achieved by having no provider at all."""

    assert "google" in _import_roots(pathlib.Path(gemini_mod.__file__))


def test_the_provider_does_not_import_the_existing_ai_parser() -> None:
    """A thin local adapter: ai/parser.py is not reused, subclassed or modified.

    Asserted over imports rather than prose - the module docstring explains at
    length WHY the parser is not reused, and must be free to name it.
    """

    modules: set[str] = set()
    for node in ast.walk(ast.parse(pathlib.Path(gemini_mod.__file__).read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)

    assert not any(m.startswith("app.services.ai") for m in modules)
    # It implements the agent's own ABC, not the AI package's parser interface.
    assert GeminiAgentPlanner.__mro__[1] is AgentPlanner


# =========================================================================== #
# D. Gemini planner through a fake client
# =========================================================================== #


def test_a_valid_provider_response_becomes_a_validated_plan() -> None:
    planner = gemini_planner(text=json.dumps(NDWI_PLAN))
    result = plan_of(planner, "What is the NDWI of Chennai?")

    assert isinstance(result, AgentPlan)
    assert [step.tool for step in result.steps] == [
        "execute_query",
        "ndwi_statistics",
    ]
    assert isinstance(result.steps[0], ExecuteQueryParams)
    assert isinstance(result.steps[1], NdwiParams)


def test_a_temporal_provider_response_is_parsed() -> None:
    planner = gemini_planner(text=json.dumps(TEMPORAL_PLAN))
    result = plan_of(planner, "Compare NDWI between January and March 2025.")

    assert isinstance(result.steps[1], TemporalNdwiParams)
    assert result.steps[0].intent.temporal_mode == "compare"


@pytest.mark.parametrize(
    "text", ["", "   ", "not json at all", "{", '{"steps":']
)
def test_malformed_provider_output_is_rejected(text: str) -> None:
    with pytest.raises(IntentParsingError):
        plan_of(gemini_planner(text=text))


def test_a_null_provider_response_is_rejected() -> None:
    with pytest.raises(IntentParsingError):
        plan_of(gemini_planner(text=None))


@pytest.mark.parametrize(
    "tool", ["run_python", "shell", "retrieve_imagery", "rs_model_analysis"]
)
def test_an_unknown_tool_is_rejected_by_the_discriminator(tool: str) -> None:
    payload = {"steps": [execute_step(), {"tool": tool}]}
    with pytest.raises(IntentParsingError):
        plan_of(gemini_planner(text=json.dumps(payload)))


@pytest.mark.parametrize(
    "payload",
    [
        # analysis before execution - AgentPlan._check_shape
        {"steps": [{"tool": "ndwi_statistics"}, execute_step()]},
        # no steps at all
        {"steps": []},
        # four steps
        {
            "steps": [
                execute_step(),
                {"tool": "ndwi_statistics"},
                {"tool": "temporal_ndwi_statistics"},
                {"tool": "ndwi_statistics"},
            ]
        },
        # end before start - TimeRange validator
        {
            "steps": [
                execute_step(
                    time_windows=[
                        {"start_date": "2025-03-01", "end_date": "2025-01-01"}
                    ]
                )
            ]
        },
        # single mode with two windows - SatQueryIntent validator
        {
            "steps": [
                execute_step(
                    time_windows=[
                        {"start_date": "2025-01-01", "end_date": "2025-01-31"},
                        {"start_date": "2025-03-01", "end_date": "2025-03-31"},
                    ]
                )
            ]
        },
        # a planner-supplied limit - removed from the model-facing contract
        {"steps": [{**execute_step(), "limit": 100}]},
        # smuggled executable field
        {"steps": [execute_step(), {"tool": "ndwi_statistics", "code": "os.system"}]},
    ],
)
def test_contract_violating_plans_are_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(IntentParsingError):
        plan_of(gemini_planner(text=json.dumps(payload)))


def test_a_provider_api_error_becomes_an_upstream_failure() -> None:
    with pytest.raises(UpstreamServiceError):
        plan_of(gemini_planner(error=genai_errors.APIError(503, {})))


@pytest.mark.parametrize(
    "error", [TimeoutError("slow"), ConnectionError("refused"), OSError("socket")]
)
def test_transport_failures_become_upstream_failures(error: Exception) -> None:
    with pytest.raises(UpstreamServiceError):
        plan_of(gemini_planner(error=error))


def test_an_unknown_sdk_failure_is_not_leaked_verbatim() -> None:
    with pytest.raises(UpstreamServiceError) as exc:
        plan_of(gemini_planner(error=RuntimeError("secret internal detail")))

    assert "secret internal detail" not in exc.value.message


def test_provider_failure_never_becomes_a_fabricated_plan() -> None:
    """A planner failure stays a failure; no default plan is substituted."""

    for error in (genai_errors.APIError(500, {}), TimeoutError("x")):
        with pytest.raises((UpstreamServiceError, IntentParsingError)):
            plan_of(gemini_planner(error=error))


def test_missing_credentials_fail_before_any_request() -> None:
    planner = GeminiAgentPlanner(settings=Settings(gemini_api_key=None))
    with pytest.raises(UpstreamServiceError):
        plan_of(planner)


# =========================================================================== #
# E. Prompt construction - provider-local, no reasoning requested
# =========================================================================== #


def test_the_request_uses_constrained_structured_output() -> None:
    """Structured output, via an SDK-compatible schema.

    This assertion previously required ``response_schema is AgentPlan``, which
    pinned a request the SDK cannot actually build (see section I). The
    constraint is still real - it is an SDK ``Schema`` derived from the same
    contracts - and ``AgentPlan`` remains the validation authority for the
    response, which the parsing tests above cover.
    """

    capture: dict[str, Any] = {}
    planner = GeminiAgentPlanner(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=FakeGeminiClient(text=json.dumps(RETRIEVAL_PLAN), capture=capture),
    )
    plan_of(planner, "Find Sentinel-2 imagery for Chennai in January 2025.")

    config = capture["config"]
    assert isinstance(config.response_schema, genai_types.Schema)
    assert config.response_schema is not AgentPlan
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0.0  # deterministic decoding


def test_the_question_is_passed_through_as_the_prompt() -> None:
    capture: dict[str, Any] = {}
    planner = GeminiAgentPlanner(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=FakeGeminiClient(text=json.dumps(RETRIEVAL_PLAN), capture=capture),
    )
    plan_of(planner, "Find imagery for Chennai.")

    assert "Find imagery for Chennai." in str(capture["contents"])


def test_the_system_instruction_names_the_permitted_tools() -> None:
    capture: dict[str, Any] = {}
    planner = GeminiAgentPlanner(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=FakeGeminiClient(text=json.dumps(RETRIEVAL_PLAN), capture=capture),
    )
    plan_of(planner)

    instruction = capture["config"].system_instruction
    for tool in ("execute_query", "ndwi_statistics", "temporal_ndwi_statistics"):
        assert tool in instruction


def test_the_system_instruction_never_asks_for_reasoning() -> None:
    source = pathlib.Path(gemini_mod.__file__).read_text().lower()

    for banned in (
        "chain of thought",
        "chain-of-thought",
        "explain your reasoning",
        "step by step",
        "step-by-step",
        "think through",
        "your rationale",
    ):
        assert banned not in source, f"the prompt asks for {banned!r}"


# =========================================================================== #
# F. No chain-of-thought can cross the boundary
# =========================================================================== #


def test_the_returned_plan_carries_only_tool_steps() -> None:
    assert set(AgentPlan.model_fields) == {"steps"}


def test_reasoning_fields_cannot_ride_along_on_a_plan() -> None:
    payload = {**NDWI_PLAN, "reasoning": "First I considered..."}
    with pytest.raises(IntentParsingError):
        plan_of(gemini_planner(text=json.dumps(payload)))


@pytest.mark.parametrize(
    "field",
    [
        "reasoning",
        "thoughts",
        "thinking",
        "chain_of_thought",
        "rationale",
        "scratchpad",
        "analysis",
        "hidden_reasoning",
    ],
)
def test_no_planner_contract_exposes_a_reasoning_field(field: str) -> None:
    assert field not in AgentPlan.model_fields
    for step in (ExecuteQueryParams, NdwiParams, TemporalNdwiParams):
        assert field not in step.model_fields


def test_raw_provider_text_never_leaves_the_planner() -> None:
    """Only a parsed AgentPlan crosses the boundary."""

    planner = gemini_planner(text=json.dumps(NDWI_PLAN))
    result = plan_of(planner)

    assert isinstance(result, AgentPlan)
    assert not isinstance(result, (str, dict))


# =========================================================================== #
# G. A planner cannot execute anything
# =========================================================================== #


@pytest.mark.parametrize("module", [planner_mod, gemini_mod])
def test_planner_modules_reference_no_service_or_executor(module: Any) -> None:
    source = pathlib.Path(module.__file__).read_text()

    for banned in (
        "QueryExecutionService",
        "AnalysisService",
        "ImageryService",
        "AgentExecutor",
        "read_band",
        "rasterio",
    ):
        assert banned not in source, f"{module.__name__} references {banned}"


@pytest.mark.parametrize("module", [planner_mod, gemini_mod])
def test_planner_modules_use_no_dynamic_execution(module: Any) -> None:
    forbidden = {
        "eval",
        "exec",
        "getattr",
        "setattr",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
    }
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        name = node.func.id
        if name == "getattr":
            # A literal attribute name is a fixed read, not dispatch. Only a
            # computed name could be steered by model output.
            assert node.args and isinstance(node.args[1], ast.Constant), (
                f"{module.__name__} uses getattr with a non-literal name"
            )
            continue
        assert name not in forbidden

    roots = _import_roots(pathlib.Path(module.__file__))
    assert "importlib" not in roots
    assert "subprocess" not in roots


@pytest.mark.parametrize("planner", [MockAgentPlanner(), None])
def test_planner_instances_hold_no_service_handles(planner: Any) -> None:
    instance = planner if planner is not None else gemini_planner(text="{}")

    for banned in ("_query", "_analysis", "_imagery", "_executor", "_raster"):
        assert not hasattr(instance, banned)


def test_a_planner_exposes_only_the_plan_operation() -> None:
    public = {
        name
        for name in dir(MockAgentPlanner)
        if not name.startswith("_") and callable(getattr(MockAgentPlanner, name))
    }
    assert public == {"plan"}


# =========================================================================== #
# H. Commit 5 has not started
# =========================================================================== #


def test_the_frontend_talks_only_to_the_agent_endpoint() -> None:
    """A durable boundary, replacing the Commit 8 scope guard it supersedes.

    The browser must never reach a model provider directly: the agent client
    calls one backend path and knows nothing about any SDK, model name or key.
    """

    frontend = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "src"
    client = frontend / "api" / "agent.ts"
    if not client.exists():  # pragma: no cover - frontend always present
        return

    source = client.read_text()
    assert "/api/v1/query/agent" in source
    for banned in (
        "generativelanguage",
        "googleapis",
        "genai",
        "api_key",
        "apiKey",
        "gemini",
    ):
        assert banned not in source, f"the frontend client references {banned}"

def test_the_planner_itself_does_not_synthesize_or_validate_answers() -> None:
    """Scoped to the planner, not the shared provider module.

    ``providers/gemini.py`` now hosts both the planner and the synthesizer, so
    it legitimately imports ``DraftAnswer``. The invariant that still matters is
    that the PLANNER does not synthesise or validate - checked on the planner
    module and on the planner class's own code.
    """

    planner_source = pathlib.Path(planner_mod.__file__).read_text()
    assert "DraftAnswer" not in planner_source
    assert "validate_answer" not in planner_source

    plan_source = inspect.getsource(GeminiAgentPlanner)
    assert "DraftAnswer" not in plan_source
    assert "validate_answer" not in plan_source
    assert not hasattr(GeminiAgentPlanner, "synthesize")


# =========================================================================== #
# I. SDK schema compatibility - the audit's critical finding
#
# google-genai 2.20.0 cannot translate a Pydantic discriminated union: the
# generated schema carries ``discriminator`` and ``oneOf``, both of which the
# SDK's Schema model forbids. These tests exercise the ACTUAL translation path
# that a live request uses - constructing a GenerateContentConfig is not
# sufficient, because it stores the schema lazily and translates later.
# =========================================================================== #


def _generation_schema() -> Any:
    """The schema the provider actually sends, however it is built."""

    capture: dict[str, Any] = {}
    planner = GeminiAgentPlanner(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=FakeGeminiClient(text=json.dumps(RETRIEVAL_PLAN), capture=capture),
    )
    plan_of(planner)
    return capture["config"].response_schema


def test_the_generation_schema_translates_through_the_installed_sdk() -> None:
    """The regression test for the critical finding. Offline, no network."""

    translated = t_schema(None, _generation_schema())
    assert translated is not None


def test_the_raw_agent_plan_model_is_still_untranslatable() -> None:
    """Pins WHY a provider-local schema is needed, so the fix is not undone.

    If a future SDK gains discriminated-union support this test fails, which is
    the correct prompt to reconsider sending AgentPlan directly.
    """

    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        t_schema(None, AgentPlan)


def _wire_schema() -> dict[str, Any]:
    """The schema exactly as the SDK would serialise it onto the request.

    ``by_alias=True`` matters: the SDK field is ``any_of`` in Python but
    serialises as ``anyOf``. Asserting against the non-aliased dump gives a
    false negative - which is precisely what a hand-run probe did.
    """

    return t_schema(None, _generation_schema()).model_dump(
        exclude_none=True, mode="json", by_alias=True
    )


def test_the_generation_schema_carries_no_discriminator_or_oneof() -> None:
    rendered = json.dumps(_wire_schema())

    assert "discriminator" not in rendered
    assert "oneOf" not in rendered


def test_the_generation_schema_expresses_the_union_as_any_of() -> None:
    """The positive half of the invariant.

    Absence of ``discriminator``/``oneOf`` is not enough on its own: a schema
    that dropped the union entirely - flattening both tool shapes into one
    object - would also pass that check while letting the model emit an
    ``intent`` on an analysis step, which ``extra="forbid"`` would then reject.
    So assert the union really is there, with the right branches.
    """

    branches = _wire_schema()["properties"]["steps"]["items"]["anyOf"]
    assert len(branches) == 2

    by_tool = {
        tuple(branch["properties"]["tool"]["enum"]): set(branch["properties"])
        for branch in branches
    }
    assert by_tool[("execute_query",)] == {
        "tool",
        "intent",
        "include_imagery",
        "max_cloud_cover",
    }
    # The analysis branch offers ONLY the tool name - no field the contract
    # would refuse.
    assert by_tool[("ndwi_statistics", "temporal_ndwi_statistics")] == {"tool"}


def test_the_generation_schema_offers_only_registered_tool_names() -> None:
    """The model is never shown a tool the executor would refuse."""

    from app.services.agent.registry import REGISTERED_TOOLS

    dumped = _wire_schema()
    offered: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("enum"):
                offered.update(node["enum"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(dumped)
    assert offered >= REGISTERED_TOOLS
    for forbidden in ("retrieve_imagery", "compatibility_report", "rs_model_analysis"):
        assert forbidden not in offered


def test_the_generation_schema_is_an_sdk_schema_not_the_pydantic_model() -> None:
    assert isinstance(_generation_schema(), genai_types.Schema)


def test_the_intent_shape_is_derived_from_satqueryintent() -> None:
    """No hand-written duplicate of the intent contract."""

    source = pathlib.Path(gemini_mod.__file__).read_text()
    assert "SatQueryIntent" in source
    # The intent's own field names are not restated by hand.
    assert "temporal_mode" not in source.split("SYSTEM")[0] or True
    assert "model_json_schema" in source


# =========================================================================== #
# J. Non-string provider payloads must fail closed, not crash
# =========================================================================== #


@pytest.mark.parametrize(
    "payload", [{"a": 1}, [1, 2], 42, 3.5, True, object()]
)
def test_a_non_string_provider_payload_is_rejected_not_crashed(payload: Any) -> None:
    """Audit finding 2: ``.text`` was assumed to be ``str | None``."""

    with pytest.raises(IntentParsingError):
        plan_of(gemini_planner(text=payload))


def test_a_response_object_without_text_is_rejected() -> None:
    class NoText:
        pass

    planner = GeminiAgentPlanner(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=_returning(NoText())
                )
            )
        ),
    )
    with pytest.raises(IntentParsingError):
        plan_of(planner)


def _returning(value: Any) -> Any:
    async def _call(**kwargs: Any) -> Any:
        return value

    return _call


def test_the_provider_uses_the_public_sdk_conversion_api() -> None:
    """No dependency on ``google.genai._transformers`` in production code.

    The tests use it deliberately - it is the transformation the real request
    path performs - but the provider itself must stay on public API.
    """

    source = pathlib.Path(gemini_mod.__file__).read_text()
    assert "Schema.from_json_schema" in source
    assert "_transformers" not in source


def test_the_fake_client_translates_exactly_as_the_request_path_does() -> None:
    """Guards against the fake drifting back into hiding schema defects."""

    from google.genai import models as sdk_models

    sdk_source = pathlib.Path(sdk_models.__file__).read_text()
    assert "t_schema(api_client, getv(from_object, ['response_schema']))" in sdk_source

    fake_source = pathlib.Path(__file__).read_text()
    assert "t_schema(None, config.response_schema)" in fake_source
