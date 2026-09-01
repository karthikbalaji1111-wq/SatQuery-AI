"""Phase 15 Commit 5 - answer synthesizer tests.

The synthesizer turns already-collected evidence into prose. That is its whole
job: it executes nothing, retrieves nothing, computes nothing, and invents no
evidence. Whatever a model returns is parsed through ``DraftAnswer`` before it
crosses the provider boundary, so raw provider output can never escape.

Grounding stays where Commit 3 put it. The synthesizer does not validate its own
output - a generator marking its own homework would be worthless - so nothing
here duplicates ``validate_answer``.

No test makes a live provider call.
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
from app.services.agent import synthesizer as synthesizer_mod
from app.services.agent.grounding import DraftAnswer
from app.services.agent.providers import gemini as gemini_mod
from app.services.agent.providers.gemini import GeminiAnswerSynthesizer
from app.services.agent.schemas import AgentEvidence
from app.services.agent.synthesizer import AnswerSynthesizer, MockAnswerSynthesizer
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from google.genai._transformers import t_schema
from pydantic import ValidationError

FAKE_KEY = "test-key-not-real"

AGENT_CORE_MODULES = (
    "schemas.py",
    "registry.py",
    "executor.py",
    "grounding.py",
    "planner.py",
    "synthesizer.py",
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


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
            },
            {
                "id": "compatibility.limitation.0",
                "source": "compatibility",
                "text": "Equal AOI coverage is NOT established.",
                "produced_by": "query.compatibility.compute_compatibility",
            },
        ]
    }
    payload.update(overrides)
    return AgentEvidence.model_validate(payload)


VALID_ANSWER: dict[str, Any] = {
    "summary": "The mean NDWI was 0.2777 index for the selected scene.",
    "evidence_refs": ["ndwi.ndwi_mean"],
}


class _FakeAioModels:
    """Stub of ``client.aio.models``; translates the schema like the real SDK."""

    def __init__(
        self,
        *,
        text: Any = None,
        error: Exception | None = None,
        capture: dict[str, Any] | None = None,
    ) -> None:
        self._text = text
        self._error = error
        self._capture = capture

    async def generate_content(self, **kwargs: Any) -> Any:
        config = kwargs.get("config")
        if config is not None and config.response_schema is not None:
            # Same translation the real request path performs, so an
            # SDK-incompatible schema cannot pass silently (Commit 4 lesson).
            t_schema(None, config.response_schema)
        if self._capture is not None:
            self._capture.update(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


class FakeGeminiClient:
    def __init__(self, **kwargs: Any) -> None:
        self.aio = SimpleNamespace(models=_FakeAioModels(**kwargs))


def gemini_synthesizer(**kwargs: Any) -> GeminiAnswerSynthesizer:
    return GeminiAnswerSynthesizer(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=FakeGeminiClient(**kwargs),
    )


def synthesize(
    synth: AnswerSynthesizer,
    question: str = "What is the NDWI here?",
    evidence: AgentEvidence | None = None,
) -> DraftAnswer:
    return asyncio.run(
        synth.synthesize(question, evidence if evidence is not None else make_evidence())
    )


# =========================================================================== #
# A. The abstract contract
# =========================================================================== #


def test_answer_synthesizer_is_an_abstract_base_class() -> None:
    assert issubclass(AnswerSynthesizer, ABC)
    assert getattr(AnswerSynthesizer.synthesize, "__isabstractmethod__", False)

    with pytest.raises(TypeError):
        AnswerSynthesizer()  # type: ignore[abstract]


def test_synthesize_takes_only_a_question_and_evidence() -> None:
    signature = inspect.signature(AnswerSynthesizer.synthesize)
    assert list(signature.parameters) == ["self", "question", "evidence"]


@pytest.mark.parametrize(
    "implementation", [MockAnswerSynthesizer, GeminiAnswerSynthesizer]
)
def test_both_synthesizers_implement_the_interface(implementation: type) -> None:
    assert issubclass(implementation, AnswerSynthesizer)
    assert inspect.iscoroutinefunction(implementation.synthesize)


# =========================================================================== #
# B. Mock synthesizer
# =========================================================================== #


def test_mock_synthesizer_returns_a_valid_draft_answer() -> None:
    result = synthesize(MockAnswerSynthesizer())

    assert isinstance(result, DraftAnswer)
    assert result.summary
    assert isinstance(result.evidence_refs, list)


def test_mock_synthesizer_returns_the_configured_answer() -> None:
    configured = DraftAnswer.model_validate(VALID_ANSWER)
    result = synthesize(MockAnswerSynthesizer(answer=configured))

    assert result == configured


def test_mock_synthesizer_is_deterministic() -> None:
    synth = MockAnswerSynthesizer(answer=DraftAnswer.model_validate(VALID_ANSWER))
    results = [synthesize(synth) for _ in range(5)]

    assert all(result == results[0] for result in results)


def test_mock_synthesizer_hands_out_independent_copies() -> None:
    synth = MockAnswerSynthesizer(answer=DraftAnswer.model_validate(VALID_ANSWER))
    first, second = synthesize(synth), synthesize(synth)

    assert first == second
    assert first is not second


def test_mock_synthesizer_makes_no_network_call(monkeypatch: Any) -> None:
    import socket

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the mock synthesizer must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert isinstance(synthesize(MockAnswerSynthesizer()), DraftAnswer)


# =========================================================================== #
# C. The DraftAnswer contract
# =========================================================================== #


def test_draft_answer_round_trips_through_json() -> None:
    answer = DraftAnswer.model_validate(VALID_ANSWER)
    assert DraftAnswer.model_validate_json(answer.model_dump_json()) == answer


def test_evidence_refs_are_strings_preserved_exactly() -> None:
    refs = ["ndwi.ndwi_mean", "compatibility.limitation.0"]
    answer = DraftAnswer(summary="x", evidence_refs=refs)

    assert answer.evidence_refs == refs
    assert all(isinstance(ref, str) for ref in answer.evidence_refs)


def test_a_missing_summary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DraftAnswer.model_validate({"evidence_refs": []})


def test_an_empty_summary_is_rejected() -> None:
    """An empty answer would still be reported as a successful one."""

    with pytest.raises(ValidationError):
        DraftAnswer.model_validate({"summary": "", "evidence_refs": []})


def test_a_missing_evidence_refs_is_rejected() -> None:
    """Citing nothing must be explicit (``[]``), never an omission."""

    with pytest.raises(ValidationError):
        DraftAnswer.model_validate({"summary": "An answer."})


def test_an_explicitly_empty_evidence_refs_is_accepted() -> None:
    assert DraftAnswer.model_validate(
        {"summary": "No numeric claims.", "evidence_refs": []}
    ).evidence_refs == []


@pytest.mark.parametrize(
    "extra",
    [
        {"reasoning": "I first considered..."},
        {"chain_of_thought": "step 1"},
        {"thoughts": "hmm"},
        {"thinking": "..."},
        {"rationale": "because"},
        {"confidence": 0.9},
        {"tool_calls": [{"tool": "run_python"}]},
        {"code": "os.system('x')"},
        {"metadata": {"k": "v"}},
    ],
)
def test_unexpected_fields_are_rejected(extra: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DraftAnswer.model_validate({**VALID_ANSWER, **extra})


def test_draft_answer_carries_only_answer_content() -> None:
    assert set(DraftAnswer.model_fields) == {"summary", "evidence_refs"}


@pytest.mark.parametrize(
    "banned",
    [
        "reasoning",
        "thoughts",
        "thinking",
        "chain_of_thought",
        "rationale",
        "scratchpad",
        "hidden_reasoning",
        "confidence",
        "tool_calls",
    ],
)
def test_no_reasoning_or_execution_field_exists(banned: str) -> None:
    assert banned not in DraftAnswer.model_fields


# =========================================================================== #
# D. Gemini synthesizer through a fake client
# =========================================================================== #


def test_a_valid_provider_response_becomes_a_draft_answer() -> None:
    result = synthesize(gemini_synthesizer(text=json.dumps(VALID_ANSWER)))

    assert isinstance(result, DraftAnswer)
    assert result.summary == VALID_ANSWER["summary"]
    assert result.evidence_refs == VALID_ANSWER["evidence_refs"]


def test_the_provider_returns_a_parsed_model_not_raw_output() -> None:
    result = synthesize(gemini_synthesizer(text=json.dumps(VALID_ANSWER)))

    assert isinstance(result, DraftAnswer)
    assert not isinstance(result, (str, dict, bytes))


@pytest.mark.parametrize(
    "text", ["", "   ", "not json", "{", '{"summary":', "[]", "null", '"hi"']
)
def test_malformed_provider_output_is_rejected(text: str) -> None:
    with pytest.raises(IntentParsingError):
        synthesize(gemini_synthesizer(text=text))


@pytest.mark.parametrize("payload", [{"a": 1}, [1], 42, 3.5, True, None])
def test_a_non_string_provider_payload_is_rejected(payload: Any) -> None:
    with pytest.raises(IntentParsingError):
        synthesize(gemini_synthesizer(text=payload))


@pytest.mark.parametrize(
    "bad",
    [
        {"evidence_refs": []},  # no summary
        {"summary": "", "evidence_refs": []},  # empty summary
        {"summary": "x"},  # no evidence_refs
        {"summary": "x", "evidence_refs": "not-a-list"},
        {"summary": "x", "evidence_refs": [1, 2]},
        {"summary": "x", "evidence_refs": [], "reasoning": "leaked"},
        {"summary": "x", "evidence_refs": [], "confidence": 0.9},
    ],
)
def test_contract_violating_provider_output_is_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(IntentParsingError):
        synthesize(gemini_synthesizer(text=json.dumps(bad)))


def test_a_provider_api_error_becomes_an_upstream_failure() -> None:
    with pytest.raises(UpstreamServiceError):
        synthesize(gemini_synthesizer(error=genai_errors.APIError(503, {})))


@pytest.mark.parametrize(
    "error", [TimeoutError("slow"), ConnectionError("refused"), OSError("socket")]
)
def test_transport_failures_become_upstream_failures(error: Exception) -> None:
    with pytest.raises(UpstreamServiceError):
        synthesize(gemini_synthesizer(error=error))


def test_an_unknown_sdk_failure_is_not_leaked_verbatim() -> None:
    with pytest.raises(UpstreamServiceError) as exc:
        synthesize(gemini_synthesizer(error=RuntimeError("secret internal detail")))

    assert "secret internal detail" not in exc.value.message


def test_provider_failure_never_becomes_a_fabricated_answer() -> None:
    for error in (genai_errors.APIError(500, {}), TimeoutError("x")):
        with pytest.raises((UpstreamServiceError, IntentParsingError)):
            synthesize(gemini_synthesizer(error=error))


def test_missing_credentials_fail_before_any_request() -> None:
    synth = GeminiAnswerSynthesizer(settings=Settings(gemini_api_key=None))
    with pytest.raises(UpstreamServiceError):
        synthesize(synth)


# =========================================================================== #
# E. Gemini schema - reuses the Commit 4 strategy
# =========================================================================== #


def _sent_config() -> Any:
    capture: dict[str, Any] = {}
    synth = GeminiAnswerSynthesizer(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=FakeGeminiClient(text=json.dumps(VALID_ANSWER), capture=capture),
    )
    synthesize(synth)
    return capture


def test_the_schema_translates_through_the_installed_sdk() -> None:
    assert t_schema(None, _sent_config()["config"].response_schema) is not None


def test_the_schema_is_an_sdk_schema_not_the_pydantic_model() -> None:
    schema = _sent_config()["config"].response_schema

    assert isinstance(schema, genai_types.Schema)
    assert schema is not DraftAnswer


def test_the_schema_carries_no_discriminator_or_oneof() -> None:
    """The Commit 4 defect must not be reintroduced."""

    wire = json.dumps(
        t_schema(None, _sent_config()["config"].response_schema).model_dump(
            exclude_none=True, mode="json", by_alias=True
        )
    )
    assert "discriminator" not in wire
    assert "oneOf" not in wire


def test_the_schema_describes_exactly_the_draft_answer_fields() -> None:
    wire = t_schema(None, _sent_config()["config"].response_schema).model_dump(
        exclude_none=True, mode="json", by_alias=True
    )
    assert set(wire["properties"]) == {"summary", "evidence_refs"}
    assert set(wire["required"]) == {"summary", "evidence_refs"}


def test_the_provider_uses_the_public_sdk_conversion_api() -> None:
    source = pathlib.Path(gemini_mod.__file__).read_text()

    assert "Schema.from_json_schema" in source
    assert "_transformers" not in source


def test_decoding_is_deterministic() -> None:
    assert _sent_config()["config"].temperature == 0.0


# =========================================================================== #
# F. Prompt - evidence-bound, no reasoning requested
# =========================================================================== #


def test_the_question_and_evidence_both_reach_the_model() -> None:
    contents = str(_sent_config()["contents"])

    assert "What is the NDWI here?" in contents
    assert "ndwi.ndwi_mean" in contents  # the citable id
    assert "0.2777" in contents  # the value it may quote


def test_limitations_are_given_to_the_model() -> None:
    contents = str(_sent_config()["contents"])
    assert "Equal AOI coverage is NOT established." in contents


def test_the_instruction_constrains_the_model_to_the_evidence() -> None:
    instruction = _sent_config()["config"].system_instruction.lower()

    assert "evidence" in instruction
    assert "cite" in instruction or "evidence_refs" in instruction


def test_the_instruction_never_asks_for_reasoning() -> None:
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
        assert banned not in source


# =========================================================================== #
# G. The synthesizer neither executes nor invents
# =========================================================================== #


def test_the_synthesizer_does_not_mutate_the_evidence() -> None:
    evidence = make_evidence()
    before = evidence.model_dump(mode="json")

    synthesize(gemini_synthesizer(text=json.dumps(VALID_ANSWER)), evidence=evidence)

    assert evidence.model_dump(mode="json") == before


def test_the_synthesizer_manufactures_no_measurements() -> None:
    """Refs are references; the synthesizer never creates evidence."""

    for module in (synthesizer_mod, gemini_mod):
        source = pathlib.Path(module.__file__).read_text()
        assert "Measurement(" not in source
        assert "EvidenceItem(" not in source


@pytest.mark.parametrize("module", [synthesizer_mod, gemini_mod])
def test_synthesizer_modules_reference_no_service_or_executor(module: Any) -> None:
    source = pathlib.Path(module.__file__).read_text()

    for banned in (
        "QueryExecutionService",
        "AnalysisService",
        "ImageryService",
        "AgentExecutor",
        "read_band",
        "rasterio",
    ):
        assert banned not in source


@pytest.mark.parametrize("module", [synthesizer_mod, gemini_mod])
def test_synthesizer_modules_use_no_dynamic_execution(module: Any) -> None:
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
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr":
                assert node.args and isinstance(node.args[1], ast.Constant)
                continue
            assert node.func.id not in forbidden


def test_synthesizer_instances_hold_no_service_handles() -> None:
    for instance in (MockAnswerSynthesizer(), gemini_synthesizer(text="{}")):
        for banned in ("_query", "_analysis", "_imagery", "_executor", "_raster"):
            assert not hasattr(instance, banned)


def test_a_synthesizer_exposes_only_the_synthesize_operation() -> None:
    public = {
        name
        for name in dir(MockAnswerSynthesizer)
        if not name.startswith("_") and callable(getattr(MockAnswerSynthesizer, name))
    }
    assert public == {"synthesize"}


# =========================================================================== #
# H. Grounding stays separate
# =========================================================================== #


def test_the_synthesizer_does_not_validate_its_own_output() -> None:
    """A generator marking its own homework would be worthless.

    Asserted over imports and calls, not prose: the module docstring explains
    at length that ``validate_answer`` lives elsewhere, and must be free to say
    so by name.
    """

    for module in (synthesizer_mod, gemini_mod):
        tree = ast.parse(pathlib.Path(module.__file__).read_text())

        imported: set[str] = set()
        called: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)

        for banned in ("validate_answer", "AnswerValidation", "FORBIDDEN_PHRASES"):
            assert banned not in imported, f"{module.__name__} imports {banned}"
            assert banned not in called, f"{module.__name__} calls {banned}"


def test_grounding_still_runs_independently_over_a_synthesized_answer() -> None:
    from app.services.agent.grounding import validate_answer

    evidence = make_evidence()
    answer = synthesize(gemini_synthesizer(text=json.dumps(VALID_ANSWER)), evidence=evidence)
    result = validate_answer(answer, evidence)

    assert result.numeric_grounding == "pass"
    assert result.evidence_refs == "pass"
    assert result.forbidden_terms == "pass"


def test_grounding_catches_a_synthesizer_that_invents_a_number() -> None:
    from app.services.agent.grounding import validate_answer

    evidence = make_evidence()
    fabricated = json.dumps(
        {"summary": "The mean NDWI was 0.99 index.", "evidence_refs": ["ndwi.ndwi_mean"]}
    )
    answer = synthesize(gemini_synthesizer(text=fabricated), evidence=evidence)

    # The synthesizer happily returns it - and grounding is what stops it.
    assert validate_answer(answer, evidence).numeric_grounding == "fail"


# =========================================================================== #
# I. SDK isolation
# =========================================================================== #


def _import_roots(path: pathlib.Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("module_name", AGENT_CORE_MODULES)
def test_agent_core_modules_import_no_provider_sdk(module_name: str) -> None:
    package = pathlib.Path(synthesizer_mod.__file__).parent
    roots = _import_roots(package / module_name)

    for forbidden in ("google", "genai"):
        assert forbidden not in roots


def test_the_sdk_is_imported_only_under_providers() -> None:
    package = pathlib.Path(synthesizer_mod.__file__).parent
    importers = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if {"google", "genai"} & _import_roots(path)
    ]
    assert importers == ["providers/gemini.py"]


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
