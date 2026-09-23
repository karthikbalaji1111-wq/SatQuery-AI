"""The Anthropic (Claude) provider: selection, isolation and honesty.

A third inference backend must not widen anything. These tests pin the four
properties that make that true:

* the SDK reaches exactly one file, and that file really does import it;
* every role a run needs is Anthropic's when Anthropic is selected, and the
  credential check fires for the whole bundle rather than only the visual step;
* the request carries the image and the question and nothing else - no
  measurement, no georeferencing, no scene id;
* what comes back is validated by the existing contracts, and a refusal, a
  truncation or an empty response is reported rather than parsed.

No test here contacts Anthropic. Every request goes through a hand-written
recording fake, in keeping with the suite's convention of no ``unittest.mock``.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import pathlib
from typing import Any

import anthropic
import httpx2
import pytest
from app.core.config import (
    AI_PROVIDER_FIELDS,
    SUPPORTED_AI_PROVIDERS,
    Settings,
)
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.services.agent.planner import AgentPlanner
from app.services.agent.providers import anthropic as anthropic_provider
from app.services.agent.providers.anthropic import (
    AnthropicAgentPlanner,
    AnthropicAnswerSynthesizer,
    AnthropicIntentParser,
    AnthropicVisualAnalyst,
    _extract_json,
    _message_text,
)
from app.services.agent.providers.catalog import find_model, models_for
from app.services.agent.providers.factory import (
    get_agent_providers,
    get_intent_parser,
    get_visual_analyst,
)
from app.services.agent.schemas import AgentEvidence, EvidenceItem
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.agent.visual import VisualAnalyst
from app.services.ai.ports import IntentParser

PNG = b"\x89PNG\r\n\x1a\nfake-bytes-for-one-scene"
QUESTION = "Is there visible water in this image?"

#: Distinctive enough that a substring hit cannot be a coincidence.
SENTINEL = "sk-ant-SENTINEL-must-never-be-disclosed-0123456789"

PLAN_JSON = (
    '{"steps": [{"tool": "execute_query", "intent": {'
    '"location_query": "Chennai", "temporal_mode": "single", '
    '"time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}], '
    '"modalities": ["sentinel-2-optical"], "task": "visualize"}}]}'
)


def settings(**overrides: object) -> Settings:
    """Anthropic credentialed unless a test says otherwise.

    ``_env_file=None`` keeps these hermetic: without it a developer's real
    ``.env`` would supply a key and the missing-credential tests would pass for
    the wrong reason on one machine and fail on another.
    """

    base: dict[str, object] = {"ANTHROPIC_API_KEY": "anthropic-test-key"}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Recording fakes - the only thing standing in for the SDK
# --------------------------------------------------------------------------- #


class FakeBlock:
    """One content block, shaped like the SDK's but owned by this file."""

    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        if text is not None:
            self.text = text


class FakeMessage:
    def __init__(
        self, blocks: list[FakeBlock], stop_reason: str = "end_turn"
    ) -> None:
        self.content = blocks
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    """Stands in for ``anthropic.AsyncAnthropic`` and records every request."""

    def __init__(self, *outcomes: Any) -> None:
        if not outcomes:
            outcomes = (FakeMessage([FakeBlock("text", "{}")]),)
        self.messages = FakeMessages(list(outcomes))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


def text_message(text: str) -> FakeMessage:
    return FakeMessage([FakeBlock("text", text)])


def status_error(status: int) -> anthropic.APIStatusError:
    response = httpx2.Response(
        status,
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return anthropic.APIStatusError("upstream", response=response, body=None)


# --------------------------------------------------------------------------- #
# SDK isolation
# --------------------------------------------------------------------------- #


def _import_roots(path: pathlib.Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def _agent_package() -> pathlib.Path:
    return pathlib.Path(anthropic_provider.__file__).parent.parent


def test_the_anthropic_sdk_is_imported_only_by_its_own_provider() -> None:
    """One file, exactly as ``google-genai`` reaches only ``gemini.py``."""

    importers = [
        path.relative_to(_agent_package()).as_posix()
        for path in _agent_package().rglob("*.py")
        if "anthropic" in _import_roots(path)
    ]
    assert importers == ["providers/anthropic.py"]


def test_the_anthropic_provider_does_import_the_sdk() -> None:
    """The boundary is real, not achieved by having no provider at all."""

    assert "anthropic" in _import_roots(
        pathlib.Path(anthropic_provider.__file__)
    )


def test_the_anthropic_provider_imports_no_other_vendor_sdk() -> None:
    source = pathlib.Path(anthropic_provider.__file__).read_text()
    for forbidden in ("google.genai", "from google", "import openai"):
        assert forbidden not in source


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_every_supported_provider_has_a_field_table_entry() -> None:
    """A provider can never be half-wired into the credential lookup."""

    assert set(AI_PROVIDER_FIELDS) == set(SUPPORTED_AI_PROVIDERS)


def test_every_field_named_in_the_table_exists_on_settings() -> None:
    configured = settings()
    for provider, fields in AI_PROVIDER_FIELDS.items():
        if fields.api_key is None:
            # A keyless provider (the local one) is reached at an endpoint
            # instead, and must never report a credential it does not have.
            assert fields.endpoint is not None, provider
            assert hasattr(configured, fields.endpoint), provider
            assert configured.api_key_for(provider) is None
        else:
            assert hasattr(configured, fields.api_key), provider
        assert isinstance(configured.model_for(provider), str)


def test_anthropic_is_a_supported_provider() -> None:
    assert "anthropic" in SUPPORTED_AI_PROVIDERS
    assert Settings(_env_file=None, AI_PROVIDER="anthropic").ai_provider == (  # type: ignore[arg-type]
        "anthropic"
    )


def test_the_default_model_is_a_catalogued_claude_model() -> None:
    default = settings().anthropic_model
    card = find_model("anthropic", default)
    assert card is not None
    assert card.is_available
    assert card.supports_image and card.supports_text


def test_every_catalogued_claude_model_can_serve_both_roles() -> None:
    """No text-only Claude entry, so no capability refusal to encode."""

    visual = {card.model_id for card in models_for("visual", "anthropic")}
    text = {card.model_id for card in models_for("text", "anthropic")}
    assert visual and visual == text


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_factory_returns_the_anthropic_visual_analyst() -> None:
    analyst = get_visual_analyst(settings=settings(AI_PROVIDER="anthropic"))
    assert isinstance(analyst, AnthropicVisualAnalyst)
    assert analyst.provider_name == "anthropic"


def test_selecting_anthropic_selects_all_three_roles() -> None:
    """The Phase 15 defect regression: not only the visual step is switched."""

    bundle = get_agent_providers(settings=settings(), provider="anthropic")
    assert bundle.provider == "anthropic"
    assert isinstance(bundle.planner, AnthropicAgentPlanner)
    assert isinstance(bundle.synthesizer, AnthropicAnswerSynthesizer)
    assert isinstance(bundle.planner, AgentPlanner)
    assert isinstance(bundle.synthesizer, AnswerSynthesizer)
    assert isinstance(bundle.visual_analyst, VisualAnalyst)
    assert bundle.visual_analyst.provider_name == "anthropic"


def test_the_intent_parser_is_anthropics_too() -> None:
    parser = get_intent_parser(settings=settings(AI_PROVIDER="anthropic"))
    assert isinstance(parser, AnthropicIntentParser)
    assert isinstance(parser, IntentParser)


def test_a_per_run_override_selects_anthropic_over_the_configured_default() -> None:
    configured = settings(AI_PROVIDER="gemini", GEMINI_API_KEY="g")
    bundle = get_agent_providers(settings=configured, provider="anthropic")
    assert bundle.provider == "anthropic"


def test_a_missing_anthropic_credential_fails_the_whole_bundle() -> None:
    with pytest.raises(UpstreamServiceError, match="ANTHROPIC_API_KEY"):
        get_agent_providers(
            settings=Settings(_env_file=None, GEMINI_API_KEY="g"),  # type: ignore[arg-type]
            provider="anthropic",
        )


def test_a_missing_anthropic_credential_never_yields_another_provider() -> None:
    """No silent fallback: an unattributable run is worse than a failed one."""

    with pytest.raises(UpstreamServiceError, match="ANTHROPIC_API_KEY"):
        get_visual_analyst(
            settings=Settings(_env_file=None, NVIDIA_API_KEY="n"),  # type: ignore[arg-type]
            provider="anthropic",
        )


def test_an_uncatalogued_model_is_allowed_through() -> None:
    """The catalog is a curated aid, not an allowlist an operator must wait on."""

    bundle = get_agent_providers(
        settings=settings(ANTHROPIC_MODEL="claude-not-yet-catalogued"),
        provider="anthropic",
    )
    assert bundle.model == "claude-not-yet-catalogued"


def test_a_model_from_another_provider_is_refused() -> None:
    with pytest.raises(UpstreamServiceError, match="not in the catalog"):
        get_agent_providers(
            settings=settings(),
            provider="anthropic",
            model="gemini-3.6-flash",
        )


def test_the_per_run_model_does_not_leak_into_the_next_run() -> None:
    configured = settings()
    get_agent_providers(
        settings=configured, provider="anthropic", model="claude-haiku-4-5"
    )
    assert configured.anthropic_model == "claude-opus-5"


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #


def test_thinking_blocks_are_never_read() -> None:
    """No reasoning is parsed, stored, returned or rendered."""

    message = FakeMessage(
        [
            FakeBlock("thinking", "internal deliberation that must not surface"),
            FakeBlock("text", "the answer"),
        ]
    )
    assert _message_text(message) == "the answer"


def test_a_refusal_is_reported_as_a_provider_failure() -> None:
    message = FakeMessage([FakeBlock("text", "I can't help with that.")],
                          stop_reason="refusal")
    with pytest.raises(UpstreamServiceError, match="declined"):
        _message_text(message)


def test_a_truncated_response_is_refused_not_parsed() -> None:
    """Blaming the model for a budget this adapter set would misreport it."""

    message = FakeMessage([FakeBlock("text", '{"steps": [')],
                          stop_reason="max_tokens")
    with pytest.raises(IntentParsingError, match="cut off"):
        _message_text(message)


def test_an_empty_response_is_refused() -> None:
    with pytest.raises(IntentParsingError):
        _message_text(FakeMessage([FakeBlock("text", "   ")]))


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the plan:\n{"a": 1}\nHope that helps.',
    ],
)
def test_json_is_located_in_the_shapes_a_model_actually_returns(raw: str) -> None:
    assert _extract_json(raw) == '{"a": 1}'


def test_malformed_json_is_located_but_never_repaired() -> None:
    """Surrounding whitespace goes; a missing brace is left missing."""

    assert _extract_json('  {"a":  ') == '{"a":'
    assert _extract_json('{"a": [1, 2') == '{"a": [1, 2'


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def test_a_valid_plan_is_parsed_through_the_closed_union() -> None:
    client = FakeClient(text_message(PLAN_JSON))
    planner = AnthropicAgentPlanner(settings=settings(), client=client)

    plan = asyncio.run(planner.plan("Show me Chennai in January 2024"))

    assert [step.tool for step in plan.steps] == ["execute_query"]


def test_an_unrecognised_tool_is_refused_before_the_executor_sees_it() -> None:
    client = FakeClient(
        text_message('{"steps": [{"tool": "run_shell", "command": "rm -rf /"}]}')
    )
    planner = AnthropicAgentPlanner(settings=settings(), client=client)

    with pytest.raises(IntentParsingError, match="did not match"):
        asyncio.run(planner.plan("anything"))


def test_prose_around_the_plan_does_not_defeat_it() -> None:
    client = FakeClient(text_message(f"Here you go:\n```json\n{PLAN_JSON}\n```"))
    planner = AnthropicAgentPlanner(settings=settings(), client=client)

    assert asyncio.run(planner.plan("q")).steps


def test_the_planner_retries_a_retryable_status_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(anthropic_provider, "_PLAN_BACKOFF_SECONDS", 0.0)
    client = FakeClient(status_error(529), text_message(PLAN_JSON))
    planner = AnthropicAgentPlanner(settings=settings(), client=client)

    assert asyncio.run(planner.plan("q")).steps
    assert len(client.calls) == 2


def test_the_planner_does_not_retry_a_request_that_is_its_own_fault() -> None:
    """Repeating a 400 would only repeat the mistake."""

    client = FakeClient(status_error(400))
    planner = AnthropicAgentPlanner(settings=settings(), client=client)

    with pytest.raises(UpstreamServiceError):
        asyncio.run(planner.plan("q"))
    assert len(client.calls) == 1


def test_the_planner_gives_up_and_raises_the_real_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(anthropic_provider, "_PLAN_BACKOFF_SECONDS", 0.0)
    client = FakeClient(text_message("not json at all"))
    planner = AnthropicAgentPlanner(settings=settings(), client=client)

    with pytest.raises(IntentParsingError):
        asyncio.run(planner.plan("q"))
    assert len(client.calls) == anthropic_provider._PLAN_ATTEMPTS


# --------------------------------------------------------------------------- #
# Synthesis and intent parsing
# --------------------------------------------------------------------------- #


def _evidence() -> AgentEvidence:
    return AgentEvidence(
        items=[
            EvidenceItem(
                id="execution.scene_count",
                source="execution",
                text="1 scene was selected.",
            )
        ]
    )


def test_a_draft_answer_is_parsed_from_the_response() -> None:
    client = FakeClient(
        text_message(
            '{"summary": "One scene was selected.", '
            '"evidence_refs": ["execution.scene_count"]}'
        )
    )
    synthesizer = AnthropicAnswerSynthesizer(settings=settings(), client=client)

    answer = asyncio.run(synthesizer.synthesize("q", _evidence()))

    assert answer.evidence_refs == ["execution.scene_count"]


def test_an_unparseable_answer_stays_a_failure() -> None:
    client = FakeClient(text_message("I could not find anything."))
    synthesizer = AnthropicAnswerSynthesizer(settings=settings(), client=client)

    with pytest.raises(IntentParsingError):
        asyncio.run(synthesizer.synthesize("q", _evidence()))


def test_an_intent_is_validated_through_the_domain_schema() -> None:
    client = FakeClient(
        text_message(
            '{"location_query": "Chennai", "temporal_mode": "single", '
            '"time_windows": [{"start_date": "2024-01-01", '
            '"end_date": "2024-01-31"}], '
            '"modalities": ["sentinel-2-optical"], "task": "visualize"}'
        )
    )
    parser = AnthropicIntentParser(settings=settings(), client=client)

    intent = asyncio.run(parser.parse_intent("Show me Chennai in January 2024"))

    assert intent.location_query == "Chennai"


# --------------------------------------------------------------------------- #
# Visual observation
# --------------------------------------------------------------------------- #


def _observe(client: FakeClient, media_type: str = "image/png"):
    analyst = AnthropicVisualAnalyst(settings=settings(), client=client)
    return asyncio.run(
        analyst.observe(question=QUESTION, image=PNG, media_type=media_type)
    )


def test_the_observation_is_returned_verbatim() -> None:
    client = FakeClient(text_message("Water is visible along the shoreline."))
    assert _observe(client).answer == "Water is visible along the shoreline."


def test_exactly_one_image_is_sent_and_it_is_the_bytes_handed_in() -> None:
    client = FakeClient(text_message("Something is visible."))
    _observe(client)

    content = client.calls[0]["messages"][0]["content"]
    images = [part for part in content if part["type"] == "image"]
    assert len(images) == 1
    assert images[0]["source"]["data"] == base64.standard_b64encode(PNG).decode(
        "ascii"
    )
    assert images[0]["source"]["media_type"] == "image/png"


def test_the_visual_request_carries_no_measurement_or_georeferencing() -> None:
    """An observation told the answer is not independent of it."""

    client = FakeClient(text_message("Something is visible."))
    _observe(client)

    sent = repr(client.calls[0])
    for forbidden in ("ndwi", "EPSG", "bbox", "transform", "scene_id", "affine"):
        assert forbidden.lower() not in sent.lower()


def test_an_unsupported_media_type_is_refused_before_any_request() -> None:
    client = FakeClient(text_message("unused"))
    with pytest.raises(UpstreamServiceError, match="cannot accept"):
        _observe(client, media_type="image/tiff")
    assert client.calls == []


def test_an_over_long_observation_is_refused_rather_than_truncated() -> None:
    """Truncating would silently alter what the model said."""

    client = FakeClient(text_message("x" * 4001))
    with pytest.raises(IntentParsingError, match="longer than the contract"):
        _observe(client)


# --------------------------------------------------------------------------- #
# Request shape - each of these is a fact about the current models
# --------------------------------------------------------------------------- #


def test_no_sampling_parameter_is_sent() -> None:
    """``temperature``/``top_p``/``top_k`` return 400 on the current models."""

    client = FakeClient(text_message(PLAN_JSON))
    asyncio.run(
        AnthropicAgentPlanner(settings=settings(), client=client).plan("q")
    )

    sent = client.calls[0]
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in sent


def test_no_thinking_or_effort_configuration_is_sent() -> None:
    """Keeps the adapter compatible with any model id an operator configures."""

    client = FakeClient(text_message(PLAN_JSON))
    asyncio.run(
        AnthropicAgentPlanner(settings=settings(), client=client).plan("q")
    )

    sent = client.calls[0]
    assert "thinking" not in sent
    assert "output_config" not in sent


def test_no_server_side_fallback_is_requested() -> None:
    """A silently substituted model would break attribution."""

    client = FakeClient(text_message(PLAN_JSON))
    asyncio.run(
        AnthropicAgentPlanner(settings=settings(), client=client).plan("q")
    )

    sent = client.calls[0]
    assert "fallbacks" not in sent
    assert "betas" not in sent


def test_the_configured_model_is_the_one_requested() -> None:
    client = FakeClient(text_message(PLAN_JSON))
    asyncio.run(
        AnthropicAgentPlanner(
            settings=settings(ANTHROPIC_MODEL="claude-sonnet-5"), client=client
        ).plan("q")
    )
    assert client.calls[0]["model"] == "claude-sonnet-5"


# --------------------------------------------------------------------------- #
# The credential
# --------------------------------------------------------------------------- #


def test_the_key_never_appears_in_a_transport_failure() -> None:
    client = FakeClient(
        anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        )
    )
    planner = AnthropicAgentPlanner(
        settings=settings(ANTHROPIC_API_KEY=SENTINEL), client=client
    )

    with pytest.raises(UpstreamServiceError) as caught:
        asyncio.run(planner.plan("q"))
    assert SENTINEL not in str(caught.value)


def test_the_key_never_appears_in_an_upstream_status_failure() -> None:
    client = FakeClient(status_error(401))
    parser = AnthropicIntentParser(
        settings=settings(ANTHROPIC_API_KEY=SENTINEL), client=client
    )

    with pytest.raises(UpstreamServiceError) as caught:
        asyncio.run(parser.parse_intent("q"))
    assert SENTINEL not in str(caught.value)


def test_a_timeout_is_reported_as_a_timeout() -> None:
    client = FakeClient(
        anthropic.APITimeoutError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        )
    )
    synthesizer = AnthropicAnswerSynthesizer(settings=settings(), client=client)

    with pytest.raises(UpstreamServiceError, match="timed out"):
        asyncio.run(synthesizer.synthesize("q", _evidence()))


def test_roles_construct_without_a_credential() -> None:
    """A deployment that never selects Anthropic must never need a key."""

    bare = Settings(_env_file=None)  # type: ignore[arg-type]
    planner = AnthropicAgentPlanner(settings=bare)

    with pytest.raises(UpstreamServiceError, match="ANTHROPIC_API_KEY"):
        asyncio.run(planner.plan("q"))
