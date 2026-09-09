"""Provider interchangeability: two inference backends, one pipeline.

The visual path is the only place a vision-language model touches SatQuery, and
these tests pin the properties that make swapping the backend safe:

* the provider is *selected*, never guessed, and never silently substituted;
* a missing credential is a clear configuration failure, not a fallback;
* the image and the question reach the provider unaltered, and nothing that
  could masquerade as a measurement travels with them;
* a provider failure stays a failure - no observation is invented to fill it.

Following the project convention: hand-written recording fakes, no
``unittest.mock``, and nothing here contacts a real provider.
"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
from app.core.config import SUPPORTED_AI_PROVIDERS, Settings
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.services.agent.grounding import DraftAnswer
from app.services.agent.planner import AgentPlanner
from app.services.agent.providers.catalog import (
    MODEL_CATALOG,
    find_model,
    models_for,
)
from app.services.agent.providers.factory import (
    get_agent_providers,
    get_intent_parser,
    get_visual_analyst,
)
from app.services.agent.providers.gemini import (
    GeminiAgentPlanner,
    GeminiAnswerSynthesizer,
    GeminiVisualAnalyst,
)
from app.services.agent.providers.nvidia import (
    NvidiaAgentPlanner,
    NvidiaAnswerSynthesizer,
    NvidiaIntentParser,
    NvidiaVisualAnalyst,
)
from app.services.agent.schemas import AgentEvidence, AgentPlan
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.agent.visual import VisualAnalyst, VisualAnswer
from app.services.ai.parser import GeminiIntentParser
from app.services.ai.ports import IntentParser

PNG = b"\x89PNG\r\n\x1a\nfake-bytes-for-one-scene"
QUESTION = "Is there visible water in this image?"


def settings(**overrides: object) -> Settings:
    """Settings with both providers credentialed unless a test says otherwise.

    ``_env_file=None`` keeps these hermetic: without it the developer's real
    ``.env`` supplies a GEMINI_API_KEY and the missing-credential tests pass
    for the wrong reason on one machine and fail on another.
    """

    base: dict[str, object] = {
        "GEMINI_API_KEY": "gemini-test-key",
        "NVIDIA_API_KEY": "nvidia-test-key",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_factory_returns_gemini_when_configured() -> None:
    analyst = get_visual_analyst(settings=settings(AI_PROVIDER="gemini"))
    assert isinstance(analyst, GeminiVisualAnalyst)
    assert analyst.provider_name == "gemini"


def test_factory_returns_nvidia_when_configured() -> None:
    analyst = get_visual_analyst(settings=settings(AI_PROVIDER="nvidia"))
    assert isinstance(analyst, NvidiaVisualAnalyst)
    assert analyst.provider_name == "nvidia"


def test_both_providers_satisfy_the_same_abstraction() -> None:
    """Interchangeable means the agent's type, not merely a similar shape."""

    for provider in sorted(SUPPORTED_AI_PROVIDERS):
        analyst = get_visual_analyst(settings=settings(AI_PROVIDER=provider))
        assert isinstance(analyst, VisualAnalyst)
        assert analyst.model_name


def test_per_run_override_beats_the_configured_default() -> None:
    configured = settings(AI_PROVIDER="gemini")
    assert isinstance(
        get_visual_analyst(settings=configured, provider="nvidia"),
        NvidiaVisualAnalyst,
    )
    assert isinstance(
        get_visual_analyst(settings=configured, provider="gemini"),
        GeminiVisualAnalyst,
    )


def test_invalid_provider_is_rejected_at_configuration_time() -> None:
    with pytest.raises(ValueError, match="AI_PROVIDER must be one of"):
        Settings(_env_file=None, AI_PROVIDER="openai")  # type: ignore[arg-type]


def test_unknown_per_run_provider_is_refused_not_defaulted() -> None:
    with pytest.raises(UpstreamServiceError, match="Unknown AI provider"):
        get_visual_analyst(settings=settings(), provider="openai")


def test_missing_nvidia_key_fails_clearly() -> None:
    with pytest.raises(UpstreamServiceError, match="NVIDIA_API_KEY"):
        get_visual_analyst(
            settings=Settings(_env_file=None, AI_PROVIDER="nvidia", GEMINI_API_KEY="g")  # type: ignore[arg-type]
        )


def test_missing_gemini_key_fails_clearly() -> None:
    with pytest.raises(UpstreamServiceError, match="GEMINI_API_KEY"):
        get_visual_analyst(
            settings=Settings(_env_file=None, AI_PROVIDER="gemini", NVIDIA_API_KEY="n")  # type: ignore[arg-type]
        )


def test_missing_key_never_falls_back_to_the_other_provider() -> None:
    """A silent substitution would make a result unattributable."""

    with pytest.raises(UpstreamServiceError):
        get_visual_analyst(
            settings=Settings(_env_file=None, AI_PROVIDER="nvidia", GEMINI_API_KEY="g")  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# The NVIDIA request
# --------------------------------------------------------------------------- #


class RecordingTransport(httpx.AsyncBaseTransport):
    """Captures the outgoing request and replays a canned completion."""

    def __init__(self, body: object, status_code: int = 200) -> None:
        self._body = body
        self._status = status_code
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self._status,
            json=self._body,
            request=request,
        )


def completion(text: str) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def observe_with(
    body: object, *, status_code: int = 200, **overrides: object
) -> tuple[VisualAnswer | Exception, RecordingTransport]:
    """Run one observation against a canned transport, following the project's
    ``asyncio.run`` convention for async services."""

    transport = RecordingTransport(body, status_code)

    async def run() -> VisualAnswer | Exception:
        async with httpx.AsyncClient(transport=transport) as client:
            analyst = NvidiaVisualAnalyst(
                settings=settings(**overrides), client=client
            )
            try:
                return await analyst.observe(
                    question=QUESTION, image=PNG, media_type="image/png"
                )
            except Exception as exc:  # returned so the caller can assert on it
                return exc

    return asyncio.run(run()), transport


def test_nvidia_calls_the_openai_compatible_endpoint() -> None:
    _, transport = observe_with(completion("Water is visible."))

    assert len(transport.requests) == 1
    url = str(transport.requests[0].url)
    assert url == "https://integrate.api.nvidia.com/v1/chat/completions"


def test_nvidia_honours_a_configured_base_url() -> None:
    _, transport = observe_with(
        completion("ok"), NVIDIA_BASE_URL="https://nim.internal/v1/"
    )
    assert str(transport.requests[0].url) == "https://nim.internal/v1/chat/completions"


def test_nvidia_sends_the_image_in_the_multimodal_format() -> None:
    """The exact content-part shape NVIDIA documents for NIM VLMs."""

    _, transport = observe_with(completion("Water is visible."))
    payload = json.loads(transport.requests[0].content)

    user = [m for m in payload["messages"] if m["role"] == "user"][0]
    parts = {part["type"] for part in user["content"]}
    assert parts == {"text", "image_url"}

    image_part = next(p for p in user["content"] if p["type"] == "image_url")
    expected = base64.b64encode(PNG).decode("ascii")
    assert image_part["image_url"]["url"] == f"data:image/png;base64,{expected}"


def test_nvidia_sends_the_exact_bytes_it_was_handed() -> None:
    """Never re-encoded, never re-fetched, never substituted."""

    _, transport = observe_with(completion("ok"))
    payload = json.loads(transport.requests[0].content)
    user = [m for m in payload["messages"] if m["role"] == "user"][0]
    image_part = next(p for p in user["content"] if p["type"] == "image_url")
    encoded = image_part["image_url"]["url"].split("base64,", 1)[1]
    assert base64.b64decode(encoded) == PNG


def test_nvidia_sends_the_question_verbatim() -> None:
    _, transport = observe_with(completion("ok"))
    payload = json.loads(transport.requests[0].content)
    user = [m for m in payload["messages"] if m["role"] == "user"][0]
    text = next(p for p in user["content"] if p["type"] == "text")["text"]
    assert QUESTION in text


def test_nvidia_leaks_no_measurement_or_georeferencing_into_the_prompt() -> None:
    """The observation must be independent of the deterministic evidence.

    A model told the NDWI would answer from that number rather than from the
    picture, which would destroy the whole point of setting the two against
    each other.
    """

    _, transport = observe_with(completion("ok"))
    sent = transport.requests[0].content.decode().lower()

    # Deterministic results and georeferencing. "sentinel" is deliberately NOT
    # in this list: the shared instruction names the sensor type as framing
    # ("one Sentinel-2 true-colour satellite image"), which is what the
    # production Gemini prompt has always said. Telling a model what kind of
    # picture it is looking at is not telling it the answer.
    for leaked in (
        "ndwi",
        "epsg",
        "crs",
        "affine",
        "transform",
        "bbox",
        "corners",
        "scene_id",
        "cloud_cover",
        "measurement",
        "valid_pixel",
    ):
        assert leaked not in sent, f"{leaked!r} must not reach the model"


def test_nvidia_uses_the_configured_model() -> None:
    _, transport = observe_with(
        completion("ok"), NVIDIA_MODEL="nvidia/some-other-vl"
    )
    payload = json.loads(transport.requests[0].content)
    assert payload["model"] == "nvidia/some-other-vl"


def test_nvidia_authorises_with_a_bearer_token() -> None:
    _, transport = observe_with(completion("ok"))
    assert transport.requests[0].headers["authorization"] == "Bearer nvidia-test-key"


# --------------------------------------------------------------------------- #
# The NVIDIA response
# --------------------------------------------------------------------------- #


def test_nvidia_returns_the_models_sentence_unedited() -> None:
    answer, _ = observe_with(
        completion("  A dark water body spans the eastern edge.  ")
    )
    assert isinstance(answer, VisualAnswer)
    assert answer.answer == "A dark water body spans the eastern edge."


def test_nvidia_accepts_a_content_part_list() -> None:
    """Some NIM models return the assistant turn as parts rather than a string."""

    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Water is visible."}],
                }
            }
        ]
    }
    answer, _ = observe_with(body)
    assert isinstance(answer, VisualAnswer)
    assert answer.answer == "Water is visible."


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"nope": True}]},
    ],
)
def test_nvidia_refuses_an_unusable_body(body: object) -> None:
    """Fail closed: an empty completion never becomes an empty observation."""

    result, _ = observe_with(body)
    assert isinstance(result, IntentParsingError)


def test_nvidia_refuses_an_over_long_observation() -> None:
    """Truncating would silently alter what the model said."""

    result, _ = observe_with(completion("x" * 4001))
    assert isinstance(result, IntentParsingError)


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_nvidia_api_error_fails_closed(status: int) -> None:
    """A provider failure stays a failure - no fabricated observation."""

    result, _ = observe_with({"error": "nope"}, status_code=status)
    assert isinstance(result, UpstreamServiceError)


def test_nvidia_error_message_does_not_leak_the_provider_body() -> None:
    """A provider error body can echo the request, which carries image bytes."""

    result, _ = observe_with(
        {"error": {"message": "secret-detail", "request": "..."}}, status_code=400
    )
    assert isinstance(result, UpstreamServiceError)
    assert "secret-detail" not in str(result)


def test_nvidia_rejects_an_unsupported_media_type() -> None:
    analyst = NvidiaVisualAnalyst(settings=settings())
    with pytest.raises(UpstreamServiceError, match="PNG or JPEG"):
        asyncio.run(
            analyst.observe(
                question=QUESTION, image=PNG, media_type="image/tiff"
            )
        )


def test_nvidia_without_a_key_never_sends_a_request() -> None:
    transport = RecordingTransport(completion("ok"))

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            analyst = NvidiaVisualAnalyst(
                settings=Settings(_env_file=None, AI_PROVIDER="nvidia"),  # type: ignore[arg-type]
                client=client,
            )
            await analyst.observe(
                question=QUESTION, image=PNG, media_type="image/png"
            )

    with pytest.raises(UpstreamServiceError, match="NVIDIA_API_KEY"):
        asyncio.run(run())
    assert transport.requests == []


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #


def test_each_provider_attributes_itself_specifically() -> None:
    """An observation whose author cannot be named is not publishable."""

    gemini = get_visual_analyst(settings=settings(AI_PROVIDER="gemini"))
    nvidia = get_visual_analyst(settings=settings(AI_PROVIDER="nvidia"))

    assert (gemini.provider_name, nvidia.provider_name) == ("gemini", "nvidia")
    assert gemini.model_name != nvidia.model_name


def test_the_nvidia_provider_imports_no_model_sdk() -> None:
    """The NVIDIA path speaks plain HTTP; no vendor SDK enters the package."""

    import pathlib

    source = pathlib.Path(
        "app/services/agent/providers/nvidia.py"
    ).read_text()
    for forbidden in ("google.genai", "from google", "import openai"):
        assert forbidden not in source


# --------------------------------------------------------------------------- #
# The model catalog
# --------------------------------------------------------------------------- #
#
# The catalog exists because "which provider" does not determine whether a run
# can happen: SatQuery's visual step sends an image, and most hosted models
# cannot accept one. These tests pin the properties that keep an incompatible
# choice from reaching the provider at all.


def test_catalogued_model_ids_are_plausible_api_ids() -> None:
    """Ids are what goes on the wire, so they carry no display formatting."""

    for card in MODEL_CATALOG:
        assert card.model_id == card.model_id.strip()
        assert " " not in card.model_id
        assert card.model_id.lower() == card.model_id
        assert card.display_name != card.model_id


def test_every_card_records_where_its_capabilities_were_checked() -> None:
    """A capability claim without a source is a guess."""

    for card in MODEL_CATALOG:
        assert len(card.verified_against) > 20


def test_multimodal_cards_support_image_and_text_cards_do_not() -> None:
    for card in MODEL_CATALOG:
        assert card.supports_image == (card.modality == "multimodal")


def test_the_catalog_offers_more_than_one_image_capable_nvidia_model() -> None:
    """A single hardcoded model is what the catalog exists to replace."""

    nvidia_visual = models_for("visual", provider="nvidia")
    assert len(nvidia_visual) >= 2


def test_text_only_models_are_excluded_from_the_visual_role() -> None:
    visual_ids = {card.model_id for card in models_for("visual")}
    text_only = [card for card in MODEL_CATALOG if not card.supports_image]

    assert text_only, "the catalog should include text-only models"
    for card in text_only:
        assert card.model_id not in visual_ids


def test_text_only_models_remain_selectable_for_a_text_role() -> None:
    """Excluded from vision is not excluded from everything."""

    text_ids = {card.model_id for card in models_for("text")}
    for card in MODEL_CATALOG:
        if not card.supports_image:
            assert card.model_id in text_ids


def test_no_embedding_or_speciality_models_are_offered() -> None:
    """Only models that can fill a SatQuery role belong in the catalog."""

    for card in MODEL_CATALOG:
        assert card.supports_text or card.supports_image
        for excluded in ("embed", "rerank", "guard", "translate", "tts", "asr"):
            assert excluded not in card.model_id


def test_find_model_is_provider_scoped() -> None:
    assert find_model("nvidia", "nvidia/nemotron-nano-12b-v2-vl") is not None
    # Right id, wrong provider - never a cross-provider match.
    assert find_model("gemini", "nvidia/nemotron-nano-12b-v2-vl") is None


# --------------------------------------------------------------------------- #
# Capability enforcement at resolution time
# --------------------------------------------------------------------------- #


def test_a_text_only_model_is_refused_for_visual_analysis() -> None:
    """Refused BEFORE any request: an image is never sent to a blind model."""

    with pytest.raises(UpstreamServiceError, match="does not support visual"):
        get_visual_analyst(
            settings=settings(AI_PROVIDER="nvidia"),
            model="nvidia/nemotron-3-super-120b-a12b",
        )


def test_an_image_capable_model_override_is_accepted() -> None:
    analyst = get_visual_analyst(
        settings=settings(AI_PROVIDER="nvidia"),
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    )
    assert analyst.model_name == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


def test_an_uncatalogued_model_override_is_refused() -> None:
    with pytest.raises(UpstreamServiceError, match="not in the catalog"):
        get_visual_analyst(
            settings=settings(AI_PROVIDER="nvidia"), model="nvidia/does-not-exist"
        )


def test_a_model_override_does_not_leak_into_other_runs() -> None:
    """Settings are cached process-wide; one run's choice must not persist."""

    base = settings(AI_PROVIDER="nvidia")
    original = base.nvidia_model

    get_visual_analyst(
        settings=base, model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    )

    assert base.nvidia_model == original
    assert get_visual_analyst(settings=base).model_name == original


def test_a_configured_text_only_model_is_refused_for_the_visual_step() -> None:
    """The same rule applies to the environment default, not just overrides."""

    with pytest.raises(UpstreamServiceError, match="does not support visual"):
        get_visual_analyst(
            settings=settings(
                AI_PROVIDER="nvidia",
                NVIDIA_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b",
            )
        )


def test_an_uncatalogued_configured_model_is_allowed_through() -> None:
    """The catalog aids selection; it is not an allowlist.

    An operator configuring a model newer than this file should not be blocked
    by the file being out of date - only a model we KNOW cannot see is refused.
    """

    analyst = get_visual_analyst(
        settings=settings(AI_PROVIDER="nvidia", NVIDIA_MODEL="nvidia/brand-new-vl")
    )
    assert analyst.model_name == "nvidia/brand-new-vl"


# --------------------------------------------------------------------------- #
# The whole AI path, not just the visual step
# --------------------------------------------------------------------------- #
#
# Selecting a provider must swap planning, seeing AND describing together.
# These tests exist because the earlier implementation routed only the visual
# analyst: choosing NVIDIA still planned with Gemini, so an NVIDIA run died at
# planning and never reached NVIDIA at all.


def _bundle(provider: str, **overrides: object):
    return get_agent_providers(settings=settings(**overrides), provider=provider)


def test_selecting_gemini_selects_the_gemini_planner_and_synthesizer() -> None:
    bundle = _bundle("gemini")
    assert isinstance(bundle.planner, GeminiAgentPlanner)
    assert isinstance(bundle.synthesizer, GeminiAnswerSynthesizer)


def test_selecting_nvidia_selects_the_nvidia_planner_and_synthesizer() -> None:
    bundle = _bundle("nvidia")
    assert isinstance(bundle.planner, NvidiaAgentPlanner)
    assert isinstance(bundle.synthesizer, NvidiaAnswerSynthesizer)


def test_no_gemini_class_survives_an_nvidia_selection() -> None:
    """No half-switched pipeline: nothing Gemini-shaped may remain."""

    bundle = _bundle("nvidia")
    for role in (bundle.planner, bundle.synthesizer):
        assert "Gemini" not in type(role).__name__
    assert bundle.visual_analyst.provider_name == "nvidia"


def test_no_nvidia_class_survives_a_gemini_selection() -> None:
    bundle = _bundle("gemini")
    for role in (bundle.planner, bundle.synthesizer):
        assert "Nvidia" not in type(role).__name__
    assert bundle.visual_analyst.provider_name == "gemini"


def test_every_role_reports_the_same_provider() -> None:
    for provider in sorted(SUPPORTED_AI_PROVIDERS):
        bundle = _bundle(provider)
        assert bundle.provider == provider
        assert bundle.visual_analyst.provider_name == provider


def test_the_bundle_exposes_the_existing_abstractions_only() -> None:
    """The orchestrator sees three ABCs, never a provider-shaped object."""

    for provider in sorted(SUPPORTED_AI_PROVIDERS):
        bundle = _bundle(provider)
        assert isinstance(bundle.planner, AgentPlanner)
        assert isinstance(bundle.visual_analyst, VisualAnalyst)
        assert isinstance(bundle.synthesizer, AnswerSynthesizer)


def test_missing_credential_fails_the_whole_bundle_not_just_the_visual_role() -> None:
    with pytest.raises(UpstreamServiceError, match="NVIDIA_API_KEY"):
        get_agent_providers(
            settings=Settings(_env_file=None, GEMINI_API_KEY="g"),  # type: ignore[arg-type]
            provider="nvidia",
        )


def test_a_missing_credential_never_yields_the_other_provider() -> None:
    """No silent fallback: an unattributable run is worse than a failed one."""

    with pytest.raises(UpstreamServiceError):
        get_agent_providers(
            settings=Settings(_env_file=None, NVIDIA_API_KEY="n"),  # type: ignore[arg-type]
            provider="gemini",
        )


def test_a_text_only_model_still_serves_the_text_roles() -> None:
    """A plan that never looks at an image is legitimate work for such a model."""

    bundle = get_agent_providers(
        settings=settings(AI_PROVIDER="nvidia"),
        provider="nvidia",
        model="nvidia/nemotron-3-super-120b-a12b",
    )
    assert isinstance(bundle.planner, NvidiaAgentPlanner)
    assert isinstance(bundle.synthesizer, NvidiaAnswerSynthesizer)


def test_a_text_only_model_is_refused_when_the_visual_step_actually_runs() -> None:
    """Deferred, but not waived - and no image is sent."""

    bundle = get_agent_providers(
        settings=settings(AI_PROVIDER="nvidia"),
        provider="nvidia",
        model="nvidia/nemotron-3-super-120b-a12b",
    )
    with pytest.raises(UpstreamServiceError, match="does not support visual"):
        asyncio.run(
            bundle.visual_analyst.observe(
                question=QUESTION, image=PNG, media_type="image/png"
            )
        )


def test_a_multimodal_model_serves_every_role() -> None:
    bundle = get_agent_providers(
        settings=settings(AI_PROVIDER="nvidia"),
        provider="nvidia",
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    )
    assert isinstance(bundle.planner, NvidiaAgentPlanner)
    assert isinstance(bundle.synthesizer, NvidiaAnswerSynthesizer)
    assert bundle.visual_analyst.model_name.endswith("omni-30b-a3b-reasoning")


def test_both_providers_are_asked_the_same_thing() -> None:
    """One copy of each prompt, shared - so the providers stay comparable."""

    from app.services.agent import prompts
    from app.services.agent.providers import gemini as gemini_mod
    from app.services.agent.providers import nvidia as nvidia_mod

    assert gemini_mod._system_instruction is prompts._system_instruction
    assert nvidia_mod._system_instruction is prompts._system_instruction
    assert gemini_mod._SYNTHESIS_INSTRUCTION == prompts._SYNTHESIS_INSTRUCTION
    assert nvidia_mod._SYNTHESIS_INSTRUCTION == prompts._SYNTHESIS_INSTRUCTION
    assert gemini_mod._VISUAL_INSTRUCTION == prompts._VISUAL_INSTRUCTION
    assert nvidia_mod._VISUAL_INSTRUCTION == prompts._VISUAL_INSTRUCTION


# --------------------------------------------------------------------------- #
# The NVIDIA text roles
# --------------------------------------------------------------------------- #


def complete_with(text: str, *, status_code: int = 200):
    transport = RecordingTransport(completion(text), status_code)
    return transport


def run_planner(text: str, *, status_code: int = 200):
    transport = complete_with(text, status_code=status_code)

    async def go() -> object:
        async with httpx.AsyncClient(transport=transport) as client:
            planner = NvidiaAgentPlanner(settings=settings(), client=client)
            try:
                return await planner.plan("Any water at Marina Beach in Jan 2024?")
            except Exception as exc:
                return exc

    return asyncio.run(go()), transport


VALID_PLAN = json.dumps(
    {
        "steps": [
            {
                "tool": "execute_query",
                "intent": {
                    "location_query": "Marina Beach, Chennai",
                    "temporal_mode": "single",
                    "time_windows": [
                        {"start_date": "2024-01-01", "end_date": "2024-01-31"}
                    ],
                    "modalities": ["sentinel-2-optical"],
                    "task": "visualize",
                },
                "include_imagery": False,
                "max_cloud_cover": None,
            }
        ]
    }
)


def test_nvidia_planner_returns_the_normalised_plan_schema() -> None:
    plan, _ = run_planner(VALID_PLAN)
    assert isinstance(plan, AgentPlan)
    assert plan.steps[0].tool == "execute_query"


def test_nvidia_planner_tolerates_a_markdown_fence() -> None:
    """Models wrap JSON; the parser locates it and never repairs it."""

    plan, _ = run_planner(f"```json\n{VALID_PLAN}\n```")
    assert isinstance(plan, AgentPlan)


def test_nvidia_planner_normalizes_only_the_parameters_envelope() -> None:
    expected = json.loads(VALID_PLAN)
    expected["steps"].append({"tool": "ndwi_statistics"})
    wrapped = {"steps": [
        {"tool": step["tool"], "parameters": {k: v for k, v in step.items() if k != "tool"}}
        for step in expected["steps"]
    ]}
    plan, _ = run_planner(json.dumps(wrapped))
    assert isinstance(plan, AgentPlan)
    assert plan == AgentPlan.model_validate(expected)


@pytest.mark.parametrize("mutation", [
    "outer_extra", "inner_extra", "inner_tool", "unknown_tool",
    "wrong_wrapper", "non_object", "missing_intent", "invalid_intent", "too_many_steps",
])
def test_nvidia_parameters_envelope_preserves_strict_validation(mutation: str) -> None:
    original = json.loads(VALID_PLAN)["steps"][0]
    step = {"tool": original.pop("tool"), "parameters": original}
    body = {"steps": [step]}
    if mutation == "outer_extra":
        step["unexpected"] = True
    elif mutation == "inner_extra":
        original["unexpected"] = True
    elif mutation == "inner_tool":
        original["tool"] = "ndwi_statistics"
    elif mutation == "unknown_tool":
        step["tool"] = "unknown"
    elif mutation == "wrong_wrapper":
        step["arguments"] = step.pop("parameters")
    elif mutation == "non_object":
        step["parameters"] = []
    elif mutation == "missing_intent":
        original.pop("intent")
    elif mutation == "invalid_intent":
        original["intent"]["location_query"] = ""
    else:
        body["steps"] *= 20
    result, _ = run_planner(json.dumps(body))
    assert isinstance(result, IntentParsingError)


def test_nvidia_planner_rejects_an_unknown_tool() -> None:
    """The closed tool union is the authority for BOTH providers."""

    rogue = json.dumps({"steps": [{"tool": "rm_rf", "intent": {}}]})
    result, _ = run_planner(rogue)
    assert isinstance(result, IntentParsingError)


def test_nvidia_planner_rejects_unparseable_output() -> None:
    result, _ = run_planner("I think we should search for imagery.")
    assert isinstance(result, IntentParsingError)


def test_nvidia_planner_fails_closed_on_an_api_error() -> None:
    result, _ = run_planner(VALID_PLAN, status_code=429)
    assert isinstance(result, UpstreamServiceError)


def test_nvidia_planner_sends_no_image_part() -> None:
    """A text role sends text. Nothing multimodal leaks into planning."""

    _, transport = run_planner(VALID_PLAN)
    body = transport.requests[0].content.decode()
    assert "image_url" not in body
    assert "base64" not in body


def test_nvidia_synthesizer_returns_the_normalised_answer_schema() -> None:
    draft = json.dumps(
        {"summary": "The mean NDWI was 0.2777 index.", "evidence_refs": ["ndwi.mean"]}
    )
    transport = complete_with(draft)

    async def go() -> object:
        async with httpx.AsyncClient(transport=transport) as client:
            synth = NvidiaAnswerSynthesizer(settings=settings(), client=client)
            return await synth.synthesize(
                "What is the NDWI?",
                AgentEvidence(items=[], execution=None, analysis=None),
            )

    answer = asyncio.run(go())
    assert isinstance(answer, DraftAnswer)
    assert answer.evidence_refs == ["ndwi.mean"]


def test_nvidia_synthesizer_fails_closed_on_an_api_error() -> None:
    transport = complete_with("{}", status_code=503)

    async def go() -> object:
        async with httpx.AsyncClient(transport=transport) as client:
            synth = NvidiaAnswerSynthesizer(settings=settings(), client=client)
            try:
                return await synth.synthesize(
                    "q", AgentEvidence(items=[], execution=None, analysis=None)
                )
            except Exception as exc:
                return exc

    assert isinstance(asyncio.run(go()), UpstreamServiceError)


# --------------------------------------------------------------------------- #
# Intent parsing uses the same provider selection
# --------------------------------------------------------------------------- #
#
# `/query/parse` predates the provider abstraction and constructed a Gemini
# parser unconditionally, so `AI_PROVIDER=nvidia` still parsed with Gemini.
# These pin that it now resolves through the one factory, with the same
# credential rules and the same refusal to fall back.


def test_intent_parser_follows_the_configured_provider() -> None:
    assert isinstance(
        get_intent_parser(settings=settings(AI_PROVIDER="gemini")),
        GeminiIntentParser,
    )
    assert isinstance(
        get_intent_parser(settings=settings(AI_PROVIDER="nvidia")),
        NvidiaIntentParser,
    )


def test_intent_parser_honours_a_per_run_override() -> None:
    configured = settings(AI_PROVIDER="gemini")
    assert isinstance(
        get_intent_parser(settings=configured, provider="nvidia"),
        NvidiaIntentParser,
    )


def test_intent_parser_satisfies_the_provider_neutral_port() -> None:
    for provider in sorted(SUPPORTED_AI_PROVIDERS):
        parser = get_intent_parser(settings=settings(AI_PROVIDER=provider))
        assert isinstance(parser, IntentParser)


def test_intent_parser_refuses_an_unknown_provider() -> None:
    with pytest.raises(UpstreamServiceError, match="Unknown AI provider"):
        get_intent_parser(settings=settings(), provider="openai")


def test_intent_parser_fails_clearly_without_the_nvidia_credential() -> None:
    with pytest.raises(UpstreamServiceError, match="NVIDIA_API_KEY"):
        get_intent_parser(
            settings=Settings(_env_file=None, AI_PROVIDER="nvidia", GEMINI_API_KEY="g")  # type: ignore[arg-type]
        )


def test_intent_parser_fails_clearly_without_the_gemini_credential() -> None:
    with pytest.raises(UpstreamServiceError, match="GEMINI_API_KEY"):
        get_intent_parser(
            settings=Settings(_env_file=None, AI_PROVIDER="gemini", NVIDIA_API_KEY="n")  # type: ignore[arg-type]
        )


def test_a_missing_credential_never_yields_the_other_parser() -> None:
    """No silent fallback on the parse path either."""

    with pytest.raises(UpstreamServiceError):
        get_intent_parser(
            settings=Settings(_env_file=None, AI_PROVIDER="nvidia", GEMINI_API_KEY="g")  # type: ignore[arg-type]
        )


def test_both_parsers_are_asked_the_same_thing() -> None:
    """One copy of the extraction instruction, shared - so they stay comparable."""

    from app.services.agent.providers import nvidia as nvidia_mod
    from app.services.ai import parser as gemini_parser_mod
    from app.services.ai import prompts as ai_prompts

    assert gemini_parser_mod._SYSTEM_INSTRUCTION is ai_prompts._SYSTEM_INSTRUCTION
    assert nvidia_mod._SYSTEM_INSTRUCTION is ai_prompts._SYSTEM_INSTRUCTION


def test_the_intent_port_module_imports_no_sdk() -> None:
    """The port is implementable without dragging a provider SDK in."""

    import ast
    import pathlib

    source = pathlib.Path("app/services/ai/ports.py").read_text()
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
    assert roots <= {"__future__", "abc", "app"}
