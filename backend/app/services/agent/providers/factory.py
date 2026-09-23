"""Provider selection: one place that decides which inference backend runs.

Callers ask for a :class:`~app.services.agent.visual.VisualAnalyst` and get one.
They do not branch on the provider, and nothing downstream - executor,
grounding, evidence, API contract - can tell which provider answered except
through the attribution that travels with the observation.

There is deliberately **no fallback**. If the configured provider cannot be
built, this raises. Quietly answering with a different provider would make a
result unattributable, which is exactly the property the visual path exists to
preserve.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import (
    AI_PROVIDER_FIELDS,
    SUPPORTED_AI_PROVIDERS,
    Settings,
    get_settings,
)
from app.core.errors import UpstreamServiceError
from app.services.agent.planner import AgentPlanner
from app.services.agent.providers.catalog import ModelCard, ModelRole, find_model
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.agent.visual import VisualAnalyst, VisualAnswer
from app.services.ai.ports import IntentParser


def get_visual_analyst(
    *,
    settings: Settings | None = None,
    provider: str | None = None,
    model: str | None = None,
    role: ModelRole = "visual",
) -> VisualAnalyst:
    """Build the configured visual analyst.

    ``provider`` overrides the configured default for a single run, which is
    what lets one request be compared against another without a restart. It is
    validated the same way the environment variable is: an unknown value is an
    error, never a silent fall back to the default.

    ``model`` overrides the provider's configured model for a single run, and
    is checked against the catalog for the requested ``role``: a text-only model
    is refused for visual analysis rather than being sent an image it cannot
    read, which would otherwise come back as a confident description of nothing.

    Raises :class:`UpstreamServiceError` when the selected provider has no
    credential, so the caller reports a provider failure rather than presenting
    an unattributable answer.
    """

    settings = settings or get_settings()
    selected = (provider or settings.ai_provider or "").strip().lower()

    if selected not in SUPPORTED_AI_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_AI_PROVIDERS))
        raise UpstreamServiceError(
            f"Unknown AI provider {selected!r}. Supported providers: {supported}."
        )

    settings = _apply_model_override(settings, selected, model, role)

    _check_configured_model(settings, selected, role)

    if not settings.is_configured(selected):
        raise UpstreamServiceError(
            f"{AI_PROVIDER_FIELDS[selected].env_var} is not configured; "
            f"visual analysis is unavailable for the {selected} provider."
        )

    # Imported here, not at module scope: a provider SDK is heavy and a build
    # that only ever runs one provider should not pay to import the others.
    if selected == "gemini":
        from app.services.agent.providers.gemini import GeminiVisualAnalyst

        return GeminiVisualAnalyst(settings=settings)

    if selected == "anthropic":
        from app.services.agent.providers.anthropic import (
            AnthropicVisualAnalyst,
        )

        return AnthropicVisualAnalyst(settings=settings)

    if selected == "local":
        from app.services.agent.providers.local import LocalVisualAnalyst

        return LocalVisualAnalyst(settings=settings)

    from app.services.agent.providers.nvidia import NvidiaVisualAnalyst

    return NvidiaVisualAnalyst(settings=settings)


def _configured_model(settings: Settings, provider: str) -> str:
    return settings.model_for(provider)


def _apply_model_override(
    settings: Settings, provider: str, model: str | None, role: ModelRole
) -> Settings:
    """Validate a per-run model and fold it into the settings the provider reads.

    Returns a copy rather than mutating: settings are cached process-wide, and
    one request's model choice must not leak into the next.
    """

    if model is None:
        return settings

    chosen = model.strip()
    card = find_model(provider, chosen)
    if card is None:
        raise UpstreamServiceError(
            f"Model {chosen!r} is not in the catalog for provider {provider!r}."
        )
    # Retirement is checked BEFORE capability. `serves()` already returns False
    # for a retired model, so without this the caller would be told the model
    # "does not support visual analysis" - a true statement about the wrong
    # problem, sending them to pick a different capability instead of a
    # different model.
    if not card.is_available:
        raise UpstreamServiceError(_retired_message(card))
    if not card.serves(role):
        raise UpstreamServiceError(_unsupported_message(card.display_name, role))

    field = AI_PROVIDER_FIELDS[provider].model
    return settings.model_copy(update={field: chosen})


def _check_configured_model(
    settings: Settings, provider: str, role: ModelRole
) -> None:
    """Refuse a configured model that cannot fill the role.

    An uncatalogued model is allowed through: the catalog is a curated aid, not
    an allowlist, and an operator who configures a newer model id should not be
    blocked by this file being out of date. A model we DO know about and know
    cannot see is refused - that is a fact, not a gap.
    """

    configured = _configured_model(settings, provider)
    card = find_model(provider, configured)
    if card is not None and not card.is_available:
        raise UpstreamServiceError(_retired_message(card))
    if card is not None and not card.serves(role):
        raise UpstreamServiceError(_unsupported_message(card.display_name, role))


def _retired_message(card: ModelCard) -> str:
    """Name the retirement and its reason, so the fix is obvious.

    The reason is catalogued static text written in this repository, not
    provider output, so quoting it leaks nothing.
    """

    return (
        f"{card.display_name} ({card.model_id}) has been retired by its "
        f"provider and can no longer be selected. {card.retired_reason} "
        "Choose a different model."
    )


def _unsupported_message(display_name: str, role: ModelRole) -> str:
    if role == "visual":
        return (
            f"{display_name} does not support visual analysis. Select an "
            "image-capable model for this step."
        )
    return f"{display_name} does not support text analysis."


@dataclass(frozen=True)
class ProviderBundle:
    """The three AI roles, all from one provider.

    The agent orchestrator receives this and cannot tell which provider built
    it: every field is one of the existing provider-neutral abstractions, and
    each returns a normalised SatQuery schema. Selecting a provider therefore
    swaps the whole reasoning path at once - planning, seeing and describing -
    rather than one step of it.
    """

    provider: str
    model: str
    planner: AgentPlanner
    visual_analyst: VisualAnalyst
    synthesizer: AnswerSynthesizer


def get_agent_providers(
    *,
    settings: Settings | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> ProviderBundle:
    """Build all three AI roles from the selected provider.

    Text roles are validated eagerly: a planner and a synthesizer both need
    text generation, and a model that cannot generate text can serve neither,
    so that is knowable before anything runs.

    The **visual** role is validated lazily, when an observation is actually
    requested. That is deliberate rather than lax: a text-only NVIDIA model is
    a legitimate choice for a run whose plan never selects the visual tool, and
    refusing the whole request up front would reject work the model can plainly
    do. When a plan *does* reach the visual step, the check fires with a
    specific message and no image is sent.
    """

    settings = settings or get_settings()
    selected = _resolve_provider(settings, provider)
    resolved = _apply_model_override(settings, selected, model, "text")
    _check_configured_model(resolved, selected, "text")
    _require_credential(resolved, selected)

    chosen_model = _configured_model(resolved, selected)

    if selected == "gemini":
        from app.services.agent.providers.gemini import (
            GeminiAgentPlanner,
            GeminiAnswerSynthesizer,
        )

        return ProviderBundle(
            provider=selected,
            model=chosen_model,
            planner=GeminiAgentPlanner(settings=resolved),
            visual_analyst=_deferred_visual(resolved, selected, model),
            synthesizer=GeminiAnswerSynthesizer(settings=resolved),
        )

    if selected == "anthropic":
        from app.services.agent.providers.anthropic import (
            AnthropicAgentPlanner,
            AnthropicAnswerSynthesizer,
        )

        return ProviderBundle(
            provider=selected,
            model=chosen_model,
            planner=AnthropicAgentPlanner(settings=resolved),
            visual_analyst=_deferred_visual(resolved, selected, model),
            synthesizer=AnthropicAnswerSynthesizer(settings=resolved),
        )

    if selected == "local":
        from app.services.agent.providers.local import (
            LocalAgentPlanner,
            LocalAnswerSynthesizer,
        )

        return ProviderBundle(
            provider=selected,
            model=chosen_model,
            planner=LocalAgentPlanner(settings=resolved),
            visual_analyst=_deferred_visual(resolved, selected, model),
            synthesizer=LocalAnswerSynthesizer(settings=resolved),
        )

    from app.services.agent.providers.nvidia import (
        NvidiaAgentPlanner,
        NvidiaAnswerSynthesizer,
    )

    return ProviderBundle(
        provider=selected,
        model=chosen_model,
        planner=NvidiaAgentPlanner(settings=resolved),
        visual_analyst=_deferred_visual(resolved, selected, model),
        synthesizer=NvidiaAnswerSynthesizer(settings=resolved),
    )


def _resolve_provider(settings: Settings, provider: str | None) -> str:
    selected = (provider or settings.ai_provider or "").strip().lower()
    if selected not in SUPPORTED_AI_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_AI_PROVIDERS))
        raise UpstreamServiceError(
            f"Unknown AI provider {selected!r}. Supported providers: {supported}."
        )
    return selected


def _require_credential(settings: Settings, provider: str) -> None:
    """Refuse a provider this deployment holds no credential for.

    Table-driven rather than a chain of ``if provider ==`` branches: a
    provider added to :data:`AI_PROVIDER_FIELDS` but forgotten here would
    silently skip its own credential check, and the message names the variable
    an operator actually sets.
    """

    if not settings.is_configured(provider):
        raise UpstreamServiceError(
            f"{AI_PROVIDER_FIELDS[provider].env_var} is not configured; the "
            f"{provider} provider is unavailable."
        )


class _DeferredVisualAnalyst(VisualAnalyst):
    """Resolves the visual role only when an observation is asked for.

    Keeps a text-only model usable for a run that never looks at an image,
    while guaranteeing that a run which *does* look at one is refused before a
    request is made rather than after.
    """

    def __init__(
        self, settings: Settings, provider: str, model: str | None
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._model = model
        self._resolved: VisualAnalyst | None = None

    def _analyst(self) -> VisualAnalyst:
        if self._resolved is None:
            self._resolved = get_visual_analyst(
                settings=self._settings,
                provider=self._provider,
                model=self._model,
                role="visual",
            )
        return self._resolved

    @property
    def provider_name(self) -> str:  # type: ignore[override]
        return self._provider

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return _configured_model(self._settings, self._provider)

    async def observe(
        self, *, question: str, image: bytes, media_type: str
    ) -> VisualAnswer:
        return await self._analyst().observe(
            question=question, image=image, media_type=media_type
        )


def _deferred_visual(
    settings: Settings, provider: str, model: str | None
) -> VisualAnalyst:
    return _DeferredVisualAnalyst(settings, provider, model)


def get_intent_parser(
    *, settings: Settings | None = None, provider: str | None = None
) -> IntentParser:
    """Build the configured intent parser.

    Reuses the same provider resolution, credential check and capability rules
    as the agent roles - there is one selection mechanism, not a second one
    living in the route. Intent parsing needs only text generation, so a
    text-only model can serve it.

    Raises :class:`UpstreamServiceError` when the selected provider has no
    credential. It never falls back to another provider: an intent silently
    parsed by a provider the operator did not select would make the whole run
    unattributable.
    """

    settings = settings or get_settings()
    selected = _resolve_provider(settings, provider)
    _check_configured_model(settings, selected, "text")
    _require_credential(settings, selected)

    if selected == "gemini":
        from app.services.ai.parser import GeminiIntentParser

        return GeminiIntentParser(settings=settings)

    if selected == "anthropic":
        from app.services.agent.providers.anthropic import (
            AnthropicIntentParser,
        )

        return AnthropicIntentParser(settings=settings)

    if selected == "local":
        from app.services.agent.providers.local import LocalIntentParser

        return LocalIntentParser(settings=settings)

    from app.services.agent.providers.nvidia import NvidiaIntentParser

    return NvidiaIntentParser(settings=settings)
