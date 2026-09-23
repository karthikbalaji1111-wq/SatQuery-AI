"""The AI model catalog endpoint.

Lets the UI offer a real choice of inference backend without shipping model ids
or capability rules to the browser, and without ever shipping a credential.

Availability is answered honestly and in three separate parts, because they are
three different questions:

* **catalogued** - the model is known and its capabilities are recorded;
* **configured** - this deployment holds a credential for its provider;
* **compatible** - the model can fill the role the caller asked about.

A model that exists is not thereby usable, and this endpoint never implies it
is. Reachability is a fourth question that only an actual request can answer,
so it is not claimed here - with one exception. The local provider runs on this
machine, so asking it is cheap and its answer is actionable: one read-only
``GET /api/tags`` says whether Ollama is running, whether it can read its
models, and which are installed - and each local model's status says so.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings, get_settings
from app.services.agent.providers.catalog import (
    MODEL_CATALOG,
    ModelCard,
    ModelRole,
)
from app.services.agent.providers.local import ProbeFailure, installed_models

router = APIRouter()


class ModelOption(BaseModel):
    """One catalogued model, with this deployment's view of it."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    display_name: str
    modality: str
    supports_image: bool
    supports_text: bool
    supports_video: bool
    supports_tools: bool
    supports_structured_output: bool
    endpoint_type: str
    #: True when this deployment holds a credential for the model's provider.
    configured: bool
    #: False when the provider has retired the model. Static, published fact -
    #: never the result of a health check.
    available: bool = True
    #: Present only for a retired model, so a UI can explain the refusal.
    retired_reason: str | None = None
    #: True when the model can fill the requested role.
    compatible: bool
    #: A short, honest status for display.
    status: str


class ModelCatalogResponse(BaseModel):
    """The catalog, plus which provider a run uses when none is named."""

    model_config = ConfigDict(extra="forbid")

    role: str
    default_provider: str
    default_model: str
    models: list[ModelOption]


def _configured(settings: Settings, provider: str) -> bool:
    """Whether this deployment holds a credential for ``provider``.

    Asks :class:`Settings` by provider name rather than branching, so a newly
    catalogued provider reports its real state instead of inheriting whichever
    branch happened to be last.
    """

    return settings.is_configured(provider)


def _status(
    *, configured: bool, compatible: bool, role: ModelRole, available: bool = True
) -> str:
    # Retirement outranks every other consideration: a model the provider no
    # longer serves cannot be Ready, Not configured, or Unsupported - it is
    # simply gone, and saying anything else invites a selection that will fail.
    if not available:
        return "Retired by provider"
    if not compatible:
        return (
            "Unsupported for visual analysis"
            if role == "visual"
            else "Unsupported for this step"
        )
    if not configured:
        return "Not configured"
    # Deliberately not "Available": nothing here has contacted the provider,
    # and a quota or an outage would make that a false claim.
    return "Ready"


def _local_status(
    card: ModelCard, installed: frozenset[str] | ProbeFailure | None
) -> str | None:
    """A local model's reachability, or ``None`` to keep the generic status."""

    if card.provider != "local":
        return None
    if installed is None:
        return "Ollama not running"
    if installed is ProbeFailure.MODELS_UNREADABLE:
        return "Ollama cannot read models"
    if card.model_id not in installed:
        return "Not installed"
    return None


def _option(
    card: ModelCard,
    *,
    settings: Settings,
    role: ModelRole,
    installed: frozenset[str] | ProbeFailure | None = None,
) -> ModelOption:
    configured = _configured(settings, card.provider)
    # Pure capability: a retired model is still image-capable, and reporting it
    # as "incompatible" would misdescribe WHY it cannot be chosen. Availability
    # is carried by `available`/`status` instead.
    compatible = card.supports_role(role)
    status = _status(
        configured=configured,
        compatible=compatible,
        role=role,
        available=card.is_available,
    )
    if status == "Ready":
        # Only a model that is otherwise ready is asked about reachability; a
        # retired, unsupported or unconfigured one already says why it cannot run.
        status = _local_status(card, installed) or status
    return ModelOption(
        provider=card.provider,
        model_id=card.model_id,
        display_name=card.display_name,
        modality=card.modality,
        supports_image=card.supports_image,
        supports_text=card.supports_text,
        supports_video=card.supports_video,
        supports_tools=card.supports_tools,
        supports_structured_output=card.supports_structured_output,
        endpoint_type=card.endpoint_type,
        configured=configured,
        compatible=compatible,
        available=card.is_available,
        retired_reason=card.retired_reason,
        status=status,
    )


@router.get("/models", response_model=ModelCatalogResponse)
async def list_models(
    role: ModelRole = Query(
        default="visual",
        description="Which step the model would fill.",
    ),
) -> ModelCatalogResponse:
    """List catalogued models and this deployment's view of each.

    No credential is returned, and none is needed to call this: the response
    says only *whether* a provider is configured, never what with.
    """

    settings = get_settings()
    default_model = settings.model_for(settings.ai_provider)
    installed = (
        await installed_models(settings)
        if settings.is_configured("local")
        else None
    )
    return ModelCatalogResponse(
        role=role,
        default_provider=settings.ai_provider,
        default_model=default_model,
        models=[
            _option(card, settings=settings, role=role, installed=installed)
            for card in MODEL_CATALOG
        ],
    )
