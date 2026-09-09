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
so it is not claimed here.
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
    if provider == "gemini":
        return bool(settings.gemini_api_key)
    return bool(settings.nvidia_api_key)


def _status(*, configured: bool, compatible: bool, role: ModelRole) -> str:
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


def _option(
    card: ModelCard, *, settings: Settings, role: ModelRole
) -> ModelOption:
    configured = _configured(settings, card.provider)
    compatible = card.serves(role)
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
        status=_status(configured=configured, compatible=compatible, role=role),
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
    default_model = (
        settings.gemini_model
        if settings.ai_provider == "gemini"
        else settings.nvidia_model
    )
    return ModelCatalogResponse(
        role=role,
        default_provider=settings.ai_provider,
        default_model=default_model,
        models=[
            _option(card, settings=settings, role=role) for card in MODEL_CATALOG
        ],
    )
