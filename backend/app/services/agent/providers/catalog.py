"""The AI model catalog: what each backend can actually do.

SatQuery's visual-analysis path sends an image and a question. Most hosted
models cannot accept an image at all, so "which provider" is not enough
information to route a run - the catalog records **capability per model**, and
the resolver refuses a model that cannot serve the role rather than silently
degrading to a text-only reading of a picture it never saw.

Why capabilities are curated rather than discovered
---------------------------------------------------
NVIDIA's hosted endpoint is OpenAI-compatible and exposes ``GET /v1/models``,
which lists the model ids an account may call. It does **not** report whether a
model accepts image input. So the two halves come from different places:

* **availability** - discoverable at run time from ``/v1/models``;
* **capability** - curated here, from NVIDIA's published model documentation.

Every entry below was checked against NVIDIA's own documentation at the time of
writing. Model ids are never guessed: an id that could not be verified is not in
this file. Availability and free-tier limits change, so a card records what a
model *can* do, never a promise that it is reachable today.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: The role a model is being asked to fill for a particular run.
ModelRole = Literal["visual", "text"]


class ModelCard(BaseModel):
    """One selectable model and what it is able to do.

    Capability flags are claims about the *model*, independent of whether the
    deployment is currently configured or reachable - those are separate
    questions, answered by the resolver and by discovery respectively.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["gemini", "nvidia"]
    #: The exact id sent in the API request. Never a display string.
    model_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    modality: Literal["multimodal", "text"]
    supports_image: bool
    supports_text: bool = True
    supports_video: bool = False
    supports_tools: bool = False
    supports_structured_output: bool = False
    #: How the model is reached, so a caller can tell the transports apart.
    endpoint_type: Literal["gemini-genai", "openai-compatible"]
    #: Where the capability claims above were checked.
    verified_against: str = Field(min_length=1)

    def serves(self, role: ModelRole) -> bool:
        """Whether this model can fill ``role``."""

        return self.supports_image if role == "visual" else self.supports_text


#: Verified models. Adding one requires no change to any provider logic.
#:
#: Deliberately NOT every model a provider hosts: embedding, translation,
#: guardrail and speech models cannot fill either SatQuery role, and listing
#: them would offer a user a choice that cannot work.
MODEL_CATALOG: tuple[ModelCard, ...] = (
    ModelCard(
        provider="gemini",
        model_id="gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        modality="multimodal",
        supports_image=True,
        supports_tools=True,
        supports_structured_output=True,
        endpoint_type="gemini-genai",
        verified_against="google-genai multimodal input; in production use here",
    ),
    # --- NVIDIA: image-capable, eligible for visual analysis ----------------
    ModelCard(
        provider="nvidia",
        model_id="nvidia/nemotron-nano-12b-v2-vl",
        display_name="Nemotron Nano 12B v2 VL",
        modality="multimodal",
        supports_image=True,
        supports_structured_output=True,
        endpoint_type="openai-compatible",
        verified_against=(
            "docs.nvidia.com NIM VLM 1.5.0 - nemotron-nano-12b-v2-vl API; "
            "base64 image_url, PNG/JPG/JPEG"
        ),
    ),
    ModelCard(
        provider="nvidia",
        model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        display_name="Nemotron 3 Nano Omni 30B A3B Reasoning",
        modality="multimodal",
        supports_image=True,
        supports_video=True,
        supports_tools=True,
        endpoint_type="openai-compatible",
        verified_against=(
            "docs.nvidia.com NIM VLM 1.7.0 - nemotron-3-nano-omni API; "
            "base64 image_url, text/image/video/audio input"
        ),
    ),
    ModelCard(
        provider="nvidia",
        model_id="meta/llama-3.2-11b-vision-instruct",
        display_name="Llama 3.2 11B Vision Instruct",
        modality="multimodal",
        supports_image=True,
        endpoint_type="openai-compatible",
        verified_against=(
            "docs.nvidia.com NIM VLM 1.2.0 - llama3-2 API; base64 image_url, "
            "PNG/JPG/JPEG"
        ),
    ),
    # --- NVIDIA: text-only. Selectable for text roles, NEVER for visual. ----
    ModelCard(
        provider="nvidia",
        model_id="nvidia/nemotron-3-super-120b-a12b",
        display_name="Nemotron 3 Super 120B A12B",
        modality="text",
        supports_image=False,
        supports_tools=True,
        endpoint_type="openai-compatible",
        verified_against=(
            "NVIDIA model card - text-only; no vision input"
        ),
    ),
    ModelCard(
        provider="nvidia",
        model_id="nvidia/nemotron-3.5-lightning-30b-a3b",
        display_name="Nemotron 3.5 Lightning 30B A3B",
        modality="text",
        supports_image=False,
        supports_tools=True,
        endpoint_type="openai-compatible",
        verified_against=(
            "NVIDIA model card - text-only reasoning/chat; no vision input"
        ),
    ),
)


def find_model(provider: str, model_id: str) -> ModelCard | None:
    """The card for one model id, or ``None`` when it is not catalogued."""

    for card in MODEL_CATALOG:
        if card.provider == provider and card.model_id == model_id:
            return card
    return None


def models_for(role: ModelRole, provider: str | None = None) -> list[ModelCard]:
    """Catalogued models able to fill ``role``, optionally for one provider."""

    return [
        card
        for card in MODEL_CATALOG
        if card.serves(role) and (provider is None or card.provider == provider)
    ]
