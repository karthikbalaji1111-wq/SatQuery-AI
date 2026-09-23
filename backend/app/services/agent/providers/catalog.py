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

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The role a model is being asked to fill for a particular run.
ModelRole = Literal["visual", "text"]


class ModelCard(BaseModel):
    """One selectable model and what it is able to do.

    Capability flags are claims about the *model*, independent of whether the
    deployment is currently configured or reachable - those are separate
    questions, answered by the resolver and by discovery respectively.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["gemini", "nvidia", "anthropic", "local"]
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
    endpoint_type: Literal[
        "gemini-genai", "openai-compatible", "anthropic-messages", "ollama-chat"
    ]
    #: Where the capability claims above were checked.
    verified_against: str = Field(min_length=1)
    #: Whether the endpoint still serves this model. A model can be perfectly
    #: capable ON PAPER and still be gone: providers retire them. Capability and
    #: availability are therefore separate facts, and the catalog states both.
    #: Static by design - this is a published retirement, not a health check, so
    #: it must not cost a network round trip on every render.
    availability: Literal["available", "retired"] = "available"
    #: Why a retired model was retired, with the evidence. Required when
    #: ``availability`` is "retired" so a future reader cannot silently revive
    #: it without confronting the reason.
    retired_reason: str | None = None

    @model_validator(mode="after")
    def _retirement_is_explained(self) -> ModelCard:
        if self.availability == "retired" and not self.retired_reason:
            raise ValueError("a retired model must carry retired_reason")
        return self

    @property
    def is_available(self) -> bool:
        return self.availability == "available"

    def supports_role(self, role: ModelRole) -> bool:
        """Whether the MODEL is capable of ``role``, ignoring availability.

        A pure statement about the model, kept separate from whether the
        provider still serves it - retiring a model does not make it blind.
        This is what the catalog API reports as ``compatible``.
        """

        return self.supports_image if role == "visual" else self.supports_text

    def serves(self, role: ModelRole) -> bool:
        """Whether this model can fill ``role`` AND is still served.

        Capability AND availability, because that conjunction is what every
        SELECTION path needs. Checking it here - the one question the resolver,
        the listing and the per-run override all already ask - makes a retired
        model unselectable by construction rather than by remembering to check.
        """

        return self.is_available and self.supports_role(role)


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
    # --- Anthropic: every catalogued Claude model accepts image input ------
    #
    # Ordered most to least capable. All three serve BOTH SatQuery roles, so
    # there is no text-only entry here and no capability refusal to encode -
    # the distinction the NVIDIA section below has to make does not arise.
    ModelCard(
        provider="anthropic",
        model_id="claude-opus-5",
        display_name="Claude Opus 5",
        modality="multimodal",
        supports_image=True,
        supports_tools=True,
        supports_structured_output=True,
        endpoint_type="anthropic-messages",
        verified_against=(
            "Anthropic Messages API - base64 image content blocks; "
            "model id from the bundled claude-api model table"
        ),
    ),
    ModelCard(
        provider="anthropic",
        model_id="claude-sonnet-5",
        display_name="Claude Sonnet 5",
        modality="multimodal",
        supports_image=True,
        supports_tools=True,
        supports_structured_output=True,
        endpoint_type="anthropic-messages",
        verified_against=(
            "Anthropic Messages API - base64 image content blocks; "
            "model id from the bundled claude-api model table"
        ),
    ),
    ModelCard(
        provider="anthropic",
        model_id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        modality="multimodal",
        supports_image=True,
        supports_tools=True,
        supports_structured_output=True,
        endpoint_type="anthropic-messages",
        verified_against=(
            "Anthropic Messages API - base64 image content blocks; "
            "model id from the bundled claude-api model table"
        ),
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
        availability="retired",
        retired_reason=(
            "Retired by NVIDIA on 2026-08-26. The live endpoint answers "
            "HTTP 410 Gone: \"The model 'nvidia/nemotron-nano-12b-v2-vl' has "
            "reached its end of life\". Verified against "
            "https://integrate.api.nvidia.com/v1 - the id is also absent from "
            "GET /v1/models. Its capability flags above remain true of the "
            "model; they are simply no longer reachable."
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
    # --- Local: open-weight Qwen3-VL served by Ollama on this machine -------
    #
    # Instruct tags only. Every plain Qwen3-VL tag on the Ollama registry is
    # byte-identical to its "-thinking" variant (same config and weights
    # digests), which reasons in hidden tokens whatever `think` is set to - a
    # real planning call did not finish in 300 s on the reference machine.
    #
    # Every size accepts image input and serves both roles; which one a machine
    # can run is a MEMORY question, not a capability one. Sizes are the
    # registry's own manifests. 4B is the default because it is the size
    # verified on the reference machine (Apple M2, 8 GB unified memory); 8B
    # needs roughly 16 GB and 30B-A3B roughly 32 GB to run beside the app.
    ModelCard(
        provider="local",
        model_id="qwen3-vl:4b-instruct",
        display_name="Qwen3-VL 4B Instruct (local)",
        modality="multimodal",
        supports_image=True,
        supports_tools=True,
        supports_structured_output=True,
        endpoint_type="ollama-chat",
        verified_against=(
            "ollama show qwen3-vl:4b-instruct - 4.4B parameters, Q4_K_M, "
            "capabilities completion/vision/tools (no thinking); 3.30 GB; run "
            "locally on an Apple M2 with 8 GB unified memory"
        ),
    ),
    ModelCard(
        provider="local",
        model_id="qwen3-vl:2b-instruct",
        display_name="Qwen3-VL 2B Instruct (local)",
        modality="multimodal",
        supports_image=True,
        supports_structured_output=True,
        endpoint_type="ollama-chat",
        verified_against="registry.ollama.ai qwen3-vl:2b-instruct manifest, 1.89 GB",
    ),
    ModelCard(
        provider="local",
        model_id="qwen3-vl:8b-instruct",
        display_name="Qwen3-VL 8B Instruct (local)",
        modality="multimodal",
        supports_image=True,
        supports_structured_output=True,
        endpoint_type="ollama-chat",
        verified_against="registry.ollama.ai qwen3-vl:8b-instruct manifest, 6.14 GB",
    ),
    ModelCard(
        provider="local",
        model_id="qwen3-vl:30b-a3b-instruct",
        display_name="Qwen3-VL 30B-A3B Instruct (local)",
        modality="multimodal",
        supports_image=True,
        supports_structured_output=True,
        endpoint_type="ollama-chat",
        verified_against="registry.ollama.ai qwen3-vl:30b-a3b-instruct manifest, 19.60 GB",
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
