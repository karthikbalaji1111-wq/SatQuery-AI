"""Environment-driven application configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: The vision-language providers this build knows how to construct.
SUPPORTED_AI_PROVIDERS: frozenset[str] = frozenset({"gemini", "nvidia"})


class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env``.

    All variables are prefixed with ``SATQUERY_`` (e.g. ``SATQUERY_LOG_LEVEL``).
    """

    model_config = SettingsConfigDict(
        env_prefix="SATQUERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SatQuery API"
    environment: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # Geospatial grounding via OpenStreetMap Nominatim.
    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = (
        "SatQuery/0.1 (SIH 2026 PS 26167; "
        "+https://github.com/karthikbalaji1111-wq)"
    )
    http_timeout_seconds: float = 10.0

    # Satellite scene discovery via the Earth Search STAC API.
    stac_base_url: str = "https://earth-search.aws.element84.com/v1"
    stac_collection: str = "sentinel-2-l2a"
    # Sentinel-1 RTC uses the fixed public Planetary Computer catalog.
    # Explicit sentinel-1-grd configuration retains Earth Search discovery.
    stac_s1_collection: str = "sentinel-1-rtc"

    # Bounded Sentinel-2 imagery retrieval (windowed COG reads).
    imagery_max_dimension: int = 1024
    imagery_hard_max_dimension: int = 2048
    imagery_max_window_pixels: int = 50_000_000

    # Natural-language intent extraction via the Google Gemini API (google-genai).
    # GEMINI_API_KEY / GEMINI_MODEL use the standard unprefixed names. Never
    # commit a real key. When GEMINI_API_KEY is unset the /query/parse endpoint
    # returns a 502 instead of crashing.
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-3.6-flash",
        validation_alias=AliasChoices("GEMINI_MODEL"),
    )
    gemini_timeout_seconds: float = 30.0

    # Which vision-language provider backs the agent's visual-analysis path.
    # Both providers run through the SAME agent orchestration, grounding and
    # evidence path - only the inference backend differs. There is deliberately
    # no fallback between them: silently answering with a provider the operator
    # did not select would make a demo or an experiment unattributable.
    ai_provider: str = Field(
        default="gemini",
        validation_alias=AliasChoices("AI_PROVIDER"),
    )

    # NVIDIA's hosted NIM endpoints, which speak the OpenAI chat-completions
    # dialect. The model must accept image input - a text-only model cannot
    # serve the visual-analysis path. Never commit a real key.
    nvidia_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NVIDIA_API_KEY"),
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("NVIDIA_BASE_URL"),
    )
    nvidia_model: str = Field(
        default="nvidia/nemotron-nano-12b-v2-vl",
        validation_alias=AliasChoices("NVIDIA_MODEL"),
    )
    nvidia_timeout_seconds: float = 60.0

    @field_validator("ai_provider", mode="before")
    @classmethod
    def _known_provider(cls, value: object) -> object:
        """Reject an unknown provider at configuration time, not at run time.

        A typo here would otherwise surface as a missing-key error much later,
        pointing at the wrong cause.
        """

        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised not in SUPPORTED_AI_PROVIDERS:
                supported = ", ".join(sorted(SUPPORTED_AI_PROVIDERS))
                raise ValueError(
                    f"AI_PROVIDER must be one of: {supported}. Got {value!r}."
                )
            return normalised
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()
