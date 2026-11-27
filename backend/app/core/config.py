"""Environment-driven application configuration."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


@dataclass(frozen=True)
class ProviderFields:
    """Where one provider's credential and model live on :class:`Settings`.

    Named rather than branched on. Every place that used to ask "is this
    gemini, else nvidia" now reads this table, so adding a provider is one
    entry here instead of an if-chain that a new provider can be half-wired
    into - the exact defect ``tests/test_provider_isolation.py`` was written
    for, where only the visual step was provider-aware and planning silently
    stayed on Gemini.
    """

    #: The :class:`Settings` field holding the API key, or ``None`` for a
    #: provider that runs on this machine and needs none.
    api_key: str | None
    #: The :class:`Settings` field holding the model id.
    model: str
    #: The environment variable an operator actually sets, for error messages.
    env_var: str
    #: For a keyless provider, the :class:`Settings` field holding the endpoint
    #: it is reached at. Such a provider is "configured" when it has somewhere
    #: to send a request - there is no credential to hold.
    endpoint: str | None = None


#: Every inference backend this build knows how to construct.
AI_PROVIDER_FIELDS: Mapping[str, ProviderFields] = MappingProxyType(
    {
        "gemini": ProviderFields(
            api_key="gemini_api_key",
            model="gemini_model",
            env_var="GEMINI_API_KEY",
        ),
        "nvidia": ProviderFields(
            api_key="nvidia_api_key",
            model="nvidia_model",
            env_var="NVIDIA_API_KEY",
        ),
        "anthropic": ProviderFields(
            api_key="anthropic_api_key",
            model="anthropic_model",
            env_var="ANTHROPIC_API_KEY",
        ),
        "local": ProviderFields(
            api_key=None,
            model="local_ai_model",
            env_var="LOCAL_AI_BASE_URL",
            endpoint="local_ai_base_url",
        ),
    }
)

#: The vision-language providers this build knows how to construct. Derived
#: from the table above so the two can never disagree.
SUPPORTED_AI_PROVIDERS: frozenset[str] = frozenset(AI_PROVIDER_FIELDS)


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
    #: ``text`` for a terminal, ``json`` for a log platform: one JSON object
    #: per line with ts, level, logger, run_id and message.
    log_format: Literal["text", "json"] = "text"
    #: Bearer token for ``GET /metrics``. Unset (the default), the route
    #: answers 404 and no metric is published.
    metrics_token: SecretStr | None = None
    api_v1_prefix: str = "/api/v1"

    #: ``NoDecode`` because pydantic-settings JSON-decodes a complex field
    #: BEFORE any validator runs, so ``SATQUERY_CORS_ORIGINS=http://host:5173``
    #: raised ``SettingsError`` at startup and the process never came up. The
    #: repository's own comma-splitting validator was unreachable for the one
    #: source that matters - the environment. Observed live while starting a
    #: second instance for a limits test, and it would equally have crashed the
    #: production container, whose compose file sets a comma-separated list.
    cors_origins: Annotated[list[str], NoDecode] = Field(
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

    #: Hosts whose rasters this deployment may open, as exact names or
    #: dot-prefixed suffixes (e.g. ``.blob.core.windows.net``).
    #:
    #: An asset href arrives from an EXTERNAL catalog response, and whatever it
    #: names is fetched by this server. EMPTY - the default - means "any public
    #: host", with private, loopback, link-local, reserved and multicast
    #: addresses refused outright; that is the protection every deployment gets
    #: without configuration. A production deployment SHOULD set this to the
    #: catalogs it actually uses, which is what the production profile does.
    #:
    #: KNOWN LIMITATION: this matches the NAME in the href. It does not pin the
    #: address that name resolves to, so it does not defeat DNS rebinding; that
    #: would require resolving and connecting to a checked address, which the
    #: raster stack (GDAL/curl) does not expose.
    trusted_asset_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )

    # Bounded Sentinel-2 imagery retrieval (windowed COG reads).
    imagery_max_dimension: int = 1024
    imagery_hard_max_dimension: int = 2048
    imagery_max_window_pixels: int = 50_000_000
    #: How large an area the standard analyses (spectral indices, the two-date
    #: comparison and SAR backscatter) may cover. Larger than one read: such an
    #: area is read in native-resolution tiles of at most
    #: ``imagery_hard_max_dimension`` px, each read, masked and measured on its
    #: own, and the tile statistics are combined exactly (pixel counts add,
    #: means are pixel-weighted). Memory stays that of one tile; time grows
    #: with the area. 5000 px is about 50 km at 10 m.
    imagery_tiled_max_dimension: int = Field(default=5000, ge=1, le=20_000)
    imagery_tiled_max_window_pixels: int = Field(default=25_000_000, ge=1)
    #: Scene validation refuses a scene whose footprint covers less than this
    #: fraction of the requested area. 0.0 (the default) refuses only a scene
    #: that does not reach the area at all and REPORTS partial coverage: there
    #: is no scientific basis in this repository for a stricter cut-off, and
    #: inventing one would be worse than stating the measured fraction.
    scene_min_aoi_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Admission control ------------------------------------------------- #
    # The per-request contracts already bound the SHAPE of one request
    # (MAX_TIME_WINDOWS, the imagery caps above). These bound how OFTEN and how
    # MANY, which no schema can express.
    #
    # Enforced per PROCESS. See app/core/limits.py for exactly what that does
    # and does not guarantee once more than one replica is running.
    #: Largest request body accepted, in bytes. Every contract here is small
    #: JSON; the only large payloads travel outward (base64 PNGs).
    max_request_bytes: int = 1_000_000
    #: Requests per client per window on the expensive routes. Generous enough
    #: that an interactive session never notices, bounded enough that a loop
    #: does.
    rate_limit_requests: int = 120
    rate_limit_window_seconds: float = 60.0
    #: Simultaneous in-flight workflows (discovery, analysis, agent runs).
    #: Four keeps a small machine responsive; each one can hold raster arrays.
    max_concurrent_workflows: int = 4
    #: How many reverse proxies in front of this process APPEND to
    #: X-Forwarded-For. 0 (the default) attributes requests to the TCP peer.
    #: Set it to the proxy count of the hosting platform so the rate limit is
    #: per caller rather than shared by everyone behind the proxy; see
    #: ``core/limits._client_of`` and DEPLOYMENT.md for how to measure it.
    trusted_proxy_hops: int = Field(default=0, ge=0, le=5)
    #: Simultaneous windowed COG reads, process-wide. Memory is bounded by the
    #: workflow count and the tile size (every band of a tile is held until it
    #: is measured, however the reads are scheduled); this bounds open remote
    #: connections. A remote read is latency-bound, so a tile's bands are read
    #: together: six halved a whole-city analysis (388 s -> 194 s, measured).
    max_concurrent_raster_reads: int = 6
    #: How long a request may wait for a slot before it is refused. Absorbs a
    #: passing burst without letting a queue grow without bound.
    admission_wait_seconds: float = 2.0
    #: How long one raster read may wait for a raster slot. Longer than the
    #: workflow wait: a read queued behind another analysis's reads is normal
    #: work in progress, and refusing it would fail a run half-way through.
    raster_admission_wait_seconds: float = Field(default=60.0, gt=0, le=600)
    #: Total budget for one workflow. A local model alone may legitimately take
    #: minutes (LOCAL_AI_TIMEOUT_SECONDS is 300), so this is deliberately far
    #: above the normal case - it exists to end a run that will never finish,
    #: not to cut short a slow one.
    workflow_budget_seconds: float = 900.0

    # OpenStreetMap's usage policy: at most one request per second from an
    # application, with caching expected. Application-wide, not per client.
    geocoder_min_interval_seconds: float = 1.0
    #: A place's geocoded extent does not change within a day; the policy asks
    #: for caching, and every avoided request is budget the shared outbound IP
    #: of a cloud host does not spend. Successes only - a failure is never
    #: cached.
    geocoder_cache_ttl_seconds: float = Field(default=86_400.0, ge=0)
    geocoder_cache_entries: int = Field(default=256, ge=1, le=10_000)
    #: Upstream requests ONE geocode may make, the first included.
    geocoder_max_attempts: int = Field(default=3, ge=1, le=5)
    #: The most one geocode will spend WAITING - on backoff, on Retry-After, on
    #: a cooldown another caller started. Past it the caller gets
    #: ``geocoding_unavailable`` at once rather than a longer wait.
    geocoder_retry_budget_seconds: float = Field(default=15.0, ge=0, le=120)
    #: First backoff after a failure without Retry-After; doubles per
    #: consecutive failure, capped by ``geocoder_max_cooldown_seconds``.
    geocoder_backoff_base_seconds: float = Field(default=2.0, ge=0, le=60)
    geocoder_max_cooldown_seconds: float = Field(default=300.0, ge=0, le=3_600)
    #: How many NEW places (not already cached) one caller may look up per
    #: window. The geocoder's budget is shared by the whole application, so
    #: without this one anonymous caller cycling place names could spend it for
    #: everyone. Cached places are free; 0 disables the cap.
    geocoder_new_places_per_caller: int = Field(default=30, ge=0, le=10_000)
    geocoder_new_places_window_seconds: float = Field(default=600.0, gt=0, le=86_400)
    #: Optional, ";"-separated place names geocoded once in the background at
    #: startup, through the same cache, pacing and cooldown as any request. A
    #: free-tier host loses its in-memory cache whenever it sleeps; this refills
    #: it with REAL answers before a user asks. Empty: nothing is warmed.
    geocoder_warm_places: str = Field(default="", max_length=2_000)

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
    # Every provider runs through the SAME agent orchestration, grounding and
    # evidence path - only the inference backend differs. There is deliberately
    # no fallback between them: silently answering with a provider the operator
    # did not select would make a demo or an experiment unattributable.
    ai_provider: str = Field(
        default="gemini",
        validation_alias=AliasChoices("AI_PROVIDER"),
    )

    #: Whether a CALLER may name an AI provider or model for a run. ``None``
    #: (the default) allows it in development and refuses it in production:
    #: the API has no accounts, so a public deployment cannot attribute or cap
    #: what an anonymous caller spends on a metered provider. The standard
    #: workflow - deterministic rules and the local intent model - needs no
    #: provider and is unaffected. Set ``SATQUERY_CLIENT_AI_SELECTION=true`` only
    #: where every caller is trusted.
    client_ai_selection: bool | None = None

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
    #: Default deliberately NOT nemotron-nano-12b-v2-vl: NVIDIA retired that
    #: model on 2026-08-26 and the endpoint answers 410 Gone, so a fresh
    #: deployment that selected NVIDIA was broken before it sent a request.
    #: This one is live, multimodal, and serves both the text and visual roles.
    nvidia_model: str = Field(
        default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        validation_alias=AliasChoices("NVIDIA_MODEL"),
    )
    #: Raised from 60s: the catalogued NIM models include *reasoning* models
    #: that emit a long internal deliberation before their JSON, and 60s was
    #: cutting them off mid-generation - observed live as repeated
    #: `httpx.TimeoutException` on planning calls that the endpoint was still
    #: servicing. A timeout that fires before the provider has answered reports
    #: a failure that did not happen.
    nvidia_timeout_seconds: float = 120.0

    # Anthropic's Claude models, reached through the official ``anthropic``
    # SDK. ANTHROPIC_API_KEY / ANTHROPIC_MODEL use the standard unprefixed
    # names the SDK itself documents. Never commit a real key.
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = Field(
        default="claude-opus-5",
        validation_alias=AliasChoices("ANTHROPIC_MODEL"),
    )
    #: Overrides the API host. Left unset in every real deployment - it exists
    #: so a test can point the provider at an unroutable address and prove a
    #: run never reaches the network, exactly as ``nvidia_base_url`` does.
    anthropic_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_BASE_URL"),
    )
    #: Claude models think before answering by default, so a planning call can
    #: legitimately take tens of seconds. Matched to the NVIDIA reasoning
    #: models' budget for the same reason: a timeout that fires while the
    #: provider is still working reports a failure that did not happen.
    anthropic_timeout_seconds: float = 120.0

    # A local open-weight model served by Ollama on this machine. No key and no
    # quota: requests go to LOCAL_AI_BASE_URL and nowhere else, and a local
    # failure is reported as one - it is never answered by a cloud provider.
    local_ai_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("LOCAL_AI_BASE_URL"),
    )
    #: The model verified on the reference development machine (Apple M2, 8 GB
    #: unified memory): qwen3-vl:8b (6.1 GB of weights) does not fit beside the
    #: OS and the app there. Machines with 16 GB or more can select
    #: qwen3-vl:8b-instruct through LOCAL_AI_MODEL.
    #:
    #: An ``-instruct`` tag on purpose. Every plain Qwen3-VL tag on the Ollama
    #: registry (qwen3-vl:4b, :8b, ...) is byte-identical to its ``-thinking``
    #: variant, which reasons in hidden tokens whatever ``think`` is set to:
    #: observed live, a one-word reply cost 220 generated tokens and a real
    #: planning call did not finish in 300 s on this machine.
    local_ai_model: str = Field(
        default="qwen3-vl:4b-instruct",
        validation_alias=AliasChoices("LOCAL_AI_MODEL"),
    )
    #: Generous on purpose: the first request after a restart loads the model
    #: into memory, and a small machine reads a long planning prompt slowly. A
    #: timeout that fires while the model is still working reports a failure
    #: that did not happen.
    local_ai_timeout_seconds: float = 300.0
    #: Context window requested from Ollama. Set explicitly because a prompt
    #: longer than the window is truncated from its START - the system
    #: instruction - without any error. 4096 fits every role with room to
    #: spare (measured: the largest prompt, temporal synthesis, is about 2.2k
    #: tokens) and halves the KV cache of 8192, which on an 8 GB machine is the
    #: difference between running and swapping.
    local_ai_num_ctx: int = 4096

    def api_key_for(self, provider: str) -> str | None:
        """This deployment's credential for ``provider``, if it holds one.

        Raises :class:`KeyError` for an unknown provider rather than returning
        ``None``: "no key" and "no such provider" are different problems, and
        collapsing them would let a typo read as an unconfigured deployment.
        """

        field = AI_PROVIDER_FIELDS[provider].api_key
        return getattr(self, field) if field is not None else None

    def is_configured(self, provider: str) -> bool:
        """Whether this deployment can send ``provider`` a request at all.

        A keyed provider needs its credential; a keyless provider running on
        this machine needs an endpoint. Asked by name, from the table, so a new
        provider cannot be half-wired into one check and missing from another.
        """

        fields = AI_PROVIDER_FIELDS[provider]
        if fields.api_key is None:
            return bool(fields.endpoint and getattr(self, fields.endpoint))
        return bool(getattr(self, fields.api_key))

    def model_for(self, provider: str) -> str:
        """The model ``provider`` uses when a request names none."""

        return getattr(self, AI_PROVIDER_FIELDS[provider].model)

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

    @field_validator("cors_origins", "trusted_asset_hosts", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string, a JSON array, or a real list.

        Both string forms are in use and both must keep working: the
        development compose file writes a JSON array, and a shell export or the
        production compose file writes a comma-separated list. ``NoDecode`` on
        the fields is what makes this validator reachable for environment
        values at all; without it pydantic-settings decoded first and failed on
        anything that was not JSON.
        """

        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return []

        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    "expected a JSON array or a comma-separated list"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array or a comma-separated list")
            return [str(item).strip() for item in parsed if str(item).strip()]

        return [item.strip() for item in text.split(",") if item.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in {"production", "prod"}

    @property
    def client_ai_selection_enabled(self) -> bool:
        if self.client_ai_selection is not None:
            return self.client_ai_selection
        return not self.is_production

    @model_validator(mode="after")
    def _production_configuration(self) -> Settings:
        if not self.is_production:
            return self
        if not self.cors_origins:
            raise ValueError("Production requires explicit SATQUERY_CORS_ORIGINS")
        for origin in self.cors_origins:
            _require_public_https(origin, "SATQUERY_CORS_ORIGINS", origin_only=True)
        if not self.trusted_asset_hosts or any(
            not host.strip() or "*" in host for host in self.trusted_asset_hosts
        ):
            raise ValueError("Production requires explicit SATQUERY_TRUSTED_ASSET_HOSTS")
        for field in ("stac_base_url", "nominatim_base_url", "nvidia_base_url",
                      "anthropic_base_url"):
            value = getattr(self, field)
            if value is not None:
                _require_public_https(value, field)
        # Local Ollama and container health checks are private transports, not
        # public browser origins. They intentionally retain HTTP support.
        return self


def _require_public_https(value: str, field: str, *, origin_only: bool = False) -> None:
    message = f"Production {field} requires HTTPS with a public DNS hostname"
    try:
        url = urlsplit(value)
        host = (url.hostname or "").lower().rstrip(".")
        port = url.port
    except ValueError:
        raise ValueError(message) from None
    if (
        re.search(r"[\s\\]", value) or url.scheme != "https"
        or not host or "." not in host or ":" in host
        or re.fullmatch(r"[\d.]+", host) or "*" in host
        or re.search(r"(^|\.)(localhost|local|internal)$", host)
        or url.username is not None or url.password is not None
        or url.query or url.fragment or port == 0
        or (origin_only and url.path)
    ):
        raise ValueError(message)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()
