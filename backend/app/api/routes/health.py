"""Liveness and readiness - two different questions, two endpoints.

``/health`` answers **is this process alive?** It touches nothing, depends on
nothing, and is what a container orchestrator restarts on. It must keep
answering while the service is busy, misconfigured or cut off from every
upstream, because none of those is a reason to kill the process.

``/ready`` answers **is this deployment actually configured to do the work it
advertises?** A process can be perfectly alive and unable to answer a single
query: no credential for the selected provider, or a local model that is not
installed. That was invisible - the frontend read ``/health``, saw ``ok``, and
displayed "Operational" beside an AI path that could not run.

**What readiness does NOT do.** It contacts no paid provider. Configuration is
not reachability, and a probe that spent quota to find out would be a bill
attached to a health check - one that a monitoring system would pay every few
seconds. So a configured cloud provider is reported as *configured*, in those
words, and nothing more is claimed.

The single exception is the local provider, and only when it is the SELECTED
one: Ollama runs on this machine, one read-only ``GET /api/tags`` costs
nothing, and its answer is the actual determinant of whether a local run can
work. That call is already bounded (2 s to connect, 10 s to answer).

**The AI provider is optional.** Natural-language questions are interpreted by
the standard, deterministic workflow unless a request names an AI provider, so
a deployment with no provider configured can still do its work. The AI
capability is therefore reported - configured or not, in the same words as
before - but marked ``required: false``, and it no longer decides readiness.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app import __version__
from app.core.config import AI_PROVIDER_FIELDS, Settings, get_settings
from app.services.agent.intent_model import load_intent_classifier
from app.services.agent.providers.local import ProbeFailure, installed_models

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class Capability(BaseModel):
    """One thing this deployment needs in order to do its work."""

    name: str
    ready: bool
    #: Why, in plain words. Always populated - a capability that is ready says
    #: what it is ready to do, so the endpoint is readable without a legend.
    detail: str
    #: Whether readiness depends on it. An optional capability that is not
    #: ready is reported, never hidden, and never makes the deployment unready.
    required: bool = True


class ReadinessResponse(BaseModel):
    ready: bool
    service: str
    version: str
    environment: str
    capabilities: list[Capability]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe. Deliberately answers from configuration alone."""

    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )


def _catalogs(settings: Settings) -> Capability:
    return Capability(
        name="satellite_catalogs",
        ready=bool(settings.stac_base_url),
        detail=(
            f"Sentinel-2 discovery via {settings.stac_base_url}; Sentinel-1 RTC "
            "via its fixed public catalog. Configured - not contacted by this "
            "probe."
        ),
    )


def _geocoder(settings: Settings) -> Capability:
    configured = bool(settings.nominatim_base_url and settings.nominatim_user_agent)
    return Capability(
        name="geocoder",
        ready=configured,
        detail=(
            f"{settings.nominatim_base_url}, at most one request every "
            f"{settings.geocoder_min_interval_seconds:.0f}s application-wide, "
            "with a cache. Configured - not contacted by this probe."
            if configured
            else "No geocoder base URL or User-Agent is configured."
        ),
    )


def _interpretation() -> Capability:
    classifier = load_intent_classifier()
    operation = (
        f"the local intent model {classifier.version} chooses the operation "
        f"when at least {classifier.threshold:.2f} confident and consistent "
        "with the rule-based interpreter, which decides otherwise"
        if classifier is not None
        else "the local intent model is unavailable, so the rule-based "
        "interpreter decides every operation"
    )
    return Capability(
        name="interpretation",
        ready=True,
        detail=(
            "Natural-language questions are interpreted locally by the "
            "standard workflow: NDVI, NDWI, NDBI, SAR backscatter, a two-period "
            f"NDWI comparison or true-colour imagery for a named place and "
            f"period. {operation[0].upper() + operation[1:]}. No external AI "
            "provider or credential is required."
        ),
    )


async def _ai_provider(settings: Settings) -> Capability:
    """The SELECTED provider only - optional, reported but not required.

    Reporting every catalogued provider would invite the reading that any of
    them could answer, and there is deliberately no fallback between them: a
    run that names AI uses the selected backend or it fails.
    """

    capability = await _ai_provider_state(settings)
    return capability.model_copy(update={"required": False})


async def _ai_provider_state(settings: Settings) -> Capability:

    provider = settings.ai_provider
    fields = AI_PROVIDER_FIELDS[provider]
    model = settings.model_for(provider)

    if not settings.is_configured(provider):
        return Capability(
            name="ai_provider",
            ready=False,
            detail=(
                f"{provider} is the selected AI provider and {fields.env_var} "
                "is not set, so optional AI interpretation is unavailable. "
                "Supported questions are answered without it."
            ),
        )

    if provider != "local":
        return Capability(
            name="ai_provider",
            ready=True,
            detail=(
                f"{provider} is configured for model {model}. Configured is not "
                "reachable: nothing here contacts the provider, because a probe "
                "that did would spend quota on every check."
            ),
        )

    installed = await installed_models(settings)
    if installed is None:
        return Capability(
            name="ai_provider",
            ready=False,
            detail=(
                f"local is selected but Ollama is not answering at "
                f"{settings.local_ai_base_url}."
            ),
        )
    if installed is ProbeFailure.MODELS_UNREADABLE:
        return Capability(
            name="ai_provider",
            ready=False,
            detail=(
                "Ollama is running but cannot read its installed models. If "
                "they are stored on an external drive, check that it is "
                "connected."
            ),
        )
    if model not in installed:
        return Capability(
            name="ai_provider",
            ready=False,
            detail=f"{model} is not installed. Install it with: ollama pull {model}",
        )
    return Capability(
        name="ai_provider",
        ready=True,
        detail=f"Ollama is answering and {model} is installed.",
    )


@router.get("/ready", response_model=ReadinessResponse)
async def ready(response: Response) -> ReadinessResponse:
    """Capability probe: can this deployment actually perform its workflow?

    Returns **503 when not ready**, so an orchestrator's default status-code
    semantics keep traffic away from a replica that cannot serve - while the
    body still says exactly which capability is missing, so a human reading it
    (or the UI) learns why rather than just that.
    """

    settings = get_settings()
    capabilities = [
        Capability(
            name="application",
            ready=True,
            detail=f"{settings.app_name} {__version__} ({settings.environment}).",
        ),
        _catalogs(settings),
        _geocoder(settings),
        _interpretation(),
        await _ai_provider(settings),
    ]
    everything = all(
        capability.ready for capability in capabilities if capability.required
    )
    if not everything:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=everything,
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
        capabilities=capabilities,
    )
