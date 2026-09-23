"""Alive is not the same as able.

``/health`` reported ``{"status": "ok"}`` from configuration alone, and the
frontend rendered "Operational" from it. A deployment with no credential for
its selected provider - or a local model that was never installed - answered
exactly the same way, so the one state an operator most needs to see was the
one state the interface could not show.

The split is the point, and each half is pinned here:

* ``/health`` must keep answering 200 when NOTHING is configured, because a
  misconfigured process is still alive and restarting it would not help;
* ``/ready`` must say no, and say WHICH capability is missing.

Also pinned: readiness never contacts a paid provider. A probe that spent quota
to answer a health check would bill the deployment for every poll a monitoring
system makes.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.api.routes import health as health_route
from app.core.config import get_settings
from app.main import create_app
from app.services.agent.providers.local import ProbeFailure
from fastapi.testclient import TestClient


def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def capabilities(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in body["capabilities"]}


def configure_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Liveness stays liveness
# --------------------------------------------------------------------------- #


def test_health_answers_even_when_nothing_is_configured() -> None:
    """No credential is configured in this suite by default."""

    response = client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_says_nothing_about_capability() -> None:
    """It must not grow provider fields: that is the other endpoint's job."""

    body = client().get("/health").json()

    assert set(body) == {"status", "service", "version", "environment"}


# --------------------------------------------------------------------------- #
# Readiness answers the question liveness could not
# --------------------------------------------------------------------------- #


def test_an_unconfigured_provider_is_not_ready_and_names_the_variable() -> None:
    response = client().get("/ready")

    assert response.status_code == 503  # orchestration reads the status code
    body = response.json()
    assert body["ready"] is False
    ai = capabilities(body)["ai_provider"]
    assert ai["ready"] is False
    # Actionable: which variable to set, not merely that something is wrong.
    assert "GEMINI_API_KEY" in ai["detail"]


def test_a_configured_provider_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-vacuity: readiness must not report "no" unconditionally."""

    configure_gemini(monkeypatch)

    response = client().get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert capabilities(body)["ai_provider"]["ready"] is True


def test_readiness_never_claims_a_cloud_provider_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured is not reachable, and the wording must not blur them."""

    configure_gemini(monkeypatch)

    detail = capabilities(client().get("/ready").json())["ai_provider"]["detail"]

    assert "configured" in detail.lower()
    assert "not reachable" in detail.lower() or "is not reachable" in detail.lower()


def test_readiness_does_not_probe_a_cloud_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local probe is the ONLY outbound call readiness may make.

    Asserted by arming the probe and checking it is never used when the
    selected provider is not the local one.
    """

    configure_gemini(monkeypatch)
    touched: list[str] = []

    async def tripwire(*_args: object, **_kwargs: object) -> None:
        touched.append("probed")
        return None

    monkeypatch.setattr(health_route, "installed_models", tripwire)

    assert client().get("/ready").status_code == 200
    assert touched == []


def test_every_advertised_capability_is_reported() -> None:
    body = client().get("/ready").json()

    assert set(capabilities(body)) == {
        "application",
        "satellite_catalogs",
        "geocoder",
        "ai_provider",
    }


# --------------------------------------------------------------------------- #
# The local provider, where a probe IS justified
# --------------------------------------------------------------------------- #


def select_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "local")
    get_settings.cache_clear()


def test_local_is_not_ready_when_ollama_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_local(monkeypatch)  # conftest already patches the probe to "not running"

    response = client().get("/ready")

    assert response.status_code == 503
    assert "not answering" in capabilities(response.json())["ai_provider"]["detail"]


def test_local_is_not_ready_when_the_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_local(monkeypatch)

    async def other_models(*_args: object, **_kwargs: object) -> frozenset[str]:
        return frozenset({"some-other-model"})

    monkeypatch.setattr(health_route, "installed_models", other_models)

    detail = capabilities(client().get("/ready").json())["ai_provider"]["detail"]
    assert "ollama pull" in detail


def test_local_names_the_external_drive_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this deployment actually hits: models on a disconnected disk."""

    select_local(monkeypatch)

    async def unreadable(*_args: object, **_kwargs: object) -> ProbeFailure:
        return ProbeFailure.MODELS_UNREADABLE

    monkeypatch.setattr(health_route, "installed_models", unreadable)

    detail = capabilities(client().get("/ready").json())["ai_provider"]["detail"]
    assert "external drive" in detail


def test_local_is_ready_when_the_model_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_local(monkeypatch)
    model = get_settings().model_for("local")

    async def installed(*_args: object, **_kwargs: object) -> frozenset[str]:
        return frozenset({model})

    monkeypatch.setattr(health_route, "installed_models", installed)

    response = client().get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True
