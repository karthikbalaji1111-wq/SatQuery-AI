"""The model-catalog endpoint.

It exists so the UI can offer a real choice without shipping model ids,
capability rules or - above all - credentials to the browser. These tests pin
that it answers three separate questions separately, and never overstates any
of them.
"""

from __future__ import annotations

from app.api.routes import ai as ai_route
from app.core.config import Settings
from app.main import app
from fastapi.testclient import TestClient


def _catalog(client: TestClient, role: str = "visual") -> dict:
    response = client.get(f"/api/v1/ai/models?role={role}")
    assert response.status_code == 200
    return response.json()


def _with_settings(settings: Settings) -> TestClient:
    app.dependency_overrides.clear()
    ai_route.get_settings = lambda: settings  # type: ignore[assignment]
    return TestClient(app)


def test_the_catalog_never_returns_a_credential() -> None:
    client = _with_settings(
        Settings(  # type: ignore[arg-type]
            _env_file=None,
            GEMINI_API_KEY="gemini-secret-value",
            NVIDIA_API_KEY="nvidia-secret-value",
        )
    )
    body = client.get("/api/v1/ai/models").text

    assert "gemini-secret-value" not in body
    assert "nvidia-secret-value" not in body
    assert "api_key" not in body.lower()


def test_unconfigured_provider_is_reported_as_not_configured() -> None:
    client = _with_settings(
        Settings(_env_file=None, GEMINI_API_KEY="g")  # type: ignore[arg-type]
    )
    models = _catalog(client)["models"]

    nvidia = [m for m in models if m["provider"] == "nvidia"]
    assert nvidia
    for model in nvidia:
        assert model["configured"] is False
        if model["supports_image"]:
            assert model["status"] == "Not configured"


def test_a_configured_provider_is_not_claimed_to_be_reachable() -> None:
    """Nothing here contacts a provider, so nothing may claim availability."""

    client = _with_settings(
        Settings(_env_file=None, NVIDIA_API_KEY="n", AI_PROVIDER="nvidia")  # type: ignore[arg-type]
    )
    models = _catalog(client)["models"]

    ready = [m for m in models if m["status"] == "Ready"]
    assert ready
    for model in models:
        assert model["status"] != "Available"


def test_text_only_models_are_flagged_unsupported_for_visual() -> None:
    client = _with_settings(
        Settings(_env_file=None, NVIDIA_API_KEY="n")  # type: ignore[arg-type]
    )
    models = _catalog(client, "visual")["models"]

    text_only = [m for m in models if not m["supports_image"]]
    assert text_only
    for model in text_only:
        assert model["compatible"] is False
        assert model["status"] == "Unsupported for visual analysis"


def test_the_same_models_are_compatible_for_a_text_role() -> None:
    client = _with_settings(
        Settings(_env_file=None, NVIDIA_API_KEY="n")  # type: ignore[arg-type]
    )
    models = _catalog(client, "text")["models"]

    for model in models:
        if model["supports_text"]:
            assert model["compatible"] is True


def test_the_catalog_reports_the_configured_default() -> None:
    client = _with_settings(
        Settings(  # type: ignore[arg-type]
            _env_file=None,
            AI_PROVIDER="nvidia",
            NVIDIA_API_KEY="n",
            NVIDIA_MODEL="nvidia/nemotron-nano-12b-v2-vl",
        )
    )
    body = _catalog(client)

    assert body["default_provider"] == "nvidia"
    assert body["default_model"] == "nvidia/nemotron-nano-12b-v2-vl"


def test_every_option_carries_the_capability_metadata_the_ui_filters_on() -> None:
    client = _with_settings(Settings(_env_file=None))  # type: ignore[arg-type]
    models = _catalog(client)["models"]

    assert models
    for model in models:
        for field in (
            "provider",
            "model_id",
            "display_name",
            "modality",
            "supports_image",
            "supports_text",
            "supports_video",
            "supports_tools",
            "supports_structured_output",
            "endpoint_type",
            "configured",
            "compatible",
            "status",
        ):
            assert field in model
