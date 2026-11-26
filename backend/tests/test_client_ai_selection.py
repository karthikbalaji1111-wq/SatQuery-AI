"""A caller cannot route work to an AI provider where the deployment forbids it.

The API has no accounts. On a public deployment, letting an anonymous caller
name a provider and model lets anyone spend whatever metered credential the
operator configured. Production therefore refuses caller-chosen AI by default;
the standard workflow, which needs no provider, is unaffected.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.api.routes import ai as ai_route
from app.api.routes import query as query_route
from app.core.config import Settings
from app.main import app
from fastapi.testclient import TestClient

PRODUCTION = {
    "environment": "production",
    "cors_origins": ["https://satquery.example.com"],
    "trusted_asset_hosts": ["sentinel-cogs.s3.us-west-2.amazonaws.com"],
}


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.fixture
def client_with(monkeypatch: pytest.MonkeyPatch):
    def build(settings: Settings) -> TestClient:
        monkeypatch.setattr(query_route, "get_settings", lambda: settings)
        monkeypatch.setattr(ai_route, "get_settings", lambda: settings)
        app.dependency_overrides.clear()
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


def test_the_default_is_open_in_development_and_closed_in_production() -> None:
    assert _settings().client_ai_selection_enabled is True
    assert _settings(**PRODUCTION).client_ai_selection_enabled is False


def test_an_explicit_setting_wins_either_way() -> None:
    assert _settings(**PRODUCTION, client_ai_selection=True).client_ai_selection_enabled
    assert not _settings(client_ai_selection=False).client_ai_selection_enabled


@pytest.mark.parametrize(
    "body",
    [
        {"question": "NDVI at Cubbon Park, Bengaluru in December 2024", "provider": "gemini"},
        {"question": "NDVI at Cubbon Park, Bengaluru in December 2024", "model": "claude-opus-5"},
    ],
)
def test_the_agent_refuses_a_named_provider_or_model(
    client_with, monkeypatch: pytest.MonkeyPatch, body: dict[str, str]
) -> None:
    built: list[Any] = []
    monkeypatch.setattr(
        query_route, "build_agent_service", lambda *a, **k: built.append(a) or None
    )
    client = client_with(_settings(client_ai_selection=False))

    response = client.post("/api/v1/query/agent", json=body)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "ai_selection_disabled"
    assert built == [], "no provider may be constructed for a refused request"


def test_parse_refuses_a_named_provider(client_with, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        query_route, "get_intent_parser", lambda **k: pytest.fail("must not build a parser")
    )
    client = client_with(_settings(client_ai_selection=False))

    response = client.post(
        "/api/v1/query/parse", json={"prompt": "NDVI at Cubbon Park", "provider": "gemini"}
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "ai_selection_disabled"


def test_the_catalog_tells_the_client_which_mode_it_is_in(client_with) -> None:
    closed = client_with(_settings(client_ai_selection=False)).get("/api/v1/ai/models")
    opened = client_with(_settings(client_ai_selection=True)).get("/api/v1/ai/models")

    assert closed.json()["client_selection_enabled"] is False
    assert opened.json()["client_selection_enabled"] is True


def test_render_does_not_reopen_ai_selection() -> None:
    from pathlib import Path

    render = (Path(__file__).resolve().parents[2] / "render.yaml").read_text()
    assert "SATQUERY_CLIENT_AI_SELECTION" not in render
