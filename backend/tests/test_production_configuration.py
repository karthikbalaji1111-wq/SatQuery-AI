"""Production must fail before serving requests with development defaults."""

import pytest
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient
from pydantic import ValidationError


def production(**overrides: object) -> Settings:
    values = {
        "environment": "production",
        "cors_origins": ["https://satquery.example"],
        "trusted_asset_hosts": ["sentinel-cogs.s3.us-west-2.amazonaws.com"],
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize("environment", ["production", "prod", " Production "])
def test_production_defaults_fail(environment: str) -> None:
    with pytest.raises(ValidationError, match="SATQUERY_CORS_ORIGINS"):
        Settings(environment=environment)


@pytest.mark.parametrize("origin", [
    "http://satquery.example", "https://localhost", "https://app.localhost",
    "https://localhost.", "https://127.0.0.1", "https://[::1]",
    "https://10.0.0.1", "https://backend", "https://host.docker.internal",
    "https://app.local", "*", "https://*.example.com",
    "https://user:secret@satquery.example", "https://satquery.example/path",
    "https://satquery.example/", "https://satquery.example?key=secret",
    "https://satquery.example#fragment", "https://satquery.example:bad",
])
def test_unsafe_production_origins_fail(origin: str) -> None:
    with pytest.raises(ValidationError, match="SATQUERY_CORS_ORIGINS"):
        production(cors_origins=[origin])


def test_empty_origins_fail() -> None:
    with pytest.raises(ValidationError, match="SATQUERY_CORS_ORIGINS"):
        production(cors_origins=[])


@pytest.mark.parametrize("hosts", [[], ["*"], [" "]])
def test_production_requires_asset_allowlist(hosts: list[str]) -> None:
    with pytest.raises(ValidationError, match="TRUSTED_ASSET_HOSTS"):
        production(trusted_asset_hosts=hosts)


@pytest.mark.parametrize("field", [
    "stac_base_url", "nominatim_base_url", "NVIDIA_BASE_URL", "ANTHROPIC_BASE_URL",
])
def test_public_upstream_http_is_refused(field: str) -> None:
    with pytest.raises(ValidationError, match="requires HTTPS"):
        production(**{field: "http://api.example.com"})


def test_valid_production_app_serves_and_enforces_cors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_ENVIRONMENT", "production")
    monkeypatch.setenv("SATQUERY_CORS_ORIGINS", "https://satquery.example")
    monkeypatch.setenv("SATQUERY_TRUSTED_ASSET_HOSTS", "sentinel-cogs.s3.us-west-2.amazonaws.com")
    with TestClient(create_app(Settings())) as client:
        assert client.get("/health").status_code == 200
        good = client.options("/api/v1/query/agent", headers={
            "Origin": "https://satquery.example", "Access-Control-Request-Method": "POST",
        })
        assert good.status_code == 200
        assert good.headers["access-control-allow-origin"] == "https://satquery.example"
        bad = client.options("/api/v1/query/agent", headers={
            "Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
        })
        assert bad.status_code == 400
        assert "access-control-allow-origin" not in bad.headers


def test_development_and_private_ollama_remain_supported() -> None:
    assert Settings().cors_origins[0] == "http://localhost:5173"
    assert production(LOCAL_AI_BASE_URL="http://ollama:11434").is_production
