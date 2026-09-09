"""Provider credentials must never reach a client, a body, or a log line.

The failure this guards against is quiet: a provider error body echoed into a
response, an exception rendered with its context, or a debug log of the request
would each publish an API key to whoever asked. None of that shows up as a
broken feature - the product keeps working while leaking.

Every check below uses a sentinel key with a distinctive shape. A single
substring search for that sentinel is a stronger assertion than checking for
field names: it catches the key wherever it surfaces, including places nobody
thought to name.

Covers the NVIDIA paths (unconfigured, unreachable, and rejected-by-provider)
and the equivalent Gemini paths where one exists.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import httpx
import pytest
from app.api.routes import ai as ai_route
from app.core.config import Settings, get_settings
from app.core.errors import UpstreamServiceError
from app.main import app, create_app
from app.services.agent.providers import factory as factory_mod
from app.services.agent.providers.factory import get_agent_providers
from app.services.agent.providers.nvidia import NvidiaVisualAnalyst
from fastapi.testclient import TestClient

#: Distinctive enough that a substring hit cannot be a coincidence, and shaped
#: like a real NVIDIA key so a naive redactor would not special-case it.
NVIDIA_SENTINEL = "nvapi-SENTINEL-must-never-be-disclosed-0123456789"
GEMINI_SENTINEL = "AIza-SENTINEL-must-never-be-disclosed-0123456789"

#: RFC 863 discard port: refused immediately, so "unavailable" needs no network.
UNREACHABLE = "http://127.0.0.1:9/v1"

PNG = b"\x89PNG\r\n\x1a\nfake-bytes"
QUESTION = "Is there visible water in this image?"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "GEMINI_API_KEY": GEMINI_SENTINEL,
        "NVIDIA_API_KEY": NVIDIA_SENTINEL,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _assert_clean(text: str, where: str) -> None:
    """No sentinel, and no obvious credential-bearing header either."""

    assert NVIDIA_SENTINEL not in text, f"NVIDIA key disclosed in {where}"
    assert GEMINI_SENTINEL not in text, f"Gemini key disclosed in {where}"
    lowered = text.lower()
    assert "nvapi-" not in lowered, f"key-shaped material in {where}"
    assert "authorization" not in lowered, f"auth header echoed in {where}"


# --------------------------------------------------------------------------- #
# The frontend-facing catalog
# --------------------------------------------------------------------------- #


def test_the_model_catalog_response_carries_no_credential(
    monkeypatch: Any,
) -> None:
    """The one provider/model payload the browser receives."""

    monkeypatch.setattr(ai_route, "get_settings", lambda: _settings())
    app.dependency_overrides.clear()
    with TestClient(app) as client:
        response = client.get("/api/v1/ai/models?role=visual")

    assert response.status_code == 200
    _assert_clean(response.text, "the model catalog response")


# --------------------------------------------------------------------------- #
# NVIDIA unconfigured
# --------------------------------------------------------------------------- #


def test_unconfigured_nvidia_names_the_variable_not_a_value(
    monkeypatch: Any,
) -> None:
    """The error must be actionable without being disclosive."""

    settings = _settings(AI_PROVIDER="nvidia", NVIDIA_API_KEY=None)
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)

    app.dependency_overrides.clear()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": "nvidia"},
        )

    body = response.text
    # Names the variable, so an operator can fix it...
    assert "NVIDIA_API_KEY" in body
    # ...but discloses nothing, including the OTHER provider's key, which is
    # configured in this fixture and must not travel with an unrelated error.
    _assert_clean(body, "the unconfigured-provider error")


# --------------------------------------------------------------------------- #
# NVIDIA configured but unreachable
# --------------------------------------------------------------------------- #


def test_an_unreachable_provider_never_echoes_its_credential(
    monkeypatch: Any,
) -> None:
    """A transport failure carries the request context; the response must not."""

    settings = _settings(AI_PROVIDER="nvidia", NVIDIA_BASE_URL=UNREACHABLE)
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)

    app.dependency_overrides.clear()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": "nvidia"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "planner_unavailable"
    _assert_clean(response.text, "the unreachable-provider response")


def test_an_unreachable_provider_logs_no_credential(
    monkeypatch: Any, caplog: Any
) -> None:
    """Logs outlive responses, so they are the more dangerous surface."""

    settings = _settings(AI_PROVIDER="nvidia", NVIDIA_BASE_URL=UNREACHABLE)
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)

    app.dependency_overrides.clear()
    with caplog.at_level(logging.DEBUG), TestClient(app) as client:
        client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": "nvidia"},
        )

    _assert_clean(caplog.text, "the application log")


# --------------------------------------------------------------------------- #
# NVIDIA rejecting the credential
# --------------------------------------------------------------------------- #


class _RejectingTransport(httpx.AsyncBaseTransport):
    """A provider that refuses the key and echoes the request back.

    Deliberately hostile: real providers do return the offending request in an
    error body, and this is exactly the shape that leaks a key if the body is
    passed through.
    """

    def __init__(self, status_code: int) -> None:
        self._status = status_code

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            self._status,
            json={
                "error": {
                    "message": "Invalid API key",
                    "authorization": request.headers.get("authorization"),
                    "echo": request.headers.get("authorization"),
                }
            },
            request=request,
        )


@pytest.mark.parametrize("status", [401, 403, 429])
def test_a_rejected_credential_is_not_repeated_in_the_exception(
    status: int,
) -> None:
    """The exception text is what reaches a caller, a log and a response."""

    transport = _RejectingTransport(status)

    async def run() -> Exception:
        async with httpx.AsyncClient(transport=transport) as client:
            analyst = NvidiaVisualAnalyst(
                settings=_settings(AI_PROVIDER="nvidia"), client=client
            )
            try:
                await analyst.observe(
                    question=QUESTION, image=PNG, media_type="image/png"
                )
            except Exception as exc:
                return exc
            raise AssertionError("a rejected credential must raise")

    error = asyncio.run(run())
    assert isinstance(error, UpstreamServiceError)
    _assert_clean(str(error), "the provider exception text")
    _assert_clean(error.message, "the provider exception message")


def test_a_rejected_credential_is_not_logged(caplog: Any) -> None:
    transport = _RejectingTransport(401)

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            analyst = NvidiaVisualAnalyst(
                settings=_settings(AI_PROVIDER="nvidia"), client=client
            )
            with pytest.raises(UpstreamServiceError):
                await analyst.observe(
                    question=QUESTION, image=PNG, media_type="image/png"
                )

    with caplog.at_level(logging.DEBUG):
        asyncio.run(run())

    _assert_clean(caplog.text, "the provider log output")


# --------------------------------------------------------------------------- #
# The equivalent Gemini path
# --------------------------------------------------------------------------- #


def test_unconfigured_gemini_names_the_variable_not_a_value() -> None:
    """Gemini's equivalent error path, held to the same standard."""

    with pytest.raises(UpstreamServiceError) as caught:
        get_agent_providers(
            settings=_settings(AI_PROVIDER="gemini", GEMINI_API_KEY=None),
            provider="gemini",
        )

    assert "GEMINI_API_KEY" in caught.value.message
    _assert_clean(str(caught.value), "the Gemini configuration error")


def test_neither_providers_key_travels_with_the_others_failure() -> None:
    """A failure in one provider must not disclose the other's credential."""

    for provider, missing in (
        ("nvidia", "NVIDIA_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
    ):
        with pytest.raises(UpstreamServiceError) as caught:
            get_agent_providers(
                settings=_settings(AI_PROVIDER=provider, **{missing: None}),
                provider=provider,
            )
        _assert_clean(str(caught.value), f"the {provider} configuration error")


def test_a_configured_key_never_reaches_a_response_or_the_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canary key must survive nowhere a user or an operator can read it.

    Provider errors are the realistic leak path: an SDK or transport exception
    often carries the request - URL, headers, sometimes the credential - and
    forwarding ``str(exc)`` upstream would publish it. This drives a real
    failure (an unroutable base URL) through the full HTTP stack and asserts
    the key appears in neither the response body nor anything logged.
    """

    canary = "sk-CANARY-d41d8cd98f00b204e9800998ecf8427e"
    monkeypatch.setenv("AI_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", canary)
    monkeypatch.setenv("GEMINI_API_KEY", canary)
    # Port 9 (discard) is reserved and unroutable, so the provider fails at the
    # transport layer without contacting anything real.
    monkeypatch.setenv("NVIDIA_BASE_URL", "http://127.0.0.1:9/v1")
    get_settings.cache_clear()

    records = io.StringIO()
    handler = logging.StreamHandler(records)
    root = logging.getLogger()
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        client = TestClient(create_app(), raise_server_exceptions=False)
        response = client.post(
            "/api/v1/query/agent", json={"question": "show me Chennai"}
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        # The canary must not survive into any later test's settings.
        get_settings.cache_clear()

    # The request fails, controlled - that is the precondition for the test.
    assert response.status_code == 200
    assert response.json()["status"] == "planner_unavailable"

    logged = records.getvalue()
    assert canary not in response.text
    assert canary not in logged
    # Nothing may leak a fragment either, which a naive truncation would.
    assert "CANARY" not in response.text
    assert "CANARY" not in logged


# --------------------------------------------------------------------------- #
# The request's own provider choice must survive an unconfigured default
# --------------------------------------------------------------------------- #
#
# FastAPI resolves a dependency BEFORE the handler body runs, so building the
# configured default provider there decided the run before the request's own
# `provider` had been read. A deployment holding only an NVIDIA key therefore
# could not use NVIDIA: constructing the unconfigured Gemini default failed
# first, and the 502 named Gemini - a provider the caller never asked for.


def test_an_explicit_provider_is_honoured_when_the_default_is_unconfigured(
    monkeypatch: Any,
) -> None:
    settings = _settings(
        AI_PROVIDER="gemini",
        GEMINI_API_KEY=None,  # the DEFAULT cannot be built
        NVIDIA_API_KEY=NVIDIA_SENTINEL,  # but the REQUESTED provider can
        NVIDIA_BASE_URL="http://127.0.0.1:9/v1",  # discard port: unreachable
    )
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)

    app.dependency_overrides.clear()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": "nvidia"},
        )

    # NVIDIA was actually selected and actually attempted: the run reaches the
    # in-band status rather than dying on the default's missing credential.
    assert response.status_code == 200
    assert response.json()["status"] == "planner_unavailable"
    # And the unconfigured default's variable is not blamed for it.
    assert "GEMINI_API_KEY" not in response.text
    _assert_clean(response.text, "the explicit-provider run")


def test_an_unconfigured_explicit_provider_names_itself_not_the_default(
    monkeypatch: Any,
) -> None:
    """The 502 must name the provider that was selected, not the default.

    Naming the wrong variable sends an operator to fix a setting that was never
    consulted.
    """

    settings = _settings(
        AI_PROVIDER="gemini", GEMINI_API_KEY=None, NVIDIA_API_KEY=None
    )
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)

    app.dependency_overrides.clear()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": "nvidia"},
        )

    assert response.status_code == 502
    assert "NVIDIA_API_KEY" in response.text
    assert "GEMINI_API_KEY" not in response.text
    _assert_clean(response.text, "the unconfigured-override error")
