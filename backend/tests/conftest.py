"""Shared test fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from app.core.config import Settings, get_settings
from app.main import create_app
from app.services.geospatial.nominatim import reset_geocoder_state
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


#: Every variable that can name a provider, a credential, an endpoint or a
#: model. ``SATQUERY_`` is the settings prefix, so it is cleared wholesale.
_CONFIGURATION_PREFIXES = ("SATQUERY_",)
_CONFIGURATION_NAMES = (
    "AI_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "NVIDIA_API_KEY",
    "NVIDIA_MODEL",
    "NVIDIA_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_BASE_URL",
    "LOCAL_AI_BASE_URL",
    "LOCAL_AI_MODEL",
)


@pytest.fixture(autouse=True)
def _configuration_is_explicit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test against a deployment that holds no credentials.

    **The defect this closes.** ``Settings`` reads the process environment and
    ``backend/.env``, and ``get_settings`` caches the result for the process. A
    developer with a real ``GEMINI_API_KEY`` therefore ran a DIFFERENT suite
    from a fresh checkout: measured on this tree, 2112 passed with the key
    present and 4 failed without it. A test suite whose result depends on who is
    running it cannot gate a release - the failures appear first in CI, where
    they read as a regression rather than as the environment difference they
    are.

    Both sources are closed here. Provider variables are removed from the
    environment, and ``env_file`` is unset so the developer's own ``.env`` - a
    DEPLOYMENT's configuration, never a test fixture - is not read at all.

    The semantics are not weakened, which is the point: no credential is
    invented, so a test that asserts the unconfigured behaviour still exercises
    it honestly. A test that needs a provider configured says so itself, with an
    obviously fake value, and clears the cache after setting it:

        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")
        get_settings.cache_clear()

    The cache is cleared on the way in AND out, so neither a leaked real value
    nor a test's own dummy can survive into the next test.
    """

    for name in list(os.environ):
        if name in _CONFIGURATION_NAMES or name.startswith(_CONFIGURATION_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    # The geocoder's politeness spacing exists for the public Nominatim
    # instance, and no test contacts it - every geocode in this suite runs
    # against an injected transport. Left at one second, the suite would sleep
    # once per geocoding test for no benefit to anybody. Set to zero
    # DELIBERATELY and explicitly here, not defaulted away: the throttle itself
    # is proven in tests/test_geocoder_policy.py, which configures a real
    # interval and asserts the wait actually happens.
    monkeypatch.setenv("SATQUERY_GEOCODER_MIN_INTERVAL_SECONDS", "0")

    get_settings.cache_clear()
    reset_geocoder_state()
    yield
    get_settings.cache_clear()
    reset_geocoder_state()


@pytest.fixture(autouse=True)
def _no_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the model catalog hermetic: never ask a real local Ollama.

    The catalog asks the local provider's endpoint which models are installed.
    Left live, the suite would call whatever Ollama happens to be running on
    the developer's machine, and a status assertion would pass or fail with it.
    Tests that care about that answer patch the probe themselves.
    """

    from app.api.routes import ai as ai_route
    from app.api.routes import health as health_route

    async def not_running(*_args: object, **_kwargs: object) -> None:
        return None

    # Both routes hold their own reference to the probe, so both are patched.
    # The readiness endpoint asks the same question for the same reason, and a
    # suite that reached a developer's running Ollama would pass or fail with
    # whatever happened to be installed on it.
    monkeypatch.setattr(ai_route, "installed_models", not_running)
    monkeypatch.setattr(health_route, "installed_models", not_running)
