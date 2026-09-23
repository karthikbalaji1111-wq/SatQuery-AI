"""The suite's configuration is its own, never the developer's.

Measured on this tree before the ``_configuration_is_explicit`` fixture existed:
2112 tests passed on a machine holding a real ``GEMINI_API_KEY``, and 2108
passed with 4 failures without one. Same commit, same command, different
answer - so a green run proved nothing about a fresh checkout, and the failures
would have surfaced first in CI, reading as a regression rather than as the
environment difference they were.

These tests pin the isolation itself, in both directions: nothing leaks IN from
the environment or from ``backend/.env``, and a test that needs a provider
configured can still configure one explicitly. Without the second half, the
first could be satisfied by a fixture that simply broke configuration.

No credential VALUE is ever read or printed here - only variable names and
whether a field ended up set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.config import AI_PROVIDER_FIELDS, Settings, get_settings

#: The repository's own env file, if this machine has one. Present on a
#: developer's checkout, absent in CI - both are normal, and neither may change
#: what the suite asserts.
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _configured_variable_names() -> set[str]:
    """Variable NAMES in the developer's env file. Values are never read."""

    names: set[str] = set()
    for line in ENV_FILE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        names.add(stripped.split("=", 1)[0].strip())
    return names


def test_no_cloud_credential_is_configured_by_default() -> None:
    """Every keyed provider is unconfigured unless a test says otherwise."""

    settings = get_settings()

    for provider, fields in AI_PROVIDER_FIELDS.items():
        if fields.api_key is None:
            continue  # the local provider holds no credential by design
        assert settings.api_key_for(provider) is None, (
            f"{provider} arrived pre-configured; a test's result would then "
            f"depend on whether {fields.env_var} happens to be set"
        )
        assert settings.is_configured(provider) is False


def test_the_local_provider_stays_configured() -> None:
    """Clearing credentials must not disable the keyless provider.

    It is configured by having an endpoint, not a key, so the fixture that
    removes credentials has nothing to remove here. Pinning it keeps the
    isolation from quietly changing what the local path reports.
    """

    assert get_settings().is_configured("local") is True


def test_a_developers_env_file_does_not_reach_the_suite() -> None:
    """A real `.env` on disk changes nothing the suite sees."""

    if not ENV_FILE.exists():
        pytest.skip("no .env on this machine, so there is nothing to leak")

    named = _configured_variable_names()
    assert named, ".env exists but configures nothing; this test proves nothing"

    settings = Settings()
    for provider, fields in AI_PROVIDER_FIELDS.items():
        if fields.api_key is None or fields.env_var not in named:
            continue
        assert settings.api_key_for(provider) is None, (
            f"{fields.env_var} is set in {ENV_FILE.name} and reached Settings"
        )


def test_a_test_can_still_configure_a_provider_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-vacuity half: isolation must not mean configuration is impossible.

    This is what a test needing a configured provider does - an obviously fake
    value, set by the test itself, with the settings cache cleared so the new
    value is actually read.
    """

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.api_key_for("gemini") == "test-key-not-a-real-credential"
    assert settings.is_configured("gemini") is True


def test_that_explicit_configuration_does_not_survive_into_the_next_test() -> None:
    """The dummy set by the test above is gone; ordering cannot leak it."""

    assert get_settings().api_key_for("gemini") is None
