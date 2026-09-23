"""A list setting must survive the way deployments actually write it.

``SATQUERY_CORS_ORIGINS=http://127.0.0.1:5173`` crashed settings construction:

    pydantic_settings.exceptions.SettingsError: error parsing value for field
    "cors_origins" from source "EnvSettingsSource"

pydantic-settings JSON-decodes a complex field BEFORE any field validator runs,
so the repository's own comma-splitting validator was unreachable for the one
source that matters - the environment. The process did not start; it was not a
misconfiguration that degraded gracefully.

Observed live while bringing up a second instance to exercise the rate limit,
and it would equally have crashed the production container: the compose file
sets ``SATQUERY_TRUSTED_ASSET_HOSTS`` as a comma-separated list.

Both string forms are in use and both are pinned here, because "fixing" one by
breaking the other would move the crash rather than remove it.
"""

from __future__ import annotations

import pytest
from app.core.config import Settings
from app.main import create_app

COMMA = "http://one.test:5173,http://two.test:5173"
JSON = '["http://one.test:5173", "http://two.test:5173"]'
EXPECTED = ["http://one.test:5173", "http://two.test:5173"]


def test_a_comma_separated_environment_value_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What a shell export or the production compose file writes."""

    monkeypatch.setenv("SATQUERY_CORS_ORIGINS", COMMA)

    assert Settings().cors_origins == EXPECTED


def test_a_json_array_environment_value_is_still_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the development compose file writes. It must not regress."""

    monkeypatch.setenv("SATQUERY_CORS_ORIGINS", JSON)

    assert Settings().cors_origins == EXPECTED


def test_a_single_value_needs_no_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact value that crashed the process."""

    monkeypatch.setenv("SATQUERY_CORS_ORIGINS", "http://127.0.0.1:5173")

    assert Settings().cors_origins == ["http://127.0.0.1:5173"]


def test_surrounding_whitespace_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A YAML folded scalar (``>-``) yields exactly this shape."""

    monkeypatch.setenv(
        "SATQUERY_TRUSTED_ASSET_HOSTS",
        " sentinel-cogs.s3.us-west-2.amazonaws.com,  .blob.core.windows.net ",
    )

    assert Settings().trusted_asset_hosts == [
        "sentinel-cogs.s3.us-west-2.amazonaws.com",
        ".blob.core.windows.net",
    ]


def test_an_empty_value_is_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_TRUSTED_ASSET_HOSTS", "")

    assert Settings().trusted_asset_hosts == []


def test_a_malformed_json_array_fails_with_a_readable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It still has to fail - just legibly, and for the right reason."""

    monkeypatch.setenv("SATQUERY_CORS_ORIGINS", '["unclosed", ')

    with pytest.raises(Exception) as raised:
        Settings()

    assert "comma-separated" in str(raised.value)


def test_a_programmatic_list_is_untouched() -> None:
    """Non-vacuity: the validator must not mangle a value that is already a list."""

    assert Settings(cors_origins=EXPECTED).cors_origins == EXPECTED


def test_the_application_starts_with_a_comma_separated_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end the failure was actually observed at: the process coming up."""

    monkeypatch.setenv("SATQUERY_CORS_ORIGINS", "http://127.0.0.1:5173")
    monkeypatch.setenv(
        "SATQUERY_TRUSTED_ASSET_HOSTS",
        "sentinel-cogs.s3.us-west-2.amazonaws.com,.blob.core.windows.net",
    )

    app = create_app(Settings())

    assert app is not None
