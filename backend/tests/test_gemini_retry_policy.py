"""What Gemini retries, and what it refuses to retry.

The distinction matters because the two retryable conditions are not alike:

* **Overload** (500/502/503/504) is transient contention. A moment later often
  works, so a short backoff is worth its latency.
* **Quota** (429) is a budget that has run out over a window of seconds to
  minutes. Retrying inside that window cannot succeed - and each attempt spends
  another unit of the very allowance that was exhausted.

Measured before this policy existed: a 429 carrying no ``RetryInfo`` burned
three attempts in 2.01 s, tripling the cost of a request that could never have
succeeded. On a free-tier key that is the difference between a demo that runs
and one that does not.

When the server DOES state a delay, that number is authoritative and is honoured
up to the provider's wait budget - tested separately below.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from app.core.errors import UpstreamServiceError
from app.services.agent.providers import gemini


class _FakeAPIError(Exception):
    """Stands in for ``genai_errors.APIError`` with a code and RetryInfo."""

    def __init__(self, code: int, retry_delay: str | None = None) -> None:
        super().__init__("upstream")
        self.code = code
        # Mirrors the google-genai error body exactly: a dict whose "error"
        # holds a "details" LIST. An earlier version of this fake used a bare
        # list and silently exercised the no-delay path instead, which would
        # have made the honoured-delay test vacuous.
        self.details = (
            {
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": retry_delay,
                        }
                    ]
                }
            }
            if retry_delay
            else {}
        )


def _run(code: int, retry_delay: str | None, monkeypatch: pytest.MonkeyPatch):
    """Drive `_generate` against an endpoint that always fails, and count tries."""

    monkeypatch.setattr(gemini.genai_errors, "APIError", _FakeAPIError, raising=False)
    attempts = {"n": 0}

    class _Models:
        async def generate_content(self, **_: object):
            attempts["n"] += 1
            raise _FakeAPIError(code, retry_delay)

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    started = time.monotonic()
    with pytest.raises(UpstreamServiceError) as caught:
        asyncio.run(
            gemini._generate(_Client(), model="m", contents="c", config=None, role="planning")
        )
    return attempts["n"], time.monotonic() - started, caught.value


def test_a_quota_error_without_a_stated_delay_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One attempt, not three. The other two could only waste quota."""

    attempts, elapsed, error = _run(429, None, monkeypatch)

    assert attempts == 1
    assert elapsed < 0.5, "a quota failure must not sleep before reporting"
    assert "rate limited" in error.message


def test_an_overload_error_is_still_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counter-case: narrowing 429 must not disable retry generally.

    Without this, deleting the retry loop entirely would still pass the test
    above - the policy would look correct while having lost the behaviour it
    exists to provide.
    """

    attempts, _, _ = _run(503, None, monkeypatch)

    assert attempts == gemini._MAX_ATTEMPTS
    assert attempts > 1


def test_a_quota_error_naming_a_short_delay_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server-stated delay inside the budget is authoritative, and is waited.

    This is what keeps the rule above narrow: it declines to GUESS a delay, it
    does not ignore one the server supplied.
    """

    attempts, elapsed, _ = _run(429, "0.4s", monkeypatch)

    assert attempts == gemini._MAX_ATTEMPTS
    assert elapsed >= 0.4


def test_a_quota_error_naming_a_long_delay_is_reported_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Beyond the wait budget the honest answer is the failure, not a held socket."""

    attempts, elapsed, error = _run(429, "600s", monkeypatch)

    assert attempts == 1
    assert elapsed < 0.5
    assert "600" in error.message or "rate limited" in error.message


def test_a_non_retryable_client_error_is_reported_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts, _, _ = _run(400, None, monkeypatch)

    assert attempts == 1


# --------------------------------------------------------------------------- #
# How the wait is WORDED. A number the reader cannot act on is not guidance.
# --------------------------------------------------------------------------- #


def test_a_sub_second_delay_is_not_reported_as_zero_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini really does return "0s", and it reached the UI as "0 seconds".

    Seen live in the browser: "Please retry in about 0 seconds", beside "The
    provider requested a wait of 0 seconds before retrying." Both are literally
    derived from the server's own number, and both read as a broken field rather
    than as advice. A wait shorter than a second is reported in words instead.
    """

    _, _, error = _run(429, "0s", monkeypatch)

    assert "0 seconds" not in error.message
    assert "shortly" in error.message
    assert "rate limited" in error.message


def test_a_delay_of_a_second_or_more_still_quotes_the_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counter-case: the rule above must not silence every useful delay."""

    _, _, error = _run(429, "45s", monkeypatch)

    assert "45 seconds" in error.message
    assert "shortly" not in error.message
