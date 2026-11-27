"""One caller cannot spend the application's whole geocoder budget.

The geocoder's budget is shared by every user of the deployment (and, on a
cloud host, by strangers on the same outbound IP). Only NEW places count
against a caller: a cached place, or one already being resolved for someone
else, reaches no upstream service and is never refused.
"""

from __future__ import annotations

import pytest
from app.core.errors import RateLimitedError
from app.core.limits import CALLER
from app.services.geospatial.nominatim import geocode, geocoder_status

from tests.test_geocoder_reliability import Upstream, clock, ok, run, settings

__all__ = ["clock"]  # the autouse fake clock and state reset


def _as(caller: str | None, query: str, upstream: Upstream, **overrides: object):
    token = CALLER.set(caller)
    try:
        return run(
            geocode(
                query,
                settings=settings(**({"geocoder_new_places_per_caller": 2} | overrides)),
                transport=upstream.transport(),
            )
        )
    finally:
        CALLER.reset(token)


def test_a_caller_may_look_up_its_allowance_of_new_places() -> None:
    upstream = Upstream(ok(), ok())
    _as("203.0.113.1", "Cubbon Park, Bengaluru", upstream)
    _as("203.0.113.1", "Lalbagh, Bengaluru", upstream)
    with pytest.raises(RateLimitedError) as refused:
        _as("203.0.113.1", "Ulsoor Lake, Bengaluru", upstream)
    assert refused.value.retry_after_seconds > 0
    assert geocoder_status().upstream_requests == 2


def test_cached_places_are_never_refused() -> None:
    upstream = Upstream(ok(), ok())
    _as("203.0.113.1", "Cubbon Park, Bengaluru", upstream)
    _as("203.0.113.1", "Lalbagh, Bengaluru", upstream)
    for _ in range(5):
        _as("203.0.113.1", "Cubbon Park, Bengaluru", upstream)
    assert geocoder_status().upstream_requests == 2


def test_another_caller_is_unaffected() -> None:
    upstream = Upstream(ok(), ok(), ok())
    _as("203.0.113.1", "Cubbon Park, Bengaluru", upstream)
    _as("203.0.113.1", "Lalbagh, Bengaluru", upstream)
    _as("198.51.100.7", "Ulsoor Lake, Bengaluru", upstream)


def test_the_allowance_returns_after_the_window(clock) -> None:  # noqa: F811
    upstream = Upstream(ok(), ok(), ok())
    _as("203.0.113.1", "Cubbon Park, Bengaluru", upstream)
    _as("203.0.113.1", "Lalbagh, Bengaluru", upstream)
    clock.t += 601.0
    _as("203.0.113.1", "Ulsoor Lake, Bengaluru", upstream)


def test_no_caller_means_no_cap() -> None:
    # The background warm-up runs outside any request.
    upstream = Upstream(ok(), ok(), ok())
    for place in ("Cubbon Park, Bengaluru", "Lalbagh, Bengaluru", "Ulsoor Lake, Bengaluru"):
        _as(None, place, upstream)
    assert geocoder_status().upstream_requests == 3


def test_zero_disables_the_cap() -> None:
    upstream = Upstream(ok(), ok(), ok())
    for place in ("Cubbon Park, Bengaluru", "Lalbagh, Bengaluru", "Ulsoor Lake, Bengaluru"):
        _as("203.0.113.1", place, upstream, geocoder_new_places_per_caller=0)
