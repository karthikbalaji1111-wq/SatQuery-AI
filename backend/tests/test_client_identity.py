"""Who a request is attributed to, behind zero or more trusted proxies.

Behind a hosting platform's load balancer the TCP peer is the balancer, so a
peer-keyed limiter gives every user one shared allowance. Counting trusted
proxy hops from the RIGHT of X-Forwarded-For recovers the caller without ever
trusting an entry the caller wrote.
"""

from __future__ import annotations

import pytest
from app.core.config import Settings
from app.core.errors import register_exception_handlers
from app.core.limits import _client_of, install_request_limits, rate_limited
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request


def _request(forwarded: str | None, peer: str = "10.0.0.1") -> Request:
    headers = [] if forwarded is None else [(b"x-forwarded-for", forwarded.encode())]
    return Request({"type": "http", "headers": headers, "client": (peer, 4321)})


def test_with_no_trusted_proxy_the_header_is_ignored() -> None:
    assert _client_of(_request("203.0.113.9"), 0) == "10.0.0.1"


def test_one_hop_uses_the_entry_the_proxy_appended() -> None:
    assert _client_of(_request("203.0.113.9"), 1) == "203.0.113.9"


def test_entries_the_caller_wrote_cannot_change_the_identity() -> None:
    first = _client_of(_request("198.51.100.1, 203.0.113.9"), 1)
    second = _client_of(_request("198.51.100.2, 198.51.100.3, 203.0.113.9"), 1)
    assert first == second == "203.0.113.9"


def test_two_hops_skip_the_inner_proxy() -> None:
    assert _client_of(_request("198.51.100.7, 203.0.113.9, 192.0.2.4"), 2) == "203.0.113.9"


@pytest.mark.parametrize("forwarded", [None, "", "203.0.113.9"])
def test_a_chain_shorter_than_the_hop_count_falls_back_to_the_peer(
    forwarded: str | None,
) -> None:
    assert _client_of(_request(forwarded), 2) == "10.0.0.1"


@pytest.mark.parametrize("entry", ["not-an-ip", "unknown", "203.0.113.9:80"])
def test_an_entry_that_is_not_an_address_falls_back_to_the_peer(entry: str) -> None:
    assert _client_of(_request(f"198.51.100.1, {entry}"), 1) == "10.0.0.1"


def test_ipv6_entries_are_normalised() -> None:
    assert _client_of(_request("2001:DB8::1"), 1) == "2001:db8::1"


def _app(hops: int) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    install_request_limits(app, Settings(rate_limit_requests=2, trusted_proxy_hops=hops))

    @app.get("/work", dependencies=[Depends(rate_limited)])
    def work() -> dict[str, str]:
        return {"ok": "yes"}

    return TestClient(app)


def test_behind_a_proxy_two_callers_get_separate_allowances() -> None:
    client = _app(hops=1)
    for _ in range(2):
        assert client.get("/work", headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 200
    assert client.get("/work", headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 429
    # Another caller behind the same proxy is unaffected.
    assert client.get("/work", headers={"X-Forwarded-For": "203.0.113.2"}).status_code == 200


def test_spoofing_the_header_does_not_buy_a_fresh_allowance() -> None:
    client = _app(hops=1)
    statuses = [
        client.get("/work", headers={"X-Forwarded-For": f"198.51.100.{i}, 203.0.113.1"}).status_code
        for i in range(3)
    ]
    assert statuses == [200, 200, 429]


def test_without_trusted_proxies_a_forged_header_changes_nothing() -> None:
    client = _app(hops=0)
    statuses = [
        client.get("/work", headers={"X-Forwarded-For": f"203.0.113.{i}"}).status_code
        for i in range(3)
    ]
    assert statuses == [200, 200, 429]
