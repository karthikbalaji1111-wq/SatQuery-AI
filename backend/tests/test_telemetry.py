"""Request ids, access lines, metrics and JSON logs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pytest
from app.core import telemetry
from app.core.config import Settings
from app.core.logging import JsonFormatter
from app.core.observability import RunIdFilter, workflow
from app.main import create_app
from fastapi.testclient import TestClient
from pydantic import SecretStr


def _client(monkeypatch: pytest.MonkeyPatch, token: str | None = None) -> TestClient:
    settings = Settings(_env_file=None, metrics_token=SecretStr(token) if token else None)  # type: ignore[arg-type]
    monkeypatch.setattr(telemetry, "get_settings", lambda: settings)
    return TestClient(create_app(settings))


def test_every_response_carries_a_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    ok = client.get("/health")
    missing = client.get("/no-such-page")
    refused = client.post("/api/v1/query/parse", json={"prompt": "x"})

    ids = [r.headers.get("x-request-id") for r in (ok, missing, refused)]
    assert all(re.fullmatch(r"[0-9a-f]{12}", i or "") for i in ids)
    assert len(set(ids)) == 3


def test_the_request_id_is_readable_by_the_browser_app(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    exposed = response.headers.get("access-control-expose-headers", "").lower()
    assert "x-request-id" in exposed
    assert "retry-after" in exposed


def test_an_oversized_body_refused_by_middleware_still_gets_an_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/query/parse",
        content=b"x" * 10,
        headers={"content-length": str(10_000_000), "content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.headers.get("x-request-id")


def test_log_lines_written_during_a_request_carry_its_id(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _client(monkeypatch)
    caplog.handler.addFilter(RunIdFilter())
    with caplog.at_level(logging.INFO):
        response = client.post("/api/v1/query/parse", json={"prompt": "x"})
    request_id = response.headers["x-request-id"]
    access = [r for r in caplog.records if r.getMessage().startswith("request method=POST")]
    assert access and access[0].run_id == request_id
    assert "route=/api/v1/query/parse status=422" in access[0].getMessage()


def test_the_access_line_never_records_the_query_or_the_body(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _client(monkeypatch)
    with caplog.at_level(logging.INFO):
        client.post("/api/v1/query/parse?secret=abc", json={"prompt": "Marina Beach secret place"})
    access = " ".join(
        r.getMessage() for r in caplog.records if r.getMessage().startswith("request ")
    )
    assert "secret" not in access
    assert "Marina" not in access


def test_a_workflow_inside_a_request_reuses_the_request_id() -> None:
    from app.core.observability import RUN_ID

    token = RUN_ID.set("abcdefabcdef")
    try:
        with workflow("agent") as run_id:
            assert run_id == "abcdefabcdef"
    finally:
        RUN_ID.reset(token)
    with workflow("agent") as standalone:
        assert re.fullmatch(r"[0-9a-f]{12}", standalone)


def test_metrics_do_not_exist_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _client(monkeypatch).get("/metrics").status_code == 404


def test_metrics_require_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, token="scrape-secret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_metrics_count_requests_by_route_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, token="scrape-secret")
    client.get("/health")
    client.post("/api/v1/query/parse", json={"prompt": "x"})
    client.get("/scanner/probe/1")
    client.get("/scanner/probe/2")

    body = client.get("/metrics", headers={"Authorization": "Bearer scrape-secret"}).text

    assert 'satquery_http_requests_total{route="/health",method="GET",status="200"} 1' in body
    assert (
        'satquery_http_requests_total{route="/api/v1/query/parse",method="POST",status="422"} 1'
        in body
    )
    # Unknown paths share one label, so probing cannot grow the series set.
    assert 'satquery_http_requests_total{route="unmatched",method="GET",status="404"} 2' in body
    assert "/scanner" not in body
    assert 'satquery_http_request_duration_seconds_count{route="/health"} 1' in body
    assert "scrape-secret" not in body


def _record(message: str, **extra: Any) -> logging.LogRecord:
    record = logging.LogRecord("satquery.test", logging.INFO, __file__, 1, message, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_logs_are_one_object_per_line_with_the_run_id() -> None:
    line = JsonFormatter().format(_record("hello \"world\"\nnext", run_id="0123456789ab"))
    assert "\n" not in line
    entry = json.loads(line)
    assert entry["message"] == 'hello "world"\nnext'
    assert entry["run_id"] == "0123456789ab"
    assert set(entry) == {"ts", "level", "logger", "run_id", "message"}
