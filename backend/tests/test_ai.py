"""Intent-parsing tests.

CRITICAL: no test contacts the real Gemini API. Every test uses
:class:`MockIntentParser`, an injected fake parser, or a fake google-genai
client. ``GEMINI_API_KEY`` is never required.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from app.api.routes.query import get_ai_service
from app.core.config import Settings
from app.core.errors import IntentParsingError, InvalidInputError, UpstreamServiceError
from app.main import create_app
from app.services.ai import (
    AiService,
    GeminiIntentParser,
    IntentParser,
    MockIntentParser,
)
from app.services.query.schemas import SatQueryIntent, TimeRange
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

PARSE_URL = "/api/v1/query/parse"

MOCK_INTENT_JSON = {
    "location_query": "Chennai",
    "temporal_mode": "single",
    "time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
    "modalities": ["sentinel-2-optical"],
    "task": "visualize",
}

GEMINI_INTENT_JSON = {
    "location_query": "Rotterdam",
    "temporal_mode": "compare",
    "time_windows": {
        "baseline": {"start_date": "2023-06-01", "end_date": "2023-06-30"},
        "target": {"start_date": "2024-06-01", "end_date": "2024-06-30"},
    },
    "modalities": ["sentinel-2-optical", "sentinel-1-sar"],
    "task": "change_detection",
}

FAKE_KEY = "test-key-not-real"


class RecordingParser(IntentParser):
    """Records prompts and returns a caller-supplied intent."""

    def __init__(self, intent: SatQueryIntent) -> None:
        self.prompts: list[str] = []
        self._intent = intent

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        self.prompts.append(prompt)
        return self._intent


class _FakeAioModels:
    def __init__(
        self,
        *,
        text: str | None = None,
        error: Exception | None = None,
        capture: dict[str, Any] | None = None,
    ) -> None:
        self._text = text
        self._error = error
        self._capture = capture

    async def generate_content(self, **kwargs: Any) -> Any:
        if self._capture is not None:
            self._capture.update(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


class FakeGeminiClient:
    """Stub of ``genai.Client`` - only ``.aio.models.generate_content`` is used."""

    def __init__(self, **kwargs: Any) -> None:
        self.aio = SimpleNamespace(models=_FakeAioModels(**kwargs))


def gemini_parser(**kwargs: Any) -> GeminiIntentParser:
    return GeminiIntentParser(
        settings=Settings(gemini_api_key=FAKE_KEY),
        client=FakeGeminiClient(**kwargs),
    )


def make_client(service: AiService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_ai_service] = lambda: service
    return TestClient(app)


# --------------------------------------------------------------------------- #
# IntentParser / MockIntentParser  (retained)
# --------------------------------------------------------------------------- #


def test_intent_parser_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IntentParser()  # type: ignore[abstract]


def test_mock_parser_returns_a_valid_intent() -> None:
    intent = asyncio.run(MockIntentParser().parse_intent("show me anything"))
    assert isinstance(intent, SatQueryIntent)
    assert intent.model_dump(mode="json") == MOCK_INTENT_JSON


def test_mock_parser_is_deterministic_and_prompt_independent() -> None:
    parser = MockIntentParser()
    a = asyncio.run(parser.parse_intent("flooding near Chennai last summer"))
    b = asyncio.run(parser.parse_intent("count ships in Rotterdam in 2024"))
    assert a == b == asyncio.run(MockIntentParser().parse_intent("x"))


def test_mock_parser_returns_independent_copies() -> None:
    parser = MockIntentParser()
    first = asyncio.run(parser.parse_intent("a"))
    first.location_query = "mutated"
    second = asyncio.run(parser.parse_intent("b"))
    assert second.location_query == "Chennai"


def test_mock_parser_accepts_an_injected_intent() -> None:
    custom = SatQueryIntent(
        location_query="Rotterdam",
        temporal_mode="compare",
        time_windows={
            "baseline": TimeRange(
                start_date=date(2023, 1, 1), end_date=date(2023, 3, 31)
            ),
            "target": TimeRange(
                start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)
            ),
        },
        modalities=["sentinel-2-optical", "sentinel-1-sar"],
        task="change_detection",
    )
    intent = asyncio.run(MockIntentParser(custom).parse_intent("anything"))
    assert intent == custom


def test_mock_parser_is_flagged_as_a_mock() -> None:
    assert getattr(MockIntentParser, "is_mock", False) is True


# --------------------------------------------------------------------------- #
# GeminiIntentParser  (fake client - never the network)
# --------------------------------------------------------------------------- #


def test_gemini_parser_success_returns_validated_intent() -> None:
    parser = gemini_parser(text=json.dumps(GEMINI_INTENT_JSON))
    intent = asyncio.run(parser.parse_intent("compare Rotterdam 2023 vs 2024"))

    assert isinstance(intent, SatQueryIntent)
    assert intent.model_dump(mode="json") == GEMINI_INTENT_JSON


def test_gemini_parser_request_shape() -> None:
    capture: dict[str, Any] = {}
    parser = GeminiIntentParser(
        settings=Settings(gemini_api_key=FAKE_KEY, gemini_model="gemini-2.5-flash"),
        client=FakeGeminiClient(text=json.dumps(MOCK_INTENT_JSON), capture=capture),
    )
    asyncio.run(parser.parse_intent("optical imagery of Chennai"))

    assert capture["model"] == "gemini-2.5-flash"
    assert capture["contents"] == "optical imagery of Chennai"

    config = capture["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is SatQueryIntent  # existing model, not a copy

    system = config.system_instruction
    for token in (
        "location_query",
        "single",
        "compare",
        "timeseries",
        "sentinel-2-optical",
        "sentinel-1-sar",
        "visualize",
        "change_detection",
        "object_identification",
        "NEVER invent coordinates",
    ):
        assert token in system


def test_gemini_parser_malformed_output_raises_intent_parse_error() -> None:
    parser = gemini_parser(text="not json at all {")
    with pytest.raises(IntentParsingError):
        asyncio.run(parser.parse_intent("anything"))


@pytest.mark.parametrize("text", ["", "   ", None])
def test_gemini_parser_empty_output_raises_intent_parse_error(text: str | None) -> None:
    parser = gemini_parser(text=text)
    with pytest.raises(IntentParsingError):
        asyncio.run(parser.parse_intent("anything"))


def test_gemini_parser_pydantic_validation_failure_raises_intent_parse_error() -> None:
    broken = {
        **MOCK_INTENT_JSON,
        "time_windows": [
            {"start_date": "2024-01-01", "end_date": "2024-01-31"},
            {"start_date": "2024-02-01", "end_date": "2024-02-29"},
        ],  # temporal_mode "single" but two windows
    }
    parser = gemini_parser(text=json.dumps(broken))
    with pytest.raises(IntentParsingError):
        asyncio.run(parser.parse_intent("anything"))


def test_gemini_parser_api_error_becomes_upstream_error() -> None:
    parser = gemini_parser(error=genai_errors.APIError(503, {}))
    with pytest.raises(UpstreamServiceError):
        asyncio.run(parser.parse_intent("anything"))


def test_gemini_parser_timeout_becomes_upstream_error() -> None:
    parser = gemini_parser(error=TimeoutError("deadline exceeded"))
    with pytest.raises(UpstreamServiceError):
        asyncio.run(parser.parse_intent("anything"))


def test_gemini_parser_unexpected_failure_becomes_upstream_error() -> None:
    parser = gemini_parser(error=RuntimeError("boom"))
    with pytest.raises(UpstreamServiceError):
        asyncio.run(parser.parse_intent("anything"))


def test_gemini_parser_without_api_key_raises_upstream_error_and_no_client() -> None:
    parser = GeminiIntentParser(settings=Settings(gemini_api_key=None), client=None)
    with pytest.raises(UpstreamServiceError):
        asyncio.run(parser.parse_intent("anything"))


def test_gemini_parser_never_logs_the_api_key(caplog: pytest.LogCaptureFixture) -> None:
    secret = "SUPER-SECRET-GEMINI-KEY-123"
    parser = GeminiIntentParser(
        settings=Settings(gemini_api_key=secret),
        client=FakeGeminiClient(text=json.dumps(MOCK_INTENT_JSON)),
    )
    with caplog.at_level(logging.DEBUG):
        asyncio.run(parser.parse_intent("show me Chennai"))
    assert secret not in caplog.text


# --------------------------------------------------------------------------- #
# AiService stays provider-agnostic
# --------------------------------------------------------------------------- #


def test_ai_service_defaults_to_the_mock_parser() -> None:
    assert isinstance(AiService()._parser, MockIntentParser)


def test_ai_service_delegates_to_the_injected_parser() -> None:
    intent = SatQueryIntent.model_validate(MOCK_INTENT_JSON)
    parser = RecordingParser(intent)
    result = asyncio.run(AiService(parser=parser).parse_intent("  hello world  "))

    assert result is intent
    assert parser.prompts == ["hello world"]


def test_ai_service_works_with_a_gemini_parser_through_the_interface() -> None:
    service = AiService(parser=gemini_parser(text=json.dumps(GEMINI_INTENT_JSON)))
    intent = asyncio.run(service.parse_intent("compare Rotterdam"))
    assert intent.model_dump(mode="json") == GEMINI_INTENT_JSON


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_ai_service_rejects_an_empty_prompt(prompt: str) -> None:
    parser = RecordingParser(SatQueryIntent.model_validate(MOCK_INTENT_JSON))
    with pytest.raises(InvalidInputError):
        asyncio.run(AiService(parser=parser).parse_intent(prompt))
    assert parser.prompts == []


# --------------------------------------------------------------------------- #
# POST /api/v1/query/parse
# --------------------------------------------------------------------------- #


def test_parse_endpoint_returns_the_intent() -> None:
    response = make_client(AiService(parser=MockIntentParser())).post(
        PARSE_URL, json={"prompt": "show me Chennai"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body == MOCK_INTENT_JSON
    # Extraction only - no geocode, no plan, no discovery, no imagery.
    for leaked in ("bbox", "center", "intent", "scenes", "catalog", "image_base64"):
        assert leaked not in body


def test_parse_endpoint_with_fake_gemini_returns_intent() -> None:
    service = AiService(parser=gemini_parser(text=json.dumps(GEMINI_INTENT_JSON)))
    response = make_client(service).post(
        PARSE_URL, json={"prompt": "compare Rotterdam 2023 vs 2024"}
    )
    assert response.status_code == 200
    assert response.json() == GEMINI_INTENT_JSON


def test_parse_endpoint_forwards_the_trimmed_prompt() -> None:
    parser = RecordingParser(SatQueryIntent.model_validate(MOCK_INTENT_JSON))
    make_client(AiService(parser=parser)).post(
        PARSE_URL, json={"prompt": "  radar over Rotterdam  "}
    )
    assert parser.prompts == ["radar over Rotterdam"]


@pytest.mark.parametrize("prompt", ["", "   "])
def test_parse_endpoint_rejects_empty_prompt(prompt: str) -> None:
    parser = RecordingParser(SatQueryIntent.model_validate(MOCK_INTENT_JSON))
    response = make_client(AiService(parser=parser)).post(
        PARSE_URL, json={"prompt": prompt}
    )
    assert response.status_code == 422
    assert parser.prompts == []


def test_parse_endpoint_rejects_missing_prompt() -> None:
    response = make_client(AiService(parser=MockIntentParser())).post(PARSE_URL, json={})
    assert response.status_code == 422


def test_parse_endpoint_gemini_failure_is_502() -> None:
    service = AiService(parser=gemini_parser(error=genai_errors.APIError(503, {})))
    response = make_client(service).post(PARSE_URL, json={"prompt": "x"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_parse_endpoint_gemini_malformed_output_is_422() -> None:
    service = AiService(parser=gemini_parser(text="}{ not json"))
    response = make_client(service).post(PARSE_URL, json={"prompt": "x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "intent_parse_error"


def test_parse_endpoint_without_gemini_key_is_502_and_no_network() -> None:
    service = AiService(parser=GeminiIntentParser(settings=Settings(gemini_api_key=None)))
    response = make_client(service).post(PARSE_URL, json={"prompt": "show Chennai"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_parse_endpoint_never_calls_geospatial_stac_or_imagery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.geospatial import nominatim
    from app.services.satellite import raster, stac

    def _forbidden(name: str) -> Any:
        def _raise(*_a: Any, **_k: Any) -> None:  # pragma: no cover - must not run
            raise AssertionError(f"/query/parse must not invoke {name}")

        return _raise

    monkeypatch.setattr(nominatim, "geocode", _forbidden("geospatial.geocode"))
    monkeypatch.setattr(stac, "search_items", _forbidden("satellite.stac.search_items"))
    monkeypatch.setattr(raster, "read_rgb_window", _forbidden("imagery.read_rgb_window"))

    parser = RecordingParser(SatQueryIntent.model_validate(MOCK_INTENT_JSON))
    response = make_client(AiService(parser=parser)).post(
        PARSE_URL, json={"prompt": "show Chennai"}
    )
    assert response.status_code == 200
    assert parser.prompts == ["show Chennai"]


# --------------------------------------------------------------------------- #
# Production wiring + provider boundary
# --------------------------------------------------------------------------- #


def test_production_get_ai_service_uses_gemini_parser() -> None:
    service = get_ai_service()
    assert isinstance(service, AiService)
    assert isinstance(service._parser, GeminiIntentParser)


def test_provider_coupling_is_confined_to_parser_module() -> None:
    ai_dir = Path(__file__).resolve().parents[1] / "app" / "services" / "ai"

    # The deprecated Google SDK and other providers must not appear anywhere.
    banned_everywhere = (
        "anthropic",
        "google.generativeai",
        "google_generativeai",
        "openai",
        "ollama",
    )
    # Only parser.py may import the (current) google-genai SDK.
    for path in ai_dir.glob("*.py"):
        source = path.read_text()
        for token in banned_everywhere:
            assert token not in source, f"{path.name} references {token!r}"
        if path.name != "parser.py":
            assert "import google" not in source and "from google" not in source, (
                f"{path.name} imports the Gemini SDK - only parser.py may"
            )

    parser_src = (ai_dir / "parser.py").read_text()
    assert "from google import genai" in parser_src
