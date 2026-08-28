"""Provider-agnostic intent-parsing tests.

Everything runs against :class:`MockIntentParser` / injected fakes. No test
contacts an external AI service - there is no provider SDK to contact.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest
from app.api.routes.query import get_ai_service
from app.core.errors import InvalidInputError
from app.main import create_app
from app.services.ai import AiService, IntentParser, MockIntentParser
from app.services.query.schemas import SatQueryIntent, TimeRange
from fastapi.testclient import TestClient

PARSE_URL = "/api/v1/query/parse"

MOCK_INTENT_JSON = {
    "location_query": "Chennai",
    "temporal_mode": "single",
    "time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
    "modalities": ["sentinel-2-optical"],
    "task": "visualize",
}


class RecordingParser(IntentParser):
    """Records prompts and returns a caller-supplied intent."""

    def __init__(self, intent: SatQueryIntent) -> None:
        self.prompts: list[str] = []
        self._intent = intent

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        self.prompts.append(prompt)
        return self._intent


def make_client(service: AiService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_ai_service] = lambda: service
    return TestClient(app)


# --------------------------------------------------------------------------- #
# IntentParser / MockIntentParser
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
    assert second.location_query == "Chennai"  # not affected by the mutation


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
# AiService
# --------------------------------------------------------------------------- #


def test_ai_service_defaults_to_the_mock_parser() -> None:
    assert isinstance(AiService()._parser, MockIntentParser)


def test_ai_service_delegates_to_the_injected_parser() -> None:
    intent = SatQueryIntent.model_validate(MOCK_INTENT_JSON)
    parser = RecordingParser(intent)
    result = asyncio.run(AiService(parser=parser).parse_intent("  hello world  "))

    assert result is intent
    assert parser.prompts == ["hello world"]  # trimmed before delegation


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_ai_service_rejects_an_empty_prompt(prompt: str) -> None:
    parser = RecordingParser(SatQueryIntent.model_validate(MOCK_INTENT_JSON))
    with pytest.raises(InvalidInputError):
        asyncio.run(AiService(parser=parser).parse_intent(prompt))
    assert parser.prompts == []


# --------------------------------------------------------------------------- #
# POST /api/v1/query/parse
# --------------------------------------------------------------------------- #


def test_parse_endpoint_returns_the_mock_intent() -> None:
    response = make_client(AiService(parser=MockIntentParser())).post(
        PARSE_URL, json={"prompt": "show me Chennai"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body == MOCK_INTENT_JSON
    # Extraction only - not a plan, not geocoded, not discovery.
    assert "bbox" not in body
    assert "intent" not in body
    assert "scenes" not in body


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


def test_parse_endpoint_default_service_uses_mock_parser() -> None:
    # No dependency override: the real provider is get_ai_service() -> AiService().
    response = TestClient(create_app()).post(
        PARSE_URL, json={"prompt": "anything at all"}
    )
    assert response.status_code == 200
    assert response.json() == MOCK_INTENT_JSON


# --------------------------------------------------------------------------- #
# Provider boundary: nothing in app/services/ai touches an AI SDK
# --------------------------------------------------------------------------- #


def test_ai_package_imports_no_external_provider() -> None:
    ai_dir = Path(__file__).resolve().parents[1] / "app" / "services" / "ai"
    banned = ("anthropic", "google.generativeai", "google_generativeai", "openai", "ollama")
    for path in ai_dir.glob("*.py"):
        source = path.read_text()
        for token in banned:
            assert token not in source, f"{path.name} references {token!r}"
