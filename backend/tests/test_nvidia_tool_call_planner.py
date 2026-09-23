"""NVIDIA planning through native tool calling.

The plain ``response_format={"type": "json_object"}`` path asks only for "some
JSON object", and the live NIM endpoint used that freedom to return plans that
cannot be repaired without inventing arguments. Both shapes below were captured
from real responses of catalogued models:

* the configured default returned ``{"steps": ["execute_query", "ndwi_statistics"]}``
  - tool NAMES as bare strings, carrying no intent at all;
* another model stringified nested values, e.g.
  ``"modalities": "['sentinel-2-optical']"`` - a Python repr, not JSON.

Forcing a tool call whose parameter schema is ``AgentPlan``'s own JSON Schema
made the same default model return a plan that validated unchanged. These tests
pin that request shape, the fallback for a model that ignores ``tools``, and -
as before - that ``AgentPlan`` remains the only authority on what is accepted.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from app.core.errors import IntentParsingError
from app.services.agent.providers import nvidia as nvidia_mod
from app.services.agent.providers.nvidia import NvidiaAgentPlanner, _tool_call_arguments
from app.services.agent.schemas import AgentPlan

from tests.test_agent_providers import completion, settings

_INTENT = {
    "location_query": "Marina Beach, Chennai",
    "temporal_mode": "single",
    "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
    "modalities": ["sentinel-2-optical"],
    "task": "visualize",
}
_PLAN = {
    "steps": [
        {"tool": "execute_query", "intent": _INTENT, "include_imagery": True},
        {"tool": "spectral_indices", "indices": ["ndwi"]},
    ]
}


def tool_call(arguments: object) -> dict[str, object]:
    """A chat completion answering with one tool call, as NIM returns it."""

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "submit_plan",
                                "arguments": (
                                    arguments
                                    if isinstance(arguments, str)
                                    else json.dumps(arguments)
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }


class SequenceTransport(httpx.AsyncBaseTransport):
    """Replays the given bodies in order (the last one repeats)."""

    def __init__(self, *bodies: object) -> None:
        self._bodies = list(bodies)
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = self._bodies[min(len(self.requests), len(self._bodies) - 1)]
        self.requests.append(request)
        return httpx.Response(200, json=body, request=request)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "_PLAN_BACKOFF_SECONDS", 0.0)


def plan_with(*bodies: object) -> tuple[AgentPlan | Exception, SequenceTransport]:
    transport = SequenceTransport(*bodies)

    async def go() -> AgentPlan | Exception:
        async with httpx.AsyncClient(transport=transport) as client:
            planner = NvidiaAgentPlanner(settings=settings(), client=client)
            try:
                return await planner.plan("What is the NDWI of Marina Beach in Jan 2025?")
            except Exception as exc:  # returned so the test can assert on it
                return exc

    return asyncio.run(go()), transport


def payload(transport: SequenceTransport, index: int) -> dict[str, object]:
    return json.loads(transport.requests[index].content)


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


def test_planning_forces_one_tool_call_constrained_by_the_plan_schema() -> None:
    result, transport = plan_with(tool_call(_PLAN))

    assert isinstance(result, AgentPlan)
    sent = payload(transport, 0)
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "submit_plan"}}
    [tool] = sent["tools"]  # type: ignore[misc]
    # The schema the endpoint is constrained to IS the contract's schema, so the
    # two can never drift apart.
    assert tool["function"]["parameters"] == AgentPlan.model_json_schema()
    assert "response_format" not in sent


def test_a_tool_call_plan_is_accepted_in_one_request() -> None:
    result, transport = plan_with(tool_call(_PLAN))

    assert isinstance(result, AgentPlan)
    assert [step.tool for step in result.steps] == ["execute_query", "spectral_indices"]
    assert len(transport.requests) == 1


# --------------------------------------------------------------------------- #
# Fallback for a model that ignores `tools`
# --------------------------------------------------------------------------- #


def test_a_model_that_answers_without_a_tool_call_falls_back_to_plain_json() -> None:
    result, transport = plan_with(completion(json.dumps(_PLAN)))

    assert isinstance(result, AgentPlan)
    assert [step.tool for step in result.steps] == ["execute_query", "spectral_indices"]
    # First the tool-call request, then the older json_object request.
    assert "tools" in payload(transport, 0)
    fallback = payload(transport, 1)
    assert "tools" not in fallback
    assert fallback["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------------------- #
# AgentPlan is still the only authority
# --------------------------------------------------------------------------- #


def test_a_tool_call_naming_an_unknown_tool_is_still_refused() -> None:
    rogue = {"steps": [{"tool": "run_shell", "command": "rm -rf /"}]}
    result, _ = plan_with(tool_call(rogue))
    assert isinstance(result, IntentParsingError)


def test_a_tool_call_whose_arguments_are_not_json_is_refused() -> None:
    result, _ = plan_with(tool_call("{not json"))
    assert isinstance(result, IntentParsingError)


def test_the_live_bare_tool_name_shape_is_not_repaired() -> None:
    """Captured live on the json_object path. There is no intent to recover,
    so accepting it would mean inventing the location and the dates."""

    bare = {"steps": ["execute_query", "ndwi_statistics"]}
    result, _ = plan_with(tool_call(bare))
    assert isinstance(result, IntentParsingError)


def test_the_live_stringified_values_shape_is_not_coerced() -> None:
    """Captured live: nested values sent as a Python repr. Coercing a repr into
    structure is content repair, which this layer deliberately never does."""

    stringified = {
        "steps": [
            {
                "tool": "execute_query",
                "intent": {**_INTENT, "modalities": "['sentinel-2-optical']"},
            }
        ]
    }
    result, _ = plan_with(tool_call(stringified))
    assert isinstance(result, IntentParsingError)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _response(body: object) -> httpx.Response:
    return httpx.Response(
        200, json=body, request=httpx.Request("POST", "https://nim.internal/v1")
    )


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"role": "assistant", "content": "{}"}}]},
        {"choices": [{"message": {"tool_calls": []}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": 7}}]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "  "}}]}}]},
    ],
)
def test_a_response_without_a_usable_tool_call_yields_none(body: object) -> None:
    assert _tool_call_arguments(_response(body)) is None


def test_the_arguments_string_is_returned_verbatim() -> None:
    assert _tool_call_arguments(_response(tool_call(_PLAN))) == json.dumps(_PLAN)


# --------------------------------------------------------------------------- #
# Budget and diagnostics
# --------------------------------------------------------------------------- #


def test_planning_leaves_headroom_for_a_reasoning_model() -> None:
    """Measured live: 1,430-2,179 completion tokens per plan, reasoning included."""

    _, transport = plan_with(tool_call(_PLAN))
    assert payload(transport, 0)["max_tokens"] == nvidia_mod._PLAN_MAX_TOKENS
    assert nvidia_mod._PLAN_MAX_TOKENS >= 2500


def test_the_fallback_request_has_the_same_budget() -> None:
    _, transport = plan_with(completion(json.dumps(_PLAN)))
    assert payload(transport, 1)["max_tokens"] == nvidia_mod._PLAN_MAX_TOKENS


def _no_tool_call(reason: object, content: str = "SECRET-PROSE") -> dict[str, object]:
    return {
        "choices": [
            {"finish_reason": reason, "message": {"role": "assistant", "content": content}}
        ]
    }


@pytest.mark.parametrize(
    ("reason", "logged"),
    [
        ("length", "length"),
        ("stop", "stop"),
        ("Bad Value!", "unrecognised"),
        (None, "unrecognised"),
    ],
)
def test_a_missing_tool_call_is_logged_by_structure_only(
    reason: object, logged: str, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    target = logging.getLogger("satquery.agent.visual.nvidia")
    target.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=target.name):
            plan_with(_no_tool_call(reason))
    finally:
        target.removeHandler(caplog.handler)

    messages = [record.getMessage() for record in caplog.records]
    assert any(f"finish_reason={logged}" in message for message in messages)
    # The body - which may echo the request - never reaches the log.
    assert not any("SECRET-PROSE" in message for message in messages)
