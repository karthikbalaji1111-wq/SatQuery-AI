"""Every plan envelope the live NVIDIA endpoint was observed to return.

The NIM chat surface pins nothing about the JSON envelope the way Gemini's
response schema does, so the SAME model returns the SAME semantic plan wrapped
several different ways from one call to the next. Each shape below was captured
from a real response during live testing.

These tests exist because the endpoint's availability is not a reliable place to
verify parsing: it intermittently answers 503 ("Worker local total request limit
reached") and times out, so a live sample proves nothing on a bad minute. The
parsing is deterministic even when the endpoint is not, so it is tested here.

The negative cases matter as much as the positive ones: this layer relocates an
already-complete step list and must never repair CONTENT. An unknown tool, a
misnamed argument or a malformed step has to keep reaching `AgentPlan` and be
rejected there.
"""

from __future__ import annotations

import json

import pytest
from app.services.agent.providers.nvidia import _extract_json, _plan_envelope
from app.services.agent.schemas import AgentPlan
from pydantic import ValidationError

_INTENT = {
    "location_query": "Marina Beach, Chennai",
    "temporal_mode": "single",
    "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
    "modalities": ["sentinel-2-optical"],
    "task": "visualize",
}
_CANONICAL = [
    {"tool": "execute_query", "intent": _INTENT},
    {"tool": "ndwi_statistics"},
]


def _plan(raw: str) -> AgentPlan:
    return AgentPlan.model_validate_json(
        json.dumps(_plan_envelope(json.loads(_extract_json(raw))))
    )


def test_the_contract_shape_passes_through_unchanged() -> None:
    plan = _plan(json.dumps({"steps": _CANONICAL}))
    assert [s.tool for s in plan.steps] == ["execute_query", "ndwi_statistics"]


def test_a_step_list_under_a_plan_key_is_unwrapped() -> None:
    plan = _plan(json.dumps({"plan": _CANONICAL}))
    assert [s.tool for s in plan.steps] == ["execute_query", "ndwi_statistics"]


def test_the_contract_shape_nested_under_plan_is_unwrapped() -> None:
    plan = _plan(json.dumps({"plan": {"steps": _CANONICAL}}))
    assert [s.tool for s in plan.steps] == ["execute_query", "ndwi_statistics"]


def test_a_bare_step_list_is_wrapped() -> None:
    """`[{...}, {...}]` with no enclosing object - the shape that used to raise
    JSONDecodeError("Extra data"), because the brace scan spanned both elements."""

    plan = _plan(json.dumps(_CANONICAL))
    assert [s.tool for s in plan.steps] == ["execute_query", "ndwi_statistics"]


def test_a_single_bare_step_is_wrapped() -> None:
    plan = _plan(json.dumps({"tool": "execute_query", "intent": _INTENT}))
    assert [s.tool for s in plan.steps] == ["execute_query"]


def test_the_openai_parameters_envelope_is_unwrapped_per_step() -> None:
    raw = json.dumps(
        {
            "steps": [
                {"tool": "execute_query", "parameters": {"intent": _INTENT}},
                {"tool": "ndwi_statistics", "parameters": {}},
            ]
        }
    )
    plan = _plan(raw)
    assert [s.tool for s in plan.steps] == ["execute_query", "ndwi_statistics"]


def test_a_markdown_fence_and_surrounding_prose_are_stripped() -> None:
    raw = 'Here is the plan:\n```json\n' + json.dumps({"steps": _CANONICAL}) + '\n```\n'
    plan = _plan(raw)
    assert [s.tool for s in plan.steps] == ["execute_query", "ndwi_statistics"]


def test_an_echoed_intent_beside_the_plan_is_dropped() -> None:
    """The intent belongs to a step, not to the plan; echoing it is not a field."""

    plan = _plan(json.dumps({"intent": _INTENT, "steps": _CANONICAL}))
    assert [s.tool for s in plan.steps] == ["execute_query", "ndwi_statistics"]


# --------------------------------------------------------------------------- #
# Content is NEVER repaired. These must still fail.
# --------------------------------------------------------------------------- #


def test_an_unknown_tool_is_still_refused() -> None:
    """The closed allowlist is a security property; no envelope work relaxes it."""

    with pytest.raises(ValidationError):
        _plan(json.dumps({"steps": [{"tool": "rm_rf", "intent": _INTENT}]}))


def test_a_misnamed_argument_is_still_refused() -> None:
    """`spectral_indices` takes `indices`; a model writing `index` is wrong and
    must be rejected rather than silently corrected."""

    with pytest.raises(ValidationError):
        _plan(
            json.dumps(
                {
                    "steps": [
                        {"tool": "execute_query", "intent": _INTENT},
                        {"tool": "spectral_indices", "index": "ndwi"},
                    ]
                }
            )
        )


def test_an_inner_tool_override_is_not_honoured() -> None:
    """A `parameters` block carrying its own `tool` is not the envelope this
    layer adapts, so it is left alone and refused."""

    with pytest.raises(ValidationError):
        _plan(
            json.dumps(
                {
                    "steps": [
                        {
                            "tool": "execute_query",
                            "parameters": {"tool": "ndwi_statistics"},
                        }
                    ]
                }
            )
        )


def test_an_unrecognised_envelope_is_left_for_the_contract_to_refuse() -> None:
    with pytest.raises(ValidationError):
        _plan(json.dumps({"actions": _CANONICAL}))
