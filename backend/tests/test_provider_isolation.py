"""With NVIDIA selected, the agent path must never touch Gemini.

This is the regression for a real defect: before the provider bundle existed,
only the visual analyst was provider-aware, so choosing NVIDIA still planned
with Gemini. The run died at Gemini's rate limit having never reached NVIDIA,
and nothing in the suite noticed - every test passed while provider selection
was, in practice, half wired.

The proof is a tripwire rather than an assertion about structure: every Gemini
entry point is replaced with something that raises, so a Gemini call *anywhere*
in the request fails the test loudly instead of quietly succeeding. Structure
can be refactored around; a tripwire cannot.

The test goes through the real HTTP route and the real composition function, so
it covers the production path and not a reconstruction of it. NVIDIA's endpoint
is pointed at the discard port, so the run fails at the planner's first
connection: no external network, and no dependence on a live provider.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.core.config import Settings
from app.main import app
from app.services.agent.providers import anthropic as anthropic_mod
from app.services.agent.providers import factory as factory_mod
from app.services.agent.providers import gemini as gemini_mod
from app.services.agent.providers import nvidia as nvidia_mod
from fastapi.testclient import TestClient

#: Unroutable by definition (RFC 863 discard). The planner's first request is
#: refused immediately, so the run ends without contacting anything real.
UNREACHABLE = "http://127.0.0.1:9/v1"

QUESTION = "Is there visible water in the Sentinel-2 image of Marina Beach?"


def _settings(provider: str) -> Settings:
    """EVERY provider fully credentialed.

    Deliberate: if Gemini were unconfigured, a Gemini-free run would prove
    nothing - the path might be avoiding Gemini for want of a key rather than
    because the provider was selected. The same reasoning applies to each
    provider added since, so all of them hold a key here and the two with a
    configurable host are pointed at the discard port.
    """

    return Settings(  # type: ignore[arg-type]
        _env_file=None,
        AI_PROVIDER=provider,
        GEMINI_API_KEY="gemini-key-that-must-never-be-used",
        NVIDIA_API_KEY="nvidia-test-key",
        NVIDIA_BASE_URL=UNREACHABLE,
        ANTHROPIC_API_KEY="anthropic-test-key",
        ANTHROPIC_BASE_URL=UNREACHABLE,
        LOCAL_AI_BASE_URL=UNREACHABLE,
    )


def _arm_gemini_tripwire(monkeypatch: Any) -> list[str]:
    """Make every Gemini entry point fail, and record what was touched."""

    touched: list[str] = []

    def forbid(name: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> None:
            touched.append(name)
            raise AssertionError(
                f"Gemini was reached via {name} while AI_PROVIDER=nvidia"
            )

        return boom

    # The SDK client itself - the last line before a real Google request.
    monkeypatch.setattr(gemini_mod.genai, "Client", forbid("genai.Client"))
    # And each provider class, so even constructing one is a failure.
    for cls in (
        "GeminiAgentPlanner",
        "GeminiAnswerSynthesizer",
        "GeminiVisualAnalyst",
    ):
        monkeypatch.setattr(gemini_mod, cls, forbid(cls))

    return touched


def _run(monkeypatch: Any, provider: str) -> tuple[int, dict[str, Any], list[str]]:
    settings = _settings(provider)
    # The factory is the single place provider selection happens; pointing its
    # settings lookup at ours exercises the real resolution logic.
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)
    touched = _arm_gemini_tripwire(monkeypatch)

    app.dependency_overrides.clear()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": provider},
        )
    return response.status_code, response.json(), touched


def test_nvidia_selection_never_reaches_gemini(monkeypatch: Any) -> None:
    """The regression: a full agent request with NVIDIA selected."""

    status, body, touched = _run(monkeypatch, "nvidia")

    assert touched == [], f"Gemini was reached via {touched}"
    # The run still completes as an agent outcome: NVIDIA was unreachable, so
    # planning failed and nothing was claimed.
    assert status == 200
    assert body["status"] == "planner_unavailable"
    assert body["answer"] is None
    assert body["evidence"]["items"] == []


def test_the_tripwire_actually_fires_for_gemini(monkeypatch: Any) -> None:
    """Proves the test above is not vacuous.

    Without this, a tripwire that silently failed to install would make the
    NVIDIA test pass for the wrong reason. Selecting Gemini must trip it.
    """

    with pytest.raises(AssertionError, match="Gemini was reached"):
        _run(monkeypatch, "gemini")


# --------------------------------------------------------------------------- #
# The /query/parse path must respect the selected provider too
# --------------------------------------------------------------------------- #
#
# This endpoint predates the provider abstraction: it constructed a Gemini
# parser unconditionally, so `AI_PROVIDER=nvidia` still parsed intents with
# Gemini. The same tripwire that guards the agent path guards it now.


def _run_parse(monkeypatch: Any, provider: str) -> tuple[int, list[str]]:
    settings = _settings(provider)
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)
    touched = _arm_gemini_tripwire(monkeypatch)

    app.dependency_overrides.clear()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query/parse",
            json={"prompt": "Show optical imagery of Chennai in January 2024"},
        )
    return response.status_code, touched


def test_parse_with_nvidia_selected_never_reaches_gemini(monkeypatch: Any) -> None:
    """The regression: NVIDIA selected must not parse with Gemini."""

    status, touched = _run_parse(monkeypatch, "nvidia")

    assert touched == [], f"Gemini was reached via {touched}"
    # NVIDIA is pointed at the discard port, so the request fails as an upstream
    # failure - never by quietly answering as the other provider.
    assert status >= 400


def test_the_parse_tripwire_actually_fires_for_gemini(monkeypatch: Any) -> None:
    """Proves the test above is not vacuous."""

    with pytest.raises(AssertionError, match="Gemini was reached"):
        _run_parse(monkeypatch, "gemini")


def test_parse_keeps_its_request_and_response_contract(monkeypatch: Any) -> None:
    """Provider selection must not change what the endpoint accepts or returns.

    Uses the route's own dependency override - the same seam the existing
    route tests use - so the contract is checked without any provider at all.
    """

    from app.api.routes import query as query_route
    from app.services.ai import AiService, MockIntentParser

    app.dependency_overrides[query_route.get_ai_service] = lambda: AiService(
        parser=MockIntentParser()
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/query/parse", json={"prompt": "imagery of Chennai"}
            )
        assert response.status_code == 200
        body = response.json()
        # The unchanged SatQueryIntent shape.
        for field in ("location_query", "temporal_mode", "time_windows", "modalities", "task"):
            assert field in body
    finally:
        app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# The other direction
# --------------------------------------------------------------------------- #
#
# Everything above proves NVIDIA-selected never reaches Gemini. That is only
# half the invariant: isolation is a property of the selection mechanism, not
# of one provider, and a mechanism that leaked only the other way would pass
# every test above. Both providers are fully credentialed here for the same
# reason as `_settings` states - a run must avoid the other provider because it
# was not selected, never for want of a key.


def _arm_nvidia_tripwire(monkeypatch: Any) -> list[str]:
    """Make every NVIDIA entry point fail, and record what was touched."""

    touched: list[str] = []

    def forbid(name: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> None:
            touched.append(name)
            raise AssertionError(
                f"NVIDIA was reached via {name} while AI_PROVIDER=gemini"
            )

        return boom

    for cls in (
        "NvidiaAgentPlanner",
        "NvidiaAnswerSynthesizer",
        "NvidiaVisualAnalyst",
        "NvidiaIntentParser",
    ):
        monkeypatch.setattr(nvidia_mod, cls, forbid(cls))
    # The shared chat client - the last line before a real NVIDIA request.
    monkeypatch.setattr(nvidia_mod, "_NvidiaChatClient", forbid("_NvidiaChatClient"))
    return touched


def _run_with_nvidia_tripwire(
    monkeypatch: Any, provider: str
) -> tuple[int, list[str]]:
    settings = _settings(provider)
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)
    touched = _arm_nvidia_tripwire(monkeypatch)
    # Gemini is credentialed but must not be contacted either. A TimeoutError
    # is what an unreachable endpoint actually produces, and the provider maps
    # it to a normal upstream failure - so the run stays offline AND stays a
    # real agent outcome rather than an unhandled crash.
    monkeypatch.setattr(
        gemini_mod.genai,
        "Client",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("offline")),
    )

    app.dependency_overrides.clear()
    # `raise_server_exceptions=False` because the sabotaged Gemini client is
    # not the subject here: whatever Gemini does with it, NVIDIA must not have
    # been touched. Letting the sabotage surface as a response keeps the
    # assertion about isolation rather than about Gemini's error handling.
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": provider},
        )
    return response.status_code, touched


def test_gemini_selection_never_reaches_nvidia(monkeypatch: Any) -> None:
    _status, touched = _run_with_nvidia_tripwire(monkeypatch, "gemini")

    assert touched == [], f"NVIDIA was reached via {touched}"


def test_the_nvidia_tripwire_actually_fires(monkeypatch: Any) -> None:
    """Proves the test above is not vacuous, mirroring the Gemini case.

    A tripwire that silently failed to install would make the test above pass
    for the wrong reason, so selecting NVIDIA must record a touch.
    """

    _status, touched = _run_with_nvidia_tripwire(monkeypatch, "nvidia")
    assert touched, "the NVIDIA tripwire was never installed"


# --------------------------------------------------------------------------- #
# The third provider
# --------------------------------------------------------------------------- #
#
# Isolation is a property of the SELECTION MECHANISM, not of any one pair of
# providers, so a third backend has to be held to the same standard in both
# directions: selecting it must reach neither of the others, and selecting one
# of the others must not reach it. Every provider is credentialed by
# `_settings`, so an absent touch is never explained by an absent key.


def _arm_anthropic_tripwire(monkeypatch: Any) -> list[str]:
    """Make every Anthropic entry point fail, and record what was touched."""

    touched: list[str] = []

    def forbid(name: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> None:
            touched.append(name)
            raise AssertionError(f"Anthropic was reached via {name}")

        return boom

    for cls in (
        "AnthropicAgentPlanner",
        "AnthropicAnswerSynthesizer",
        "AnthropicVisualAnalyst",
        "AnthropicIntentParser",
    ):
        monkeypatch.setattr(anthropic_mod, cls, forbid(cls))
    # The shared messages client - the last line before a real Anthropic request.
    monkeypatch.setattr(
        anthropic_mod, "_AnthropicMessagesClient", forbid("_AnthropicMessagesClient")
    )
    return touched


def _run_with_anthropic_tripwire(
    monkeypatch: Any, provider: str
) -> tuple[int, list[str]]:
    settings = _settings(provider)
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)
    touched = _arm_anthropic_tripwire(monkeypatch)
    # Gemini is credentialed but must not be contacted either; NVIDIA already
    # points at the discard port. Same reasoning as the NVIDIA-direction test.
    monkeypatch.setattr(
        gemini_mod.genai,
        "Client",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("offline")),
    )

    app.dependency_overrides.clear()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": provider},
        )
    return response.status_code, touched


def test_gemini_selection_never_reaches_anthropic(monkeypatch: Any) -> None:
    _status, touched = _run_with_anthropic_tripwire(monkeypatch, "gemini")

    assert touched == [], f"Anthropic was reached via {touched}"


def test_nvidia_selection_never_reaches_anthropic(monkeypatch: Any) -> None:
    _status, touched = _run_with_anthropic_tripwire(monkeypatch, "nvidia")

    assert touched == [], f"Anthropic was reached via {touched}"


def test_the_anthropic_tripwire_actually_fires(monkeypatch: Any) -> None:
    """Proves the two tests above are not vacuous."""

    _status, touched = _run_with_anthropic_tripwire(monkeypatch, "anthropic")
    assert touched, "the Anthropic tripwire was never installed"


def test_anthropic_selection_reaches_neither_other_provider(
    monkeypatch: Any,
) -> None:
    """The other direction, with BOTH other providers armed at once."""

    settings = _settings("anthropic")
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)
    gemini_touched = _arm_gemini_tripwire(monkeypatch)
    nvidia_touched = _arm_nvidia_tripwire(monkeypatch)

    app.dependency_overrides.clear()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/query/agent",
            json={"question": QUESTION, "provider": "anthropic"},
        )

    assert gemini_touched == [], f"Gemini was reached via {gemini_touched}"
    assert nvidia_touched == [], f"NVIDIA was reached via {nvidia_touched}"
    # The run still completes as an honest agent outcome: Anthropic is pointed
    # at the discard port, so planning failed and nothing was claimed.
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "planner_unavailable"
    assert body["answer"] is None
    assert body["evidence"]["items"] == []


def test_parse_with_anthropic_selected_never_reaches_gemini(
    monkeypatch: Any,
) -> None:
    """`/query/parse` respects the third provider too."""

    status, touched = _run_parse(monkeypatch, "anthropic")

    assert touched == [], f"Gemini was reached via {touched}"
    assert status >= 400


# --------------------------------------------------------------------------- #
# The local (Ollama) provider, in both directions
#
# Its whole purpose is to answer without a cloud quota, so the two failures
# that matter are symmetrical: a cloud run must never wander onto this machine's
# model, and a local run must never be quietly answered by a cloud provider when
# the local one is down. Every cloud provider is credentialed here, exactly as
# `_settings` says, so an untouched tripwire means "not selected", never "no key".
# --------------------------------------------------------------------------- #


def _arm_local_tripwire(monkeypatch: Any) -> list[str]:
    from app.services.agent.providers import local as local_mod

    touched: list[str] = []

    def forbid(name: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> None:
            touched.append(name)
            raise AssertionError(f"the local provider was reached via {name}")

        return boom

    for cls in (
        "LocalAgentPlanner",
        "LocalAnswerSynthesizer",
        "LocalVisualAnalyst",
        "LocalIntentParser",
    ):
        monkeypatch.setattr(local_mod, cls, forbid(cls))
    monkeypatch.setattr(local_mod, "_OllamaChatClient", forbid("_OllamaChatClient"))
    return touched


@pytest.mark.parametrize("provider", ["gemini", "nvidia"])
def test_a_cloud_selection_never_reaches_the_local_model(
    monkeypatch: Any, provider: str
) -> None:
    settings = _settings(provider)
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)
    touched = _arm_local_tripwire(monkeypatch)
    monkeypatch.setattr(
        gemini_mod.genai,
        "Client",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("offline")),
    )

    app.dependency_overrides.clear()
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/api/v1/query/agent", json={"question": QUESTION, "provider": provider})

    assert touched == []


def test_the_local_tripwire_actually_fires(monkeypatch: Any) -> None:
    settings = _settings("local")
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)
    touched = _arm_local_tripwire(monkeypatch)

    app.dependency_overrides.clear()
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/api/v1/query/agent", json={"question": QUESTION, "provider": "local"})

    assert touched, "the local tripwire was never installed"


def test_a_local_run_with_ollama_down_fails_honestly_and_touches_no_cloud(
    monkeypatch: Any,
) -> None:
    """Ollama unreachable, every cloud provider credentialed AND armed."""

    from app.services.agent.providers.local import UNAVAILABLE_MESSAGE

    settings = _settings("local")
    monkeypatch.setattr(factory_mod, "get_settings", lambda: settings)
    touched = (
        _arm_gemini_tripwire(monkeypatch)
        + _arm_nvidia_tripwire(monkeypatch)
        + _arm_anthropic_tripwire(monkeypatch)
    )

    app.dependency_overrides.clear()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query/agent", json={"question": QUESTION, "provider": "local"}
        )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "planner_unavailable"
    assert body["failure"]["message"] == UNAVAILABLE_MESSAGE
    assert body["answer"] is None
    assert touched == []
