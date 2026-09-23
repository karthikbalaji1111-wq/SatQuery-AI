"""Health endpoint tests."""

from __future__ import annotations

from app.api.routes.query import get_ai_service
from app.main import create_app
from app.services.ai import AiService, MockIntentParser
from fastapi.testclient import TestClient


def envelope_client() -> TestClient:
    """A client whose AI provider is wired with a fake parser.

    The tests below are about the shape of an ERROR RESPONSE, not about any
    provider. Left to the real factory, the request never reaches body
    validation on a deployment holding no credential: the dependency is
    resolved first and raises, so the assertion sees the upstream 502 rather
    than the 422 it is written about. Injecting the fake makes these tests
    say what they mean, on any machine.
    """

    app = create_app()
    app.dependency_overrides[get_ai_service] = lambda: AiService(
        parser=MockIntentParser()
    )
    return TestClient(app, raise_server_exceptions=False)


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"]
    assert body["version"]
    assert body["environment"]


# --------------------------------------------------------------------------- #
# One error envelope
# --------------------------------------------------------------------------- #
#
# A 422 used to arrive in one of two shapes: Pydantic's ``{"detail": [...]}``
# for a structurally malformed body, and ``{"error": {...}}`` for an AppError.
# A client reading ``error.message`` got nothing from the first kind and fell
# back to a status-code-only string, throwing away the one useful thing a
# validation failure carries - which field, and why.
#
# The two failures stay distinguishable by CODE. Only the envelope is shared.


def test_a_malformed_body_uses_the_same_envelope_as_every_other_error() -> None:
    client = envelope_client()
    response = client.post("/api/v1/query/parse", json={})

    assert response.status_code == 422
    body = response.json()
    assert "detail" not in body
    assert body["error"]["code"] == "validation_error"
    # The detail survives: a client can say which field was wrong.
    assert "prompt" in body["error"]["message"]


def test_a_schema_rejection_stays_distinguishable_from_a_semantic_one() -> None:
    """Same envelope, same status - deliberately different codes.

    A structurally malformed body and a well-formed but semantically invalid
    one are different failures, and unifying the envelope must not erase that.
    """

    client = envelope_client()
    malformed = client.post("/api/v1/query/parse", json={})
    semantic = client.post(
        "/api/v1/satellite/imagery",
        json={
            "scene_id": "../../../search",
            "bbox": {"west": 80.0, "south": 13.0, "east": 80.1, "north": 13.1},
            "asset": "visual",
        },
    )

    assert malformed.status_code == semantic.status_code == 422
    assert malformed.json()["error"]["code"] == "validation_error"
    assert semantic.json()["error"]["code"] == "invalid_input"


def test_a_validation_error_never_echoes_the_offending_value() -> None:
    """Echoing input back is how a request's own contents return to it.

    A body may carry a credential; the sender must not be handed it back
    inside an error, so only the location and the reason are reported.
    """

    client = envelope_client()
    secret = "sk-CANARY-must-not-be-echoed"
    response = client.post(
        "/api/v1/query/parse", json={"prompt": {"nested": secret}}
    )

    assert response.status_code == 422
    assert secret not in response.text
    assert "CANARY" not in response.text
