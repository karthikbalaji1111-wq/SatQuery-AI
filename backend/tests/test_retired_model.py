"""A model the provider has retired must be impossible to select.

NVIDIA retired ``nvidia/nemotron-nano-12b-v2-vl`` on 2026-08-26; the endpoint
answers ``410 Gone`` and the id is absent from ``GET /v1/models``. The catalog
still described it as fully capable, which was true and useless: the UI offered
it as "Ready" and the resolver would happily send a request to it.

Two facts are now kept apart, because conflating them misdescribes the failure:

* **capability** - what the model can do. Retirement does not make it blind, so
  ``supports_role`` still reports the truth, and the API's ``compatible`` field
  with it.
* **availability** - whether the provider still serves it. This is what makes a
  model selectable, and it is a published static fact, never a health check.

``serves()`` is the conjunction, and it is the question every selection path
already asks - so unselectability holds by construction rather than by each
caller remembering to check.
"""

from __future__ import annotations

import pytest
from app.core.config import Settings
from app.core.errors import UpstreamServiceError
from app.services.agent.providers.catalog import MODEL_CATALOG, find_model
from app.services.agent.providers.factory import get_visual_analyst

RETIRED = "nvidia/nemotron-nano-12b-v2-vl"
LIVE_NVIDIA = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
LIVE_GEMINI = "gemini-3.6-flash"


def _settings(**kw: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "GEMINI_API_KEY": "g",
        "NVIDIA_API_KEY": "n",
    }
    base.update(kw)
    return Settings(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1. The EOL model is unavailable, and says why.
# --------------------------------------------------------------------------- #


def test_the_retired_model_is_marked_unavailable_with_its_reason() -> None:
    card = find_model("nvidia", RETIRED)

    assert card is not None, "the card stays catalogued, so the reason survives"
    assert card.is_available is False
    assert card.retired_reason and "410" in card.retired_reason
    assert not card.serves("visual")
    assert not card.serves("text")


def test_retirement_does_not_rewrite_the_model_s_capabilities() -> None:
    """The card must not lie in the other direction either.

    Zeroing the capability flags would be an easy way to make it unselectable,
    and it would make the catalog wrong about the model.
    """

    card = find_model("nvidia", RETIRED)
    assert card is not None
    assert card.supports_image is True
    assert card.supports_role("visual") is True


def test_a_retired_card_cannot_be_declared_without_a_reason() -> None:
    """Reviving one silently would otherwise be a one-word edit."""

    card = find_model("nvidia", RETIRED)
    assert card is not None
    with pytest.raises(ValueError):
        card.model_copy(update={"retired_reason": None}).model_validate(
            card.model_dump() | {"retired_reason": None}
        )


# --------------------------------------------------------------------------- #
# 2. The backend refuses it, by whichever route it is asked for.
# --------------------------------------------------------------------------- #


def test_an_explicitly_requested_retired_model_is_refused() -> None:
    with pytest.raises(UpstreamServiceError) as caught:
        get_visual_analyst(settings=_settings(), provider="nvidia", model=RETIRED)

    message = caught.value.message
    assert "retired" in message.lower()
    # The refusal must name the real problem. `serves()` is False for a retired
    # model, so without an explicit check the caller was told it "does not
    # support visual analysis" - true of the wrong thing, and it sends them to
    # change capability instead of model.
    assert "does not support visual analysis" not in message
    assert RETIRED in message


def test_a_retired_model_configured_as_the_default_is_refused_too() -> None:
    """The route that actually bit us: it was the shipped default."""

    with pytest.raises(UpstreamServiceError) as caught:
        get_visual_analyst(
            settings=_settings(NVIDIA_MODEL=RETIRED), provider="nvidia"
        )

    assert "retired" in caught.value.message.lower()


def test_the_shipped_default_nvidia_model_is_not_retired() -> None:
    """A fresh deployment must not be broken before it sends a request."""

    default = Settings(_env_file=None).nvidia_model  # type: ignore[arg-type]
    card = find_model("nvidia", default)

    assert card is not None, f"the default {default!r} is not catalogued"
    assert card.is_available, f"the default {default!r} is retired"


# --------------------------------------------------------------------------- #
# 3 + 4 + 5. Live models stay selectable and truthfully described.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("provider", "model"), [("gemini", LIVE_GEMINI), ("nvidia", LIVE_NVIDIA)]
)
def test_a_live_model_remains_selectable(provider: str, model: str) -> None:
    """The counter-case. Without it, refusing everything would pass above."""

    analyst = get_visual_analyst(
        settings=_settings(), provider=provider, model=model
    )

    assert analyst.model_name == model


def test_every_catalogued_model_is_either_available_or_explained() -> None:
    for card in MODEL_CATALOG:
        if not card.is_available:
            assert card.retired_reason, f"{card.model_id} retired without a reason"


def test_at_least_one_model_per_provider_is_still_available() -> None:
    """A provider whose every model retired is a deployment that cannot run."""

    for provider in ("gemini", "nvidia"):
        live = [
            c for c in MODEL_CATALOG if c.provider == provider and c.is_available
        ]
        assert live, f"no available model remains for {provider}"
