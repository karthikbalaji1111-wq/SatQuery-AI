"""Phase 18.1 - the provider-neutral vision-language boundary.

:class:`VisualAnalyst` is the abstraction the agent depends on. **This module
contains no provider SDK.** Every provider-specific import lives in
:mod:`app.services.agent.providers`, and an AST test enforces that boundary.

The analyst is handed **bytes**. It cannot fetch, resolve, download or choose an
image: there is no URL parameter, no path, no scene id and no catalog handle in
this interface. The executor decodes the image the query pipeline already
retrieved and passes it in, which is what makes "the server decides what the
model looks at" a property of the type rather than a convention.

What comes back is a *claim*, not a measurement. :class:`VisualAnswer` carries a
sentence and nothing that could be mistaken for an established quantity - see
``grounding._allowed_values``, which refuses to let model-sourced evidence
authorise any number, including one the model states inside that sentence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class VisualAnswer(BaseModel):
    """One vision-language model's answer about one image.

    Deliberately a single field. A confidence score was considered and rejected:
    a number a model assigns to its own certainty has no defined semantics here,
    cannot be calibrated against anything this system measures, and would invite
    exactly the reading - "0.9 means probably true" - that the rest of this
    phase exists to prevent. A qualifier the model chooses to put *into* its
    sentence ("appears to be", "likely") is preserved verbatim instead, where it
    is visibly the model's own hedging rather than a system-issued figure.
    """

    model_config = ConfigDict(extra="forbid")

    #: Exactly what the model said, unedited.
    answer: str = Field(min_length=1, max_length=4000)


class VisualAnalyst(ABC):
    """Abstract observer of one already-retrieved image.

    Implementations must not leak provider concepts through this interface -
    swapping the provider must not change the agent, the evidence shape or the
    API contract.

    ``provider_name`` and ``model_name`` exist so a claim can be attributed to
    something specific. An observation whose author cannot be named is not
    publishable evidence, so both travel with it into the trace.
    """

    #: Provider family, e.g. ``"gemini"``. Overridden by implementations.
    provider_name: str = "unknown"
    #: The specific model and version, e.g. ``"gemini-3.6-flash"``.
    model_name: str = "unknown"

    @abstractmethod
    async def observe(
        self, *, question: str, image: bytes, media_type: str
    ) -> VisualAnswer:
        """Answer ``question`` about ``image``.

        ``image`` is the exact PNG the query pipeline already produced. An
        implementation must send those bytes and must not re-fetch, re-encode
        or substitute anything.

        Implementations raise rather than inventing an answer: a provider
        failure must stay a failure, so the executor can record it and emit no
        evidence at all.
        """


_DEFAULT_ANSWER = VisualAnswer(
    answer=(
        "This is a mock visual analyst; no image was inspected and this "
        "sentence describes nothing about the scene."
    )
)


class MockVisualAnalyst(VisualAnalyst):
    """Deterministic stand-in that never looks at the image.

    Its answer says so in as many words, so a mock observation can never be
    mistaken for a real one if it reaches a screen.
    """

    is_mock = True
    provider_name = "mock"
    model_name = "mock-visual-analyst"

    def __init__(self, answer: VisualAnswer | None = None) -> None:
        self._answer = answer or _DEFAULT_ANSWER

    async def observe(
        self, *, question: str, image: bytes, media_type: str
    ) -> VisualAnswer:
        # Every input is intentionally ignored - this mock inspects nothing, and
        # it must never touch the bytes it is handed.
        del question, image, media_type
        return self._answer.model_copy(deep=True)
