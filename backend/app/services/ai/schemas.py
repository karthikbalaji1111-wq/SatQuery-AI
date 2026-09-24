"""Request models for the AI (intent-extraction) service."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ParsePromptRequest(BaseModel):
    """A raw natural-language request to translate into a ``SatQueryIntent``."""

    prompt: str = Field(min_length=1, max_length=4000)
    #: An AI provider to parse with. ``None`` - the default - uses the standard
    #: deterministic parser, which needs no model and no credential.
    provider: str | None = Field(default=None, max_length=32)

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower() or None
        return value

    @field_validator("prompt", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value
