"""Request models for the AI (intent-extraction) service."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ParsePromptRequest(BaseModel):
    """A raw natural-language request to translate into a ``SatQueryIntent``."""

    prompt: str = Field(min_length=1, max_length=4000)

    @field_validator("prompt", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value
