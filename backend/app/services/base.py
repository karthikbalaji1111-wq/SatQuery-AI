"""Shared contract for domain services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.errors import NotImplementedFeatureError


class DomainService(ABC):
    """Base class for all SatQuery domain services.

    Concrete implementations arrive in later phases. Until then, subclasses
    inherit :meth:`run`, which raises :class:`NotImplementedFeatureError`.
    """

    #: Stable identifier used in logs, routing, and telemetry.
    name: str = "domain"

    @abstractmethod
    def describe(self) -> str:
        """Return a short human-readable description of this service."""

    def run(self, *_args: Any, **_kwargs: Any) -> Any:
        """Execute the service. Not implemented in the foundation build."""

        raise NotImplementedFeatureError(
            f"The '{self.name}' service is not implemented yet."
        )
