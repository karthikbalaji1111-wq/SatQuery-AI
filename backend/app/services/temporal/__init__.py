"""Reserved home for time-series analysis. NOT IMPLEMENTED."""

from __future__ import annotations

from app.services.base import DomainService


class TemporalService(DomainService):
    """Reserved package for time-series analysis. **Nothing is implemented.**

    The multitemporal capability that does exist - Temporal NDWI Statistics
    and NDWI change over one deterministic Sentinel-2 pair - lives in
    ``services/analysis``, and refuses to compare grids it cannot prove are
    identical. This package holds no change-detection logic.

    Kept as an explicit extension point for a general time-series framework.
    """

    name = "temporal"

    def describe(self) -> str:
        return (
            "Reserved for time-series analysis. Not implemented: the "
            "temporal NDWI capability lives in services/analysis."
        )


__all__ = ["TemporalService"]
