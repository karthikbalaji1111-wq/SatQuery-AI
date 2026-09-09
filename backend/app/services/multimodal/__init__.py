"""Reserved home for cross-modal analysis. NOT IMPLEMENTED."""

from __future__ import annotations

from app.services.base import DomainService


class MultimodalService(DomainService):
    """Reserved package for cross-modal analysis. **Nothing is implemented.**

    The name is aspirational and the class is a placeholder: there is no
    optical/SAR fusion, no co-registration and no joint pixel analysis
    anywhere in this repository. Sentinel-1 and Sentinel-2 are discovered and
    analysed independently, and any answer that draws on both does so at the
    level of separately attributed evidence, never fused pixels.

    Kept as an explicit extension point so the boundary is visible rather than
    implied. See ``services/analysis`` for what is actually computed.
    """

    name = "multimodal"

    def describe(self) -> str:
        return (
            "Reserved for cross-modal analysis. Not implemented: no "
            "optical/SAR fusion or co-registration exists."
        )


__all__ = ["MultimodalService"]
