"""Reserved home for server-side map preparation. NOT IMPLEMENTED."""

from __future__ import annotations

from app.services.base import DomainService


class MapService(DomainService):
    """Reserved package for server-side map preparation. **Nothing is
    implemented.**

    No tiles are generated or served. The frontend renders an external raster
    basemap directly and positions imagery from the four WGS 84 corners the
    imagery response already carries, so no server-side map layer is needed
    today.

    Kept as an explicit extension point should tile serving ever be required.
    """

    name = "map"

    def describe(self) -> str:
        return (
            "Reserved for server-side map preparation. Not implemented: "
            "no tiles are generated or served."
        )


__all__ = ["MapService"]
