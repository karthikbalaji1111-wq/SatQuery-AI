"""Fixed public Sentinel-1 RTC catalog and bounded asset signing.

The provider supplies radiometrically terrain-corrected gamma-naught rasters.
SatQuery only renders their values; it performs no SAR calibration itself.
"""
from urllib.parse import urlsplit

import httpx

from app.core.config import Settings
from app.core.errors import InvalidInputError, UpstreamServiceError

RTC_COLLECTION = "sentinel-1-rtc"
RTC_CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"
_SIGN_ENDPOINT = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
_RTC_HOSTS = frozenset({
    "sentinel1euwestrtc.blob.core.windows.net",
    "sentinel1westeurope.blob.core.windows.net",
})


def catalog_for(collection: str, settings: Settings) -> str:
    return RTC_CATALOG if collection == RTC_COLLECTION else settings.stac_base_url


def sign_rtc_asset(
    href: str, *, settings: Settings,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Sign only known RTC storage paths; never return tokens to the browser."""
    parsed = urlsplit(href)
    if (
        parsed.scheme != "https" or parsed.hostname not in _RTC_HOSTS
        or parsed.netloc != parsed.hostname
        or not parsed.path.startswith("/sentinel1-grd-rtc/GRD/")
        or not parsed.path.endswith(("/iw-vv.rtc.tiff", "/iw-vh.rtc.tiff"))
        or parsed.query or parsed.fragment
        or any(c in href for c in "\x00\r\n\t")
    ):
        raise InvalidInputError("The Sentinel-1 RTC asset is not an approved public raster.")
    try:
        with httpx.Client(timeout=settings.http_timeout_seconds, transport=transport) as client:
            response = client.get(_SIGN_ENDPOINT, params={"href": href})
        response.raise_for_status()
        signed = response.json()["href"]
        resolved = urlsplit(signed)
        if (
            resolved.scheme != parsed.scheme or resolved.netloc != parsed.netloc
            or resolved.path != parsed.path or resolved.fragment or not resolved.query
            or any(c in signed for c in "\x00\r\n\t")
        ):
            raise ValueError("Unexpected signed asset")
        return signed
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise UpstreamServiceError(
            "The public Sentinel-1 RTC asset signing service is unavailable."
        ) from exc
