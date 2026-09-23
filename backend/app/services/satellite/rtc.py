"""Fixed public Sentinel-1 RTC catalog and bounded asset signing.

The provider supplies radiometrically terrain-corrected gamma-naught rasters.
SatQuery only renders their values; it performs no SAR calibration itself.

WHY SIGNING EXISTS. The RTC rasters live in an Azure storage account that
refuses public access - a plain GET returns ``409 PublicAccessNotPermitted`` -
so the href published in the STAC item cannot be opened as it stands. The
Planetary Computer exposes an ANONYMOUS signing endpoint that exchanges that
href for a short-lived read-only SAS URL for the same blob. No account, key or
credential is held by this deployment; the exchange is a public service call.

The signed URL is used to open the raster and is never returned to a client:
``ImageryResponse.asset_href`` carries the unsigned href, and the raster layer
strips the query string before logging.
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
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            # A distinct, recoverable condition: the provider rate-limits
            # anonymous signing. Collapsing it into "unavailable" would tell an
            # operator to investigate an outage that is not happening.
            raise UpstreamServiceError(
                "The Sentinel-1 RTC provider is rate-limiting anonymous asset "
                "signing right now. Discovery and scene metadata are "
                "unaffected; retry the imagery request shortly."
            )
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
            "The public Sentinel-1 RTC asset signing service did not return a "
            "usable read URL, so the raster could not be opened. Sentinel-1 "
            "discovery and scene metadata are unaffected."
        ) from exc


def require_linear_power_encoding(
    *, scale: object = 1.0, offset: object = 0.0, unit: object = None,
) -> None:
    """Refuse encoded/scaled or decibel data instead of applying a second conversion."""
    if scale != 1.0 or offset != 0.0 or unit not in (None, "", "1", "linear", "power"):
        raise InvalidInputError(
            "Sentinel-1 RTC quantitative analysis requires unscaled linear gamma-naught "
            "power. The raster metadata advertises a different scale, offset or unit; "
            "no backscatter statistics were computed."
        )
