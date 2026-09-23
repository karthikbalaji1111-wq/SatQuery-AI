"""Where a catalog may send this server to read pixels.

An asset href is not this application's string. It arrives inside an external
catalog's response and is then opened BY THIS PROCESS, from inside whatever
network it runs in. The scheme check established that the URL was HTTP(S); it
said nothing about where. ``https://169.254.169.254/latest/meta-data/`` is a
perfectly well-formed HTTPS URL, and on a cloud host it is the instance
metadata service.

Two rules of deliberately different strength, and both are pinned here:

* **always** - a private, loopback, link-local, reserved, multicast or
  unspecified IP address is refused. No public satellite catalog publishes
  rasters there, so this costs nothing legitimate;
* **when configured** - ``trusted_asset_hosts`` narrows reads to named hosts.
  Empty (the default) allows any public host, so existing deployments and the
  test fixtures that use example hosts are unaffected.

KNOWN LIMITATION, asserted nowhere because it is not fixed here: the check is on
the NAME, not on the address it resolves to, so a DNS rebind between this check
and the read is not prevented. The raster stack does not expose the connection
that would be needed to pin it.
"""

from __future__ import annotations

import pytest
from app.core.config import Settings
from app.core.errors import InvalidInputError
from app.services.satellite.imagery import (
    ImageryService,
    _require_permitted_host,
)

COG = "image/tiff; application=geotiff; profile=cloud-optimized"
EARTH_SEARCH_ASSET = (
    "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
    "44/P/MV/2025/1/S2B_44PMV_20250104_0_L2A/TCI.tif"
)
PLANETARY_ASSET = (
    "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/"
    "S1A_IW_GRDH_20250111/measurement/iw-vv.rtc.tiff"
)


# --------------------------------------------------------------------------- #
# Always refused, whatever is configured
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "href"),
    [
        ("cloud metadata service", "https://169.254.169.254/latest/meta-data/"),
        ("loopback", "https://127.0.0.1/secret.tif"),
        ("loopback with a port", "https://127.0.0.1:8000/x.tif"),
        ("private class A", "https://10.0.0.5/x.tif"),
        ("private class C", "https://192.168.1.10/x.tif"),
        ("carrier-grade NAT", "https://100.64.0.1/x.tif"),
        ("unspecified", "https://0.0.0.0/x.tif"),
        ("multicast", "https://224.0.0.1/x.tif"),
        ("IPv6 loopback", "https://[::1]/x.tif"),
        ("IPv6 link-local", "https://[fe80::1]/x.tif"),
    ],
)
def test_a_non_public_address_is_refused(label: str, href: str) -> None:
    with pytest.raises(InvalidInputError) as raised:
        _require_permitted_host("visual", href, [])

    assert "non-public address" in str(raised.value), label


def test_a_href_without_a_host_is_refused() -> None:
    with pytest.raises(InvalidInputError):
        _require_permitted_host("visual", "https:///x.tif", [])


# --------------------------------------------------------------------------- #
# Public hosts, and the optional allowlist
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("href", [EARTH_SEARCH_ASSET, PLANETARY_ASSET])
def test_a_real_catalog_asset_is_allowed_without_configuration(href: str) -> None:
    """Non-vacuity: the default must not refuse the assets this system reads."""

    _require_permitted_host("visual", href, [])  # must not raise


def test_an_allowlist_refuses_a_host_that_is_not_on_it() -> None:
    with pytest.raises(InvalidInputError) as raised:
        _require_permitted_host(
            "visual",
            "https://evil.example/x.tif",
            ["sentinel-cogs.s3.us-west-2.amazonaws.com"],
        )

    assert "not among the asset hosts" in str(raised.value)


def test_an_allowlist_admits_the_hosts_on_it() -> None:
    _require_permitted_host(
        "visual", EARTH_SEARCH_ASSET, ["sentinel-cogs.s3.us-west-2.amazonaws.com"]
    )


def test_a_dot_prefixed_entry_matches_subdomains_only() -> None:
    trusted = [".blob.core.windows.net"]

    _require_permitted_host("vv", PLANETARY_ASSET, trusted)  # a real subdomain

    with pytest.raises(InvalidInputError):
        # The classic near-miss: the trusted name as a PREFIX of another domain.
        _require_permitted_host(
            "vv", "https://blob.core.windows.net.evil.example/x.tif", trusted
        )


def test_matching_ignores_case() -> None:
    _require_permitted_host(
        "visual",
        "https://SENTINEL-COGS.s3.us-west-2.amazonaws.com/x.tif",
        ["sentinel-cogs.s3.us-west-2.amazonaws.com"],
    )


# --------------------------------------------------------------------------- #
# Wired into the path that actually opens rasters
# --------------------------------------------------------------------------- #


def item(href: str) -> dict[str, object]:
    return {"assets": {"visual": {"href": href, "type": COG}}}


def test_the_service_refuses_a_redirected_asset_host() -> None:
    """The check has to run where the href is resolved, not only in isolation."""

    service = ImageryService(
        Settings(trusted_asset_hosts=["sentinel-cogs.s3.us-west-2.amazonaws.com"])
    )

    with pytest.raises(InvalidInputError):
        service._resolve_asset_href(item("https://evil.example/x.tif"), "visual")


def test_the_service_still_resolves_a_legitimate_asset() -> None:
    service = ImageryService(
        Settings(trusted_asset_hosts=["sentinel-cogs.s3.us-west-2.amazonaws.com"])
    )

    assert (
        service._resolve_asset_href(item(EARTH_SEARCH_ASSET), "visual")
        == EARTH_SEARCH_ASSET
    )


def test_the_scheme_and_control_character_rules_still_apply() -> None:
    """The host policy is additive; it must not replace what was there."""

    service = ImageryService(Settings())

    with pytest.raises(InvalidInputError):
        service._resolve_asset_href(item("s3://requester-pays/x.tif"), "visual")
    with pytest.raises(InvalidInputError):
        service._resolve_asset_href(
            item("https://sentinel-cogs.s3.us-west-2.amazonaws.com/x.tif\x00.s3"),
            "visual",
        )
