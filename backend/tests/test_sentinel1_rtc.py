"""RTC catalog, signing, provenance and real raster-boundary regressions."""
from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np
import pytest
from app.core.config import Settings
from app.core.errors import InvalidInputError, UpstreamServiceError
from app.services.query.execution import QueryExecutionService
from app.services.query.schemas import QueryExecutionRequest
from app.services.query.service import QueryService
from app.services.satellite import raster
from app.services.satellite.imagery import ImageryService
from app.services.satellite.rtc import RTC_CATALOG, RTC_COLLECTION, catalog_for, sign_rtc_asset
from app.services.satellite.schemas import ImageryRequest, SceneSearchRequest
from app.services.satellite.service import SatelliteService
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

from tests.test_imagery import bbox_for_window, synthetic_raster
from tests.test_query_execution import FakeGeospatialService, FakeSatelliteService, make_intent

HREF = (
    "https://sentinel1euwestrtc.blob.core.windows.net/"
    "sentinel1-grd-rtc/GRD/2025/1/11/IW/DV/scene/measurement/iw-vv.rtc.tiff"
)


def test_rtc_catalog_is_local_to_its_collection() -> None:
    settings = Settings()
    assert settings.stac_s1_collection == RTC_COLLECTION
    assert catalog_for(RTC_COLLECTION, settings) == RTC_CATALOG
    assert catalog_for("sentinel-1-grd", settings) == settings.stac_base_url
    assert catalog_for("sentinel-2-l2a", settings) == settings.stac_base_url
    satellite = FakeSatelliteService()
    execution = QueryExecutionService(
        settings=settings,
        query_service=QueryService(geospatial_service=FakeGeospatialService()),
        satellite_service=satellite,
    )
    asyncio.run(execution.execute(QueryExecutionRequest(
        intent=make_intent(modalities=["sentinel-1-sar"]),
    )))
    assert satellite.requests[0].collection == RTC_COLLECTION


def test_rtc_discovery_preserves_metadata_and_excludes_optical_cloud_filter() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == RTC_CATALOG + "/search"
        body = json.loads(request.content)
        assert body["collections"] == [RTC_COLLECTION]
        assert "query" not in body
        return httpx.Response(200, json={"features": [{
            "id": "test_rtc", "collection": RTC_COLLECTION,
            "properties": {
                "datetime": "2025-01-11T00:31:00Z", "sar:polarizations": ["VV", "VH"],
                "sar:instrument_mode": "IW", "sat:orbit_state": "descending",
            },
            "assets": {"vv": {"href": HREF, "type": "image/tiff; application=geotiff"}},
        }]})
    response = asyncio.run(SatelliteService(transport=httpx.MockTransport(respond)).search(
        SceneSearchRequest(
            bbox=bbox_for_window(Window(0, 0, 20, 20)),
            start_date="2025-01-01", end_date="2025-01-31",
            collection=RTC_COLLECTION, max_cloud_cover=1,
        )
    ))
    scene = response.scenes[0]
    assert response.catalog == RTC_CATALOG
    assert scene.sar_polarizations == ["VV", "VH"]
    assert scene.sar_instrument_mode == "IW"
    assert scene.orbit_state == "descending"
    assert "gamma naught" in scene.processing_level
    assert scene.assets[0].key == "vv"


@pytest.mark.parametrize("asset", ["vv", "vh"])
def test_rtc_reads_bounded_georeferenced_pixels_without_exposing_signature(
    monkeypatch: pytest.MonkeyPatch, asset: str,
) -> None:
    href = HREF.replace("iw-vv", f"iw-{asset}")
    signed = href + "?sig=test-only-token"
    seen = []
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sign"):
            assert request.url.params["href"] == href
            return httpx.Response(200, json={"href": signed})
        assert str(request.url) == RTC_CATALOG + "/collections/sentinel-1-rtc/items/test_rtc"
        return httpx.Response(200, json={"assets": {
            asset: {"href": href, "type": "image/tiff; application=geotiff"},
        }})
    data = np.tile(np.linspace(0.001, 0.2, 20, dtype="float32"), (20, 1))
    with synthetic_raster(count=1, dtype="float32", data=data) as mem:
        def open_raster(url: str):
            seen.append(url)
            return mem.open()
        monkeypatch.setattr(raster, "_open_raster", open_raster)
        result = ImageryService(transport=httpx.MockTransport(respond)).retrieve(
            ImageryRequest(
                scene_id="test_rtc", collection=RTC_COLLECTION, asset=asset,
                bbox=bbox_for_window(Window(0, 0, 20, 20)),
            )
        )
    assert seen == [signed]
    assert result.asset_href == href
    assert "test-only-token" not in result.model_dump_json()
    assert result.bands == [asset] * 3
    assert result.crs == "EPSG:32644"
    assert len(result.corners_wgs84) == 4
    assert result.image_base64
    assert "Provider RTC gamma naught" in result.normalization
    assert "no local calibration" in result.normalization


@pytest.mark.parametrize("href", [
    HREF.replace("https:", "http:"), HREF.replace("sentinel1euwestrtc", "evil"),
    HREF.replace(".net/", ".net.evil/"), HREF.replace(".net/", ".net:443/"),
    HREF.replace("/sentinel1-grd-rtc/", "/private/"),
    HREF + "?sig=untrusted", HREF + "#fragment", HREF + "\n",
])
def test_signing_rejects_unapproved_urls_before_network(href: str) -> None:
    def forbidden(_: httpx.Request) -> httpx.Response:
        pytest.fail("Unapproved URL reached signing service")
    with pytest.raises(InvalidInputError):
        sign_rtc_asset(href, settings=Settings(), transport=httpx.MockTransport(forbidden))


@pytest.mark.parametrize("signed", [
    "https://evil.test/x?sig=token", HREF.replace("/measurement/", "/other/") + "?sig=x",
    HREF, HREF + "?sig=x#fragment", None,
])
def test_signing_rejects_changed_resource_and_malformed_response(signed: str | None) -> None:
    with pytest.raises(UpstreamServiceError):
        sign_rtc_asset(HREF, settings=Settings(), transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"href": signed}),
        ))


def test_raster_failure_logs_do_not_disclose_signed_query(monkeypatch, caplog) -> None:
    signed = HREF + "?sig=test-only-token"
    def fail(_: str):
        raise RasterioIOError(signed)
    monkeypatch.setattr(raster, "_open_raster", fail)
    with pytest.raises(UpstreamServiceError):
        raster.read_rgb_window(
            signed, bbox_for_window(Window(0, 0, 20, 20)),
            max_dimension=1024, max_window_pixels=50_000_000,
        )
    assert "test-only-token" not in caplog.text
