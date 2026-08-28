"""Bounded Sentinel-2 imagery retrieval tests.

No test touches live imagery or the live catalog: the STAC item lookup is
stubbed with ``httpx.MockTransport`` or an injected fetcher, and windowed
raster reads run against a small synthetic in-memory GeoTIFF.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import rasterio
from app.api.routes.satellite import get_imagery_service
from app.core.errors import InvalidInputError, UpstreamServiceError
from app.main import create_app
from app.services.geospatial.schemas import BoundingBox
from app.services.satellite import ImageryService
from app.services.satellite import imagery as imagery_mod
from app.services.satellite import raster as raster_mod
from app.services.satellite.raster import RgbWindow, read_rgb_window
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

IMAGERY_URL = "/api/v1/satellite/imagery"

VISUAL_HREF = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/x/y/z/TCI.tif"

# A synthetic raster tile: UTM 44N, 10 m pixels, same origin as MGRS 44PMV.
_ORIGIN_X, _ORIGIN_Y, _RES = 399960.0, 1500000.0, 10.0
_SRC_W, _SRC_H = 1200, 1600


@contextmanager
def synthetic_raster(
    *, width: int = _SRC_W, height: int = _SRC_H, count: int = 3, dtype: str = "uint8"
) -> Iterator[MemoryFile]:
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, _RES, _RES)
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=width,
            height=height,
            count=count,
            dtype=dtype,
            crs="EPSG:32644",
            transform=transform,
        ) as dataset:
            for band in range(1, count + 1):
                fill = (band * 60) % 256
                dataset.write(np.full((height, width), fill, dtype=dtype), band)
        yield mem


def bbox_for_window(window: Window) -> BoundingBox:
    """WGS84 bbox that maps back onto ``window`` of the synthetic raster."""

    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, _RES, _RES)
    left, bottom, right, top = window_bounds(window, transform)
    west, south, east, north = transform_bounds(
        "EPSG:32644", "EPSG:4326", left, bottom, right, top
    )
    return BoundingBox(west=west, south=south, east=east, north=north)


def stac_item(
    *,
    scene_id: str = "S2A_44PMV_20240214_0_L2A",
    assets: dict[str, Any] | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": scene_id,
        "collection": "sentinel-2-l2a",
        "bbox": bbox if bbox is not None else [80.0, 12.5, 81.1, 13.6],
        "properties": {"datetime": "2024-02-14T05:15:10Z"},
        "assets": assets
        if assets is not None
        else {
            "visual": {
                "href": VISUAL_HREF,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["visual"],
            }
        },
    }


def fake_rgb_window(width: int = 6, height: int = 4) -> RgbWindow:
    array = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
    return RgbWindow(
        array=array,
        width=width,
        height=height,
        crs="EPSG:32644",
        resolution=10.0,
        bands=["red", "green", "blue"],
        window={"col_off": 100, "row_off": 200, "width": width, "height": height},
        source_shape=[_SRC_H, _SRC_W],
        normalization="none (source is 8-bit RGB)",
    )


class RecordingReader:
    def __init__(self, result: RgbWindow | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result or fake_rgb_window()

    def __call__(self, href: str, bbox: BoundingBox, **kwargs: Any) -> RgbWindow:
        self.calls.append({"href": href, "bbox": bbox, **kwargs})
        return self._result


class RecordingFetcher:
    def __init__(self, item: dict[str, Any] | None = None) -> None:
        self.scene_ids: list[str] = []
        self._item = item if item is not None else stac_item()

    def __call__(self, scene_id: str) -> dict[str, Any]:
        self.scene_ids.append(scene_id)
        return {**self._item, "id": scene_id}


def make_client(
    *,
    fetcher: Any = None,
    reader: Any = None,
    transport: Any = None,
    settings: Any = None,
) -> TestClient:
    app = create_app()
    if fetcher is None and transport is None:
        # No real HTTP wanted -> default to an in-memory fetcher.
        fetcher = RecordingFetcher()
    app.dependency_overrides[get_imagery_service] = lambda: ImageryService(
        settings=settings,
        transport=transport,
        stac_item_fetcher=fetcher,  # None -> real _default_fetch_item (uses transport)
        raster_reader=reader if reader is not None else RecordingReader(),
    )
    return TestClient(app)


def valid_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "scene_id": "S2A_44PMV_20240214_0_L2A",
        "bbox": {"west": 80.20, "south": 13.00, "east": 80.24, "north": 13.04},
    }
    body.update(overrides)
    return body


def decode_png(image_base64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_base64)))


# --------------------------------------------------------------------------- #
# 1-6, 15: route behaviour with injected fakes
# --------------------------------------------------------------------------- #


def test_successful_bounded_retrieval() -> None:
    reader = RecordingReader(fake_rgb_window(width=6, height=4))
    response = make_client(reader=reader).post(IMAGERY_URL, json=valid_body())

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "S2A_44PMV_20240214_0_L2A"
    assert body["asset"] == "visual"
    assert body["asset_href"] == VISUAL_HREF
    assert body["format"] == "png"
    assert body["media_type"] == "image/png"
    assert body["bands"] == ["red", "green", "blue"]
    assert body["crs"] == "EPSG:32644"
    assert body["resolution"] == pytest.approx(10.0)
    assert body["normalization"] == "none (source is 8-bit RGB)"
    assert body["source_shape"] == [_SRC_H, _SRC_W]

    image = decode_png(body["image_base64"])
    assert image.format == "PNG"
    assert image.mode == "RGB"
    assert image.size == (body["width"], body["height"]) == (6, 4)


def test_correct_scene_and_asset_selection() -> None:
    fetcher = RecordingFetcher()
    reader = RecordingReader()
    make_client(fetcher=fetcher, reader=reader).post(
        IMAGERY_URL, json=valid_body(scene_id="MY_SCENE_ID")
    )

    assert fetcher.scene_ids == ["MY_SCENE_ID"]
    assert reader.calls[0]["href"] == VISUAL_HREF
    assert isinstance(reader.calls[0]["bbox"], BoundingBox)
    assert reader.calls[0]["bbox"].west == pytest.approx(80.20)


def test_metadata_response_contract() -> None:
    body = make_client().post(IMAGERY_URL, json=valid_body()).json()
    for key in (
        "scene_id",
        "bbox",
        "asset",
        "asset_href",
        "width",
        "height",
        "format",
        "media_type",
        "bands",
        "crs",
        "resolution",
        "normalization",
        "window",
        "source_shape",
        "image_base64",
    ):
        assert key in body, key
    assert set(body["window"]) == {"col_off", "row_off", "width", "height"}
    assert body["bbox"] == valid_body()["bbox"]


def test_rgb_output_is_three_channel() -> None:
    body = make_client().post(IMAGERY_URL, json=valid_body()).json()
    array = np.asarray(decode_png(body["image_base64"]))
    assert array.ndim == 3
    assert array.shape[2] == 3
    assert array.dtype == np.uint8


def test_max_dimension_override_is_forwarded_and_capped() -> None:
    reader = RecordingReader()
    make_client(reader=reader).post(
        IMAGERY_URL, json=valid_body(max_dimension=4096)
    )
    # request 4096 -> capped to imagery_hard_max_dimension (2048)
    assert reader.calls[0]["max_dimension"] == 2048


# --------------------------------------------------------------------------- #
# 7-10: input / resource validation
# --------------------------------------------------------------------------- #


def test_invalid_bbox_reversed_is_rejected() -> None:
    reader = RecordingReader()
    response = make_client(reader=reader).post(
        IMAGERY_URL,
        json=valid_body(
            bbox={"west": 80.30, "south": 13.00, "east": 80.20, "north": 13.04}
        ),
    )
    assert response.status_code == 422
    assert reader.calls == []


def test_invalid_bbox_out_of_range_is_rejected() -> None:
    response = make_client().post(
        IMAGERY_URL,
        json=valid_body(
            bbox={"west": -181.0, "south": 13.0, "east": 80.2, "north": 13.04}
        ),
    )
    assert response.status_code == 422


def test_bbox_outside_scene_is_rejected_before_any_raster_read() -> None:
    fetcher = RecordingFetcher(stac_item(bbox=[10.0, 40.0, 11.0, 41.0]))
    reader = RecordingReader()
    response = make_client(fetcher=fetcher, reader=reader).post(
        IMAGERY_URL, json=valid_body()
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert reader.calls == []


def test_unknown_scene_returns_404() -> None:
    def missing(_scene_id: str) -> dict[str, Any]:
        from app.core.errors import NotFoundError

        raise NotFoundError("Scene not found in the catalog.")

    response = make_client(fetcher=missing).post(IMAGERY_URL, json=valid_body())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_missing_asset_returns_404() -> None:
    fetcher = RecordingFetcher(stac_item(assets={"thumbnail": {"href": "x", "type": "image/jpeg"}}))
    reader = RecordingReader()
    response = make_client(fetcher=fetcher, reader=reader).post(
        IMAGERY_URL, json=valid_body()
    )
    assert response.status_code == 404
    assert reader.calls == []


def test_unsupported_asset_key_rejected_without_catalog_call() -> None:
    fetcher = RecordingFetcher()
    reader = RecordingReader()
    response = make_client(fetcher=fetcher, reader=reader).post(
        IMAGERY_URL, json=valid_body(asset="thumbnail")
    )
    assert response.status_code == 422
    assert fetcher.scene_ids == []
    assert reader.calls == []


def test_unsupported_asset_media_type_rejected() -> None:
    fetcher = RecordingFetcher(
        stac_item(assets={"visual": {"href": "https://x/y.jp2", "type": "image/jp2"}})
    )
    reader = RecordingReader()
    response = make_client(fetcher=fetcher, reader=reader).post(
        IMAGERY_URL, json=valid_body()
    )
    assert response.status_code == 422
    assert reader.calls == []


# --------------------------------------------------------------------------- #
# 11-13: upstream failures via the real httpx STAC-item fetcher
# --------------------------------------------------------------------------- #


def _imagery_client_with_transport(handler: Any) -> TestClient:
    import httpx

    transport = httpx.MockTransport(handler)
    return make_client(fetcher=None, reader=RecordingReader(), transport=transport)


def test_malformed_stac_item_returns_502() -> None:
    import httpx

    response = _imagery_client_with_transport(
        lambda _req: httpx.Response(200, json={"type": "Feature"})  # no assets
    ).post(IMAGERY_URL, json=valid_body())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_stac_item_http_404_returns_404() -> None:
    import httpx

    response = _imagery_client_with_transport(
        lambda _req: httpx.Response(404, text="not found")
    ).post(IMAGERY_URL, json=valid_body())
    assert response.status_code == 404


def test_stac_item_network_failure_returns_502() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    response = _imagery_client_with_transport(handler).post(
        IMAGERY_URL, json=valid_body()
    )
    assert response.status_code == 502


def test_stac_item_timeout_returns_502() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    response = _imagery_client_with_transport(handler).post(
        IMAGERY_URL, json=valid_body()
    )
    assert response.status_code == 502


def test_imagery_endpoint_makes_no_nl_search_only_item_lookup() -> None:
    import httpx

    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json=stac_item())

    _imagery_client_with_transport(handler).post(IMAGERY_URL, json=valid_body())

    assert len(seen) == 1
    method, path = seen[0]
    assert method == "GET"
    assert path == "/v1/collections/sentinel-2-l2a/items/S2A_44PMV_20240214_0_L2A"


def test_raster_reader_upstream_error_propagates_as_502() -> None:
    def boom(*_args: Any, **_kwargs: Any) -> RgbWindow:
        raise UpstreamServiceError("Could not open the remote raster.")

    response = make_client(reader=boom).post(IMAGERY_URL, json=valid_body())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_conversion_failure_returns_imagery_error() -> None:
    bad = RgbWindow(
        array=np.zeros((4, 4, 5), dtype=np.uint8),  # 5 channels: Pillow rejects
        width=4,
        height=4,
        crs="EPSG:32644",
        resolution=10.0,
        bands=["red", "green", "blue"],
        window={"col_off": 0, "row_off": 0, "width": 4, "height": 4},
        source_shape=[_SRC_H, _SRC_W],
        normalization="none (source is 8-bit RGB)",
    )
    response = make_client(reader=lambda *a, **k: bad).post(
        IMAGERY_URL, json=valid_body()
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "imagery_error"


# --------------------------------------------------------------------------- #
# 3-5, 14, 15: real windowed reads against a synthetic in-memory GeoTIFF
# --------------------------------------------------------------------------- #


def test_window_math_reads_only_the_requested_sub_window(monkeypatch: Any) -> None:
    target = Window(col_off=400, row_off=500, width=300, height=400)
    req_bbox = bbox_for_window(target)

    seen: list[dict[str, Any]] = []
    real_read = raster_mod._read_dataset_window

    def spy(src: Any, indexes: Any, window: Any, out_shape: Any) -> Any:
        seen.append({"window": window, "out_shape": out_shape})
        return real_read(src, indexes, window, out_shape)

    with synthetic_raster() as mem:
        monkeypatch.setattr(raster_mod, "_open_raster", lambda _href: mem.open())
        monkeypatch.setattr(raster_mod, "_read_dataset_window", spy)

        result = read_rgb_window(
            VISUAL_HREF,
            req_bbox,
            max_dimension=1024,
            max_window_pixels=50_000_000,
        )

    # Exactly one read, and it was a Window - never a full-dataset read.
    assert len(seen) == 1
    window = seen[0]["window"]
    assert isinstance(window, Window)
    # floor(offset)/ceil(length) clamping + WGS84<->UTM reprojection expand the
    # window by a few pixels; it must still tightly bracket the target.
    assert abs(window.col_off - 400) <= 8
    assert abs(window.row_off - 500) <= 8
    assert abs(window.width - 300) <= 8
    assert abs(window.height - 400) <= 8

    # The window is a tiny fraction of the full raster.
    src_pixels = _SRC_W * _SRC_H
    assert window.width * window.height < 0.1 * src_pixels

    assert result.source_shape == [_SRC_H, _SRC_W]
    assert result.window["width"] == int(window.width)
    assert result.array.shape == (result.height, result.width, 3)
    assert result.array.dtype == np.uint8
    assert result.crs == "EPSG:32644"
    assert result.resolution == pytest.approx(10.0)
    assert result.normalization == "none (source is 8-bit RGB)"


def test_output_dimensions_are_capped(monkeypatch: Any) -> None:
    target = Window(col_off=100, row_off=100, width=600, height=800)
    req_bbox = bbox_for_window(target)

    with synthetic_raster() as mem:
        monkeypatch.setattr(raster_mod, "_open_raster", lambda _href: mem.open())
        result = read_rgb_window(
            VISUAL_HREF, req_bbox, max_dimension=64, max_window_pixels=50_000_000
        )

    assert max(result.width, result.height) <= 64
    # Native window is still recorded at full resolution.
    assert result.window["height"] > 64


def test_oversized_window_is_rejected(monkeypatch: Any) -> None:
    target = Window(col_off=0, row_off=0, width=800, height=800)
    req_bbox = bbox_for_window(target)

    with synthetic_raster() as mem:
        monkeypatch.setattr(raster_mod, "_open_raster", lambda _href: mem.open())
        with pytest.raises(InvalidInputError):
            read_rgb_window(
                VISUAL_HREF, req_bbox, max_dimension=1024, max_window_pixels=1000
            )


def test_bbox_fully_outside_raster_raises_invalid_input(monkeypatch: Any) -> None:
    far_bbox = BoundingBox(west=-50.0, south=-30.0, east=-49.0, north=-29.0)
    with synthetic_raster() as mem:
        monkeypatch.setattr(raster_mod, "_open_raster", lambda _href: mem.open())
        with pytest.raises(InvalidInputError):
            read_rgb_window(
                VISUAL_HREF, far_bbox, max_dimension=1024, max_window_pixels=50_000_000
            )


def test_raster_open_failure_raises_upstream_error(monkeypatch: Any) -> None:
    def boom(_href: str) -> Any:
        raise rasterio.errors.RasterioIOError("cannot open")

    monkeypatch.setattr(raster_mod, "_open_raster", boom)
    with pytest.raises(UpstreamServiceError):
        read_rgb_window(
            VISUAL_HREF,
            bbox_for_window(Window(10, 10, 50, 50)),
            max_dimension=1024,
            max_window_pixels=50_000_000,
        )


def test_invalid_raster_metadata_too_few_bands(monkeypatch: Any) -> None:
    with synthetic_raster(count=1) as mem:
        monkeypatch.setattr(raster_mod, "_open_raster", lambda _href: mem.open())
        with pytest.raises(UpstreamServiceError):
            read_rgb_window(
                VISUAL_HREF,
                bbox_for_window(Window(10, 10, 50, 50)),
                max_dimension=1024,
                max_window_pixels=50_000_000,
            )


def test_unsupported_datatype_is_refused(monkeypatch: Any) -> None:
    with synthetic_raster(dtype="uint16") as mem:
        monkeypatch.setattr(raster_mod, "_open_raster", lambda _href: mem.open())
        with pytest.raises(UpstreamServiceError):
            read_rgb_window(
                VISUAL_HREF,
                bbox_for_window(Window(10, 10, 50, 50)),
                max_dimension=1024,
                max_window_pixels=50_000_000,
            )


def test_full_pipeline_against_synthetic_raster(monkeypatch: Any) -> None:
    """End-to-end through the route: fake STAC item + real windowed read."""

    target = Window(col_off=200, row_off=300, width=120, height=90)
    req_bbox = bbox_for_window(target)

    with synthetic_raster() as mem:
        monkeypatch.setattr(raster_mod, "_open_raster", lambda _href: mem.open())
        client = make_client(reader=read_rgb_window)  # real reader
        response = client.post(
            IMAGERY_URL,
            json={
                "scene_id": "S2A_44PMV_20240214_0_L2A",
                "bbox": {
                    "west": req_bbox.west,
                    "south": req_bbox.south,
                    "east": req_bbox.east,
                    "north": req_bbox.north,
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    image = decode_png(body["image_base64"])
    assert image.mode == "RGB"
    assert image.size == (body["width"], body["height"])
    assert abs(body["window"]["width"] - 120) <= 3
    assert abs(body["window"]["height"] - 90) <= 3
    assert body["source_shape"] == [_SRC_H, _SRC_W]
    assert imagery_mod is not None  # module import sanity
