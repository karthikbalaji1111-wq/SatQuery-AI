"""Stage 2 of the scientific pipeline: scene + asset validation.

The selected scene is validated against the CATALOG'S item - identity, sensor,
footprint over the requested area, processing metadata and every asset the
operation reads - before any pixel is read. Fixture items are shaped on the
live catalogs (read 2026-09-23): Earth Search ``S2B_44PMV_20250104_0_L2A`` and a
Planetary Computer ``sentinel-1-rtc`` item over the same area.

Also here: the agent's pre-discovery check, which applies the Stage 1 area rule
to the grounded plan before any catalog search.
"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import date
from typing import Any

import httpx
import pytest
from app.core.config import get_settings
from app.core.errors import NotFoundError
from app.services.agent.executor import AgentExecutor
from app.services.agent.schemas import AgentPlan
from app.services.analysis import AnalysisRequest, AnalysisService
from app.services.analysis.validation import AnalysisRequestRejectedError
from app.services.geospatial.schemas import BoundingBox
from app.services.query import QueryService
from app.services.query.execution import QueryExecutionService
from app.services.query.schemas import (
    ExecutedWindow,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    TimeRange,
)
from app.services.satellite import SatelliteService
from app.services.satellite.imagery import ImageryService
from app.services.satellite.rtc import RTC_COLLECTION
from app.services.satellite.scene_validation import (
    SceneValidationError,
    ValidatedScene,
    aoi_coverage_fraction,
    validate_scene_item,
    validate_scene_pair,
)
from app.services.satellite.schemas import Scene

from tests.test_analysis import band
from tests.test_query_execution import FakeGeospatialService

S2 = "sentinel-2-l2a"
OPTICAL = "sentinel-2-optical"
SAR = "sentinel-1-sar"
COG = "image/tiff; application=geotiff; profile=cloud-optimized"
HOST = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2c-cogs/44/P/MV"

AOI = BoundingBox(west=80.25, south=13.00, east=80.30, north=13.05)
LARGE = BoundingBox(west=79.0, south=12.0, east=81.0, north=14.0)
SCENE_ID = "S2B_44PMV_20250104_0_L2A"
LATER_ID = "S2A_44PMV_20250119_0_L2A"
RTC_ID = "S1A_IW_GRDH_1SDV_20250123T003152_20250123T003217_057564_071748_rtc"


def polygon(west: float, south: float, east: float, north: float) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north],
                         [west, south]]],
    }


def optical_band(key: str, resolution: int = 10) -> dict[str, Any]:
    return {
        "href": f"{HOST}/{key}.tif",
        "type": COG,
        "roles": ["data", "reflectance"],
        "raster:bands": [{
            "nodata": 0, "data_type": "uint16", "bits_per_sample": 15,
            "spatial_resolution": resolution, "scale": 0.0001, "offset": -0.1,
        }],
    }


def s2_item(scene_id: str = SCENE_ID, *, when: str = "2025-01-04T05:15:39.024000Z") -> dict:
    return {
        "type": "Feature",
        "id": scene_id,
        "collection": S2,
        "bbox": [79.9, 12.6, 80.9, 13.6],
        "geometry": polygon(80.0, 12.8, 80.6, 13.4),
        "properties": {
            "datetime": when,
            "platform": "sentinel-2b",
            "constellation": "sentinel-2",
            "instruments": ["msi"],
            "eo:cloud_cover": 3.2,
            "s2:processing_baseline": "05.11",
            "s2:product_type": "S2MSI2A",
            "earthsearch:boa_offset_applied": True,
            "processing:software": {"sentinel2-to-stac": "0.1.1"},
        },
        "assets": {
            "red": optical_band("B04"),
            "green": optical_band("B03"),
            "nir": optical_band("B08"),
            "swir16": optical_band("B11", 20),
            # As published: a uint8 class map, 20 m, nodata 0.
            "scl": {
                "href": f"{HOST}/SCL.tif",
                "type": COG,
                "roles": ["data", "reflectance"],
                "raster:bands": [
                    {"nodata": 0, "data_type": "uint8", "spatial_resolution": 20}
                ],
            },
            "visual": {"href": f"{HOST}/TCI.tif", "type": COG, "roles": ["visual"]},
        },
    }


def rtc_item() -> dict:
    sar_band = {"nodata": -32768, "data_type": "float32", "spatial_resolution": 10}
    return {
        "type": "Feature",
        "id": RTC_ID,
        "collection": RTC_COLLECTION,
        "bbox": [79.6, 12.4, 82.1, 14.3],
        "geometry": polygon(79.6, 12.4, 82.1, 14.3),
        "properties": {
            "datetime": "2025-01-23T00:32:04.5Z",
            "platform": "SENTINEL-1A",
            "constellation": "Sentinel-1",
            # Planetary Computer's RTC items name the product they were derived
            # from, and publish no processing baseline.
            "sar:product_type": "GRD",
            "s1:processing_level": "1",
        },
        "assets": {
            pol: {
                "href": f"https://sentinel1euwestrtc.blob.core.windows.net/rtc/{pol}.tif",
                "type": COG,
                "roles": ["data"],
                "raster:bands": [dict(sar_band)],
            }
            for pol in ("vv", "vh")
        },
    }


def no_href_check(_key: str, _href: str) -> None:
    return None


def validate(
    item: dict,
    *,
    scene_id: str = SCENE_ID,
    collection: str = S2,
    modality: str = OPTICAL,
    assets: tuple[str, ...] = ("red", "nir"),
    aoi: BoundingBox = AOI,
    **kwargs: Any,
) -> ValidatedScene:
    return validate_scene_item(
        item, scene_id=scene_id, collection=collection, modality=modality,  # type: ignore[arg-type]
        assets=assets, aoi=aoi, check_href=no_href_check, **kwargs,
    )


def refused(item: dict, code: str, **kwargs: Any) -> SceneValidationError:
    with pytest.raises(SceneValidationError) as info:
        validate(item, **kwargs)
    assert info.value.code == code, info.value.message
    assert info.value.status_code == 422
    assert f"({code})" in info.value.message
    return info.value


# =========================================================================== #
# The scene
# =========================================================================== #


def test_a_live_shaped_sentinel2_scene_validates_with_its_catalog_metadata() -> None:
    scene = validate(s2_item())
    assert scene.scene_id == SCENE_ID
    assert scene.collection == S2
    assert scene.modality == OPTICAL
    assert (scene.platform, scene.constellation) == ("sentinel-2b", "sentinel-2")
    assert scene.acquired_at == "2025-01-04T05:15:39.024000Z"
    assert scene.geometry == polygon(80.0, 12.8, 80.6, 13.4)
    assert scene.validation_stage == "scene"
    assert [a.key for a in scene.assets] == ["red", "nir"]


def test_processing_metadata_is_recorded_not_applied() -> None:
    processing = validate(s2_item()).processing
    assert processing.processing_baseline == "05.11"
    assert processing.product_type == "S2MSI2A"
    assert processing.boa_offset_applied is True
    assert processing.software == "sentinel2-to-stac 0.1.1"


def test_item_answering_for_a_different_id_is_not_the_scene() -> None:
    refused(s2_item("S2A_OTHER_20250104_0_L2A"), "scene_not_found")


def test_item_in_another_collection_is_refused() -> None:
    item = s2_item()
    item["collection"] = "sentinel-2-l1c"
    refused(item, "incompatible_collection")


def test_collection_without_a_profile_is_refused() -> None:
    refused(s2_item(), "incompatible_collection", collection="landsat-c2-l2")


@pytest.mark.parametrize(
    ("constellation", "platform"),
    [("sentinel-1", "sentinel-1a"), (None, "landsat-9"), ("Sentinel-1", None)],
)
def test_wrong_sensor_is_refused(constellation: str | None, platform: str | None) -> None:
    item = s2_item()
    item["properties"]["constellation"] = constellation
    item["properties"]["platform"] = platform
    refused(item, "incompatible_sensor")


def test_absent_sensor_metadata_is_not_a_wrong_sensor() -> None:
    item = s2_item()
    del item["properties"]["constellation"]
    del item["properties"]["platform"]
    assert validate(item).constellation is None


@pytest.mark.parametrize("baseline", [None, "5.11", "baseline-05.11", 5.11, ""])
def test_unknown_or_unreadable_processing_baseline_is_refused(baseline: object) -> None:
    item = s2_item()
    if baseline is None:
        del item["properties"]["s2:processing_baseline"]
    else:
        item["properties"]["s2:processing_baseline"] = baseline
    refused(item, "unknown_processing_baseline")


def test_absent_boa_offset_flag_is_recorded_as_unknown() -> None:
    item = s2_item()
    del item["properties"]["earthsearch:boa_offset_applied"]
    assert validate(item).processing.boa_offset_applied is None


# =========================================================================== #
# AOI coverage
# =========================================================================== #


def test_footprint_containing_the_aoi_is_full_coverage() -> None:
    coverage = validate(s2_item()).aoi_coverage
    assert (coverage.basis, coverage.status) == ("geometry", "full")
    assert coverage.fraction == pytest.approx(1.0)


def test_footprint_missing_the_aoi_is_refused() -> None:
    item = s2_item()
    item["geometry"] = polygon(81.0, 13.5, 81.5, 14.0)
    refused(item, "scene_does_not_cover_aoi")


def test_footprint_touching_only_an_edge_does_not_cover() -> None:
    item = s2_item()
    item["geometry"] = polygon(80.30, 13.0, 80.6, 13.4)  # shares the AOI's east edge
    refused(item, "scene_does_not_cover_aoi")


def test_partial_coverage_is_accepted_and_reported() -> None:
    item = s2_item()
    item["geometry"] = polygon(80.275, 12.8, 80.6, 13.4)  # the eastern half of the AOI
    coverage = validate(item).aoi_coverage
    assert coverage.status == "partial"
    assert coverage.fraction == pytest.approx(0.5, abs=1e-9)


def test_a_configured_minimum_coverage_refuses_too_little() -> None:
    item = s2_item()
    item["geometry"] = polygon(80.275, 12.8, 80.6, 13.4)
    error = refused(item, "insufficient_aoi_coverage", min_coverage=0.9)
    assert "50.0%" in error.message
    assert validate(item, min_coverage=0.5).aoi_coverage.status == "partial"


def test_geometry_is_preferred_over_the_bbox() -> None:
    """A tilted swath's bbox covers the AOI; its real footprint does not."""

    item = s2_item()
    item["bbox"] = [79.0, 12.0, 81.0, 14.0]
    item["geometry"] = {
        "type": "Polygon",
        "coordinates": [[[80.4, 12.0], [81.0, 12.0], [80.6, 14.0], [80.4, 12.0]]],
    }
    refused(item, "scene_does_not_cover_aoi")


def test_bbox_is_the_fallback_basis_when_geometry_is_absent() -> None:
    item = s2_item()
    del item["geometry"]
    coverage = validate(item).aoi_coverage
    assert (coverage.basis, coverage.status) == ("bbox", "full")


def test_no_readable_footprint_is_unknown_not_uncovered() -> None:
    item = s2_item()
    del item["geometry"]
    del item["bbox"]
    refused(item, "scene_coverage_unknown")
    item["geometry"] = {"type": "Polygon", "coordinates": [[["a", "b"]]]}
    refused(item, "scene_coverage_unknown")


def test_coverage_fraction_handles_holes_and_multipolygons() -> None:
    # A hole over the western half of the AOI.
    with_hole = {
        "type": "Polygon",
        "coordinates": [
            [[80.0, 12.8], [80.6, 12.8], [80.6, 13.4], [80.0, 13.4], [80.0, 12.8]],
            [[80.2, 12.9], [80.275, 12.9], [80.275, 13.1], [80.2, 13.1], [80.2, 12.9]],
        ],
    }
    assert aoi_coverage_fraction(with_hole, AOI) == pytest.approx(0.5, abs=1e-9)
    # Two parts, each over a quarter of the AOI.
    parts = {
        "type": "MultiPolygon",
        "coordinates": [
            polygon(80.25, 13.00, 80.275, 13.025)["coordinates"],
            polygon(80.275, 13.025, 80.30, 13.05)["coordinates"],
        ],
    }
    assert aoi_coverage_fraction(parts, AOI) == pytest.approx(0.5, abs=1e-9)
    # A concave footprint.
    concave = {
        "type": "Polygon",
        "coordinates": [[[80.2, 12.9], [80.4, 12.9], [80.4, 13.2], [80.275, 13.0],
                         [80.2, 13.2], [80.2, 12.9]]],
    }
    fraction = aoi_coverage_fraction(concave, AOI)
    assert fraction is not None and 0.0 < fraction < 1.0


def test_unsupported_geometry_type_has_unknown_coverage() -> None:
    assert aoi_coverage_fraction({"type": "Point", "coordinates": [80.27, 13.02]}, AOI) is None


# =========================================================================== #
# Required assets
# =========================================================================== #


@pytest.mark.parametrize(
    ("assets", "keys"),
    [
        (("red", "nir"), ["red", "nir"]),  # NDVI
        (("green", "nir"), ["green", "nir"]),  # NDWI
        (("nir", "swir16"), ["nir", "swir16"]),  # NDBI
    ],
)
def test_the_bands_each_index_needs_are_validated(
    assets: tuple[str, ...], keys: list[str]
) -> None:
    assert [a.key for a in validate(s2_item(), assets=assets).assets] == keys


def test_asset_encoding_is_recorded_as_known() -> None:
    swir = validate(s2_item(), assets=("swir16",)).asset("swir16")
    assert swir is not None
    assert swir.metadata_status == "known"
    assert (swir.data_type, swir.nodata, swir.spatial_resolution) == ("uint16", 0.0, 20.0)
    assert (swir.scale, swir.offset) == (0.0001, -0.1)
    assert swir.roles == ("data", "reflectance")


def test_missing_required_band_is_refused() -> None:
    item = s2_item()
    del item["assets"]["nir"]
    error = refused(item, "required_asset_missing")
    assert "'nir'" in error.message


def test_band_without_href_is_missing() -> None:
    item = s2_item()
    item["assets"]["red"]["href"] = ""
    refused(item, "required_asset_missing")


@pytest.mark.parametrize("media_type", ["image/jp2", "application/xml", None])
def test_non_cog_media_type_is_unsupported(media_type: str | None) -> None:
    item = s2_item()
    item["assets"]["red"]["type"] = media_type
    refused(item, "unsupported_asset_type")


def test_asset_not_published_as_data_is_unsupported() -> None:
    refused(s2_item(), "unsupported_asset_type", assets=("visual",))


def test_href_rules_of_the_reader_apply_before_any_read() -> None:
    item = s2_item()
    item["assets"]["red"]["href"] = "s3://sentinel-cogs/B04.tif"
    imagery = ImageryService(stac_item_fetcher=lambda *_: item)
    with pytest.raises(SceneValidationError) as info:
        imagery.validate_scene(
            scene_id=SCENE_ID, collection=S2, modality=OPTICAL,
            assets=("red", "nir"), bbox=AOI,
        )
    assert info.value.code == "unsupported_asset_type"


def test_declared_data_type_other_than_the_engine_reads_is_unsupported() -> None:
    item = s2_item()
    item["assets"]["nir"]["raster:bands"][0]["data_type"] = "float32"
    error = refused(item, "unsupported_asset_encoding")
    assert "uint16" in error.message


def test_absent_raster_metadata_is_recorded_as_unknown_not_refused() -> None:
    item = s2_item()
    del item["assets"]["red"]["raster:bands"]
    red = validate(item).asset("red")
    assert red is not None
    assert red.metadata_status == "unknown"
    assert red.scale is None


def test_partial_availability_records_what_cannot_be_read() -> None:
    item = s2_item()
    del item["assets"]["swir16"]
    scene = validate(item, assets=("red", "nir", "swir16"), require_all_assets=False)
    assert [a.key for a in scene.assets] == ["red", "nir"]
    assert "swir16" in scene.unavailable_assets
    assert "required_asset_missing" in scene.unavailable_assets["swir16"]


def test_partial_availability_still_needs_one_readable_asset() -> None:
    item = s2_item()
    item["assets"] = {}
    refused(item, "required_asset_missing", require_all_assets=False)


# =========================================================================== #
# Sensor / collection compatibility
# =========================================================================== #


def test_optical_request_on_sentinel2_is_accepted() -> None:
    assert validate(s2_item()).modality == OPTICAL


def test_optical_request_on_sentinel1_is_refused() -> None:
    refused(rtc_item(), "incompatible_collection", scene_id=RTC_ID, collection=RTC_COLLECTION)


def test_sar_request_on_sentinel1_rtc_is_accepted() -> None:
    scene = validate(
        rtc_item(), scene_id=RTC_ID, collection=RTC_COLLECTION, modality=SAR,
        assets=("vv", "vh"),
    )
    assert scene.modality == SAR
    assert scene.processing.processing_baseline is None  # none is published
    assert scene.processing.product_type == "GRD"
    assert scene.processing.processing_level == "1"
    assert {a.data_type for a in scene.assets} == {"float32"}


def test_sar_request_on_sentinel2_is_refused() -> None:
    refused(s2_item(), "incompatible_collection", modality=SAR, assets=("vv", "vh"))


def test_rtc_asset_declaring_integer_pixels_is_unsupported() -> None:
    item = rtc_item()
    item["assets"]["vv"]["raster:bands"][0]["data_type"] = "uint16"
    refused(
        item, "unsupported_asset_encoding", scene_id=RTC_ID, collection=RTC_COLLECTION,
        modality=SAR, assets=("vv", "vh"),
    )


# =========================================================================== #
# Scene pairs
# =========================================================================== #


def pair_scenes(
    first: dict | None = None, second: dict | None = None
) -> tuple[ValidatedScene, ValidatedScene]:
    first = first or s2_item()
    second = second or s2_item(LATER_ID, when="2025-01-19T05:15:41Z")
    return (
        validate(first, scene_id=first["id"], assets=("green", "nir")),
        validate(second, scene_id=second["id"], assets=("green", "nir")),
    )


def test_two_valid_scenes_in_acquisition_order_are_compatible() -> None:
    pair = validate_scene_pair(*pair_scenes())
    assert pair.status == "compatible"
    assert (pair.earlier.scene_id, pair.later.scene_id) == (SCENE_ID, LATER_ID)
    assert pair.notes == ()


def test_the_same_scene_twice_is_not_a_pair() -> None:
    scene, _ = pair_scenes()
    with pytest.raises(SceneValidationError) as info:
        validate_scene_pair(scene, scene)
    assert info.value.code == "temporal_scene_incompatible"


@pytest.mark.parametrize("later_when", ["2025-01-01T00:00:00Z", "2025-01-04T05:15:39.024Z"])
def test_catalog_dates_out_of_order_or_equal_are_refused(later_when: str) -> None:
    first, second = pair_scenes(second=s2_item(LATER_ID, when=later_when))
    with pytest.raises(SceneValidationError) as info:
        validate_scene_pair(first, second)
    assert info.value.code == "temporal_scene_incompatible"


def test_naive_or_missing_catalog_time_is_refused() -> None:
    first, second = pair_scenes(second=s2_item(LATER_ID, when="2025-01-19T05:15:41"))
    with pytest.raises(SceneValidationError):
        validate_scene_pair(first, second)


def test_different_reflectance_offset_state_is_incompatible() -> None:
    second = s2_item(LATER_ID, when="2025-01-19T05:15:41Z")
    second["properties"]["earthsearch:boa_offset_applied"] = False
    with pytest.raises(SceneValidationError) as info:
        validate_scene_pair(*pair_scenes(second=second))
    assert "offset" in info.value.message


def test_different_baselines_and_coverage_are_noted_not_refused() -> None:
    second = s2_item(LATER_ID, when="2025-01-19T05:15:41Z")
    second["properties"]["s2:processing_baseline"] = "05.10"
    second["geometry"] = polygon(80.275, 12.8, 80.6, 13.4)
    pair = validate_scene_pair(*pair_scenes(second=second))
    assert len(pair.notes) == 2
    assert "baselines" in pair.notes[0]


# =========================================================================== #
# The live contract: AnalysisService validates before reading
# =========================================================================== #


class Catalog:
    """The STAC item fetch inside the REAL ImageryService, plus a band reader."""

    def __init__(self, items: dict[str, dict]) -> None:
        self.items = items
        self.events: list[str] = []

    def fetch(self, scene_id: str, collection: str) -> dict:
        self.events.append(f"fetch:{scene_id}")
        if scene_id not in self.items:
            raise NotFoundError(f"Scene {scene_id!r} was not found in the catalog.")
        return copy.deepcopy(self.items[scene_id])

    def read(self, href: str, *_: Any, **__: Any) -> Any:
        name = href.rsplit("/", 1)[-1]
        self.events.append(f"read:{name}")
        if name == "SCL.tif":
            return band([4, 4, 4], dtype="uint8")  # vegetation: every pixel usable
        return band([3, 4, 5])

    @property
    def reads(self) -> list[str]:
        return [e for e in self.events if e.startswith("read:")]

    def service(self) -> AnalysisService:
        return AnalysisService(
            imagery_service=ImageryService(stac_item_fetcher=self.fetch, band_reader=self.read)
        )


def client_scene(scene_id: str, **overrides: Any) -> Scene:
    """What a CLIENT says about a scene. Validation must not believe any of it."""

    payload: dict[str, Any] = {
        "id": scene_id,
        "datetime": "2025-01-04T05:15:39Z",
        "bbox": BoundingBox(west=80.0, south=12.8, east=80.6, north=13.4),
        "geometry": polygon(80.0, 12.8, 80.6, 13.4),
        "cloud_cover": 0.0,
        "collection": S2,
        "platform": "sentinel-2b",
        "processing_level": "L2A",
        "thumbnail_url": None,
        "assets": [],
    }
    payload.update(overrides)
    return Scene(**payload)


def window(
    scene: Scene, *, label: str = "single", start: str = "2025-01-01",
    end: str = "2025-01-31", modality: str = OPTICAL,
) -> ExecutedWindow:
    return ExecutedWindow(
        modality=modality,  # type: ignore[arg-type]
        label=label,
        time_range=TimeRange(
            start_date=date.fromisoformat(start), end_date=date.fromisoformat(end)
        ),
        scene_count=1,
        scenes=[scene],
        selected_scene_id=scene.id,
    )


def execution(windows: list[ExecutedWindow], *, temporal: bool = False) -> QueryExecutionResult:
    modality = windows[0].modality
    body: dict[str, Any] = {
        "location_query": "Chennai",
        "modalities": [modality],
        "task": "change_detection" if temporal else "visualize",
    }
    if temporal:
        body.update(
            temporal_mode="compare",
            time_windows={
                "baseline": {"start_date": "2025-01-01", "end_date": "2025-01-10"},
                "target": {"start_date": "2025-01-11", "end_date": "2025-01-31"},
            },
        )
    else:
        body.update(
            temporal_mode="single",
            time_windows=[{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
        )
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=SatQueryIntent.model_validate(body), bbox=AOI),
        executed_modalities=[modality],  # type: ignore[list-item]
        skipped_modalities=[],
        windows=windows,
        catalog="https://earth-search.aws.element84.com/v1",
    )


def analyze(service: AnalysisService, request: AnalysisRequest):
    return asyncio.run(service.analyze(request))


def test_a_valid_scene_is_validated_before_the_first_band_read() -> None:
    catalog = Catalog({SCENE_ID: s2_item()})
    result = analyze(
        catalog.service(),
        AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), indices=["ndvi"]),
    )
    assert "ndvi_mean" in {m.name for m in result.measurements}
    assert catalog.events[0] == f"fetch:{SCENE_ID}"
    assert catalog.reads[0] == "read:SCL.tif"  # pixel quality before any band
    assert sorted(catalog.reads[1:]) == ["read:B04.tif", "read:B08.tif"]


def test_scene_not_in_the_catalog_reads_nothing() -> None:
    catalog = Catalog({})
    result = analyze(
        catalog.service(),
        AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), include_ndwi=True),
    )
    assert catalog.reads == []
    (outcome,) = result.analysis_outcomes
    assert outcome.status == "unavailable"
    assert "scene_not_found" in (outcome.reason or "")


def test_client_scene_metadata_cannot_override_the_catalog() -> None:
    """The client's Scene says it covers the area, cloud-free; the catalog disagrees."""

    item = s2_item()
    item["geometry"] = polygon(81.0, 13.5, 81.5, 14.0)
    item["bbox"] = [81.0, 13.5, 81.5, 14.0]
    catalog = Catalog({SCENE_ID: item})
    liar = client_scene(SCENE_ID, cloud_cover=0.0, geometry=polygon(80.0, 12.8, 80.6, 13.4))
    result = analyze(
        catalog.service(),
        AnalysisRequest(execution=execution([window(liar)]), indices=["ndvi", "ndwi"]),
    )
    assert catalog.reads == []
    assert result.measurements == []
    assert any("scene_does_not_cover_aoi" in w for w in result.warnings)


def test_validated_scene_carries_catalog_values_not_client_values() -> None:
    catalog = Catalog({SCENE_ID: s2_item()})
    imagery = ImageryService(stac_item_fetcher=catalog.fetch)
    scene = imagery.validate_scene(
        scene_id=SCENE_ID, collection=S2, modality=OPTICAL, assets=("red",), bbox=AOI
    )
    assert scene.acquired_at == "2025-01-04T05:15:39.024000Z"
    # The client's cloud cover never reaches the validated scene.
    assert "cloud_cover" not in json.dumps(scene.model_dump(mode="json"))


def test_catalog_not_found_maps_to_scene_not_found() -> None:
    imagery = ImageryService(stac_item_fetcher=Catalog({}).fetch)
    with pytest.raises(SceneValidationError) as info:
        imagery.validate_scene(
            scene_id=SCENE_ID, collection=S2, modality=OPTICAL, assets=("red",), bbox=AOI
        )
    assert info.value.code == "scene_not_found"


def test_missing_swir_costs_ndbi_alone_without_reading_it() -> None:
    item = s2_item()
    del item["assets"]["swir16"]
    catalog = Catalog({SCENE_ID: item})
    result = analyze(
        catalog.service(),
        AnalysisRequest(
            execution=execution([window(client_scene(SCENE_ID))]), indices=["ndvi", "ndbi"]
        ),
    )
    statuses = {o.name: o.status for o in result.analysis_outcomes}
    assert statuses == {"ndvi": "completed", "ndbi": "unavailable"}
    assert "read:B11.tif" not in catalog.reads


def test_bands_with_different_declared_scales_are_not_indexed() -> None:
    item = s2_item()
    item["assets"]["red"]["raster:bands"][0]["scale"] = 0.001
    catalog = Catalog({SCENE_ID: item})
    result = analyze(
        catalog.service(),
        AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), indices=["ndvi"]),
    )
    assert result.measurements == []
    assert any("different scales" in w for w in result.warnings)


def test_partial_coverage_is_surfaced_as_a_warning() -> None:
    item = s2_item()
    item["geometry"] = polygon(80.275, 12.8, 80.6, 13.4)
    catalog = Catalog({SCENE_ID: item})
    result = analyze(
        catalog.service(),
        AnalysisRequest(execution=execution([window(client_scene(SCENE_ID))]), include_ndwi=True),
    )
    assert any("covers about 50.0%" in w for w in result.warnings)


def test_sar_scene_failing_validation_reads_no_polarization() -> None:
    item = rtc_item()
    item["properties"]["constellation"] = "sentinel-2"
    catalog = Catalog({RTC_ID: item})
    sar_scene = client_scene(RTC_ID, collection=RTC_COLLECTION, platform="sentinel-1a")
    result = analyze(
        catalog.service(),
        AnalysisRequest(
            execution=execution([window(sar_scene, modality=SAR)]),
            include_sar_backscatter=True,
        ),
    )
    assert catalog.reads == []
    assert result.sar_backscatter is None
    assert any("incompatible_sensor" in w for w in result.warnings)


def test_sar_scene_validates_through_the_real_service() -> None:
    imagery = ImageryService(stac_item_fetcher=Catalog({RTC_ID: rtc_item()}).fetch)
    scene = imagery.validate_scene(
        scene_id=RTC_ID, collection=RTC_COLLECTION, modality=SAR, assets=("vv", "vh"), bbox=AOI
    )
    assert [a.key for a in scene.assets] == ["vv", "vh"]


# -- temporal ---------------------------------------------------------------- #


def temporal_request(first_item: dict | None, second_item: dict | None) -> tuple[Catalog, Any]:
    items = {}
    if first_item is not None:
        items[SCENE_ID] = first_item
    if second_item is not None:
        items[LATER_ID] = second_item
    catalog = Catalog(items)
    windows = [
        window(client_scene(SCENE_ID, datetime="2025-01-04T05:15:39Z"),
               label="baseline", start="2025-01-01", end="2025-01-10"),
        window(client_scene(LATER_ID, datetime="2025-01-19T05:15:41Z"),
               label="target", start="2025-01-11", end="2025-01-31"),
    ]
    result = analyze(
        catalog.service(),
        AnalysisRequest(execution=execution(windows, temporal=True), include_temporal_ndwi=True),
    )
    return catalog, result


def later_item() -> dict:
    return s2_item(LATER_ID, when="2025-01-19T05:15:41Z")


def test_temporal_pair_with_two_valid_scenes_is_compared() -> None:
    catalog, result = temporal_request(s2_item(), later_item())
    assert result.temporal_comparison is not None
    assert len(catalog.reads) == 6  # scl + green + nir, per observation
    # Both scenes were validated before the first of the four reads.
    first_read = catalog.events.index(catalog.reads[0])
    assert {f"fetch:{SCENE_ID}", f"fetch:{LATER_ID}"} <= set(catalog.events[:first_read])


def test_temporal_first_scene_invalid_reads_nothing() -> None:
    first = s2_item()
    first["geometry"] = polygon(81.0, 13.5, 81.5, 14.0)
    catalog, result = temporal_request(first, later_item())
    assert result.temporal_comparison is None
    assert catalog.reads == []


def test_temporal_second_scene_invalid_reads_nothing() -> None:
    second = later_item()
    del second["properties"]["s2:processing_baseline"]
    catalog, result = temporal_request(s2_item(), second)
    assert result.temporal_comparison is None
    assert catalog.reads == []
    assert any("unknown_processing_baseline" in w for w in result.warnings)


def test_temporal_scene_missing_a_band_reads_nothing() -> None:
    second = later_item()
    del second["assets"]["green"]
    catalog, result = temporal_request(s2_item(), second)
    assert result.temporal_comparison is None
    assert catalog.reads == []


def test_temporal_scene_missing_from_catalog_reads_nothing() -> None:
    catalog, result = temporal_request(s2_item(), None)
    assert result.temporal_comparison is None
    assert catalog.reads == []


def test_temporal_incompatible_pair_reads_nothing() -> None:
    second = later_item()
    second["properties"]["earthsearch:boa_offset_applied"] = False
    catalog, result = temporal_request(s2_item(), second)
    assert result.temporal_comparison is None
    assert catalog.reads == []
    assert any("temporal_scene_incompatible" in w for w in result.warnings)


def test_temporal_catalog_order_contradicting_client_order_is_refused() -> None:
    """The client dated the scenes one way; the catalog dates them the other."""

    first = s2_item(when="2025-01-25T05:15:39Z")
    catalog, result = temporal_request(first, later_item())
    assert result.temporal_comparison is None
    assert catalog.reads == []


def test_temporal_pair_notes_travel_with_the_comparison() -> None:
    second = later_item()
    second["properties"]["s2:processing_baseline"] = "05.10"
    _, result = temporal_request(s2_item(), second)
    assert result.temporal_comparison is not None
    assert any("different baselines" in w for w in result.temporal_comparison.warnings)


# =========================================================================== #
# The agent: the area rule runs before discovery
# =========================================================================== #


class StacSearchRecorder:
    """The REAL SatelliteService, behind a transport that records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"type": "FeatureCollection", "features": []})

    @property
    def searches(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("/search")]


def agent(
    bbox: BoundingBox, *, analysis: Any | None = None
) -> tuple[AgentExecutor, StacSearchRecorder]:
    recorder = StacSearchRecorder()
    satellite = SatelliteService(transport=httpx.MockTransport(recorder.handler))
    query = QueryExecutionService(
        query_service=QueryService(geospatial_service=FakeGeospatialService(bbox=bbox)),  # type: ignore[arg-type]
        satellite_service=satellite,
    )
    executor = AgentExecutor(
        query_execution_service=query,
        analysis_service=analysis or AnalysisService(
            imagery_service=ImageryService(stac_item_fetcher=Catalog({}).fetch)
        ),
    )
    return executor, recorder


def plan(*analysis_steps: dict[str, Any], modalities: list[str] | None = None) -> AgentPlan:
    intent = {
        "location_query": "Tamil Nadu",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
        "modalities": modalities or [OPTICAL],
        "task": "visualize",
    }
    return AgentPlan.model_validate(
        {"steps": [{"tool": "execute_query", "intent": intent}, *analysis_steps]}
    )


@pytest.mark.parametrize(
    ("step", "modalities"),
    [
        ({"tool": "spectral_indices", "indices": ["ndvi"]}, [OPTICAL]),
        ({"tool": "ndwi_statistics"}, [OPTICAL]),
        ({"tool": "sar_backscatter_statistics"}, [SAR]),
    ],
)
def test_oversized_area_makes_zero_stac_searches(
    step: dict[str, Any], modalities: list[str]
) -> None:
    executor, recorder = agent(LARGE)
    outcome = asyncio.run(executor.execute(plan(step, modalities=modalities)))
    assert recorder.requests == []
    discovery = outcome.steps[0]
    assert discovery.status == "failed"
    assert "execution.plan.bbox" not in (discovery.error_message or "")
    assert "plan.bbox" in (discovery.error_message or "")
    assert all(s.status == "skipped" for s in outcome.steps[1:])


def test_the_precheck_is_the_same_rule_the_analysis_applies() -> None:
    service = AnalysisService()
    with pytest.raises(AnalysisRequestRejectedError) as info:
        service.precheck_plan(LARGE, indices=("ndvi",))
    assert info.value.code == "aoi_too_large"
    service.precheck_plan(AOI, indices=("ndvi",))  # a normal area passes
    service.precheck_plan(LARGE)  # nothing to measure, nothing to refuse


def test_valid_area_searches_exactly_once_per_window() -> None:
    executor, recorder = agent(AOI)
    outcome = asyncio.run(
        executor.execute(plan({"tool": "spectral_indices", "indices": ["ndvi"]}))
    )
    assert outcome.steps[0].status == "ok"
    assert len(recorder.searches) == 1


def test_valid_area_over_two_modalities_searches_once_each() -> None:
    executor, recorder = agent(AOI)
    asyncio.run(
        executor.execute(
            plan({"tool": "spectral_indices", "indices": ["ndvi"]}, modalities=[OPTICAL, SAR])
        )
    )
    assert len(recorder.searches) == 2


def test_discovery_without_analysis_is_not_prechecked() -> None:
    """A plan that measures nothing is not refused for an area it will not read."""

    executor, recorder = agent(LARGE)
    outcome = asyncio.run(executor.execute(plan()))
    assert outcome.steps[0].status == "ok"
    assert len(recorder.searches) == 1


def test_analysis_service_without_a_precheck_keeps_the_old_path() -> None:
    class NoPrecheck:
        async def analyze(self, request: AnalysisRequest) -> Any:
            raise NotFoundError("unused")

    executor, recorder = agent(LARGE, analysis=NoPrecheck())
    asyncio.run(executor.execute(plan({"tool": "ndwi_statistics"})))
    assert len(recorder.searches) == 1


def test_scene_min_coverage_setting_defaults_to_no_invented_threshold() -> None:
    assert get_settings().scene_min_aoi_coverage == 0.0
