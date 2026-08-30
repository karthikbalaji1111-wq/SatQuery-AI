"""Phase 14 - Temporal NDWI Statistics.

Two Sentinel-2 observations, each indexed **independently** at native 10 m
resolution, reported side by side with the Phase 13 compatibility report. The
only derived value is ``mean_ndwi_difference = second.ndwi_mean -
first.ndwi_mean``, a difference between two aggregate statistics.

Nothing here is a spatial comparison: no pixel is ever compared against
another pixel, no grid is aligned, no raster is resampled. The forbidden-
characterisation scan in section A is the machine-checked form of that promise.

No test contacts Gemini, Nominatim, STAC, or real imagery.
"""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import rasterio
from app.api.routes.query import get_analysis_service
from app.core.errors import (
    ImageryError,
    InvalidInputError,
    NotFoundError,
    UpstreamServiceError,
)
from app.main import create_app
from app.services.analysis import AnalysisRequest, AnalysisResult, AnalysisService
from app.services.analysis import service as service_mod
from app.services.analysis.engines import (
    HIGH_CLOUD_COVER_PERCENT,
    MEAN_NDWI_DIFFERENCE,
    compare_ndwi_observations,
)
from app.services.analysis.schemas import (
    Measurement,
    ObservationIndexResult,
    TemporalIndexComparison,
)
from app.services.geospatial.schemas import BoundingBox
from app.services.query.compatibility import compute_compatibility
from app.services.query.schemas import (
    ExecutedWindow,
    Observation,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    TimeRange,
)
from app.services.satellite import ImageryService
from app.services.satellite.raster import BandWindow
from app.services.satellite.schemas import Scene
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

ANALYZE_URL = "/api/v1/query/analyze"

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
CATALOG = "https://earth-search.aws.element84.com/v1"
S2 = "sentinel-2-optical"
S1 = "sentinel-1-sar"

#: Phrases that would mischaracterise the aggregate difference. A disclaimer is
#: written so that it never needs any of them, which keeps this scan strict.
FORBIDDEN_PHRASES = (
    "per-pixel",
    "per pixel",
    "change detection",
    "change mask",
    "land-cover change",
    "land cover change",
    "detected change",
    "spatial change",
    "changed pixels",
    "pixel-level",
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def band(
    values: list[float] | np.ndarray,
    *,
    dtype: str = "uint16",
    nodata: float | None = 0.0,
) -> BandWindow:
    array = np.asarray(values, dtype=dtype).reshape(1, -1)
    valid = (
        np.isfinite(array)
        if np.issubdtype(array.dtype, np.floating)
        else np.ones(array.shape, dtype=bool)
    )
    if nodata is not None:
        valid = valid & (array != nodata)
    return BandWindow(
        values=array,
        valid=valid,
        width=array.shape[1],
        height=1,
        crs="EPSG:32644",
        transform=from_origin(399960.0, 1500000.0, 10.0, 10.0),
        resolution=10.0,
        nodata=nodata,
        window={"col_off": 0, "row_off": 0, "width": array.shape[1], "height": 1},
        source_shape=[1, array.shape[1]],
    )


def named(measurements: list[Measurement]) -> dict[str, float]:
    return {m.name: m.value for m in measurements}


def make_scene(
    scene_id: str,
    *,
    datetime_: str | None = "2024-01-15T05:00:00Z",
    cloud_cover: float | None = 1.0,
    bbox: BoundingBox | None = DEFAULT_BBOX,
    collection: str | None = "sentinel-2-l2a",
) -> Scene:
    return Scene(
        id=scene_id,
        datetime=datetime_,
        bbox=bbox,
        geometry=None,
        cloud_cover=cloud_cover,
        collection=collection,
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
    )


def make_observation(
    *,
    scene_id: str = "scene-a",
    window_label: str = "baseline",
    modality: str = S2,
    datetime_: str | None = "2024-01-15T05:00:00Z",
    cloud_cover: float | None = 1.0,
    bbox: BoundingBox | None = DEFAULT_BBOX,
) -> Observation:
    return Observation(
        modality=modality,  # type: ignore[arg-type]
        window_label=window_label,
        requested_window=TimeRange.model_validate(
            {"start_date": "2024-01-01", "end_date": "2024-01-31"}
        ),
        scene=make_scene(
            scene_id, datetime_=datetime_, cloud_cover=cloud_cover, bbox=bbox
        ),
        imagery=None,
    )


def index_result(
    *,
    window_label: str = "baseline",
    scene_id: str = "scene-a",
    mean: float | None = 0.2,
    count: float = 100.0,
    cloud_cover: float | None = 1.0,
) -> ObservationIndexResult:
    measurements = [
        Measurement(name="ndwi_valid_pixel_count", value=count, unit="pixels")
    ]
    if mean is not None:
        measurements.append(Measurement(name="ndwi_mean", value=mean, unit="index"))
    return ObservationIndexResult(
        window_label=window_label,
        scene_id=scene_id,
        acquired_at=None,
        cloud_cover=cloud_cover,
        measurements=measurements,
    )


def compat(
    *,
    first: Observation | None = None,
    second: Observation | None = None,
):
    return compute_compatibility(
        first or make_observation(scene_id="a", window_label="baseline"),
        second or make_observation(scene_id="b", window_label="target"),
    )


def compare(
    *,
    first: ObservationIndexResult | None = None,
    second: ObservationIndexResult | None = None,
    compatibility: Any = None,
) -> tuple[list[Measurement], list[str]]:
    return compare_ndwi_observations(
        first=first or index_result(mean=0.2, scene_id="a"),
        second=second or index_result(mean=0.5, scene_id="b", window_label="target"),
        compatibility=compatibility if compatibility is not None else compat(),
    )


def make_window(
    *,
    modality: str = S2,
    label: str = "baseline",
    scene: Scene,
    start: str = "2024-01-01",
    end: str = "2024-01-31",
) -> ExecutedWindow:
    return ExecutedWindow(
        modality=modality,  # type: ignore[arg-type]
        label=label,
        time_range=TimeRange.model_validate({"start_date": start, "end_date": end}),
        scene_count=1,
        scenes=[scene],
        selected_scene_id=scene.id,
        imagery=None,
        imagery_error=None,
    )


def make_execution(
    *, windows: list[ExecutedWindow], task: str = "visualize"
) -> QueryExecutionResult:
    intent = SatQueryIntent.model_validate(
        {
            "location_query": "Chennai",
            "temporal_mode": "single",
            "time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
            "modalities": [S2],
            "task": task,
        }
    )
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=intent, bbox=DEFAULT_BBOX),
        executed_modalities=[S2],
        skipped_modalities=[],
        windows=windows,
        catalog=CATALOG,
    )


def two_window_execution() -> QueryExecutionResult:
    return make_execution(
        windows=[
            make_window(
                label="baseline",
                scene=make_scene("scene-a", datetime_="2024-01-05T05:00:00Z"),
            ),
            make_window(
                label="target",
                scene=make_scene("scene-b", datetime_="2024-03-05T05:00:00Z"),
            ),
        ]
    )


class FakeImageryService:
    """Records read_band calls; returns canned bands or raises."""

    def __init__(
        self,
        *,
        bands: dict[str, dict[str, BandWindow]] | None = None,
        error: Exception | None = None,
        error_on_scene: str | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.retrieve_calls = 0
        self._bands = bands
        self._error = error
        self._error_on_scene = error_on_scene

    def retrieve(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        self.retrieve_calls += 1
        raise AssertionError("the temporal path must never call retrieve()")

    def read_band(
        self,
        *,
        scene_id: str,
        bbox: Any,
        asset: str,
        collection: str | None = None,
    ) -> BandWindow:
        self.calls.append(
            {
                "scene_id": scene_id,
                "bbox": bbox,
                "asset": asset,
                "collection": collection,
            }
        )
        if self._error is not None and (
            self._error_on_scene is None or self._error_on_scene == scene_id
        ):
            raise self._error
        if self._bands is not None:
            return self._bands[scene_id][asset]
        # scene-a -> NDWI 0.5 ; scene-b -> NDWI 0.0
        if scene_id == "scene-a":
            return band([3]) if asset == "green" else band([1])
        return band([5]) if asset == "green" else band([5])


def analyze_temporal(
    execution: QueryExecutionResult,
    imagery: FakeImageryService | None = None,
    *,
    include_ndwi: bool = False,
) -> tuple[AnalysisResult, FakeImageryService]:
    imagery = imagery or FakeImageryService()
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                execution=execution,
                include_ndwi=include_ndwi,
                include_temporal_ndwi=True,
            )
        )
    )
    return result, imagery


def all_strings(result: AnalysisResult) -> list[str]:
    strings = [result.answer, *result.warnings]
    comparison = result.temporal_comparison
    if comparison is not None:
        strings.extend(comparison.warnings)
        strings.extend(comparison.compatibility.limitations)
        strings.extend(m.name for m in comparison.differences)
        for side in (comparison.first, comparison.second):
            strings.extend(m.name for m in side.measurements)
    return strings


# =========================================================================== #
# A. Pure engine - compare_ndwi_observations
# =========================================================================== #


def test_difference_is_second_minus_first() -> None:
    differences, _ = compare(
        first=index_result(mean=0.2, scene_id="a"),
        second=index_result(mean=0.5, scene_id="b"),
    )
    assert named(differences)[MEAN_NDWI_DIFFERENCE] == pytest.approx(0.3)


def test_difference_sign_is_negative_when_the_index_falls() -> None:
    differences, _ = compare(
        first=index_result(mean=0.5, scene_id="a"),
        second=index_result(mean=0.2, scene_id="b"),
    )
    assert named(differences)[MEAN_NDWI_DIFFERENCE] == pytest.approx(-0.3)


def test_difference_is_zero_for_identical_means() -> None:
    differences, _ = compare(
        first=index_result(mean=0.4, scene_id="a"),
        second=index_result(mean=0.4, scene_id="b"),
    )
    assert named(differences)[MEAN_NDWI_DIFFERENCE] == pytest.approx(0.0)


def test_difference_is_named_and_carries_the_index_unit() -> None:
    differences, _ = compare()
    assert len(differences) == 1
    assert differences[0].name == MEAN_NDWI_DIFFERENCE
    assert differences[0].unit == "index"


def test_no_difference_when_the_first_observation_has_no_mean() -> None:
    differences, warnings = compare(
        first=index_result(mean=None, count=0.0, scene_id="a"),
        second=index_result(mean=0.5, scene_id="b"),
    )
    assert differences == []
    assert any("valid pixels" in w for w in warnings)


def test_no_difference_when_the_second_observation_has_no_mean() -> None:
    differences, warnings = compare(
        first=index_result(mean=0.5, scene_id="a"),
        second=index_result(mean=None, count=0.0, scene_id="b"),
    )
    assert differences == []
    assert any("valid pixels" in w for w in warnings)


def test_difference_suppressed_when_footprints_do_not_overlap() -> None:
    far = make_observation(
        scene_id="b",
        window_label="target",
        bbox=BoundingBox(west=90.0, south=20.0, east=90.5, north=20.5),
    )
    differences, warnings = compare(
        compatibility=compat(second=far),
    )
    assert differences == []
    assert any("do not overlap" in w for w in warnings)


def test_difference_emitted_with_a_warning_for_partial_overlap() -> None:
    shifted = make_observation(
        scene_id="b",
        window_label="target",
        bbox=BoundingBox(west=80.20, south=12.95, east=80.40, north=13.30),
    )
    differences, warnings = compare(compatibility=compat(second=shifted))

    assert len(differences) == 1
    assert any("partially" in w for w in warnings)


def test_difference_suppressed_when_both_observations_share_a_scene() -> None:
    differences, warnings = compare(
        first=index_result(mean=0.2, scene_id="same"),
        second=index_result(mean=0.5, scene_id="same"),
    )
    assert differences == []
    assert any("same scene" in w for w in warnings)


def test_every_suppression_explains_itself() -> None:
    for kwargs in (
        {"first": index_result(mean=None, count=0.0, scene_id="a")},
        {
            "first": index_result(mean=0.2, scene_id="s"),
            "second": index_result(mean=0.5, scene_id="s"),
        },
    ):
        differences, warnings = compare(**kwargs)  # type: ignore[arg-type]
        assert differences == []
        assert warnings


def test_warnings_always_frame_the_value_as_an_aggregate_difference() -> None:
    _, warnings = compare()
    assert any("aggregate statistics" in w for w in warnings)


def test_engine_output_never_mischaracterises_the_difference() -> None:
    differences, warnings = compare()
    for text in [*warnings, *(m.name for m in differences)]:
        lowered = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in lowered, f"{phrase!r} in {text!r}"


def test_high_cloud_cover_warns_on_the_affected_observation() -> None:
    _, warnings = compare(
        first=index_result(
            mean=0.2, scene_id="a", cloud_cover=HIGH_CLOUD_COVER_PERCENT + 10
        )
    )
    assert any("cloud" in w and "baseline" in w for w in warnings)


def test_unknown_cloud_cover_is_reported_as_unknown_not_clear() -> None:
    _, warnings = compare(first=index_result(mean=0.2, scene_id="a", cloud_cover=None))
    assert any("cloud" in w and "unknown" in w for w in warnings)


def test_low_cloud_cover_produces_no_cloud_warning() -> None:
    _, warnings = compare(
        first=index_result(mean=0.2, scene_id="a", cloud_cover=1.0),
        second=index_result(mean=0.5, scene_id="b", cloud_cover=2.0),
    )
    assert not any("cloud cover" in w for w in warnings)


def test_single_valid_pixel_warns_that_statistics_are_not_meaningful() -> None:
    _, warnings = compare(first=index_result(mean=0.2, count=1.0, scene_id="a"))
    assert any("not meaningful" in w for w in warnings)


def test_engine_does_not_mutate_its_inputs() -> None:
    first = index_result(mean=0.2, scene_id="a")
    second = index_result(mean=0.5, scene_id="b")
    before = (first.model_dump(), second.model_dump())

    compare(first=first, second=second)

    assert (first.model_dump(), second.model_dump()) == before


def test_engine_is_deterministic() -> None:
    a = compare()
    b = compare()
    assert [m.model_dump() for m in a[0]] == [m.model_dump() for m in b[0]]
    assert a[1] == b[1]


# =========================================================================== #
# B. AnalysisService dispatch
# =========================================================================== #


def test_flag_off_performs_no_band_reads_and_yields_no_comparison() -> None:
    imagery = FakeImageryService()
    service = AnalysisService(imagery_service=imagery)  # type: ignore[arg-type]
    result = asyncio.run(
        service.analyze(AnalysisRequest(execution=two_window_execution()))
    )

    assert imagery.calls == []
    assert result.temporal_comparison is None
    assert result.status == "ok"


def test_two_optical_observations_produce_one_comparison() -> None:
    result, _ = analyze_temporal(two_window_execution())
    comparison = result.temporal_comparison

    assert comparison is not None
    assert (comparison.first.scene_id, comparison.second.scene_id) == (
        "scene-a",
        "scene-b",
    )
    # scene-a NDWI 0.5, scene-b NDWI 0.0 -> difference -0.5
    assert named(comparison.differences)[MEAN_NDWI_DIFFERENCE] == pytest.approx(-0.5)


def test_exactly_four_band_reads_two_per_observation() -> None:
    _, imagery = analyze_temporal(two_window_execution())

    assert len(imagery.calls) == 4
    assert [c["asset"] for c in imagery.calls] == ["green", "nir", "green", "nir"]
    assert [c["scene_id"] for c in imagery.calls] == [
        "scene-a",
        "scene-a",
        "scene-b",
        "scene-b",
    ]


def test_reads_use_the_scene_collection_and_the_plan_bbox() -> None:
    _, imagery = analyze_temporal(two_window_execution())

    assert {c["collection"] for c in imagery.calls} == {"sentinel-2-l2a"}
    assert all(c["bbox"] == DEFAULT_BBOX for c in imagery.calls)


def test_the_temporal_path_never_calls_retrieve() -> None:
    _, imagery = analyze_temporal(two_window_execution())
    assert imagery.retrieve_calls == 0


def test_observations_carry_acquisition_time_and_cloud_cover() -> None:
    result, _ = analyze_temporal(two_window_execution())
    comparison = result.temporal_comparison

    assert comparison is not None
    assert comparison.first.acquired_at is not None
    assert comparison.first.cloud_cover == 1.0
    assert comparison.first.window_label == "baseline"
    assert comparison.second.window_label == "target"


def test_compatibility_report_is_attached() -> None:
    result, _ = analyze_temporal(two_window_execution())
    comparison = result.temporal_comparison

    assert comparison is not None
    assert comparison.compatibility.same_modality is True
    assert comparison.compatibility.co_registration_status == "not_evaluated"
    assert comparison.compatibility.limitations


def test_zero_optical_observations_warns_and_reads_nothing() -> None:
    result, imagery = analyze_temporal(make_execution(windows=[]))

    assert result.temporal_comparison is None
    assert imagery.calls == []
    assert any("Temporal NDWI" in w for w in result.warnings)


def test_one_optical_observation_warns_and_reads_nothing() -> None:
    execution = make_execution(
        windows=[make_window(label="single", scene=make_scene("scene-a"))]
    )
    result, imagery = analyze_temporal(execution)

    assert result.temporal_comparison is None
    assert imagery.calls == []
    assert any("Temporal NDWI" in w for w in result.warnings)


def test_three_observations_analyse_the_first_pair_and_warn_about_the_rest() -> None:
    execution = make_execution(
        windows=[
            make_window(
                label="series[0]",
                scene=make_scene("scene-a", datetime_="2024-01-05T05:00:00Z"),
            ),
            make_window(
                label="series[1]",
                scene=make_scene("scene-b", datetime_="2024-02-05T05:00:00Z"),
            ),
            make_window(
                label="series[2]",
                scene=make_scene("scene-c", datetime_="2024-03-05T05:00:00Z"),
            ),
        ]
    )
    result, imagery = analyze_temporal(execution)
    comparison = result.temporal_comparison

    assert comparison is not None
    assert (comparison.first.scene_id, comparison.second.scene_id) == (
        "scene-a",
        "scene-b",
    )
    assert len(imagery.calls) == 4  # only the first pair is read
    assert any("not analysed" in w for w in result.warnings)


def test_sar_observations_are_excluded_from_pairing() -> None:
    execution = make_execution(
        windows=[
            make_window(label="baseline", scene=make_scene("scene-a")),
            make_window(
                label="target",
                scene=make_scene("scene-b", datetime_="2024-03-05T05:00:00Z"),
            ),
            make_window(
                modality=S1,
                label="baseline",
                scene=make_scene("sar-a", collection="sentinel-1-grd"),
            ),
        ]
    )
    result, imagery = analyze_temporal(execution)
    comparison = result.temporal_comparison

    assert comparison is not None
    assert {c["scene_id"] for c in imagery.calls} == {"scene-a", "scene-b"}
    assert comparison.compatibility.same_modality is True


def test_sar_only_execution_warns_and_reads_nothing() -> None:
    execution = make_execution(
        windows=[
            make_window(
                modality=S1,
                label="baseline",
                scene=make_scene("sar-a", collection="sentinel-1-grd"),
            ),
            make_window(
                modality=S1,
                label="target",
                scene=make_scene("sar-b", collection="sentinel-1-grd"),
            ),
        ]
    )
    result, imagery = analyze_temporal(execution)

    assert result.temporal_comparison is None
    assert imagery.calls == []


@pytest.mark.parametrize(
    "error",
    [
        NotFoundError("Asset 'green' is not available on this scene."),
        InvalidInputError("The requested bbox does not intersect the selected scene."),
        UpstreamServiceError("Could not open the remote raster."),
        ImageryError("Failed to read the requested raster window."),
    ],
)
def test_band_read_failure_warns_instead_of_raising(error: Exception) -> None:
    result, _ = analyze_temporal(
        two_window_execution(), FakeImageryService(error=error)
    )

    assert result.temporal_comparison is None
    assert any(str(error) in w for w in result.warnings)
    assert result.status == "ok"  # the analysis itself still succeeded


def test_oversized_window_rejection_surfaces_the_actionable_message() -> None:
    error = InvalidInputError(
        "The requested window exceeds the maximum quantitative read dimension "
        "(4096 px > 2048 px). Quantitative reads are not decimated; use a "
        "smaller bbox."
    )
    result, _ = analyze_temporal(
        two_window_execution(), FakeImageryService(error=error)
    )

    assert result.temporal_comparison is None
    assert any("smaller bbox" in w for w in result.warnings)


def test_failure_on_the_second_observation_produces_no_comparison() -> None:
    imagery = FakeImageryService(
        error=NotFoundError("Asset 'nir' is not available on this scene."),
        error_on_scene="scene-b",
    )
    result, _ = analyze_temporal(two_window_execution(), imagery)

    assert result.temporal_comparison is None
    assert any("nir" in w for w in result.warnings)


def test_answer_mentions_temporal_ndwi_statistics() -> None:
    result, _ = analyze_temporal(two_window_execution())
    assert "Temporal NDWI statistics" in result.answer


def test_answer_is_untouched_when_no_comparison_was_produced() -> None:
    baseline = asyncio.run(
        AnalysisService(imagery_service=FakeImageryService()).analyze(  # type: ignore[arg-type]
            AnalysisRequest(execution=two_window_execution())
        )
    )
    result, _ = analyze_temporal(
        two_window_execution(), FakeImageryService(error=NotFoundError("nope"))
    )
    assert result.answer == baseline.answer


def test_service_output_never_mischaracterises_the_difference() -> None:
    result, _ = analyze_temporal(two_window_execution())
    for text in all_strings(result):
        lowered = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in lowered, f"{phrase!r} in {text!r}"


def test_the_two_ndwi_flags_are_independent() -> None:
    result, imagery = analyze_temporal(two_window_execution(), include_ndwi=True)

    assert result.measurements  # single-scene NDWI still produced
    assert result.temporal_comparison is not None
    assert len(imagery.calls) == 6  # 2 single-scene + 4 temporal


def test_unimplemented_tasks_still_receive_a_comparison() -> None:
    execution = make_execution(
        windows=[
            make_window(label="baseline", scene=make_scene("scene-a")),
            make_window(
                label="target",
                scene=make_scene("scene-b", datetime_="2024-03-05T05:00:00Z"),
            ),
        ],
        task="change_detection",
    )
    result, _ = analyze_temporal(execution)

    assert result.status == "not_implemented"
    assert result.temporal_comparison is not None


def test_service_performs_no_pixel_arithmetic() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()

    assert "import numpy" not in source
    assert "np." not in source
    assert "compare_ndwi_observations" in source


def test_service_does_not_modify_the_phase_13_compatibility_module() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()

    # Read-only collaborator: imported and called, never reached into.
    assert "compute_compatibility" in source
    assert "pair_observations" in source
    assert "compatibility." not in source.replace("query.compatibility", "")


# =========================================================================== #
# C. Integration - raw values survive to the engine
# =========================================================================== #

_ORIGIN_X, _ORIGIN_Y, _RES = 399960.0, 1500000.0, 10.0


@contextmanager
def synthetic_band(fill: int) -> Iterator[MemoryFile]:
    """A single-band uint16 GeoTIFF whose every pixel is ``fill``."""

    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=200,
            height=200,
            count=1,
            dtype="uint16",
            crs="EPSG:32644",
            transform=from_origin(_ORIGIN_X, _ORIGIN_Y, _RES, _RES),
            nodata=0,
        ) as dataset:
            dataset.write(np.full((200, 200), fill, dtype=np.uint16), 1)
        yield mem


def bbox_for_window(window: Window) -> BoundingBox:
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, _RES, _RES)
    left, bottom, right, top = window_bounds(window, transform)
    west, south, east, north = transform_bounds(
        "EPSG:32644", "EPSG:4326", left, bottom, right, top
    )
    return BoundingBox(west=west, south=south, east=east, north=north)


def test_raw_uint16_values_reach_the_engine_at_native_resolution(
    monkeypatch: Any,
) -> None:
    """Green 3000 / NIR 1000 must give the raw-DN NDWI of 0.5, not a display value."""

    import app.services.satellite.raster as raster_mod

    item = {
        "type": "Feature",
        "id": "scene-a",
        "collection": "sentinel-2-l2a",
        "bbox": [80.0, 12.5, 81.1, 13.6],
        "properties": {"datetime": "2024-02-14T05:15:10Z"},
        "assets": {
            "green": {
                "href": "https://example.test/B03.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            },
            "nir": {
                "href": "https://example.test/B08.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            },
        },
    }
    service = ImageryService(stac_item_fetcher=lambda _s, _c: item)
    aoi = bbox_for_window(Window(col_off=10, row_off=10, width=40, height=40))

    with synthetic_band(3000) as green_mem, synthetic_band(1000) as nir_mem:
        def open_raster(href: str) -> Any:
            return (green_mem if href.endswith("B03.tif") else nir_mem).open()

        monkeypatch.setattr(raster_mod, "_open_raster", open_raster)

        green = service.read_band(
            scene_id="scene-a", bbox=aoi, asset="green", collection="sentinel-2-l2a"
        )
        nir = service.read_band(
            scene_id="scene-a", bbox=aoi, asset="nir", collection="sentinel-2-l2a"
        )

    assert green.values.dtype == np.uint16
    assert green.resolution == 10.0
    assert int(green.values.max()) == 3000  # never normalised to 0..255

    from app.services.analysis.engines import compute_ndwi_measurements

    values = named(compute_ndwi_measurements(green, nir))
    assert values["ndwi_mean"] == pytest.approx((3000 - 1000) / (3000 + 1000))


# =========================================================================== #
# D. API - /query/analyze
# =========================================================================== #


def make_client(imagery: FakeImageryService | None = None) -> TestClient:
    app = create_app()
    service = AnalysisService(imagery_service=imagery or FakeImageryService())  # type: ignore[arg-type]
    app.dependency_overrides[get_analysis_service] = lambda: service
    return TestClient(app)


def analyze_body(execution: QueryExecutionResult, **extra: Any) -> dict[str, Any]:
    return {"execution": execution.model_dump(mode="json"), **extra}


def test_flag_off_keeps_every_existing_field_and_adds_a_null_comparison() -> None:
    response = make_client().post(
        ANALYZE_URL, json=analyze_body(two_window_execution())
    )
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {
        "status",
        "task",
        "answer",
        "windows_considered",
        "warnings",
        "measurements",
        "temporal_comparison",
    }
    # The only intentional serialized difference.
    assert body["temporal_comparison"] is None
    assert body["status"] == "ok"
    assert body["task"] == "visualize"
    assert body["measurements"] == []
    assert len(body["windows_considered"]) == 2


def test_flag_on_returns_a_populated_comparison() -> None:
    response = make_client().post(
        ANALYZE_URL,
        json=analyze_body(two_window_execution(), include_temporal_ndwi=True),
    )
    body = response.json()

    assert response.status_code == 200
    comparison = body["temporal_comparison"]
    assert comparison is not None
    assert comparison["first"]["scene_id"] == "scene-a"
    assert comparison["second"]["scene_id"] == "scene-b"
    assert comparison["differences"][0]["name"] == MEAN_NDWI_DIFFERENCE
    assert comparison["compatibility"]["co_registration_status"] == "not_evaluated"
    assert comparison["warnings"]


def test_invalid_body_still_returns_422() -> None:
    response = make_client().post(ANALYZE_URL, json={"execution": {"plan": {}}})
    assert response.status_code == 422


def test_flag_is_optional_in_the_request_contract() -> None:
    request = AnalysisRequest.model_validate(
        {"execution": two_window_execution().model_dump(mode="json")}
    )
    assert request.include_temporal_ndwi is False


def test_comparison_model_is_optional_on_the_result() -> None:
    assert TemporalIndexComparison is not None
    result = AnalysisResult(
        status="ok",
        task="visualize",
        answer="x",
        windows_considered=[],
    )
    assert result.temporal_comparison is None


def test_rasterio_is_not_imported_by_the_analysis_service() -> None:
    source = pathlib.Path(service_mod.__file__).read_text()
    assert "rasterio" not in source
    assert rasterio is not None  # the import above is for the integration test only


# =========================================================================== #
# Phase 14.1 - AOI coverage evidence
#
# Scene footprint overlap is NOT AOI coverage. Two observations can have
# identical footprints (bbox_overlap == "full") and still have analysed very
# different numbers of AOI pixels, because the quantitative read clamps to the
# source and masks nodata independently for each scene. The evidence needed to
# say so already exists in the BandWindow; these tests pin that it reaches the
# report instead of being discarded.
#
# No threshold is invented and no new suppression rule is introduced: the
# correction is to state the coverage, not to judge it.
# =========================================================================== #


def index_result_with_evidence(
    *,
    window_label: str = "baseline",
    scene_id: str = "scene-a",
    mean: float | None = 0.2,
    valid: float = 4200.0,
    window_pixels: int | None = 4489,
    crs: str | None = "EPSG:32644",
    resolution: float | None = 10.0,
) -> ObservationIndexResult:
    measurements = [
        Measurement(name="ndwi_valid_pixel_count", value=valid, unit="pixels")
    ]
    if mean is not None:
        measurements.append(Measurement(name="ndwi_mean", value=mean, unit="index"))
    return ObservationIndexResult(
        window_label=window_label,
        scene_id=scene_id,
        acquired_at=None,
        cloud_cover=1.0,
        measurements=measurements,
        crs=crs,
        resolution=resolution,
        window_pixel_count=window_pixels,
    )


def test_coverage_is_reported_even_when_footprints_fully_overlap() -> None:
    """The adversarial case: identical footprints, wildly different coverage."""

    differences, warnings = compare_ndwi_observations(
        first=index_result_with_evidence(scene_id="a", valid=4200.0),
        second=index_result_with_evidence(
            window_label="target", scene_id="b", mean=0.5, valid=210.0
        ),
        compatibility=compat(),
    )

    assert compat().bbox_overlap == "full"  # footprints say "fully overlapping"
    # ...yet the analysed samples differ by an order of magnitude, and the
    # report must say so with the actual numbers.
    coverage = [w for w in warnings if "coverage" in w.lower()]
    assert coverage, "expected an explicit AOI-coverage statement"
    assert "4200" in coverage[0] and "210" in coverage[0]
    assert "4489" in coverage[0]
    # No invented threshold: the difference is still reported, not suppressed.
    assert len(differences) == 1


def test_coverage_statement_names_both_window_labels() -> None:
    _, warnings = compare_ndwi_observations(
        first=index_result_with_evidence(window_label="baseline", scene_id="a"),
        second=index_result_with_evidence(
            window_label="target", scene_id="b", mean=0.5
        ),
        compatibility=compat(),
    )
    coverage = [w for w in warnings if "coverage" in w.lower()][0]
    assert "baseline" in coverage and "target" in coverage


def test_actual_read_crs_and_resolution_are_reported_when_known() -> None:
    _, warnings = compare_ndwi_observations(
        first=index_result_with_evidence(scene_id="a", crs="EPSG:32644"),
        second=index_result_with_evidence(
            window_label="target", scene_id="b", mean=0.5, crs="EPSG:32643"
        ),
        compatibility=compat(),
    )
    # The metadata-only compatibility report cannot know this; the reads can.
    assert compat().crs_match == "unknown"
    grid = [w for w in warnings if "EPSG:32644" in w and "EPSG:32643" in w]
    assert grid, "expected the actual read CRSs to be reported"


def test_coverage_statement_is_absent_when_evidence_is_missing() -> None:
    _, warnings = compare(
        first=index_result(mean=0.2, scene_id="a"),
        second=index_result(mean=0.5, scene_id="b"),
    )
    # Nothing is fabricated when the BandWindow evidence was not carried.
    assert not [w for w in warnings if "AOI pixels" in w]


def test_service_carries_band_window_evidence_into_the_result() -> None:
    result, _ = analyze_temporal(two_window_execution())
    comparison = result.temporal_comparison

    assert comparison is not None
    for side in (comparison.first, comparison.second):
        assert side.crs == "EPSG:32644"
        assert side.resolution == 10.0
        assert side.window_pixel_count == 1  # the 1x1 synthetic band fixture
