"""Observation compatibility tests (Phase 13).

The compatibility layer is metadata-only and pure: it opens no raster, imports
no rasterio, performs no reprojection or resampling, and computes nothing at
the pixel level. Every fixture here is built in memory; no test contacts
Nominatim, the STAC API, Gemini, or any imagery.

The central invariant under test is *anti-inference*: matching CRS, matching
resolution, matching footprint and matching modality must never, in any
combination, be reported as co-registration.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest
from app.services.geospatial.schemas import BoundingBox
from app.services.query import compatibility as compatibility_mod
from app.services.query.compatibility import (
    CompatibilityReport,
    ObservationPair,
    PairingFailure,
    compute_compatibility,
    pair_observations,
)
from app.services.query.schemas import (
    Observation,
    ObservationSet,
    TimeRange,
)
from app.services.satellite.schemas import ImageryResponse, Scene, WindowInfo
from pydantic import ValidationError

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
WINDOW = TimeRange.model_validate(
    {"start_date": "2024-01-01", "end_date": "2024-01-31"}
)

S2 = "sentinel-2-optical"
S1 = "sentinel-1-sar"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_scene(
    *,
    scene_id: str = "scene-a",
    datetime_: str | None = "2024-01-05T05:15:21Z",
    bbox: BoundingBox | None = DEFAULT_BBOX,
    collection: str | None = "sentinel-2-l2a",
    processing_level: str | None = "L2A",
) -> Scene:
    return Scene(
        id=scene_id,
        datetime=datetime_,
        bbox=bbox,
        geometry=None,
        cloud_cover=None,
        collection=collection,
        platform=None,
        processing_level=processing_level,
        thumbnail_url=None,
        assets=[],
    )


def make_imagery(
    *,
    scene_id: str = "scene-a",
    crs: str | None = "EPSG:32644",
    resolution: float | None = 10.0,
    asset: str = "visual",
) -> ImageryResponse:
    return ImageryResponse(
        scene_id=scene_id,
        bbox=DEFAULT_BBOX,
        asset=asset,
        asset_href="https://example.test/TCI.tif",
        width=4,
        height=4,
        format="png",
        media_type="image/png",
        bands=["red", "green", "blue"],
        crs=crs,
        resolution=resolution,
        normalization="none (source is 8-bit RGB)",
        window=WindowInfo(col_off=0, row_off=0, width=4, height=4),
        source_shape=[10, 10],
        image_base64="AAAA",
    )


def make_observation(
    *,
    modality: str = S2,
    window_label: str = "single",
    scene_id: str = "scene-a",
    datetime_: str | None = "2024-01-05T05:15:21Z",
    bbox: BoundingBox | None = DEFAULT_BBOX,
    collection: str | None = "sentinel-2-l2a",
    processing_level: str | None = "L2A",
    imagery: ImageryResponse | None = None,
) -> Observation:
    return Observation(
        modality=modality,  # type: ignore[arg-type]
        window_label=window_label,
        requested_window=WINDOW,
        scene=make_scene(
            scene_id=scene_id,
            datetime_=datetime_,
            bbox=bbox,
            collection=collection,
            processing_level=processing_level,
        ),
        imagery=imagery,
    )


def make_set(*observations: Observation) -> ObservationSet:
    return ObservationSet(
        requested_bbox=DEFAULT_BBOX, observations=list(observations)
    )


# --------------------------------------------------------------------------- #
# A. Schema / model contract
# --------------------------------------------------------------------------- #


def test_match_status_contract_is_exactly_three_values() -> None:
    assert set(compatibility_mod.MatchStatus.__args__) == {
        "same",
        "different",
        "unknown",
    }


def test_bbox_overlap_contract_is_exactly_four_values() -> None:
    assert set(compatibility_mod.BboxOverlapStatus.__args__) == {
        "none",
        "partial",
        "full",
        "unknown",
    }


def test_co_registration_status_contract_is_exactly_two_values() -> None:
    assert set(compatibility_mod.CoRegistrationStatus.__args__) == {
        "not_evaluated",
        "not_supported_cross_modal",
    }


def test_report_exposes_the_agreed_fields() -> None:
    report = compute_compatibility(make_observation(), make_observation())

    assert set(CompatibilityReport.model_fields) == {
        "same_modality",
        "temporal_separation_days",
        "bbox_overlap",
        "crs_match",
        "resolution_match",
        "processing_level_match",
        "limitations",
        "co_registration_status",
    }
    assert isinstance(report, CompatibilityReport)


def test_report_carries_no_intersection_geometry() -> None:
    # Phase 13 reports a coarse overlap status only - no geometry is emitted.
    assert "intersection" not in CompatibilityReport.model_fields
    assert "bbox_intersection" not in CompatibilityReport.model_fields


@pytest.mark.parametrize("field", ["crs_match", "resolution_match", "processing_level_match"])
def test_match_fields_reject_values_outside_the_contract(field: str) -> None:
    base: dict[str, Any] = {
        "same_modality": True,
        "temporal_separation_days": None,
        "bbox_overlap": "unknown",
        "crs_match": "unknown",
        "resolution_match": "unknown",
        "processing_level_match": "unknown",
        "limitations": [],
        "co_registration_status": "not_evaluated",
    }
    base[field] = "maybe"
    with pytest.raises(ValidationError):
        CompatibilityReport.model_validate(base)


def test_pairing_failure_fields() -> None:
    failure = PairingFailure(modality=S2, reason="only one observation")  # type: ignore[arg-type]
    assert failure.modality == S2
    assert failure.reason

    unattributed = PairingFailure(modality=None, reason="nothing to pair")
    assert unattributed.modality is None


def test_observation_pair_holds_two_observations() -> None:
    first = make_observation(scene_id="a")
    second = make_observation(scene_id="b")
    pair = ObservationPair(first=first, second=second)

    assert pair.first.scene_id == "a"
    assert pair.second.scene_id == "b"


# --------------------------------------------------------------------------- #
# B. Anti-inference - the core invariant of this phase
# --------------------------------------------------------------------------- #


def test_matching_everything_still_never_implies_co_registration() -> None:
    """Same modality, same CRS, same resolution, identical footprint."""

    imagery = make_imagery(crs="EPSG:32644", resolution=10.0)
    report = compute_compatibility(
        make_observation(scene_id="a", imagery=imagery),
        make_observation(scene_id="b", imagery=imagery),
    )

    assert report.crs_match == "same"
    assert report.resolution_match == "same"
    assert report.processing_level_match == "same"
    assert report.bbox_overlap == "full"
    assert report.same_modality is True
    # None of the above upgrades this.
    assert report.co_registration_status == "not_evaluated"


def test_same_modality_reports_not_evaluated() -> None:
    report = compute_compatibility(
        make_observation(modality=S2), make_observation(modality=S2)
    )
    assert report.same_modality is True
    assert report.co_registration_status == "not_evaluated"


def test_cross_modal_reports_not_supported() -> None:
    report = compute_compatibility(
        make_observation(modality=S2, collection="sentinel-2-l2a"),
        make_observation(modality=S1, collection="sentinel-1-grd"),
    )
    assert report.same_modality is False
    assert report.co_registration_status == "not_supported_cross_modal"


def test_cross_modal_is_symmetric_in_both_argument_orders() -> None:
    s2 = make_observation(modality=S2, collection="sentinel-2-l2a")
    s1 = make_observation(modality=S1, collection="sentinel-1-grd")

    forward = compute_compatibility(s2, s1)
    backward = compute_compatibility(s1, s2)

    assert forward.co_registration_status == backward.co_registration_status
    assert forward.same_modality == backward.same_modality is False


def test_co_registration_status_never_takes_any_other_value() -> None:
    combos = [
        (make_observation(modality=S2), make_observation(modality=S2)),
        (make_observation(modality=S1), make_observation(modality=S1)),
        (make_observation(modality=S2), make_observation(modality=S1)),
        (make_observation(modality=S1), make_observation(modality=S2)),
    ]
    for first, second in combos:
        status = compute_compatibility(first, second).co_registration_status
        assert status in {"not_evaluated", "not_supported_cross_modal"}


# --------------------------------------------------------------------------- #
# C. CRS - determinable only from retrieved imagery
# --------------------------------------------------------------------------- #


def test_crs_same_when_both_observations_carry_matching_imagery_crs() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(crs="EPSG:32644")),
        make_observation(imagery=make_imagery(crs="EPSG:32644")),
    )
    assert report.crs_match == "same"


def test_crs_different_when_imagery_reports_different_strings() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(crs="EPSG:32644")),
        make_observation(imagery=make_imagery(crs="EPSG:32643")),
    )
    assert report.crs_match == "different"


def test_crs_unknown_without_imagery_never_becomes_different() -> None:
    """The common case: include_imagery=False, so no CRS exists anywhere."""

    report = compute_compatibility(make_observation(), make_observation())
    assert report.crs_match == "unknown"


def test_crs_unknown_when_only_one_side_has_imagery() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(crs="EPSG:32644")),
        make_observation(imagery=None),
    )
    assert report.crs_match == "unknown"


def test_crs_unknown_when_imagery_reports_no_crs() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(crs=None)),
        make_observation(imagery=make_imagery(crs="EPSG:32644")),
    )
    assert report.crs_match == "unknown"


def test_crs_comparison_ignores_surrounding_whitespace_and_case() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(crs=" epsg:32644 ")),
        make_observation(imagery=make_imagery(crs="EPSG:32644")),
    )
    assert report.crs_match == "same"


# --------------------------------------------------------------------------- #
# D. Resolution
# --------------------------------------------------------------------------- #


def test_resolution_same_for_identical_values() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(resolution=10.0)),
        make_observation(imagery=make_imagery(resolution=10.0)),
    )
    assert report.resolution_match == "same"


def test_resolution_same_within_tolerance() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(resolution=10.0)),
        make_observation(imagery=make_imagery(resolution=10.0 + 1e-9)),
    )
    assert report.resolution_match == "same"


def test_resolution_different_for_10m_vs_20m() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(resolution=10.0)),
        make_observation(imagery=make_imagery(resolution=20.0)),
    )
    assert report.resolution_match == "different"


def test_resolution_unknown_without_imagery() -> None:
    report = compute_compatibility(make_observation(), make_observation())
    assert report.resolution_match == "unknown"


def test_resolution_unknown_when_value_is_absent() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(resolution=None)),
        make_observation(imagery=make_imagery(resolution=10.0)),
    )
    assert report.resolution_match == "unknown"


def test_resolution_unknown_for_non_finite_values() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(resolution=float("nan"))),
        make_observation(imagery=make_imagery(resolution=10.0)),
    )
    assert report.resolution_match == "unknown"


# --------------------------------------------------------------------------- #
# E. Processing level
# --------------------------------------------------------------------------- #


def test_processing_level_same() -> None:
    report = compute_compatibility(
        make_observation(processing_level="L2A"),
        make_observation(processing_level="L2A"),
    )
    assert report.processing_level_match == "same"


def test_processing_level_different() -> None:
    report = compute_compatibility(
        make_observation(processing_level="L2A"),
        make_observation(processing_level="L1C"),
    )
    assert report.processing_level_match == "different"


def test_processing_level_unknown_when_absent() -> None:
    report = compute_compatibility(
        make_observation(processing_level=None),
        make_observation(processing_level="L2A"),
    )
    assert report.processing_level_match == "unknown"


def test_processing_level_unknown_for_blank_string() -> None:
    report = compute_compatibility(
        make_observation(processing_level="   "),
        make_observation(processing_level="L2A"),
    )
    assert report.processing_level_match == "unknown"


def test_processing_level_match_carries_the_derived_value_caveat() -> None:
    """`processing_level` may come from the collection name, not the item."""

    report = compute_compatibility(
        make_observation(processing_level="L2A"),
        make_observation(processing_level="L2A"),
    )
    assert any("baseline" in note for note in report.limitations)


# --------------------------------------------------------------------------- #
# F. Temporal separation
# --------------------------------------------------------------------------- #


def test_temporal_separation_in_days() -> None:
    report = compute_compatibility(
        make_observation(datetime_="2024-01-01T00:00:00Z"),
        make_observation(datetime_="2024-01-11T00:00:00Z"),
    )
    assert report.temporal_separation_days == pytest.approx(10.0)


def test_temporal_separation_is_absolute_and_order_independent() -> None:
    early = make_observation(datetime_="2024-01-01T00:00:00Z")
    late = make_observation(datetime_="2024-01-11T12:00:00Z")

    forward = compute_compatibility(early, late).temporal_separation_days
    backward = compute_compatibility(late, early).temporal_separation_days

    assert forward == backward == pytest.approx(10.5)


def test_temporal_separation_unknown_when_datetime_is_missing() -> None:
    report = compute_compatibility(
        make_observation(datetime_=None),
        make_observation(datetime_="2024-01-11T00:00:00Z"),
    )
    assert report.temporal_separation_days is None


def test_temporal_separation_unknown_when_datetime_is_unparseable() -> None:
    report = compute_compatibility(
        make_observation(datetime_="last tuesday"),
        make_observation(datetime_="2024-01-11T00:00:00Z"),
    )
    assert report.temporal_separation_days is None


def test_temporal_separation_unknown_for_mixed_awareness() -> None:
    """Naive minus aware raises TypeError; it must be reported as unknown."""

    report = compute_compatibility(
        make_observation(datetime_="2024-01-01T00:00:00"),
        make_observation(datetime_="2024-01-11T00:00:00Z"),
    )
    assert report.temporal_separation_days is None
    assert any("time zone" in note for note in report.limitations)


def test_temporal_separation_works_for_two_naive_datetimes() -> None:
    report = compute_compatibility(
        make_observation(datetime_="2024-01-01T00:00:00"),
        make_observation(datetime_="2024-01-03T00:00:00"),
    )
    assert report.temporal_separation_days == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# G. Bounding-box overlap - coarse status only
# --------------------------------------------------------------------------- #


def test_bbox_full_for_identical_footprints() -> None:
    report = compute_compatibility(make_observation(), make_observation())
    assert report.bbox_overlap == "full"


def test_bbox_full_when_one_contains_the_other() -> None:
    inner = BoundingBox(west=80.15, south=12.95, east=80.25, north=13.10)
    report = compute_compatibility(
        make_observation(bbox=DEFAULT_BBOX),
        make_observation(bbox=inner),
    )
    assert report.bbox_overlap == "full"


def test_bbox_partial_for_a_genuine_intersection() -> None:
    shifted = BoundingBox(west=80.20, south=12.95, east=80.40, north=13.30)
    report = compute_compatibility(
        make_observation(bbox=DEFAULT_BBOX),
        make_observation(bbox=shifted),
    )
    assert report.bbox_overlap == "partial"


def test_bbox_none_for_disjoint_footprints() -> None:
    far = BoundingBox(west=90.0, south=20.0, east=90.5, north=20.5)
    report = compute_compatibility(
        make_observation(bbox=DEFAULT_BBOX),
        make_observation(bbox=far),
    )
    assert report.bbox_overlap == "none"


def test_bbox_none_for_edge_touching_footprints() -> None:
    """A shared edge has zero area; it is not an overlap."""

    touching = BoundingBox(west=80.30, south=12.90, east=80.50, north=13.20)
    report = compute_compatibility(
        make_observation(bbox=DEFAULT_BBOX),
        make_observation(bbox=touching),
    )
    assert report.bbox_overlap == "none"


def test_bbox_unknown_when_a_footprint_is_missing() -> None:
    report = compute_compatibility(
        make_observation(bbox=None),
        make_observation(bbox=DEFAULT_BBOX),
    )
    assert report.bbox_overlap == "unknown"


def test_bbox_overlap_is_symmetric() -> None:
    shifted = BoundingBox(west=80.20, south=12.95, east=80.40, north=13.30)
    a = make_observation(bbox=DEFAULT_BBOX)
    b = make_observation(bbox=shifted)

    assert (
        compute_compatibility(a, b).bbox_overlap
        == compute_compatibility(b, a).bbox_overlap
    )


# --------------------------------------------------------------------------- #
# H. Limitations and determinism
# --------------------------------------------------------------------------- #


def test_limitations_always_state_the_metadata_only_boundary() -> None:
    report = compute_compatibility(make_observation(), make_observation())
    assert report.limitations
    assert any("metadata" in note for note in report.limitations)


def test_unknown_crs_is_explained_in_limitations() -> None:
    report = compute_compatibility(make_observation(), make_observation())
    assert report.crs_match == "unknown"
    assert any("CRS" in note for note in report.limitations)


def test_known_crs_is_flagged_as_a_display_asset_property() -> None:
    report = compute_compatibility(
        make_observation(imagery=make_imagery(crs="EPSG:32644")),
        make_observation(imagery=make_imagery(crs="EPSG:32644")),
    )
    assert any("display" in note for note in report.limitations)


def test_cross_modal_limitations_mention_terrain_correction() -> None:
    report = compute_compatibility(
        make_observation(modality=S2),
        make_observation(modality=S1),
    )
    assert any("terrain correction" in note for note in report.limitations)


def test_report_is_deterministic() -> None:
    first = make_observation(scene_id="a", imagery=make_imagery(crs="EPSG:32644"))
    second = make_observation(scene_id="b", imagery=make_imagery(crs="EPSG:32643"))

    assert compute_compatibility(first, second) == compute_compatibility(first, second)


# --------------------------------------------------------------------------- #
# I. Pairing - same-modality only
# --------------------------------------------------------------------------- #


def test_two_same_modality_observations_form_one_pair() -> None:
    pairs, failures = pair_observations(
        make_set(
            make_observation(scene_id="a", window_label="baseline"),
            make_observation(scene_id="b", window_label="target"),
        )
    )
    assert len(pairs) == 1
    assert failures == []
    assert (pairs[0].first.scene_id, pairs[0].second.scene_id) == ("a", "b")


def test_three_observations_form_two_consecutive_pairs() -> None:
    pairs, failures = pair_observations(
        make_set(
            make_observation(scene_id="a", datetime_="2024-01-01T00:00:00Z"),
            make_observation(scene_id="b", datetime_="2024-01-05T00:00:00Z"),
            make_observation(scene_id="c", datetime_="2024-01-09T00:00:00Z"),
        )
    )
    assert failures == []
    assert [(p.first.scene_id, p.second.scene_id) for p in pairs] == [
        ("a", "b"),
        ("b", "c"),
    ]


def test_pairs_are_ordered_by_acquisition_time_not_input_order() -> None:
    pairs, _ = pair_observations(
        make_set(
            make_observation(scene_id="late", datetime_="2024-03-01T00:00:00Z"),
            make_observation(scene_id="early", datetime_="2024-01-01T00:00:00Z"),
        )
    )
    assert (pairs[0].first.scene_id, pairs[0].second.scene_id) == ("early", "late")


def test_observations_with_unknown_acquisition_time_sort_last() -> None:
    pairs, _ = pair_observations(
        make_set(
            make_observation(scene_id="undated", datetime_=None),
            make_observation(scene_id="dated", datetime_="2024-01-01T00:00:00Z"),
        )
    )
    assert (pairs[0].first.scene_id, pairs[0].second.scene_id) == ("dated", "undated")


def test_single_observation_yields_a_failure_not_a_pair() -> None:
    pairs, failures = pair_observations(make_set(make_observation(modality=S2)))
    assert pairs == []
    assert len(failures) == 1
    assert failures[0].modality == S2
    assert failures[0].reason


def test_empty_observation_set_yields_an_unattributed_failure() -> None:
    pairs, failures = pair_observations(make_set())
    assert pairs == []
    assert len(failures) == 1
    assert failures[0].modality is None


def test_pairing_never_crosses_modalities() -> None:
    pairs, _ = pair_observations(
        make_set(
            make_observation(modality=S2, scene_id="s2a"),
            make_observation(modality=S2, scene_id="s2b"),
            make_observation(modality=S1, scene_id="s1a"),
            make_observation(modality=S1, scene_id="s1b"),
        )
    )
    assert len(pairs) == 2
    for pair in pairs:
        assert pair.first.modality == pair.second.modality


def test_a_lone_sar_observation_fails_while_optical_still_pairs() -> None:
    pairs, failures = pair_observations(
        make_set(
            make_observation(modality=S2, scene_id="s2a"),
            make_observation(modality=S2, scene_id="s2b"),
            make_observation(modality=S1, scene_id="s1a"),
        )
    )
    assert len(pairs) == 1
    assert pairs[0].first.modality == S2
    assert [f.modality for f in failures] == [S1]


def test_cross_modal_compatibility_is_still_reachable_directly() -> None:
    """Pairing is same-modality only, but the report is not."""

    pairs, _ = pair_observations(
        make_set(
            make_observation(modality=S2, scene_id="s2a"),
            make_observation(modality=S1, scene_id="s1a"),
        )
    )
    assert pairs == []

    report = compute_compatibility(
        make_observation(modality=S2), make_observation(modality=S1)
    )
    assert report.co_registration_status == "not_supported_cross_modal"


def test_pairing_is_deterministic() -> None:
    observations = make_set(
        make_observation(scene_id="a", datetime_="2024-01-01T00:00:00Z"),
        make_observation(scene_id="b", datetime_="2024-01-05T00:00:00Z"),
    )
    assert pair_observations(observations) == pair_observations(observations)


def test_modality_groups_follow_first_appearance_order() -> None:
    _, failures = pair_observations(
        make_set(
            make_observation(modality=S1, scene_id="s1a"),
            make_observation(modality=S2, scene_id="s2a"),
        )
    )
    assert [f.modality for f in failures] == [S1, S2]


# --------------------------------------------------------------------------- #
# J. Purity - the phase boundary, asserted against the source
# --------------------------------------------------------------------------- #


def _imported_roots(module: Any) -> set[str]:
    """Top-level package of every import in ``module``'s source."""

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_compatibility_module_imports_nothing_that_can_do_i_o() -> None:
    """Asserted over imports, not prose - the docstring may name what it avoids."""

    roots = _imported_roots(compatibility_mod)

    for forbidden in ("rasterio", "numpy", "httpx", "PIL", "fastapi", "os", "pathlib"):
        assert forbidden not in roots, f"{forbidden!r} must not be imported"

    # Everything it may legitimately need: stdlib primitives, pydantic, and the
    # query/geospatial domain models it reports over.
    assert roots <= {"__future__", "math", "datetime", "typing", "pydantic", "app"}


def test_compatibility_module_defines_no_raster_infrastructure() -> None:
    source = pathlib.Path(compatibility_mod.__file__).read_text()

    for forbidden in ("WarpedVRT", "GridSpec", "ImageryService", "read_band"):
        assert forbidden not in source, f"{forbidden!r} must not appear"


def test_compatibility_layer_is_owned_by_the_query_domain() -> None:
    # Dependency direction: analysis -> query. query must never import analysis.
    source = pathlib.Path(compatibility_mod.__file__).read_text()
    assert "services.analysis" not in source
    assert compatibility_mod.__name__ == "app.services.query.compatibility"
