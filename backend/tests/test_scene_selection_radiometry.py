"""Optical scene selection prefers scenes whose radiometry can be verified.

Before: the least-cloudy scene was always chosen, even when radiometric
validation would then refuse it (baseline >= 04.00 with the provider flag
false is "undetermined") and another scene of the same month was usable. The
user got "analysis not computed" for a month that had a measurable scene.
The preference reorders and never drops; validation stays the authority.
"""

from __future__ import annotations

import pytest
from app.services.query.execution import _select_scene, _select_scene_sar
from app.services.satellite.radiometry import offset_free_by_metadata
from app.services.satellite.schemas import Scene
from app.services.satellite.service import _normalize_scene


def _scene(
    scene_id: str,
    cloud: float | None,
    baseline: str | None = None,
    flag: str | None = None,
    when: str = "2025-01-10T05:00:00Z",
) -> Scene:
    return Scene(
        id=scene_id,
        datetime=when,
        bbox=None,
        geometry=None,
        cloud_cover=cloud,
        collection="sentinel-2-l2a",
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
        processing_baseline=baseline,
        boa_offset_flag=flag,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("baseline", "flag", "expected"),
    [
        ("05.11", "true", True),  # provider removed the offset
        ("03.01", "false", True),  # baseline predates the offset
        ("05.11", "false", False),  # undetermined: refused by validation
        ("04.00", "absent", False),
        ("05.11", "unparseable", False),
        ("03.01", "true", False),  # contradictory
        (None, "true", None),
        ("5.11x", "true", None),
        ("05.11", None, None),
    ],
)
def test_metadata_verdict_follows_the_central_offset_rule(
    baseline: str | None, flag: str | None, expected: bool | None
) -> None:
    assert offset_free_by_metadata(baseline, flag) is expected


def test_a_verifiable_scene_wins_over_a_clearer_unverifiable_one() -> None:
    clear_but_undetermined = _scene("S2C_clear", 0.5, "05.11", "false")
    cloudier_but_usable = _scene("S2B_usable", 12.0, "05.11", "true")

    chosen = _select_scene([clear_but_undetermined, cloudier_but_usable])

    assert chosen is not None and chosen.id == "S2B_usable"


def test_among_verifiable_scenes_cloud_order_still_decides() -> None:
    scenes = [
        _scene("a", 30.0, "05.11", "true"),
        _scene("b", 4.0, "03.01", "false"),
        _scene("c", 0.1, "05.11", "false"),
    ]
    chosen = _select_scene(scenes)
    assert chosen is not None and chosen.id == "b"


def test_with_no_metadata_the_order_is_exactly_the_old_one() -> None:
    scenes = [
        _scene("late", 2.0, when="2025-01-20T05:00:00Z"),
        _scene("early", 2.0, when="2025-01-05T05:00:00Z"),
        _scene("cloudy", 40.0),
        _scene("unknown", None),
    ]
    chosen = _select_scene(scenes)
    assert chosen is not None and chosen.id == "early"


def test_no_scene_is_ever_dropped() -> None:
    only = _scene("only", 3.0, "05.11", "false")
    chosen = _select_scene([only])
    assert chosen is not None and chosen.id == "only"


def test_sar_selection_is_unaffected() -> None:
    first = _scene("s1-early", None, when="2025-01-01T00:00:00Z")
    second = _scene("s1-late", None, "05.11", "true", when="2025-01-09T00:00:00Z")
    chosen = _select_scene_sar([second, first])
    assert chosen is not None and chosen.id == "s1-early"


def _feature(properties: dict) -> dict:
    return {"id": "x", "collection": "sentinel-2-l2a", "properties": properties}


def test_discovery_reads_the_baseline_and_flag_as_validation_does() -> None:
    usable = _normalize_scene(
        _feature({"s2:processing_baseline": "05.11", "earthsearch:boa_offset_applied": True})
    )
    undetermined = _normalize_scene(
        _feature({"s2:processing_baseline": "05.11", "earthsearch:boa_offset_applied": False})
    )
    silent = _normalize_scene(_feature({"s2:processing_baseline": "04.00"}))
    sar = _normalize_scene(_feature({"sar:polarizations": ["VV", "VH"]}))

    assert (usable.processing_baseline, usable.boa_offset_flag) == ("05.11", "true")
    assert undetermined.boa_offset_flag == "false"
    assert silent.boa_offset_flag == "absent"
    assert (sar.processing_baseline, sar.boa_offset_flag) == (None, None)
