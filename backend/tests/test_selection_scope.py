"""A selection is the best of what was RETURNED, not of what exists.

Discovery asks the catalog for one bounded page - ten scenes by default, one
hundred at most - and deterministic selection then picks the lowest cloud cover
among them. The result carried ``scene_count``, which is how many scenes came
back, and nothing at all about how many matched. Ten of ten and ten of nine
hundred are different claims, and the response made them look identical.

The catalog already reports the difference (``numberMatched``, or ``context.
matched`` on older deployments). It was read off the response and discarded.

Nothing here changes which scene is chosen: the point is that the SCOPE of the
choice is now visible, so a run can be read as "the least cloudy of the ten
examined" rather than as "the least cloudy there is".
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from app.core.config import Settings
from app.services.geospatial.schemas import BoundingBox
from app.services.satellite.schemas import SceneSearchRequest
from app.services.satellite.service import SatelliteService
from app.services.satellite.stac import search_items, search_page

BBOX = BoundingBox(west=80.1, south=12.9, east=80.3, north=13.2)

FEATURE: dict[str, Any] = {
    "id": "S2B_44PMV_20250104_0_L2A",
    "collection": "sentinel-2-l2a",
    "bbox": [80.0, 12.8, 80.5, 13.3],
    "geometry": None,
    "properties": {"datetime": "2025-01-04T05:12:34Z", "eo:cloud_cover": 4.2},
    "assets": {},
}


def payload(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"type": "FeatureCollection", "features": [FEATURE]}
    body.update(extra)
    return body


def transport_for(body: dict[str, Any]) -> httpx.MockTransport:
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=body))


def page(body: dict[str, Any]):  # type: ignore[no-untyped-def]
    return asyncio.run(
        search_page(
            settings=Settings(),
            body={"collections": ["sentinel-2-l2a"]},
            transport=transport_for(body),
        )
    )


# --------------------------------------------------------------------------- #
# Reading the count the catalog already reports
# --------------------------------------------------------------------------- #


def test_the_match_count_is_read() -> None:
    result = page(payload(numberMatched=137))

    assert result.matched == 137
    assert len(result.features) == 1  # the page itself is unchanged


def test_an_older_catalog_reports_it_under_context() -> None:
    result = page(payload(context={"matched": 42, "returned": 1}))

    assert result.matched == 42


def test_a_catalog_that_says_nothing_leaves_it_unknown() -> None:
    """Unknown is not zero, and it is certainly not "all of them"."""

    assert page(payload()).matched is None


def test_a_nonsense_count_is_treated_as_unknown() -> None:
    assert page(payload(numberMatched="lots")).matched is None
    assert page(payload(numberMatched=-1)).matched is None


def test_the_features_only_helper_still_works() -> None:
    """Backward compatibility: callers wanting just the features are unaffected."""

    features = asyncio.run(
        search_items(
            settings=Settings(),
            body={"collections": ["sentinel-2-l2a"]},
            transport=transport_for(payload(numberMatched=7)),
        )
    )

    assert [f["id"] for f in features] == [FEATURE["id"]]


# --------------------------------------------------------------------------- #
# ...and carrying it out to where a reader can see it
# --------------------------------------------------------------------------- #


def search(body: dict[str, Any]):  # type: ignore[no-untyped-def]
    service = SatelliteService(Settings(), transport=transport_for(body))
    return asyncio.run(
        service.search(
            SceneSearchRequest(
                bbox=BBOX,
                start_date="2025-01-01",  # type: ignore[arg-type]
                end_date="2025-01-31",  # type: ignore[arg-type]
            )
        )
    )


def test_the_search_response_states_the_scope_of_its_page() -> None:
    response = search(payload(numberMatched=137))

    # What was chosen between...
    assert response.scene_count == 1
    # ...and what it was chosen FROM.
    assert response.scenes_matched == 137


def test_the_scope_is_unknown_when_the_catalog_does_not_report_it() -> None:
    assert search(payload()).scenes_matched is None


def test_a_single_page_result_is_not_dressed_up_as_the_whole_archive() -> None:
    """The case that matters: one returned scene, many matching ones.

    Before this, the response said "1 scene" and a reader could only conclude
    that one scene existed for that window.
    """

    response = search(payload(numberMatched=900))

    assert response.scene_count < response.scenes_matched  # type: ignore[operator]
