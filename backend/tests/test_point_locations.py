"""A point is not an area: point-like geocoder matches never become image AOIs.

Found in production (2026-09-25): "Show water (NDWI) around sahara desert
january 2025" geocoded to the single OpenStreetMap NODE that labels the Sahara
("Sahara, Tazrouk ... (natural · desert)"), whose box Nominatim returns as
about 10 x 11 m. That box became the analysis area, and SatQuery read, measured
and displayed a 2 x 2 pixel raster as if it were the Sahara.

The fix, in the geocoder and the plan builder - never in the image:

* A node that is not a settlement (``place=*``) is POINT-LIKE: the geocoder has
  only a point, and the box around it is a display box, not the feature's
  extent. Geometry decides this, not a size threshold.
* Up to five candidates are read (one request). A point-like first result is
  replaced ONLY by a later candidate that has a real extent AND carries the name
  asked for - the geocoder's own order decides, never the size of a box.
* A point that remains is refused before any catalog search, by name. No box is
  ever widened, and no coordinate is ever produced that the geocoder did not
  return.

Every payload here is a verbatim Nominatim answer captured on 2026-09-25
(``tests/nominatim_candidates.json``); no test contacts the network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from app.core.config import Settings
from app.core.errors import PointLocationError, UpstreamServiceError
from app.services.agent.executor import AgentExecutor
from app.services.agent.schemas import AgentQuestionRequest
from app.services.agent.service import AgentService
from app.services.agent.standard import StandardPlanner, StandardReport
from app.services.analysis import AnalysisService
from app.services.analysis.validation import AnalysisRequestRejectedError
from app.services.geospatial import GeospatialService, ResolveRequest
from app.services.geospatial.nominatim import (
    CANDIDATE_LIMIT,
    NominatimPlace,
    _parse_candidates,
    reset_geocoder_state,
    select_candidate,
)
from app.services.query import QueryService
from app.services.query.execution import QueryExecutionService
from app.services.query.schemas import SatQueryIntent

CAPTURED: dict[str, list[dict[str, Any]]] = {
    key: value
    for key, value in json.loads(
        (Path(__file__).parent / "nominatim_candidates.json").read_text(encoding="utf-8")
    ).items()
    if not key.startswith("_")
}

INTENT = {
    "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
    "modalities": ["sentinel-2-optical"],
    "task": "visualize",
    "temporal_mode": "single",
}


@pytest.fixture(autouse=True)
def _fresh_geocoder() -> None:
    reset_geocoder_state()


class Nominatim:
    """A scripted Nominatim that records every request it is sent."""

    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self.payload = payload
        self.requests: list[httpx.Request] = []

    def service(self) -> GeospatialService:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json=self.payload)

        return GeospatialService(
            settings=Settings(geocoder_min_interval_seconds=0.0),
            transport=httpx.MockTransport(handle),
        )


def resolve(query: str, payload: list[dict[str, Any]] | None = None):
    geo = Nominatim(CAPTURED[query] if payload is None else payload)
    return asyncio.run(geo.service().resolve(ResolveRequest(place=query))), geo


def build_plan(query: str, payload: list[dict[str, Any]] | None = None):
    geo = Nominatim(CAPTURED[query] if payload is None else payload)
    intent = SatQueryIntent.model_validate({**INTENT, "location_query": query})
    return asyncio.run(QueryService(geospatial_service=geo.service()).build_plan(intent))


def as_box(item: dict[str, Any]) -> tuple[float, float, float, float]:
    south, north, west, east = (float(value) for value in item["boundingbox"])
    return west, south, east, north


def box(response: Any) -> tuple[float, float, float, float]:
    b = response.bbox
    return b.west, b.south, b.east, b.north


# =========================================================================== #
# 1 + 5. The Sahara: a point, refused - never widened, never a 2 x 2 image
# =========================================================================== #


def test_the_sahara_resolves_only_to_a_point_and_says_so() -> None:
    resolved, geo = resolve("sahara desert")
    [only] = CAPTURED["sahara desert"]
    assert (only["osm_type"], only["class"], only["type"]) == ("node", "natural", "desert")
    assert resolved.point_like is True
    assert resolved.osm_type == "node"
    # Still RESOLVABLE - the point is reported as the geocoder gave it.
    assert box(resolved) == as_box(only)
    assert len(geo.requests) == 1


def test_an_analysis_of_the_sahara_is_refused_by_name_before_any_search() -> None:
    with pytest.raises(PointLocationError) as refused:
        build_plan("sahara desert")
    assert refused.value.code == "location_is_point"
    message = refused.value.message
    assert message.startswith("'sahara desert' resolved only to a single point on the map: Sahara")
    assert "(natural · desert)" in message
    assert "does not draw one around it" in message


def test_sahara_is_never_replaced_by_an_unrelated_area() -> None:
    """The geocoder also returns New York for "Sahara" - a real, large extent
    that is not the place asked for. Having an extent does not make it so."""

    resolved, _ = resolve("Sahara")
    names = [(c["name"], c["osm_type"]) for c in CAPTURED["Sahara"]]
    assert names == [("Sahara", "node"), ("New York", "relation")]
    assert resolved.display_name.startswith("Sahara")
    assert resolved.point_like is True
    with pytest.raises(PointLocationError):
        build_plan("Sahara")


def test_the_sahara_through_the_agent_is_a_question_with_no_search_and_no_image() -> None:
    """End to end: the REAL geocoder and plan builder inside the REAL query
    execution service. The catalog and imagery are recorders that must stay
    untouched - a 2 x 2 success image cannot be produced if nothing is read."""

    class Untouched:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def search(self, *args: Any, **kwargs: Any) -> Any:
            self.calls.append((args, kwargs))
            raise AssertionError("no catalog search may run for a point")

        def retrieve(self, *args: Any, **kwargs: Any) -> Any:
            self.calls.append((args, kwargs))
            raise AssertionError("no imagery may be read for a point")

    geo = Nominatim(CAPTURED["sahara desert"])
    satellite, imagery = Untouched(), Untouched()
    execution = QueryExecutionService(
        query_service=QueryService(geospatial_service=geo.service()),
        satellite_service=satellite,  # type: ignore[arg-type]
        imagery_service=imagery,  # type: ignore[arg-type]
    )
    service = AgentService(
        planner=StandardPlanner(),
        executor=AgentExecutor(
            query_execution_service=execution, analysis_service=AnalysisService()
        ),
        synthesizer=StandardReport(),
    )

    result = asyncio.run(
        service.answer(
            AgentQuestionRequest(question="Show water (NDWI) around sahara desert january 2025")
        )
    )

    assert result.status == "needs_clarification"
    assert result.answer is None
    assert result.clarification is not None
    assert result.clarification.reason == "location_is_point"
    assert "single point" in result.clarification.message
    assert "Name an area instead" in result.clarification.message
    assert result.clarification.understood_location == "sahara desert"
    assert satellite.calls == [] and imagery.calls == []
    assert not any(item.measurement for item in result.evidence.items)
    assert result.evidence.execution is None


# =========================================================================== #
# 2. The same named place with a real extent replaces a point-like first result
# =========================================================================== #


def test_lalbagh_uses_the_gardens_the_geocoder_also_returned_not_the_station() -> None:
    candidates = CAPTURED["Lalbagh, Bengaluru"]
    assert (candidates[0]["osm_type"], candidates[0]["class"]) == ("node", "railway")
    gardens = candidates[3]
    assert gardens["name"] == "Lalbagh Botanical Gardens"

    plan = build_plan("Lalbagh, Bengaluru")

    assert plan.matched_name == gardens["display_name"]
    assert (plan.matched_class, plan.matched_type) == ("leisure", "park")
    assert (plan.bbox.west, plan.bbox.south, plan.bbox.east, plan.bbox.north) == as_box(gardens)


def test_bandra_uses_the_suburb_polygon_not_the_station_node() -> None:
    plan = build_plan("Bandra, Mumbai")
    assert plan.matched_name is not None and plan.matched_name.startswith("Bandra West")
    assert (plan.matched_class, plan.matched_type) == ("place", "suburb")


def test_a_named_feature_kind_counts_as_part_of_the_name() -> None:
    """ "Lalbagh park" names the park: the candidate's own type supplies "park"."""

    point = NominatimPlace.model_validate(
        _parse_candidates(CAPTURED["Lalbagh, Bengaluru"])[0].model_dump()
    )
    gardens = _parse_candidates(CAPTURED["Lalbagh, Bengaluru"])[3]
    assert select_candidate("Lalbagh park", [point, gardens]) is gardens
    assert select_candidate("Lalbagh lake", [point, gardens]) is point


# =========================================================================== #
# 3 + 9. Real small geometries and settlements are unchanged; points resolve
# =========================================================================== #


def test_a_landmark_with_its_own_small_footprint_is_still_measurable() -> None:
    """India Gate is a WAY: its 19 x 28 m box is its real footprint, not a
    display box. Small is not point-like; the small-sample note says the rest."""

    plan = build_plan("India Gate")
    first = CAPTURED["India Gate"][0]
    assert (first["osm_type"], first["class"]) == ("way", "tourism")
    assert (plan.bbox.west, plan.bbox.south, plan.bbox.east, plan.bbox.north) == as_box(first)


def test_a_settlement_mapped_as_a_point_keeps_the_geocoders_settlement_box() -> None:
    """Ameerpet is a place=suburb NODE - how OSM maps a settlement - and the
    geocoder answers it with a settlement-scale box. Unchanged, and a demo query."""

    plan = build_plan("Ameerpet, Hyderabad")
    first = CAPTURED["Ameerpet, Hyderabad"][0]
    assert (first["osm_type"], first["class"], first["type"]) == ("node", "place", "suburb")
    assert (plan.bbox.west, plan.bbox.south, plan.bbox.east, plan.bbox.north) == as_box(first)


def test_a_lone_point_still_resolves_and_its_refusal_names_what_was_matched() -> None:
    """A railway stop with nothing else of that name: resolvable, visible, and
    refused as an analysis area - with the match in the refusal, verbatim."""

    stop = [CAPTURED["Lalbagh, Bengaluru"][2]]
    resolved, _ = resolve("Lalbagh, Bengaluru", stop)
    assert resolved.point_like is True
    assert resolved.display_name == stop[0]["display_name"]

    with pytest.raises(PointLocationError) as refused:
        build_plan("Lalbagh, Bengaluru", stop)
    assert "Lalbagh, Rashtriya Vidyalaya Road (railway · stop)" in refused.value.message


def test_marina_beach_without_its_comma_is_a_point_and_is_refused() -> None:
    [node] = CAPTURED["Marina Beach Chennai"]
    assert (node["osm_type"], node["type"]) == ("node", "beach")
    with pytest.raises(PointLocationError):
        build_plan("Marina Beach Chennai")


# =========================================================================== #
# 7 + 8. Cubbon Park and Marina Beach: exactly the geocoder's first result
# =========================================================================== #


@pytest.mark.parametrize("query", ["Cubbon Park, Bengaluru", "Marina Beach, Chennai"])
def test_existing_places_resolve_exactly_as_before(query: str) -> None:
    first = CAPTURED[query][0]
    assert first["osm_type"] == "way"
    plan = build_plan(query)
    assert plan.matched_name == first["display_name"]
    assert (plan.bbox.west, plan.bbox.south, plan.bbox.east, plan.bbox.north) == as_box(first)


# =========================================================================== #
# 6. A real large extent is still refused as too large, not as a point
# =========================================================================== #


def test_a_real_large_extent_is_still_refused_as_too_large() -> None:
    plan = build_plan("Chennai")
    assert CAPTURED["Chennai"][0]["osm_type"] == "relation"
    with pytest.raises(AnalysisRequestRejectedError) as refused:
        AnalysisService().precheck_plan(plan.bbox, indices=("ndwi",))
    assert refused.value.code == "aoi_too_large"


# =========================================================================== #
# 4 + 5. Nothing fabricated: every box and centre is one the geocoder returned
# =========================================================================== #


@pytest.mark.parametrize("query", sorted(CAPTURED))
def test_the_chosen_box_and_centre_are_a_candidates_own_values(query: str) -> None:
    resolved, _ = resolve(query)
    returned = {
        (as_box(item), (float(item["lat"]), float(item["lon"]))) for item in CAPTURED[query]
    }
    chosen = (box(resolved), (resolved.center.lat, resolved.center.lon))
    assert chosen in returned


def test_selection_returns_a_candidate_object_never_a_new_one() -> None:
    for query, payload in CAPTURED.items():
        candidates = _parse_candidates(payload)
        assert any(select_candidate(query, candidates) is c for c in candidates)


# =========================================================================== #
# 10. Deterministic, bounded, and order - not size - decides
# =========================================================================== #


def area(name: str, box: list[str], osm_type: str = "way") -> dict[str, Any]:
    return {
        "lat": "12.95", "lon": "77.58", "boundingbox": box, "display_name": f"{name}, Bengaluru",
        "class": "leisure", "type": "park", "osm_type": osm_type, "name": name,
    }


def test_the_geocoders_order_decides_between_equals_never_the_larger_box() -> None:
    point = CAPTURED["Lalbagh, Bengaluru"][1]
    small = area("Lalbagh North", ["12.950", "12.951", "77.580", "77.581"])
    large = area("Lalbagh South", ["12.90", "13.00", "77.50", "77.60"])

    first = select_candidate("Lalbagh", _parse_candidates([point, small, large]))
    assert first.name == "Lalbagh North"
    swapped = select_candidate("Lalbagh", _parse_candidates([point, large, small]))
    assert swapped.name == "Lalbagh South"
    # Same input, same answer.
    for _ in range(5):
        again = select_candidate("Lalbagh", _parse_candidates([point, small, large]))
        assert again.name == "Lalbagh North"


def test_a_non_point_first_result_is_never_second_guessed() -> None:
    first = area("Cubbon Park", ["12.970", "12.983", "77.587", "77.599"])
    bigger = area("Cubbon Park", ["12.90", "13.10", "77.50", "77.70"], osm_type="relation")
    assert select_candidate("Cubbon Park", _parse_candidates([first, bigger])).bbox.north == 12.983


def test_one_bounded_request_for_five_candidates() -> None:
    _, geo = resolve("Lalbagh, Bengaluru")
    [request] = geo.requests
    assert request.url.params["limit"] == str(CANDIDATE_LIMIT) == "5"


def test_a_malformed_alternative_is_dropped_but_a_malformed_first_result_is_an_error() -> None:
    good = CAPTURED["Cubbon Park, Bengaluru"][0]
    assert len(_parse_candidates([good, {"lat": "x"}])) == 1
    with pytest.raises(UpstreamServiceError):
        _parse_candidates([{"lat": "x"}, good])
