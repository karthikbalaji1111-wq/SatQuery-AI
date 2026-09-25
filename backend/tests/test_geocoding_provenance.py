"""What the geocoder matched, and what the planner may not override.

Nominatim is asked for one result and the first is used. That is a real
limitation - "Springfield" has many - and this file does NOT pretend to solve
it, because solving it needs either a second geocoder or a confidence measure,
and inventing a score would be worse than the gap.

What it does establish is that the choice is VISIBLE and the choice is the
GEOCODER'S. Two failure modes are covered:

* a match that looks right and is not - a shop named after a beach returns a
  plausible display name and a valid bbox, so the geocoder's own class/type is
  recorded to tell them apart;
* a planner supplying its own coordinates - now refused outright rather than
  accepted and silently dropped.
"""

from __future__ import annotations

import httpx
import pytest
from app.core.config import Settings
from app.services.geospatial.nominatim import geocode, reset_geocoder_state
from app.services.query.schemas import SatQueryIntent
from pydantic import ValidationError

BASE_INTENT = {
    "location_query": "Chennai",
    "temporal_mode": "single",
    "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
    "modalities": ["sentinel-2-optical"],
    "task": "visualize",
}


def _result(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "lat": "13.05",
        "lon": "80.28",
        "boundingbox": ["13.03", "13.07", "80.27", "80.29"],
        "display_name": "Marina Beach, Chennai, Tamil Nadu, India",
        "class": "natural",
        "type": "beach",
    }
    item.update(overrides)
    return item


def _geocode(item: dict[str, object]):
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[item])

    import asyncio

    # Each call simulates an INDEPENDENT lookup. The geocoder now caches by
    # query, so without this the second call in a test would be answered from
    # the first one's result - these tests are about what the geocoder reports,
    # not about caching, which is proven in tests/test_geocoder_policy.py.
    reset_geocoder_state()

    return asyncio.run(
        geocode(
            "Marina Beach",
            settings=Settings(_env_file=None),  # type: ignore[arg-type]
            transport=httpx.MockTransport(respond),
        )
    )


# --------------------------------------------------------------------------- #
# The geocoder's own classification survives into provenance.
# --------------------------------------------------------------------------- #


def test_the_matched_class_and_type_are_preserved() -> None:
    place = _geocode(_result())

    assert place.place_class == "natural"
    assert place.place_type == "beach"


def test_a_differently_classified_match_is_distinguishable() -> None:
    """The failure this exists for: right name, wrong thing.

    Both results carry a plausible display name and a valid bounding box, so
    without the classification they are indistinguishable in provenance.
    """

    beach = _geocode(_result())
    shop = _geocode(
        _result(
            **{
                "class": "shop",
                "type": "supermarket",
                "display_name": "Marina Beach Stores, Chennai",
            }
        )
    )

    assert beach.place_class != shop.place_class
    assert (shop.place_class, shop.place_type) == ("shop", "supermarket")


def test_a_missing_classification_is_unknown_rather_than_invented() -> None:
    item = _result()
    del item["class"]
    del item["type"]

    place = _geocode(item)

    assert place.place_class is None
    assert place.place_type is None
    # And the resolution still succeeds - absent metadata is not a failure.
    assert place.display_name


def test_a_blank_classification_is_normalised_to_unknown() -> None:
    """An empty string would render as a present-but-meaningless field."""

    place = _geocode(_result(**{"class": "   ", "type": ""}))

    assert place.place_class is None
    assert place.place_type is None


@pytest.mark.parametrize(
    "place_name", ["Springfield", "Paris", "London", "Central Park"]
)
def test_an_ambiguous_name_still_reports_what_it_matched(place_name: str) -> None:
    """These names are genuinely ambiguous and this system picks the first.

    The honest guarantee is not that the choice is right - it is that the choice
    is stated: the display name and the classification both travel with it, so a
    reader can see that "Paris" resolved to a place/city in France rather than a
    hamlet in Texas.
    """

    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "lat": "48.85",
                    "lon": "2.35",
                    "boundingbox": ["48.8", "48.9", "2.2", "2.4"],
                    "display_name": f"{place_name}, France",
                    "class": "place",
                    "type": "city",
                }
            ],
        )

    import asyncio

    place = asyncio.run(
        geocode(
            place_name,
            settings=Settings(_env_file=None),  # type: ignore[arg-type]
            transport=httpx.MockTransport(respond),
        )
    )

    assert place_name in place.display_name
    assert (place.place_class, place.place_type) == ("place", "city")


# --------------------------------------------------------------------------- #
# The planner grounds nothing. It names a place; it does not supply one.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field", ["bbox", "center", "lat", "lon", "coordinates", "aoi", "geometry"]
)
def test_a_planner_cannot_supply_its_own_geography(field: str) -> None:
    """Previously accepted and silently dropped; now refused.

    Nothing downstream ever read these - the location comes from
    `location_query` alone - so no substitution occurred. But "accepted and
    ignored" is the wrong answer to a model trying to supply coordinates, and
    every other agent contract in this system is closed.
    """

    with pytest.raises(ValidationError):
        SatQueryIntent(**(BASE_INTENT | {field: [1, 2, 3, 4]}))  # type: ignore[arg-type]


def test_the_legitimate_intent_still_validates() -> None:
    """Counter-case: forbidding everything would pass the test above."""

    assert SatQueryIntent(**BASE_INTENT) is not None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Final demo audit: the plan carries WHAT was matched, not only the typed name
# --------------------------------------------------------------------------- #


def test_the_plan_names_the_feature_the_geocoder_actually_matched() -> None:
    """Live: "Lalbagh, Bengaluru" matched a railway STOP, not the garden, and
    the workspace could only show the typed name. The match travels with the
    plan now, verbatim - the geocoder stays the authority, nothing is judged."""

    import asyncio

    from app.services.geospatial import GeospatialService
    from app.services.query import QueryService

    reset_geocoder_state()
    stop = [{
        "lat": "12.9507", "lon": "77.5848",
        "boundingbox": ["12.95065", "12.95075", "77.58475", "77.58485"],
        "display_name": "Lalbagh, Rashtriya Vidyalaya Road, Bengaluru, Karnataka, India",
        "class": "railway", "type": "stop", "osm_type": "node",
    }]
    geo = GeospatialService(
        settings=Settings(geocoder_min_interval_seconds=0.0),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=stop)),
    )
    intent = SatQueryIntent.model_validate({**BASE_INTENT, "location_query": "Lalbagh, Bengaluru"})

    plan = asyncio.run(QueryService(geospatial_service=geo).build_plan(intent))

    assert plan.intent.location_query == "Lalbagh, Bengaluru"  # what was typed
    assert plan.matched_name == stop[0]["display_name"]         # what was matched
    assert (plan.matched_class, plan.matched_type) == ("railway", "stop")
    # The geocoder's box, untouched - no area invented around the point.
    assert (plan.bbox.south, plan.bbox.north) == (12.95065, 12.95075)
