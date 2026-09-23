"""The SAR polarization actually requested is the asset actually retrieved.

VV was hardcoded in `QueryExecutionService`, so VH was unreachable through
`/query/execute` even though `ImageryService` had supported it all along. These
tests pin the wiring and the default, so a regression cannot quietly strand VH
again.
"""

from __future__ import annotations

import asyncio

from app.services.query.schemas import QueryExecutionRequest, SatQueryIntent, TimeRange
from app.services.satellite.schemas import SAR_IMAGERY_ASSET

from tests.test_query_execution import (  # reuse the established fakes verbatim
    FakeImageryService,
    FakeSatelliteService,
    build_service,
    make_scene,
    make_search_response,
)


def _sar_intent() -> SatQueryIntent:
    return SatQueryIntent(
        location_query="Marina Beach, Chennai",
        temporal_mode="single",
        time_windows=[TimeRange(start_date="2025-03-01", end_date="2025-03-31")],
        modalities=["sentinel-1-sar"],
        task="visualize",
    )


def test_the_default_polarization_is_vv() -> None:
    """Existing callers that never heard of this field keep their behaviour."""

    request = QueryExecutionRequest(intent=_sar_intent(), include_imagery=True)

    assert request.sar_polarization == SAR_IMAGERY_ASSET == "vv"


def test_vh_is_accepted_and_preserved() -> None:
    """The request carries VH through rather than silently normalising it."""

    request = QueryExecutionRequest(
        intent=_sar_intent(), include_imagery=True, sar_polarization="vh"
    )

    assert request.sar_polarization == "vh"


def test_an_unknown_polarization_is_refused() -> None:
    """Only the two polarizations the imagery service supports are accepted."""

    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QueryExecutionRequest(
            intent=_sar_intent(), include_imagery=True, sar_polarization="hh"
        )


def test_the_requested_polarization_reaches_the_imagery_service() -> None:
    """The end the wiring exists for: VH asked for is VH fetched.

    Recorded rather than asserted on a return value, because the defect this
    guards against produced a perfectly good response - it was simply the wrong
    band, which no output-shaped assertion would have caught.
    """

    for polarization, expected in (("vv", "vv"), ("vh", "vh")):
        imagery = FakeImageryService()
        service = build_service(
            satellite=FakeSatelliteService(
                responses=make_search_response(make_scene("s1"))
            ),
            imagery=imagery,
        )
        asyncio.run(
            service.execute(
                QueryExecutionRequest(
                    intent=_sar_intent(),
                    include_imagery=True,
                    sar_polarization=polarization,
                )
            )
        )
        assert [r.asset for r in imagery.requests] == [expected]
