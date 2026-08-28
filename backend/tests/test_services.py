"""Contract tests for the domain services."""

from __future__ import annotations

import pytest
from app.core.errors import NotImplementedFeatureError
from app.services.ai import AiService
from app.services.geospatial import GeospatialService
from app.services.map import MapService
from app.services.multimodal import MultimodalService
from app.services.query import QueryService
from app.services.satellite import SatelliteService
from app.services.temporal import TemporalService

# Services still awaiting implementation - their generic `run` hook must raise.
STUBBED_SERVICES = [
    QueryService,
    MultimodalService,
    TemporalService,
    AiService,
    MapService,
]

# Implemented services expose a typed entry point instead of the generic `run`.
IMPLEMENTED_SERVICES = [GeospatialService, SatelliteService]

ALL_SERVICES = [*STUBBED_SERVICES, *IMPLEMENTED_SERVICES]


@pytest.mark.parametrize("service_cls", ALL_SERVICES)
def test_service_describes_itself(service_cls: type) -> None:
    service = service_cls()
    assert service.name
    assert isinstance(service.describe(), str)
    assert service.describe()


@pytest.mark.parametrize("service_cls", STUBBED_SERVICES)
def test_service_run_not_implemented(service_cls: type) -> None:
    with pytest.raises(NotImplementedFeatureError):
        service_cls().run()
