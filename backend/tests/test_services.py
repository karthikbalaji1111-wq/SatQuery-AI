"""Contract tests for the domain services."""

from __future__ import annotations

import pytest
from app.core.errors import NotImplementedFeatureError
from app.services.ai import AiService
from app.services.analysis import AnalysisService
from app.services.geospatial import GeospatialService
from app.services.map import MapService
from app.services.multimodal import MultimodalService
from app.services.query import QueryExecutionService, QueryService
from app.services.satellite import SatelliteService
from app.services.temporal import TemporalService

# Services still awaiting implementation - their generic `run` hook must raise.
STUBBED_SERVICES = [
    MultimodalService,
    TemporalService,
    MapService,
]

# Implemented services expose a typed entry point instead of the generic `run`.
IMPLEMENTED_SERVICES = [
    GeospatialService,
    SatelliteService,
    QueryService,
    QueryExecutionService,
    AiService,
    AnalysisService,
]

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


# --------------------------------------------------------------------------- #
# Capability truth: a reserved stub must not describe work it does not do
# --------------------------------------------------------------------------- #
#
# These three packages are empty placeholders. Their ``describe()`` strings
# previously read "Multimodal fusion...", "Multitemporal change detection..."
# and "Map layer and tile preparation..." - three capabilities that do not
# exist anywhere in this repository. ``describe()`` is the service contract's
# own statement of purpose, so that was the architecture lying about itself.


def test_reserved_stubs_declare_themselves_unimplemented() -> None:
    for service in (MultimodalService(), TemporalService(), MapService()):
        described = service.describe().lower()
        assert "not implemented" in described, (
            f"{service.name} must say it is unimplemented, got: {service.describe()}"
        )


def test_reserved_stubs_claim_no_capability_the_system_lacks() -> None:
    """Fusion and co-registration do not exist; nothing may imply they do."""

    for service in (MultimodalService(), TemporalService(), MapService()):
        described = service.describe().lower()
        for overclaim in ("fusion of", "co-registered", "co-registration of"):
            assert overclaim not in described, (
                f"{service.name} claims {overclaim!r}, which is not implemented"
            )
        # A bare mention is allowed only in an explicit denial.
        for term in ("fusion", "co-registration"):
            if term in described:
                assert "not implemented" in described


def test_reserved_stubs_really_are_empty() -> None:
    """If one of these ever gains logic, the honesty note must be revisited."""

    for service in (MultimodalService(), TemporalService(), MapService()):
        public = [
            name
            for name in dir(service)
            if not name.startswith("_")
            and callable(getattr(service, name))
            and name not in {"describe", "run"}
        ]
        assert public == [], f"{service.name} gained behaviour: {public}"
