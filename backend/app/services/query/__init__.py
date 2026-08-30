"""Structured query intent contracts, plan resolution, execution, and
metadata-only observation compatibility reporting."""

from app.services.query.compatibility import (
    BboxOverlapStatus,
    CompatibilityReport,
    CoRegistrationStatus,
    MatchStatus,
    ObservationPair,
    PairingFailure,
    compute_compatibility,
    pair_observations,
)
from app.services.query.execution import QueryExecutionService
from app.services.query.schemas import (
    ExecutedWindow,
    Modality,
    Observation,
    ObservationSet,
    QueryExecutionRequest,
    QueryExecutionResult,
    QueryTask,
    ResolvedQueryPlan,
    SatQueryIntent,
    SkippedModality,
    TemporalComparison,
    TemporalMode,
    TimeRange,
)
from app.services.query.service import QueryService

__all__ = [
    "BboxOverlapStatus",
    "CoRegistrationStatus",
    "CompatibilityReport",
    "ExecutedWindow",
    "MatchStatus",
    "Modality",
    "Observation",
    "ObservationPair",
    "ObservationSet",
    "PairingFailure",
    "QueryExecutionRequest",
    "QueryExecutionResult",
    "QueryExecutionService",
    "QueryService",
    "QueryTask",
    "ResolvedQueryPlan",
    "SatQueryIntent",
    "SkippedModality",
    "TemporalComparison",
    "TemporalMode",
    "TimeRange",
    "compute_compatibility",
    "pair_observations",
]
