"""Analysis boundary over an already-computed query execution.

Implemented so far: the contract itself (:class:`AnalysisRequest` ->
:class:`AnalysisResult`), a deterministic ``visualize`` summary, and one pure
engine - opt-in single-scene Sentinel-2 NDWI statistics, and opt-in Temporal
NDWI Statistics for one deterministic Sentinel-2 observation pair
(:mod:`app.services.analysis.engines`). ``multimodal`` (fusion) and ``temporal``
(change detection) remain the future homes for engines needing more than one
scene.

Dependency direction is ``analysis -> query -> satellite``; nothing in ``query``
may import this package.
"""

from app.services.analysis.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisStatus,
    AnalysisWindowRef,
    Measurement,
    ObservationIndexResult,
    TemporalIndexComparison,
)
from app.services.analysis.service import AnalysisService

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisService",
    "AnalysisStatus",
    "AnalysisWindowRef",
    "Measurement",
    "ObservationIndexResult",
    "TemporalIndexComparison",
]
