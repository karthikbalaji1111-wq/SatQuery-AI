"""Analysis boundary over an already-computed query execution.

Implemented so far: the contract itself (:class:`AnalysisRequest` ->
:class:`AnalysisResult`) plus a deterministic ``visualize`` summary. No analysis
engine exists yet; ``multimodal`` (fusion) and ``temporal`` (change detection)
remain the future homes for the engines this service will dispatch to.

Dependency direction is ``analysis -> query -> satellite``; nothing in ``query``
may import this package.
"""

from app.services.analysis.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisStatus,
    AnalysisWindowRef,
    Measurement,
)
from app.services.analysis.service import AnalysisService

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisService",
    "AnalysisStatus",
    "AnalysisWindowRef",
    "Measurement",
]
