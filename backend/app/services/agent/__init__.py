"""Agentic orchestration over the existing deterministic remote-sensing tools.

    question -> plan -> validated tool calls -> deterministic execution
             -> evidence -> grounded answer

Implemented so far: the contracts only (this commit). There is no planner, no
executor, no provider and no route yet.

The layer exists so that a language model may *choose* which of the existing
deterministic analyses to run, while the server keeps every guarantee: the tool
set is closed, parameters are validated against the existing domain models, and
execution happens through the same services the manual endpoints use. The model
selects; it never computes.

Deliberately NOT here, and not planned for this layer: image input to any model,
vision-language claims, fusion, co-registration, change detection, or any
model-generated code. A future remote-sensing model is intended to arrive as an
additional evidence-producing tool (its reserved home is ``services/multimodal``)
without changing these contracts - which is why :data:`EvidenceSource` already
carries a reserved ``"model"`` value.

Dependency direction is ``api -> agent -> {analysis, query} -> satellite``;
nothing in ``analysis``, ``query``, ``satellite`` or ``core`` may import this
package.
"""

from app.services.agent.schemas import (
    AgentEvidence,
    AgentPlan,
    AgentQuestionRequest,
    AgentResult,
    AgentStatus,
    AgentToolStep,
    AgentTrace,
    AnswerValidation,
    EvidenceItem,
    EvidenceSource,
    ExecuteQueryParams,
    NdwiParams,
    TemporalNdwiParams,
    ToolCall,
    ToolName,
    ToolStepStatus,
    ValidationOutcome,
)

__all__ = [
    "AgentEvidence",
    "AgentPlan",
    "AgentQuestionRequest",
    "AgentResult",
    "AgentStatus",
    "AgentToolStep",
    "AgentTrace",
    "AnswerValidation",
    "EvidenceItem",
    "EvidenceSource",
    "ExecuteQueryParams",
    "NdwiParams",
    "TemporalNdwiParams",
    "ToolCall",
    "ToolName",
    "ToolStepStatus",
    "ValidationOutcome",
]
