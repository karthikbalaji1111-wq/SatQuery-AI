"""The planning boundary: a question in, a validated plan out.

    question -> AgentPlanner.plan() -> AgentPlan

:class:`AgentPlanner` is the provider-neutral abstraction. **This module
contains no provider SDK import and never will** - concrete providers live in
:mod:`app.services.agent.providers`, and an AST test enforces that the SDK
appears in exactly one file of the agent package.

A planner *proposes*; it does not act. It holds no service handle, cannot reach
the network except through its own provider, and returns only a parsed
:class:`AgentPlan`. Raw provider text, dictionaries, free-form tool names and
model reasoning all stop at this boundary: whatever a model emits is validated
through the closed discriminated union before it becomes a plan, so an
unrecognised tool or a malformed parameter is a failure rather than something
downstream has to defend against.

There is deliberately no field anywhere in this path for chain-of-thought. A
planner returns what it proposes to do, never an account of why.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.agent.schemas import AgentPlan, ExecuteQueryParams


class AgentPlanner(ABC):
    """Turns a natural-language question into a validated :class:`AgentPlan`.

    Implementations must not leak provider concepts through this interface -
    swapping one planner for another must not change anything downstream. The
    contract is deliberately tiny: one question in, one validated plan out, and
    no other capability.
    """

    @abstractmethod
    async def plan(self, question: str) -> AgentPlan:
        """Propose a validated plan for ``question``.

        Implementations raise rather than inventing a plan when they cannot
        produce one: a planner failure must stay a planner failure, so that the
        caller can report it honestly instead of executing something the user
        never asked for.
        """


#: The plan a planner proposes when nothing more specific is configured:
#: discovery only, no analysis. Chosen because it is the least the system can
#: do that is still useful, and because it asserts no analytical intent the
#: caller did not express.
_DEFAULT_PLAN = AgentPlan(
    steps=[
        ExecuteQueryParams.model_validate(
            {
                "tool": "execute_query",
                "intent": {
                    "location_query": "Chennai",
                    "temporal_mode": "single",
                    "time_windows": [
                        {"start_date": "2024-01-01", "end_date": "2024-01-31"}
                    ],
                    "modalities": ["sentinel-2-optical"],
                    "task": "visualize",
                },
            }
        )
    ]
)


class MockAgentPlanner(AgentPlanner):
    """TEST / DEVELOPMENT ONLY - performs no language understanding.

    Returns a fixed, already-validated plan regardless of the question, so the
    agent pipeline can be exercised with no provider and no network. It does
    **not** simulate model reasoning and does not inspect the question at all;
    pretending otherwise would make tests pass for reasons that have nothing to
    do with how a real planner behaves.

    Each call returns an independent copy, so a caller that mutates a plan
    cannot corrupt the fixture for the next one.
    """

    is_mock = True

    def __init__(self, plan: AgentPlan | None = None) -> None:
        self._plan = plan if plan is not None else _DEFAULT_PLAN

    async def plan(self, question: str) -> AgentPlan:
        # The question is intentionally ignored - this mock does no planning.
        del question
        return self._plan.model_copy(deep=True)
