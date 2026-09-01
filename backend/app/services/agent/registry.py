"""The tool allowlist: which capabilities a planner is permitted to select.

This is an explicit capability allowlist, deliberately not a plugin system.
There is no dynamic import, no reflection, no entry-point scanning and no
handler callable anywhere in this module. A tool name maps to an **inert
descriptor** describing what kind of operation it is; :mod:`executor` decides
what to do with that, and only the executor holds service handles.

The security property is narrow and absolute: **a model-controlled string can
never reach Python execution.** The worst an unrecognised name can do is fail
to resolve.

Registering a name here is the deliberate act of granting a language model
access to a capability. Three capabilities that exist in the system are
deliberately absent:

* ``retrieve_imagery`` - imagery is a *parameter* on ``execute_query``. The
  model never receives the image, so fetching one is a UI-directed action
  rather than a tool the model reasons with.
* ``compatibility_report`` - already produced automatically inside the temporal
  analysis and attached to its result. A separate tool would duplicate a code
  path and let a planner request it where no pair exists.
* ``rs_model_analysis`` - reserved for a future remote-sensing model. Not
  implemented, and deliberately not reachable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from app.core.errors import InvalidInputError
from app.services.agent.schemas import ToolName

#: ``discovery`` runs the query-execution pipeline; ``analysis`` interprets its
#: result. The distinction is what lets the executor coalesce every analysis
#: tool into a single ``AnalysisService.analyze`` call.
ToolOperation = Literal["discovery", "analysis"]

#: The ``AnalysisRequest`` flag an analysis tool maps to. Declared here so the
#: executor reads the mapping rather than hardcoding a second copy of it.
AnalysisFlag = Literal["include_ndwi", "include_temporal_ndwi"]


@dataclass(frozen=True)
class ToolSpec:
    """An inert description of one permitted capability.

    Deliberately holds no callable, no service handle and no import path - only
    data. A test asserts that every field is non-callable, which is what makes
    "the registry cannot execute anything" a checked property rather than a
    claim.
    """

    name: ToolName
    operation: ToolOperation
    #: ``None`` for the discovery operation; the flag to set for analysis tools.
    analysis_flag: AnalysisFlag | None
    description: str


_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="execute_query",
        operation="discovery",
        analysis_flag=None,
        description=(
            "Ground the location, discover Sentinel-1/Sentinel-2 scenes per "
            "temporal window, select one scene per window deterministically, "
            "and optionally retrieve bounded display imagery."
        ),
    ),
    ToolSpec(
        name="ndwi_statistics",
        operation="analysis",
        analysis_flag="include_ndwi",
        description=(
            "Single-scene Sentinel-2 NDWI statistics at native 10 m resolution, "
            "computed on raw digital numbers. Index statistics only - not a "
            "validated water or flood classification."
        ),
    ),
    ToolSpec(
        name="temporal_ndwi_statistics",
        operation="analysis",
        analysis_flag="include_temporal_ndwi",
        description=(
            "Temporal NDWI Statistics for one deterministic Sentinel-2 pair, "
            "each observation indexed independently and reported with its "
            "compatibility report. No pixels are compared and nothing is "
            "co-registered."
        ),
    ),
)

#: The allowlist. Read-only at runtime: a ``MappingProxyType`` cannot be
#: assigned into, so no code path can widen the model's capabilities.
TOOL_REGISTRY: Mapping[ToolName, ToolSpec] = MappingProxyType(
    {spec.name: spec for spec in _SPECS}
)

#: The permitted names, for membership checks and error messages.
REGISTERED_TOOLS: frozenset[str] = frozenset(TOOL_REGISTRY)


def is_registered(name: Any) -> bool:
    """Whether ``name`` is a permitted tool. Total: any input is safe to pass."""

    return isinstance(name, str) and name in TOOL_REGISTRY


def resolve_tool(name: Any) -> ToolSpec:
    """Return the descriptor for ``name``, or refuse.

    A plain dictionary lookup - never ``getattr``, never a dynamic import.
    Anything not on the allowlist raises :class:`InvalidInputError`, which the
    error handlers already map to 422.
    """

    if not is_registered(name):
        permitted = ", ".join(sorted(REGISTERED_TOOLS))
        raise InvalidInputError(
            f"Tool {name!r} is not available. Permitted tools: {permitted}."
        )
    return TOOL_REGISTRY[name]
