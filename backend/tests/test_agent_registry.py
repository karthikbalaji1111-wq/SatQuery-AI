"""Phase 15 Commit 2 - tool registry tests.

The registry is an explicit capability allowlist, not a plugin system. It maps
a closed set of tool names to inert descriptors; it never holds a callable,
never imports anything dynamically, and never executes a service.

The security property under test is narrow and absolute: **there is no path
from a model-controlled string to Python execution.** A model emits a tool
name; the worst that name can do is fail to resolve.

Nothing here touches the network, a provider, or the filesystem.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
from typing import Any, get_args

import pytest
from app.core.errors import InvalidInputError
from app.services.agent import registry as registry_mod
from app.services.agent.registry import (
    REGISTERED_TOOLS,
    TOOL_REGISTRY,
    ToolSpec,
    is_registered,
    resolve_tool,
)
from app.services.agent.schemas import ToolName

APPROVED = {"execute_query", "ndwi_statistics", "temporal_ndwi_statistics"}

#: Capabilities that exist in the system but are deliberately NOT model-callable.
#: ``retrieve_imagery`` is a parameter on execute_query (the model never sees
#: the image); ``compatibility_report`` is an automatic byproduct of the
#: temporal tool; ``rs_model_analysis`` is a reserved future capability.
DELIBERATELY_UNREGISTERED = {
    "retrieve_imagery",
    "compatibility_report",
    "rs_model_analysis",
}


# =========================================================================== #
# A. The allowlist is exactly the approved set
# =========================================================================== #


def test_registry_exposes_exactly_the_three_approved_tools() -> None:
    assert set(TOOL_REGISTRY) == APPROVED
    assert frozenset(APPROVED) == REGISTERED_TOOLS


def test_registry_keys_match_the_contract_tool_names() -> None:
    """The allowlist and the discriminated union cannot drift apart."""

    assert set(TOOL_REGISTRY) == set(get_args(ToolName))


@pytest.mark.parametrize("name", sorted(APPROVED))
def test_each_approved_tool_resolves_to_a_known_operation(name: str) -> None:
    spec = resolve_tool(name)

    assert isinstance(spec, ToolSpec)
    assert spec.name == name
    assert spec.operation in {"discovery", "analysis"}
    assert spec.description


def test_execute_query_is_the_discovery_operation() -> None:
    spec = resolve_tool("execute_query")
    assert spec.operation == "discovery"
    # Discovery is not an analysis flag toggle.
    assert spec.analysis_flag is None


@pytest.mark.parametrize(
    "name,flag",
    [
        ("ndwi_statistics", "include_ndwi"),
        ("temporal_ndwi_statistics", "include_temporal_ndwi"),
    ],
)
def test_analysis_tools_declare_the_flag_they_map_to(name: str, flag: str) -> None:
    """The executor coalesces flags from here rather than hardcoding a map."""

    spec = resolve_tool(name)
    assert spec.operation == "analysis"
    assert spec.analysis_flag == flag


def test_analysis_flags_are_distinct() -> None:
    flags = [
        spec.analysis_flag
        for spec in TOOL_REGISTRY.values()
        if spec.analysis_flag is not None
    ]
    assert len(flags) == len(set(flags)) == 2


# =========================================================================== #
# B. Nothing outside the allowlist resolves
# =========================================================================== #


@pytest.mark.parametrize("name", sorted(DELIBERATELY_UNREGISTERED))
def test_deliberately_unregistered_capabilities_cannot_resolve(name: str) -> None:
    assert name not in TOOL_REGISTRY
    assert not is_registered(name)
    with pytest.raises(InvalidInputError):
        resolve_tool(name)


@pytest.mark.parametrize(
    "name",
    [
        "run_python",
        "shell",
        "read_file",
        "os.system",
        "__import__",
        "eval",
        "EXECUTE_QUERY",
        "execute_query ",
        "",
        "..",
        "app.services.satellite.imagery.ImageryService",
    ],
)
def test_unknown_names_cannot_resolve(name: str) -> None:
    assert not is_registered(name)
    with pytest.raises(InvalidInputError):
        resolve_tool(name)


@pytest.mark.parametrize("value", [None, 1, 1.5, [], {}, object()])
def test_non_string_names_cannot_resolve(value: Any) -> None:
    assert not is_registered(value)
    with pytest.raises(InvalidInputError):
        resolve_tool(value)


def test_resolution_failure_names_the_allowlist_not_the_input() -> None:
    with pytest.raises(InvalidInputError) as exc:
        resolve_tool("run_python")
    assert "execute_query" in exc.value.message


# =========================================================================== #
# C. The registry cannot become an execution vector
# =========================================================================== #


def test_registry_holds_no_callables() -> None:
    """A descriptor table, not a dispatch table - nothing here can be invoked."""

    for spec in TOOL_REGISTRY.values():
        for field_name in spec.__dataclass_fields__:
            value = getattr(spec, field_name)
            assert not callable(value), (
                f"{spec.name}.{field_name} is callable; the registry must hold "
                "inert descriptors only"
            )


def test_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        TOOL_REGISTRY["run_python"] = TOOL_REGISTRY["ndwi_statistics"]  # type: ignore[index]


def test_tool_specs_are_frozen() -> None:
    spec = resolve_tool("ndwi_statistics")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "execute_query"  # type: ignore[misc]


def test_registry_performs_no_dynamic_lookup_or_execution() -> None:
    """AST-level: no reflection, no dynamic import, no execution primitives."""

    tree = ast.parse(pathlib.Path(registry_mod.__file__).read_text())

    forbidden_calls = {
        "eval",
        "exec",
        "getattr",
        "setattr",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls, (
                f"registry.py calls {node.func.id}()"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"importlib", "subprocess"}
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in {"importlib", "subprocess"}


def test_registry_imports_no_service_and_no_provider() -> None:
    """The registry describes capabilities; it must not be able to run them."""

    tree = ast.parse(pathlib.Path(registry_mod.__file__).read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    for forbidden in ("google", "genai", "httpx", "rasterio", "numpy", "fastapi"):
        assert forbidden not in roots
    assert roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "types",
        "typing",
        "app",
    }
