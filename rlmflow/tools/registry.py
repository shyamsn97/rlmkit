"""Shared tool registry projections for runtime and REPL plumbing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from rlmflow.tools.context import ToolContext

SHOW_VARS_NAME = "SHOW_VARS"
LAUNCHER_TOOLS = frozenset({"launch_subagent", "launch_subagents"})
CONTROL_PROXY_TOOLS = frozenset({"done", "rlm_delegate", "rlm_wait"})
HIDDEN_REPL_TOOL_NAMES = frozenset({"rlm_delegate", "rlm_wait"})


@dataclass(frozen=True)
class ToolPartition:
    visible: dict[str, Any]
    hidden: dict[str, Any]

    def context(self) -> ToolContext:
        return ToolContext(tools=self.visible, hidden_tools=self.hidden)

    def names(self) -> dict[str, list[str]]:
        return {
            "visible": list(self.visible),
            "hidden": list(self.hidden),
        }


def partition_tool_defs(tool_defs: Iterable[Any]) -> ToolPartition:
    """Split registered ToolDef-like objects into visible/hidden callables."""

    visible: dict[str, Any] = {}
    hidden: dict[str, Any] = {}
    for td in tool_defs:
        fn = getattr(td, "fn", None)
        if fn is None:
            continue
        name = getattr(td, "name")
        if getattr(td, "hidden", False):
            hidden[name] = fn
        else:
            visible[name] = fn
    return ToolPartition(visible=visible, hidden=hidden)


def partition_repl_namespace(
    namespace: Mapping[str, Any],
    *,
    visible_names: set[str] | None = None,
    hidden_names: set[str] | None = None,
) -> ToolPartition:
    """Build the active REPL tool map from registered names and namespace values."""

    if visible_names is not None or hidden_names is not None:
        visible = {
            name: namespace[name]
            for name in (visible_names or set())
            if callable(namespace.get(name))
        }
        hidden = {
            name: namespace[name]
            for name in (hidden_names or set())
            if callable(namespace.get(name))
        }
        return ToolPartition(visible=visible, hidden=hidden)

    visible = {}
    hidden = {}
    for name, value in namespace.items():
        if name.startswith("_") or name == SHOW_VARS_NAME or not callable(value):
            continue
        if name in HIDDEN_REPL_TOOL_NAMES:
            hidden[name] = value
        else:
            visible[name] = value
    return ToolPartition(visible=visible, hidden=hidden)


def public_tool_defs(
    tool_defs: Iterable[Any], *, include_hidden: bool = False
) -> list[Any]:
    tools = list(tool_defs)
    if include_hidden:
        return tools
    return [td for td in tools if not getattr(td, "hidden", False)]


__all__ = [
    "CONTROL_PROXY_TOOLS",
    "HIDDEN_REPL_TOOL_NAMES",
    "LAUNCHER_TOOLS",
    "SHOW_VARS_NAME",
    "ToolPartition",
    "partition_repl_namespace",
    "partition_tool_defs",
    "public_tool_defs",
]
