"""Active tool context for REPL tool-to-tool composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolContext:
    tools: Mapping[str, Callable] = field(default_factory=dict)
    hidden_tools: Mapping[str, Callable] = field(default_factory=dict)


_CURRENT_TOOL_CONTEXT: ContextVar[ToolContext | None] = ContextVar(
    "rlmflow_tool_context",
    default=None,
)


def current_tool_context() -> ToolContext:
    ctx = _CURRENT_TOOL_CONTEXT.get()
    if ctx is None:
        raise RuntimeError("No active RLMFlow tool context")
    return ctx


def set_tool_context(ctx: ToolContext):
    return _CURRENT_TOOL_CONTEXT.set(ctx)


def reset_tool_context(token) -> None:
    _CURRENT_TOOL_CONTEXT.reset(token)


def get_repl_tools(*, include_hidden: bool = False) -> dict[str, Callable]:
    ctx = current_tool_context()
    tools = dict(ctx.tools)
    if include_hidden:
        tools.update(ctx.hidden_tools)
    return tools


__all__ = [
    "ToolContext",
    "current_tool_context",
    "get_repl_tools",
    "reset_tool_context",
    "set_tool_context",
]
