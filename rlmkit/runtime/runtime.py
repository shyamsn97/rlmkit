"""Base runtime — execute code, inject values, register tools."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rlmkit.tools import get_tool_metadata
from rlmkit.tools import tool as tool_decorator


@dataclass
class ToolDef:
    name: str
    signature: str
    description: str
    core: bool = False


def resolve_tool_name(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is not None:
        return meta.name
    name = fn.__name__
    return name[5:] if name.startswith("tool_") else name


def resolve_tool_description(fn: Callable, override: str | None = None) -> str:
    if override is not None:
        return override
    meta = get_tool_metadata(fn)
    if meta is not None:
        return meta.description
    return (fn.__doc__ or "").strip().split("\n")[0]


def resolve_tool_signature(fn: Callable) -> str:
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "(...)"


class Runtime(ABC):
    """Execution environment for the agent.

    Only ``execute`` and ``inject`` are abstract.  Everything else
    has a default implementation or raises ``NotImplementedError``.
    """

    def __init__(self, workspace: str | Path = ".") -> None:
        self.workspace = Path(workspace).resolve()
        self.tools: dict[str, tuple[Callable, str, bool]] = {}

    # ── required ─────────────────────────────────────────────────────

    @abstractmethod
    def execute(self, code: str, timeout: float | None = None) -> str:
        """Run code and return output as a string."""

    @abstractmethod
    def inject(self, name: str, value: Any) -> None:
        """Inject a named value into the runtime namespace."""

    # ── generator-based execution (override for remote runtimes) ────

    def start_code(self, code: str) -> tuple[bool, object]:
        """Execute a code block that may suspend at yield points.

        Returns ``(True, WaitRequest)`` if the code yielded, or
        ``(False, stdout_str)`` when it ran to completion.
        """
        raise NotImplementedError

    def resume_code(self, send_value=None) -> tuple[bool, object]:
        """Resume a suspended code block with child results.

        Same return convention as :meth:`start_code`.
        """
        raise NotImplementedError

    # ── clone ─────────────────────────────────────────────────────────

    def clone(self) -> Runtime:
        """Fresh runtime sharing the same tool registrations and workspace.

        Namespace is empty — injected values do NOT carry over.
        Tools are re-registered so the new instance can discover them.
        Subclasses should override if they have extra state to copy.
        """
        new = type(self)(workspace=self.workspace)
        for name, (fn, doc, core) in self.tools.items():
            new.tools[name] = (fn, doc, core)
            new.inject(name, fn)
        return new

    # ── tool registration ────────────────────────────────────────────

    def register_tool(
        self,
        fn: Callable,
        description: str | None = None,
        *,
        core: bool = False,
    ) -> None:
        """Register a function as a tool — injects it and makes it discoverable."""
        name = resolve_tool_name(fn)
        doc = resolve_tool_description(fn, description)
        self.tools[name] = (fn, doc, core)
        self.inject(name, fn)

    def register_tools(self, tools: list[Callable]) -> None:
        """Register a list of tools."""
        for tool in tools:
            self.register_tool(tool)
        return tools

    def tool(self, description: str, *, name: str | None = None, core: bool = False):
        """Decorator that registers a function as a tool on this runtime.

        Usage::

            @runtime.tool("Run a shell command.")
            def shell(cmd: str) -> str:
                ...
        """

        def decorator(fn: Callable) -> Callable:
            fn = tool_decorator(description, name=name)(fn)
            self.register_tool(fn, core=core)
            return fn

        return decorator

    def get_tool_defs(self) -> list[ToolDef]:
        return [
            ToolDef(
                name=n, signature=resolve_tool_signature(fn), description=doc, core=c
            )
            for n, (fn, doc, c) in self.tools.items()
        ]

    def available_modules(self) -> list[str]:
        """Modules already imported into the execution namespace."""
        return []
