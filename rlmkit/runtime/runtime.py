"""Base runtime — execute code, inject values, register tools."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rlmkit.tools import get_tool_metadata
from rlmkit.tools import tool as tool_decorator

DEFAULT_MODULES: list[str] = [
    "re",
    "os",
    "json",
    "math",
    "collections",
    "itertools",
    "functools",
]


@dataclass
class ToolDef:
    name: str
    signature: str
    description: str
    fn: Callable | None = None
    core: bool = False

    @classmethod
    def from_fn(
        cls, fn: Callable, description: str | None = None, *, core: bool = False
    ) -> ToolDef:
        """Build a ``ToolDef`` from a function, preferring ``@tool`` metadata."""
        meta = get_tool_metadata(fn)
        name = meta.name if meta else fn.__name__.removeprefix("tool_")
        if description is None:
            description = (
                meta.description if meta else (fn.__doc__ or "").strip().split("\n")[0]
            )
        try:
            signature = str(inspect.signature(fn))
        except (TypeError, ValueError):
            signature = "(...)"
        return cls(
            name=name,
            signature=signature,
            description=description,
            fn=fn,
            core=core,
        )


class Runtime(ABC):
    """Execution environment for the agent.

    Only ``execute`` and ``inject`` are abstract.  Everything else
    has a default implementation or raises ``NotImplementedError``.
    """

    def __init__(self, workspace: str | Path = ".") -> None:
        self.workspace = Path(workspace).resolve()
        self.tools: dict[str, ToolDef] = {}

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
        for name, td in self.tools.items():
            new.tools[name] = td
            new.inject(name, td.fn)
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
        td = ToolDef.from_fn(fn, description, core=core)
        self.tools[td.name] = td
        self.inject(td.name, fn)

    def register_tools(self, tools: list[Callable]) -> None:
        """Register a list of tools."""
        for tool in tools:
            self.register_tool(tool)

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
        return list(self.tools.values())

    def available_modules(self) -> list[str]:
        """Modules already imported into the execution namespace."""
        return []
