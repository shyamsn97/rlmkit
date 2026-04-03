"""Base runtime — execute code, inject values, and work with files."""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..utils import get_tool_metadata
from ..utils import tool as tool_decorator


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

    Only ``execute`` and ``inject`` are abstract.

    File I/O helpers (read_file, write_file, etc.) have default
    implementations that work on the local filesystem. Override for
    sandboxed or remote runtimes.
    """

    def __init__(self, workspace: Path | str = ".") -> None:
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
        """Fresh runtime sharing the same workspace and tool registrations.

        Namespace is empty — injected values do NOT carry over.
        Tools are re-registered so the new instance can discover them.
        Subclasses should override if they have extra state to copy.
        """
        new = type(self)(workspace=self.workspace)
        for name, (fn, doc, core) in self.tools.items():
            new.tools[name] = (fn, doc, core)
            new.inject(name, fn)
        return new

    # ── file I/O (override for sandboxed runtimes) ───────────────────

    @tool_decorator(
        """Read a file and return its contents as a string.
Args:
- path (str): File path, relative to workspace or absolute.
Returns:
- str: The file contents."""
    )
    def read_file(self, path: str) -> str:
        return self.resolve_path(path).read_text()

    @tool_decorator(
        """Write content to a file, creating parent directories if needed.
Args:
- path (str): File path, relative to workspace or absolute.
- content (str): The full text to write.
Returns:
- str: Confirmation with byte count."""
    )
    def write_file(self, path: str, content: str) -> str:
        resolved = self.resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    @tool_decorator(
        """Append content to the end of a file, creating it if needed.
Args:
- path (str): File path, relative to workspace or absolute.
- content (str): The text to append.
Returns:
- str: Confirmation with byte count."""
    )
    def append_file(self, path: str, content: str) -> str:
        resolved = self.resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a") as f:
            f.write(content)
        return f"Appended {len(content)} bytes to {path}"

    @tool_decorator(
        """Apply find-and-replace edits to a file. Each edit is an (old, new) pair.
Args:
- path (str): File path, relative to workspace or absolute.
- *edits (tuple[str, str]): One or more (old_string, new_string) pairs. Each replaces the first occurrence.
Returns:
- str: Summary of how many edits were applied."""
    )
    def edit_file(self, path: str, *edits: tuple[str, str]) -> str:
        resolved = self.resolve_path(path)
        text = resolved.read_text()
        count = 0
        for old, new in edits:
            if old in text:
                text = text.replace(old, new, 1)
                count += 1
        resolved.write_text(text)
        return f"Applied {count}/{len(edits)} edits to {path}"

    @tool_decorator(
        """List files and directories at a path.
Args:
- path (str): Directory (or file) path, relative to workspace or absolute.
Returns:
- list[str]: Sorted list of entry names."""
    )
    def ls(self, path) -> list[str]:
        resolved = self.resolve_path(path)
        if resolved.is_file():
            return [resolved.name]
        return sorted(p.name for p in resolved.iterdir())

    @tool_decorator(
        """Search file contents for lines matching a regex pattern.
Args:
- pattern (str): A Python regex pattern.
- path (str): File or directory to search (default: workspace root).
- max_results (int): Stop after this many matches (default: 50).
Returns:
- str: Matching lines as "relpath:lineno: line", or empty string if none."""
    )
    def grep(self, pattern: str, path: str = ".", *, max_results: int = 50) -> str:
        resolved = self.resolve_path(path)
        regex = re.compile(pattern)
        matches: list[str] = []
        files = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if regex.search(line):
                        rel = f.relative_to(self.workspace)
                        matches.append(f"{rel}:{i}: {line}")
                        if len(matches) >= max_results:
                            return "\n".join(matches)
            except (UnicodeDecodeError, PermissionError):
                continue
        return "\n".join(matches)

    def resolve_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.workspace / p

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

    def register_builtins(self) -> None:
        """Register the file I/O tools. Call in subclass __init__."""
        for method in (
            self.read_file,
            self.write_file,
            self.append_file,
            self.edit_file,
            self.ls,
            self.grep,
        ):
            self.register_tool(method)

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
