"""Base runtime — execute code, inject values, and work with files."""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rlmkit.utils import get_tool_metadata


@dataclass
class ToolDef:
    name: str
    signature: str
    description: str
    core: bool = False


def _tool_name(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is not None:
        return meta.name
    name = fn.__name__
    return name[5:] if name.startswith("tool_") else name


def _tool_description(fn: Callable, override: str | None) -> str:
    if override is not None:
        return override
    meta = get_tool_metadata(fn)
    if meta is not None:
        return meta.description
    return (fn.__doc__ or "").strip().split("\n")[0]


def _tool_signature(fn: Callable) -> str:
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
        self._tools: dict[str, tuple[Callable, str, bool]] = {}

    # ── required ─────────────────────────────────────────────────────

    @abstractmethod
    def execute(self, code: str, timeout: float | None = None) -> str:
        """Run code and return output as a string."""

    @abstractmethod
    def inject(self, name: str, value: Any) -> None:
        """Inject a named value into the runtime namespace."""

    # ── file I/O (override for sandboxed runtimes) ───────────────────

    def read_file(self, path: str) -> str:
        return self._resolve(path).read_text()

    def write_file(self, path: str, content: str) -> str:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    def append_file(self, path: str, content: str) -> str:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a") as f:
            f.write(content)
        return f"Appended {len(content)} bytes to {path}"

    def edit_file(self, path: str, *edits: tuple[str, str]) -> str:
        resolved = self._resolve(path)
        text = resolved.read_text()
        count = 0
        for old, new in edits:
            if old in text:
                text = text.replace(old, new, 1)
                count += 1
        resolved.write_text(text)
        return f"Applied {count}/{len(edits)} edits to {path}"

    def ls(self, path: str = ".") -> list[str]:
        resolved = self._resolve(path)
        if resolved.is_file():
            return [resolved.name]
        return sorted(p.name for p in resolved.iterdir())

    def grep(self, pattern: str, path: str = ".", *, max_results: int = 50) -> str:
        resolved = self._resolve(path)
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
        return "\n".join(matches) if matches else "(no matches)"

    def _resolve(self, path: str) -> Path:
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
        name = _tool_name(fn)
        doc = _tool_description(fn, description)
        self._tools[name] = (fn, doc, core)
        self.inject(name, fn)

    def tool(self, description: str, *, name: str | None = None, core: bool = False):
        """Decorator that registers a function as a tool on this runtime.

        Usage::

            @runtime.tool("Run a shell command.")
            def shell(cmd: str) -> str:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            from rlmkit.utils import tool as tool_decorator
            fn = tool_decorator(description, name=name)(fn)
            self.register_tool(fn, core=core)
            return fn
        return decorator

    def register_builtins(self) -> None:
        """Register the file I/O tools. Call in subclass __init__."""
        self.register_tool(self.read_file, "Read a file's contents.")
        self.register_tool(self.write_file, "Write content to a file.")
        self.register_tool(self.append_file, "Append content to a file.")
        self.register_tool(self.edit_file, "Apply find-and-replace edits to a file.")
        self.register_tool(self.ls, "List files and directories.")
        self.register_tool(self.grep, "Search file contents with a regex.")

    def get_tool_defs(self) -> list[ToolDef]:
        return [
            ToolDef(name=n, signature=_tool_signature(fn), description=doc, core=c)
            for n, (fn, doc, c) in self._tools.items()
        ]
