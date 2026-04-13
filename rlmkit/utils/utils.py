"""Utility helpers for the RLM runtime."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str


def _default_tool_name(name: str) -> str:
    return name[5:] if name.startswith("tool_") else name


def tool(description: str, *, name: str | None = None):
    """Mark a method as a discoverable tool."""

    def decorator(fn):
        setattr(
            fn,
            "__rlmkit_tool__",
            ToolMetadata(
                name=name or _default_tool_name(fn.__name__),
                description=description.strip(),
            ),
        )
        return fn

    return decorator


def get_tool_metadata(fn: Any) -> ToolMetadata | None:
    """Return tool metadata for a function or bound method if present."""
    target = getattr(fn, "__func__", fn)
    return getattr(target, "__rlmkit_tool__", None)


class OrphanedDelegatesError(RuntimeError):
    """Raised when delegate() is called without a matching yield wait()."""


_REPL_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```\s*(?:\n|$)", re.DOTALL)


def find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks wrapped in triple backticks."""
    return [m.group(1).strip() for m in _REPL_BLOCK_RE.finditer(text)]


def check_yield_errors(code: str) -> str | None:
    """Return an error string if any ``wait()`` calls lack ``yield``, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    yielded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Yield) and isinstance(node.value, ast.Call):
            yielded.add(id(node.value))

    errors = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "wait"
            and id(node) not in yielded
        ):
            errors.append(
                f"Line {node.lineno}: `wait(...)` must be prefixed with `yield`"
            )

    return "ERROR: " + "; ".join(errors) if errors else None
