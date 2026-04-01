"""Utility helpers for the RLM runtime."""

from __future__ import annotations

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
    """Raised when delegate(wait=False) is called without a matching wait_all()."""


def find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks wrapped in triple backticks."""
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results
