"""Tool decorator, metadata, and bundled tool collections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str


def _default_tool_name(name: str) -> str:
    return name[5:] if name.startswith("tool_") else name


def tool(description: str, *, name: str | None = None):
    """Mark a function as a discoverable tool."""

    def decorator(fn):
        setattr(
            fn,
            "_tool_meta",
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
    return getattr(target, "_tool_meta", None)


from rlmkit.tools.filesystem import FILE_TOOLS  # noqa: E402

__all__ = ["FILE_TOOLS", "ToolMetadata", "get_tool_metadata", "tool"]
