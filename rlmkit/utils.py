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


def find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks wrapped in triple backticks."""
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def add_execution_result_to_messages(
    messages: list[dict[str, str]],
    code: str,
    result: str,
    max_character_length: int = 100000,
) -> list[dict[str, str]]:
    """Append a formatted execution result message."""
    if len(result) > max_character_length:
        result = result[:max_character_length] + "..."

    execution_message = {
        "role": "user",
        "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
    }
    messages.append(execution_message)
    return messages


def format_execution_result(
    stdout: str,
    stderr: str,
    locals_dict: dict[str, Any],
    truncate_length: int = 100,
) -> str:
    """Format execution output and a small variable summary."""
    result_parts = []

    if stdout:
        result_parts.append(f"\n{stdout}")

    if stderr:
        result_parts.append(f"\n{stderr}")

    important_vars = {}
    for key, value in locals_dict.items():
        if key.startswith("_") or key in {"__builtins__", "__name__", "__doc__"}:
            continue
        try:
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                if isinstance(value, str) and len(value) > truncate_length:
                    important_vars[key] = f"'{value[:truncate_length]}...'"
                else:
                    important_vars[key] = repr(value)
        except Exception:
            important_vars[key] = f"<{type(value).__name__}>"

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"
