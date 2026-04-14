"""Utility helpers for the RLM runtime."""

from __future__ import annotations

import ast
import re


class OrphanedDelegatesError(RuntimeError):
    """Raised when delegate() is called without a matching yield wait()."""


_REPL_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```\s*(?:\n|$)", re.DOTALL)


def find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks wrapped in triple backticks."""
    return [m.group(1).strip() for m in _REPL_BLOCK_RE.finditer(text)]


def replace_code_block(text: str, new_code: str) -> str:
    """Keep text up to and including the first ```repl``` block, drop the rest.

    The block's content is replaced with *new_code*.
    """
    m = _REPL_BLOCK_RE.search(text)
    if not m:
        return text
    return text[: m.start()] + f"```repl\n{new_code}\n```"


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
