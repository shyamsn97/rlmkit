"""Utility helpers for the RLM runtime."""

from __future__ import annotations

import ast
import re


class OrphanedDelegatesError(RuntimeError):
    """Raised when rlm_delegate() is called without a matching yield rlm_wait()."""


_REPL_OPEN_RE = re.compile(r"```repl[ \t]*\n")
_REPL_CLOSE_RE = re.compile(r"\n?```[ \t]*(?:\n|$)")


def find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks, handling nested triple-backtick content.

    Uses greedy matching per block so that markdown fences inside
    Python strings (e.g. ```bash ... ```) don't prematurely close
    the repl block.
    """
    blocks: list[str] = []
    pos = 0
    while True:
        opening = _REPL_OPEN_RE.search(text, pos)
        if not opening:
            break
        code_start = opening.end()
        # Greedy: find the *last* closing fence after the opening
        last_close = None
        for m in _REPL_CLOSE_RE.finditer(text, code_start):
            last_close = m
        if last_close is None:
            break
        blocks.append(text[code_start : last_close.start()].strip())
        pos = last_close.end()
    return blocks


def replace_code_block(text: str, new_code: str) -> str:
    """Keep text up to and including the first ```repl``` block, drop the rest.

    The block's content is replaced with *new_code*.
    """
    opening = _REPL_OPEN_RE.search(text)
    if not opening:
        return text
    code_start = opening.end()
    last_close = None
    for m in _REPL_CLOSE_RE.finditer(text, code_start):
        last_close = m
    if last_close is None:
        return text
    return text[: opening.start()] + f"```repl\n{new_code}\n```"


def check_yield_errors(code: str) -> str | None:
    """Return an error string if any ``rlm_wait()`` calls lack ``yield``, else None.

    A ``rlm_wait(...)`` call is considered yielded if it appears anywhere inside
    the expression tree of a ``Yield`` node. This permits idiomatic forms like::

        results = yield rlm_wait(*handles) if handles else []
        results = yield (rlm_wait(h1), rlm_wait(h2))

    where the syntactic value of the ``Yield`` is an ``IfExp`` / ``Tuple`` /
    parenthesized expression rather than a bare ``Call``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    yielded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Yield) and node.value is not None:
            for sub in ast.walk(node.value):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "rlm_wait"
                ):
                    yielded.add(id(sub))

    errors = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "rlm_wait"
            and id(node) not in yielded
        ):
            errors.append(
                f"Line {node.lineno}: `rlm_wait(...)` must be prefixed with `yield`"
            )

    return "ERROR: " + "; ".join(errors) if errors else None
