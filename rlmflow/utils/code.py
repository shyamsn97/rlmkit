"""Utility helpers for the RLM runtime."""

from __future__ import annotations

import ast
import re

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


def check_wait_syntax(code: str) -> str | None:
    """Return an error string for unsupported ``await`` syntax."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    checker = _WaitSyntaxChecker()
    checker.visit(tree)
    return "ERROR: " + "; ".join(checker.errors) if checker.errors else None


# The only calls an agent may ``await`` at action-block top level. The
# launchers are the public surface; ``rlm_wait`` is the internal primitive they
# compose over (kept awaitable for the engine's own replay path).
_AWAITABLE_CALLS = {"launch_subagent", "launch_subagents", "rlm_wait"}


def _is_awaitable_call(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _AWAITABLE_CALLS
    )


class _WaitSyntaxChecker(ast.NodeVisitor):
    boundary = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    comprehension = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def __init__(self) -> None:
        self.await_depth = 0
        self.errors: list[str] = []

    def _line(self, node: ast.AST) -> int | None:
        return getattr(node, "lineno", None)

    def _add(self, node: ast.AST, message: str) -> None:
        line = self._line(node)
        prefix = f"Line {line}: " if line is not None else ""
        self.errors.append(prefix + message)

    def visit_Await(self, node: ast.Await) -> None:  # noqa: N802
        if not _is_awaitable_call(node.value):
            self._add(
                node,
                "only `await launch_subagent(...)` / `await launch_subagents(...)` "
                "is supported",
            )
        self.await_depth += 1
        self.generic_visit(node)
        self.await_depth -= 1

    def visit_Yield(self, node: ast.Yield) -> None:  # noqa: N802
        self._add(
            node,
            "use `await launch_subagent(...)`; top-level `yield` is not supported",
        )

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:  # noqa: N802
        self._add(node, "top-level `yield from` is not supported")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if _is_awaitable_call(node) and self.await_depth == 0:
            name = node.func.id  # type: ignore[union-attr]
            self._add(node, f"`{name}(...)` must be awaited: `await {name}(...)`")
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._check_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._check_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._check_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
        self._check_comprehension(node)

    def _check_comprehension(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Await) or _is_awaitable_call(child):
                self._add(
                    node,
                    "`await launch_subagent(...)` is not supported in comprehensions",
                )
                return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._check_nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._check_nested(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._check_nested(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._check_nested(node)

    def _check_nested(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Await) or _is_awaitable_call(child):
                self._add(
                    node,
                    "`launch_subagent(...)` is only supported at action-block "
                    "top level",
                )
                return
