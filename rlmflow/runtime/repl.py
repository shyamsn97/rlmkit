"""REPL — code execution with generator-based suspension.

Handles stdout capture, generator wrapping, the yield/resume protocol,
and the JSON-over-stdio bridge used by remote runtimes.

Container entrypoint: ``python -m rlmflow.runtime.repl [--workdir DIR]``.
Reads JSON commands from stdin, writes JSON responses to stdout.

Commands:
  - ``{"cmd": "inject",              "name": N, "value": expr}``       bind ``N = eval(expr)``
  - ``{"cmd": "inject_proxy",        "name": N}``                      bind ``N`` as proxy fn
  - ``{"cmd": "inject_object_proxy", "name": N, "methods": [...]}``    bind ``N`` with each method as a proxy fn
  - ``{"cmd": "run",                 "code": src}``                    exec; may suspend
  - ``{"cmd": "resume",              "value": v}``                     resume suspended gen

Responses for run/resume: ``{"suspended": true, "agent_ids": [...]}`` or
``{"suspended": false, "output": "..."}``.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import threading
import types
from contextlib import contextmanager
from typing import Any, TextIO

from rlmflow.node import ChildHandle, WaitRequest

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


class _AnnotationStripper(ast.NodeTransformer):
    """Rewrite ``x: T = v`` to ``x = v`` on bare names, and drop ``x: T``.

    Needed because the generator wrapper declares assigned names ``global``,
    and Python forbids a name being annotated and ``global`` in the same
    function scope. We stop at nested ``def`` / ``async def`` / ``class``
    because those open their own scopes.
    """

    def visit_FunctionDef(self, node):  # noqa: N802
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_AnnAssign(self, node):  # noqa: N802
        if not isinstance(node.target, ast.Name):
            return node
        if node.value is None:
            return None
        new = ast.Assign(targets=[node.target], value=node.value)
        return ast.copy_location(new, node)


def _strip_name_annotations(stmt: ast.stmt) -> ast.stmt | None:
    return _AnnotationStripper().visit(stmt)


# ── thread-safe stdout capture ────────────────────────────────────────

_capture = threading.local()


class _StdoutProxy:
    """Thread-aware stdout: routes to a per-thread buffer when one is active."""

    def __init__(self, real):
        self.real = real

    def write(self, s):
        buf = getattr(_capture, "buf", None)
        return buf.write(s) if buf is not None else self.real.write(s)

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


def serialize(val: Any) -> Any:
    """Convert rlmflow objects to JSON-safe structures. Recurses into lists/dicts."""
    if isinstance(val, (ChildHandle, WaitRequest)):
        return val.to_dict()
    if isinstance(val, (list, tuple)):
        return [serialize(v) for v in val]
    if isinstance(val, dict):
        return {k: serialize(v) for k, v in val.items()}
    return val


def deserialize(val: Any) -> Any:
    """Reconstruct rlmflow objects from JSON structures. Recurses into lists/dicts."""
    if isinstance(val, list):
        return [deserialize(v) for v in val]
    if isinstance(val, dict):
        if "child_handle" in val:
            return ChildHandle.from_dict(val)
        if "wait_request" in val:
            return WaitRequest.from_dict(val)
        return {k: deserialize(v) for k, v in val.items()}
    return val


class REPL:
    """Execute code in a Python namespace with generator suspension.

    Wraps code in a generator function so ``yield`` suspends execution.
    Captures stdout via a thread-local proxy so parallel REPLs running in
    a thread pool don't clobber each other's output.

    Local use: :meth:`start` / :meth:`resume` / :meth:`execute`.
    Remote containers: :meth:`serve` enters the JSON-over-stdio loop.
    """

    def __init__(
        self,
        namespace: dict[str, Any] | None = None,
        protocol_out: TextIO | None = None,
    ) -> None:
        self.namespace = namespace or {"__builtins__": __builtins__}
        self.protocol_out = protocol_out or sys.stdout
        self.gen = None
        self.buf: io.StringIO | None = None
        if not isinstance(sys.stdout, _StdoutProxy):
            sys.stdout = _StdoutProxy(sys.stdout)

    # ── code execution ────────────────────────────────────────────────

    @contextmanager
    def captured(self):
        """Activate the thread-local buffer for stdout capture."""
        _capture.buf = self.buf
        try:
            yield
        except Exception as exc:
            self.buf.write(f"\n{type(exc).__name__}: {exc}")
        finally:
            _capture.buf = None

    def execute(self, code: str) -> str:
        """Execute code without generator support. Returns stdout."""
        self.buf = io.StringIO()
        with self.captured():
            exec(code, self.namespace)
        return strip_ansi(self.buf.getvalue().strip())

    def _wrap_generator(self, code: str, tree: ast.Module):
        """Wrap ``code`` in a generator fn; ``global`` every assigned name so
        variables persist in ``self.namespace`` across yields.

        ``from X import *`` is illegal inside functions, so any such
        statements are hoisted out and exec'd at module level first;
        the imported names then live on ``self.namespace`` and are
        visible to the generator via globals.

        ``AnnAssign`` on a bare name (``x: int = 1``) is rewritten to a
        plain ``Assign`` because Python forbids a name being both
        ``global`` and annotated in the same scope.
        """
        star_imports: list[ast.stmt] = []
        body_stmts: list[ast.stmt] = []
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and any(
                a.name == "*" for a in stmt.names
            ):
                star_imports.append(stmt)
            else:
                body_stmts.append(_strip_name_annotations(stmt))
        body_stmts = [s for s in body_stmts if s is not None]

        if star_imports:
            imports_mod = ast.Module(body=star_imports, type_ignores=[])
            ast.fix_missing_locations(imports_mod)
            exec(compile(imports_mod, "<rlm-imports>", "exec"), self.namespace)

        assigned = {
            node.id
            for stmt in body_stmts
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        func = ast.parse("def __rlm_gen__(): pass").body[0]
        body: list[ast.stmt] = []
        if assigned:
            body.append(ast.Global(names=sorted(assigned)))
        body.extend(body_stmts or [ast.Pass()])
        func.body = body
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, "<rlm>", "exec"), self.namespace)
        return self.namespace.pop("__rlm_gen__")

    def start(self, code: str) -> tuple[bool, object]:
        """Execute ``code``. Returns ``(True, yielded)`` or ``(False, stdout)``."""
        self.buf = io.StringIO()
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.buf.write(f"\nSyntaxError: {exc}")
            return False, self.buf.getvalue().strip()

        has_yield = any(
            isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(tree)
        )

        if not has_yield:
            with self.captured():
                exec(code, self.namespace)
            return False, strip_ansi(self.buf.getvalue().strip())

        fn = self._wrap_generator(code, tree)
        self.gen = fn()
        return self.advance()

    def resume(self, send_value=None) -> tuple[bool, object]:
        """Resume a suspended generator. Same return convention as :meth:`start`."""
        self.buf = io.StringIO()
        return self.advance(send_value)

    def advance(self, send_value=None) -> tuple[bool, object]:
        """Drive the generator one step. Yield → suspend, StopIteration → done."""
        with self.captured():
            try:
                request = self.gen.send(send_value)
                pre_output = strip_ansi(self.buf.getvalue().strip())
                return True, (request, pre_output)
            except StopIteration:
                pass
        return False, strip_ansi(self.buf.getvalue().strip())

    # ── JSON-over-stdio protocol (remote containers) ──────────────────

    def write(self, msg: dict) -> None:
        self.protocol_out.write(json.dumps(msg) + "\n")
        self.protocol_out.flush()

    def make_proxy(self, name: str):
        """Create a function that forwards calls to the engine over stdio."""
        protocol_out = self.protocol_out

        def proxy(*args, **kwargs):
            msg = {
                "proxy": name,
                "args": [serialize(a) for a in args],
                "kwargs": {k: serialize(v) for k, v in kwargs.items()},
            }
            protocol_out.write(json.dumps(msg) + "\n")
            protocol_out.flush()
            line = sys.stdin.readline()
            resp = json.loads(line)
            if "error" in resp:
                raise RuntimeError(resp["error"])
            return deserialize(resp["value"])

        return proxy

    def format_result(self, suspended: bool, result: object) -> dict:
        if suspended:
            request, pre_output = result
            resp = {"suspended": True, "agent_ids": request.agent_ids}
            if pre_output:
                resp["pre_output"] = pre_output
            return resp
        return {"suspended": False, "output": result}

    def handle(self, msg: dict) -> dict:
        """Process a single command and return the response dict."""
        cmd = msg["cmd"]
        if cmd == "run":
            return self.format_result(*self.start(msg["code"]))
        if cmd == "resume":
            return self.format_result(*self.resume(send_value=msg.get("value")))
        if cmd == "inject":
            self.namespace[msg["name"]] = eval(msg["value"], self.namespace)
        elif cmd == "inject_proxy":
            self.namespace[msg["name"]] = self.make_proxy(msg["name"])
        elif cmd == "inject_object_proxy":
            name = msg["name"]
            obj = types.SimpleNamespace()
            for method in msg["methods"]:
                setattr(obj, method, self.make_proxy(f"{name}.{method}"))
            self.namespace[name] = obj
        else:
            return {"error": f"unknown command: {cmd}"}
        return {"ok": True}

    def serve(self) -> None:
        """Read JSON commands from stdin until EOF."""
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            msg = json.loads(line)
            self.write(self.handle(msg))


def main():
    import os

    parser = argparse.ArgumentParser(description="rlmflow repl")
    parser.add_argument("--workdir", help="chdir before starting")
    args = parser.parse_args()
    if args.workdir:
        os.chdir(args.workdir)
    REPL().serve()


if __name__ == "__main__":
    main()
