"""Sandbox — code execution with generator-based suspension.

Handles stdout capture, generator wrapping, the yield/resume protocol,
and the JSON-over-stdio bridge used by remote runtimes.

Run as a container entrypoint::

    python -m rlmkit.runtime.sandbox
    python -m rlmkit.runtime.sandbox --workdir /repo

Reads JSON commands from stdin, writes JSON responses to stdout.

Commands
--------
``inject``
    Evaluate a value string in the namespace and bind it::

        {"cmd": "inject", "name": "X", "value": "42"}

``inject_proxy``
    Create a proxy function that forwards calls back to the engine
    over the protocol channel.  The engine handles the call and
    sends the result back::

        {"cmd": "inject_proxy", "name": "done"}

``run``
    Execute a code block (may suspend via ``yield``)::

        {"cmd": "run", "code": "..."}

    Response: ``{"suspended": true, "agent_ids": [...]}``
    or ``{"suspended": false, "output": "..."}``.

``resume``
    Resume a suspended generator::

        {"cmd": "resume", "value": <any>}
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import threading
from contextlib import contextmanager
from typing import Any, TextIO

from ..state import ChildHandle, WaitRequest

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
    """Convert rlmkit objects to JSON-safe dicts."""
    if isinstance(val, (ChildHandle, WaitRequest)):
        return val.to_dict()
    return val


def deserialize(val: Any) -> Any:
    """Reconstruct rlmkit objects from JSON dicts."""
    if isinstance(val, dict):
        if "child_handle" in val:
            return ChildHandle.from_dict(val)
        if "wait_request" in val:
            return WaitRequest.from_dict(val)
    return val


class Sandbox:
    """Execute code in a Python namespace with generator suspension.

    Wraps code in a generator function so ``yield`` suspends execution.
    Captures stdout via a thread-local proxy so parallel sandboxes
    running in a thread pool don't clobber each other's output.

    For local use, call :meth:`start`, :meth:`resume`, :meth:`execute`
    directly.  For remote containers, call :meth:`serve` to enter the
    JSON-over-stdio loop.

    Usage::

        sandbox = Sandbox()
        suspended, result = sandbox.start("print('hello')")
        # suspended=False, result='hello'

        suspended, result = sandbox.start("x = yield wait(h)")
        # suspended=True, result=<WaitRequest>

        suspended, result = sandbox.resume(["child result"])
        # suspended=False, result='...'
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
        return self.buf.getvalue().strip()

    def _wrap_generator(self, code: str, tree: ast.Module):
        """Wrap code in a generator function with auto-global declarations.

        Every assigned name gets a ``global`` declaration so variables
        persist in ``self.namespace`` across generator yields.
        """
        assigned = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        func = ast.parse("def __rlm_gen__(): pass").body[0]
        body: list[ast.stmt] = []
        if assigned:
            body.append(ast.Global(names=sorted(assigned)))
        body.extend(tree.body or [ast.Pass()])
        func.body = body
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, "<rlm>", "exec"), self.namespace)
        return self.namespace.pop("__rlm_gen__")

    def start(self, code: str) -> tuple[bool, object]:
        """Start executing a code block. Returns ``(suspended, result)``.

        If the code yields, returns ``(True, <yielded value>)``.
        Otherwise returns ``(False, stdout_string)``.
        """
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
            return False, self.buf.getvalue().strip()

        fn = self._wrap_generator(code, tree)
        self.gen = fn()
        return self.advance()

    def resume(self, send_value=None) -> tuple[bool, object]:
        """Resume a suspended generator. Same return convention as :meth:`start`."""
        return self.advance(send_value)

    def advance(self, send_value=None) -> tuple[bool, object]:
        """Drive the generator one step — yield suspends, StopIteration completes."""
        with self.captured():
            try:
                request = self.gen.send(send_value)
                return True, request
            except StopIteration:
                pass
        return False, self.buf.getvalue().strip()

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
            return deserialize(resp["value"])

        return proxy

    def format_result(self, suspended: bool, result: object) -> dict:
        if suspended:
            return {"suspended": True, "agent_ids": result.agent_ids}
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
            return {"ok": True}
        if cmd == "inject_proxy":
            self.namespace[msg["name"]] = self.make_proxy(msg["name"])
            return {"ok": True}
        return {"error": f"unknown command: {cmd}"}

    def serve(self) -> None:
        """Read JSON commands from stdin until EOF."""
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            msg = json.loads(line)
            self.write(self.handle(msg))


def main():
    parser = argparse.ArgumentParser(description="rlmkit sandbox")
    parser.add_argument("--workdir", default=None, help="chdir before starting")
    args = parser.parse_args()

    if args.workdir:
        import os

        os.chdir(args.workdir)

    sandbox = Sandbox()
    sandbox.serve()


if __name__ == "__main__":
    main()
