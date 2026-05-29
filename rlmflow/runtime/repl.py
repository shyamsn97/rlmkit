"""REPL — code execution with top-level-await suspension.

Handles stdout capture, coroutine suspension, the await/resume protocol,
and the JSON-over-stdio bridge used by remote runtimes.

Container entrypoint: ``python -m rlmflow.runtime.repl [--workdir DIR]``.
Reads JSON commands from stdin, writes JSON responses to stdout.

Commands:
  - ``{"cmd": "inject",              "name": N, "value": expr}``       bind ``N = eval(expr)``
  - ``{"cmd": "inject_proxy",        "name": N}``                      bind ``N`` as proxy fn
  - ``{"cmd": "inject_launcher",     "name": N}``                      bind launcher from proxy primitives
  - ``{"cmd": "inject_object_proxy", "name": N, "methods": [...]}``    bind ``N`` with each method as a proxy fn
  - ``{"cmd": "read",                "name": N}``                      returns ``{"value": namespace.get(N)}``
  - ``{"cmd": "run",                 "code": src}``                    exec; may suspend
  - ``{"cmd": "resume",              "value": v}``                     resume suspended block

Responses for run/resume: ``{"suspended": true, "agent_ids": [...]}`` or
``{"suspended": false, "output": "..."}``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import io
import itertools
import json
import linecache
import os
import re
import sys
import threading
import traceback
import types
from contextlib import contextmanager
from typing import Any, TextIO

from rlmflow.graph import ChildHandle, WaitRequest
from rlmflow.tools.builtins import DoneSignal
from rlmflow.tools.context import ToolContext, reset_tool_context, set_tool_context
from rlmflow.utils.code import check_wait_syntax

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_THIS_FILE = os.path.normpath(__file__)
_REPL_ID_COUNTER = itertools.count(1)
_HIDDEN_REPL_NAMES = {"rlm_delegate", "rlm_wait"}


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _register_source(filename: str, code: str) -> None:
    """Register `code` under `filename` so traceback can render source lines.

    `linecache.cache` is the same hook real Python files use; once a frame's
    filename is in there, ``traceback.format_exception`` will include the
    actual source line for each frame (e.g. ``    return x['key']``).
    """
    lines = code.splitlines(keepends=True)
    linecache.cache[filename] = (len(code), None, lines, filename)


def _format_repl_traceback(exc: BaseException) -> str:
    """Format `exc`'s traceback, hiding `repl.py`'s own machinery frames.

    The agent only cares about frames in its own code (``<rlm:...>``) and in
    tool implementations. Frames from ``rlmflow/runtime/repl.py`` itself
    (``self.gen.send(...)``, ``exec(...)``) are pure noise — strip them.
    """
    te = traceback.TracebackException.from_exception(exc)
    te.stack = traceback.StackSummary.from_list(
        [frame for frame in te.stack if os.path.normpath(frame.filename) != _THIS_FILE]
    )
    return "".join(te.format()).rstrip()


def _has_top_level_await(tree: ast.AST) -> bool:
    """True iff ``tree`` has top-level ``await`` outside nested scopes."""

    boundary = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Await):
            return True
        for child in ast.iter_child_nodes(node):
            if isinstance(child, boundary):
                continue
            stack.append(child)
    return False


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
    """Execute code in a Python namespace with top-level-await suspension.

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
        self.coro = None
        self.buf: io.StringIO | None = None
        self.errored: bool = False
        self._visible_tool_names: set[str] | None = None
        self._hidden_tool_names: set[str] | None = None
        self._repl_id = next(_REPL_ID_COUNTER)
        self._src_counter = 0
        if not isinstance(sys.stdout, _StdoutProxy):
            sys.stdout = _StdoutProxy(sys.stdout)

    def _show_vars(self) -> dict[str, str]:
        """Return public REPL names and their type names without dumping values."""

        return {
            name: type(value).__name__
            for name, value in sorted(self.namespace.items())
            if not name.startswith("_")
            and name not in _HIDDEN_REPL_NAMES
            and name != "SHOW_VARS"
        }

    def _tool_context(self) -> ToolContext:
        if self._visible_tool_names is not None or self._hidden_tool_names is not None:
            visible_names = self._visible_tool_names or set()
            hidden_names = self._hidden_tool_names or set()
            return ToolContext(
                tools={
                    name: self.namespace[name]
                    for name in visible_names
                    if callable(self.namespace.get(name))
                },
                hidden_tools={
                    name: self.namespace[name]
                    for name in hidden_names
                    if callable(self.namespace.get(name))
                },
            )

        visible = {}
        hidden = {}
        for name, value in self.namespace.items():
            if name.startswith("_") or name == "SHOW_VARS" or not callable(value):
                continue
            if name in _HIDDEN_REPL_NAMES:
                hidden[name] = value
            else:
                visible[name] = value
        return ToolContext(tools=visible, hidden_tools=hidden)

    # ── code execution ────────────────────────────────────────────────

    @contextmanager
    def captured(self):
        """Activate the thread-local buffer for stdout capture.

        Catches three classes of bubble-up:

        * :class:`DoneSignal` — terminal control flow from ``done()``;
          the engine reads ``DONE_RESULT`` from ``runtime.env`` after.
        * :class:`Exception` — any normal runtime error from agent code
          becomes a captured traceback + ``errored=True`` for the next
          turn to read.
        * :class:`SystemExit` / :class:`KeyboardInterrupt` — these
          inherit from ``BaseException`` (not ``Exception``), so a
          stray ``raise SystemExit(...)`` or ``sys.exit(...)`` inside
          an agent's REPL would escape the engine and **exit the host
          process**. We treat them like any other agent error: format
          a traceback, set ``errored``, and resume the engine.
          ``BaseException`` itself is *not* caught — that would also
          swallow ``GeneratorExit`` etc., which we want to bubble.
        """
        _capture.buf = self.buf
        token = set_tool_context(self._tool_context())
        try:
            yield
        except DoneSignal:
            pass
        except (SystemExit, KeyboardInterrupt) as exc:
            self.buf.write("\n" + _format_repl_traceback(exc))
            self.errored = True
        except Exception as exc:
            self.buf.write("\n" + _format_repl_traceback(exc))
            self.errored = True
        finally:
            reset_tool_context(token)
            _capture.buf = None

    def _source_filename(self, code: str) -> str:
        """Stable per-call filename so traceback frames show the right source."""
        self._src_counter += 1
        filename = f"<rlm-{self._repl_id}.{self._src_counter}>"
        _register_source(filename, code)
        return filename

    def execute(self, code: str) -> str:
        """Execute code without generator support. Returns stdout."""
        self.buf = io.StringIO()
        filename = self._source_filename(code)
        with self.captured():
            exec(compile(code, filename, "exec"), self.namespace)
        return strip_ansi(self.buf.getvalue().strip())

    def start(self, code: str) -> tuple[bool, object]:
        """Execute ``code``. Returns ``(True, awaited)`` or ``(False, stdout)``."""
        self.buf = io.StringIO()
        self.errored = False
        self.coro = None
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.buf.write(f"\nSyntaxError: {exc}")
            self.errored = True
            return False, self.buf.getvalue().strip()
        if err := check_wait_syntax(code):
            self.buf.write(err)
            self.errored = True
            return False, self.buf.getvalue().strip()

        filename = self._source_filename(code)

        if not _has_top_level_await(tree):
            with self.captured():
                exec(compile(code, filename, "exec"), self.namespace)
            return False, strip_ansi(self.buf.getvalue().strip())

        with self.captured():
            code_obj = compile(
                code,
                filename,
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
            result = eval(code_obj, self.namespace)
            if not inspect.iscoroutine(result):
                return False, strip_ansi(self.buf.getvalue().strip())
            self.coro = result
        if self.errored:
            return False, strip_ansi(self.buf.getvalue().strip())
        return self.advance()

    def resume(self, send_value=None) -> tuple[bool, object]:
        """Resume a suspended top-level-await block."""
        self.buf = io.StringIO()
        self.errored = False
        return self.advance(send_value)

    def advance(self, send_value=None) -> tuple[bool, object]:
        """Drive the coroutine. Suspend only on awaited ``WaitRequest`` values."""
        if self.coro is None:
            self.buf.write("\nRuntimeError: no suspended awaitable")
            self.errored = True
            return False, strip_ansi(self.buf.getvalue().strip())
        with self.captured():
            try:
                request = self.coro.send(send_value)
                if not isinstance(request, WaitRequest):
                    raise TypeError("Only `await rlm_wait(...)` is supported")
                pre_output = strip_ansi(self.buf.getvalue().strip())
                return True, (request, pre_output)
            except StopIteration:
                self.coro = None
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
            if resp.get("done"):
                raise DoneSignal()
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
            if self.errored:
                resp["errored"] = True
            return resp
        resp = {"suspended": False, "output": result}
        if self.errored:
            resp["errored"] = True
        return resp

    def handle(self, msg: dict) -> dict:
        """Process a single command and return the response dict."""
        cmd = msg["cmd"]
        tool_context = msg.get("tool_context")
        if tool_context is not None:
            self._visible_tool_names = set(tool_context.get("visible", []))
            self._hidden_tool_names = set(tool_context.get("hidden", []))
        if cmd == "run":
            return self.format_result(*self.start(msg["code"]))
        if cmd == "resume":
            return self.format_result(*self.resume(send_value=msg.get("value")))
        if cmd == "read":
            return {"value": serialize(self.namespace.get(msg["name"]))}
        if cmd == "inject":
            self.namespace[msg["name"]] = eval(msg["value"], self.namespace)
        elif cmd == "inject_proxy":
            self.namespace[msg["name"]] = self.make_proxy(msg["name"])
        elif cmd == "inject_launcher":
            self._inject_launcher(msg["name"])
        elif cmd == "inject_show_vars":
            self.namespace["SHOW_VARS"] = self._show_vars
        elif cmd == "inject_object_proxy":
            name = msg["name"]
            obj = types.SimpleNamespace()
            for method in msg["methods"]:
                setattr(obj, method, self.make_proxy(f"{name}.{method}"))
            self.namespace[name] = obj
        elif cmd == "reset":
            # Wipe all agent-visible state so the host can reuse this REPL
            # process across independent tasks/tests without paying another
            # interpreter cold start. ``__builtins__`` is preserved.
            self.namespace = {"__builtins__": __builtins__}
            self.coro = None
            self.buf = None
            self.errored = False
            self._src_counter = 0
        else:
            return {"error": f"unknown command: {cmd}"}
        return {"ok": True}

    def _inject_launcher(self, name: str) -> None:
        from rlmflow.tools.builtins import make_launch_subagent, make_launch_subagents

        delegate = self.namespace["rlm_delegate"]
        wait = self.namespace["rlm_wait"]
        if name == "launch_subagent":
            self.namespace[name] = make_launch_subagent(delegate, wait)
            return
        if name == "launch_subagents":
            self.namespace[name] = make_launch_subagents(delegate, wait)
            return
        raise KeyError(f"unknown launcher: {name}")

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
