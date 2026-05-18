"""Runtime — ship a JSON message to a REPL, get the response back.

The REPL itself (code execution, generator suspension, tool-call
proxying) lives in :mod:`rlmflow.runtime.repl`.  A :class:`Runtime`
subclass just decides *how* to talk to it: in-process, over a
subprocess pipe, over a container's stdio, over SSH, whatever.
"""

from __future__ import annotations

import ast
import inspect
import os
import shutil
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rlmflow.graph import WaitRequest
from rlmflow.runtime.repl import deserialize, serialize
from rlmflow.tools import get_tool_metadata
from rlmflow.tools import tool as tool_decorator
from rlmflow.tools.builtins import DoneSignal

# Process-global lock guarding ``os.chdir`` in :meth:`Runtime._in_workspace`.
#
# ``os.chdir`` mutates a process-wide piece of state. With more than one
# runtime active at a time (e.g. ``max_concurrency > 1`` driving several
# child agents in parallel), naive try/finally chdir is racy: thread A
# captures ``prev = getcwd()`` and chdirs into the workspace; thread B
# then captures ``prev = workspace`` (because A has already chdir'd) and
# also chdirs into the workspace; A's finally restores the original cwd;
# B's finally restores… the workspace. The host process is now stuck on
# the workspace cwd. Serializing the chdir + body with this lock prevents
# the interleaving. The body is short (file I/O or one REPL ``run``
# message) so the perf impact is negligible in practice.
_CWD_LOCK = threading.Lock()

DEFAULT_MODULES: list[str] = [
    "re",
    "os",
    "json",
    "math",
    "collections",
    "itertools",
    "functools",
]


@dataclass
class ToolDef:
    name: str
    signature: str
    description: str
    fn: Callable | None = None
    core: bool = False

    @classmethod
    def from_fn(
        cls, fn: Callable, description: str | None = None, *, core: bool = False
    ) -> ToolDef:
        """Build a ``ToolDef`` from a function, preferring ``@tool`` metadata."""
        meta = get_tool_metadata(fn)
        name = meta.name if meta else fn.__name__.removeprefix("tool_")
        if description is None:
            description = (
                meta.description if meta else (fn.__doc__ or "").strip().split("\n")[0]
            )
        try:
            signature = str(inspect.signature(fn))
        except (TypeError, ValueError):
            signature = "(...)"
        return cls(
            name=name,
            signature=signature,
            description=description,
            fn=fn,
            core=core,
        )


def workspace_path(workspace: Any) -> Path:
    """Return the filesystem working tree for a runtime workspace handle."""
    if isinstance(workspace, str | Path):
        return Path(workspace).resolve()
    if hasattr(workspace, "root"):
        return Path(workspace.root).resolve()
    return Path(workspace).resolve()


class Runtime(ABC):
    """Where agent code runs.

    Subclass and implement two methods: :meth:`send` ships a dict to
    a REPL and :meth:`recv` reads the next dict back.  Everything else
    — the proxy loop, code execution, delegation, tool registration,
    cloning — is handled by this base class.

    See :class:`SubprocessRuntime` for the canonical remote example and
    :class:`LocalRuntime` for the in-process one.
    """

    def __init__(self, workspace: str | Path | Any = ".") -> None:
        self.workspace = workspace_path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.tools: dict[str, ToolDef] = {}
        self.proxied: dict[str, Callable] = {}
        # Per-runtime mutable state shared between the engine and any
        # core tool closures (``done``, ``delegate``) the engine binds to
        # this runtime. The engine resets this between executions.
        self.env: dict[str, Any] = {}
        # ``True`` while the REPL holds a generator paused at a ``yield``.
        # The engine reads this to detect a lost suspension (e.g. after
        # fork or process restart) and trigger replay-of-one to rebuild
        # the generator before calling ``resume_code``.
        self.suspended: bool = False

    # ── subclasses implement these two ────────────────────────────────

    @abstractmethod
    def send(self, msg: dict) -> None:
        """Ship one JSON-serializable dict to the REPL."""

    @abstractmethod
    def recv(self) -> dict:
        """Block until the next dict arrives from the REPL, return it."""

    # ── REPL protocol (shared by every runtime) ───────────────────────

    def call(self, msg: dict) -> dict:
        """Send ``msg`` and drive the proxy loop until the REPL replies.

        Proxied tools are invoked with the host CWD chdir'd into
        ``self.workspace`` so relative paths resolve the same way they
        would inside a local runtime.  Exceptions are shipped back as
        ``{"error": "..."}`` so agent code sees a normal Python
        exception instead of the host crashing and leaving the REPL
        wedged on stdin.
        """
        self.send(msg)
        while True:
            resp = self.recv()
            if "proxy" not in resp:
                return resp
            fn = self.proxied[resp["proxy"]]
            args = [deserialize(a) for a in resp.get("args", [])]
            kwargs = {k: deserialize(v) for k, v in resp.get("kwargs", {}).items()}
            try:
                with self._in_workspace():
                    result = fn(*args, **kwargs)
            except DoneSignal:
                self.send({"done": True})
            except Exception as exc:
                self.send({"error": f"{type(exc).__name__}: {exc}"})
            else:
                self.send({"value": serialize(result)})

    @contextmanager
    def _in_workspace(self):
        """Temporarily chdir into ``self.workspace`` for a proxy call.

        Serialized via the module-level :data:`_CWD_LOCK` because
        ``os.chdir`` is process-global; concurrent runtimes (parallel
        children under a non-trivial ``max_concurrency``) would
        otherwise race and leak the workspace cwd back to the caller.
        """
        with _CWD_LOCK:
            prev = os.getcwd()
            try:
                os.chdir(self.workspace)
                yield
            finally:
                os.chdir(prev)

    def execute(self, code: str) -> str:
        """Run ``code`` and return captured stdout."""
        return self.call({"cmd": "run", "code": code}).get("output", "")

    def start_code(self, code: str) -> tuple[bool, object, bool]:
        """Run code that may ``yield``.

        Returns ``(suspended, payload, errored)``. ``payload`` is
        ``(WaitRequest, pre_output)`` when ``suspended`` else captured
        stdout. ``errored`` is ``True`` when user code raised an
        exception (or had a ``SyntaxError``); the traceback is included
        in the captured output.
        """
        suspended, payload, errored = parse_response(
            self.call({"cmd": "run", "code": code})
        )
        self.suspended = suspended
        return suspended, payload, errored

    def resume_code(self, send_value=None) -> tuple[bool, object, bool]:
        """Resume a suspended generator. Same return shape as :meth:`start_code`."""
        suspended, payload, errored = parse_response(
            self.call({"cmd": "resume", "value": send_value})
        )
        self.suspended = suspended
        return suspended, payload, errored

    def read(self, name: str) -> Any:
        """Return the REPL-namespace value bound to ``name`` (``None`` if missing).

        Symmetric to :meth:`inject`. Used by the engine to read back per-execution
        state (``env``) after running agent code. ``LocalRuntime`` overrides
        this with a direct dict lookup.
        """
        from rlmflow.runtime.repl import deserialize

        return deserialize(self.call({"cmd": "read", "name": name}).get("value"))

    def inject(self, name: str, value: Any) -> None:
        """Bind ``name`` to ``value`` in the REPL's namespace.

        Three paths depending on the value:

        - **Callable** — installed as a stub inside the REPL that
          round-trips calls back to the host.
        - **Literal** (anything whose ``repr`` is a valid Python
          expression — numbers, strings, bools, ``None``, and nested
          lists/tuples/dicts of the same) — shipped as
          ``repr(value)`` and ``eval``'d on the far side.
        - **Object** — each public method becomes its own callable
          proxy, and a ``SimpleNamespace`` is installed in the REPL so
          ``name.method(...)`` works transparently.

        Override only if you can bind directly without the round-trip
        (e.g. :class:`LocalRuntime`, which runs in-process).
        """
        if callable(value):
            self.proxied[name] = value
            self.call({"cmd": "inject_proxy", "name": name})
            return
        try:
            ast.parse(repr(value), mode="eval")
            literal = True
        except SyntaxError:
            literal = False
        if literal:
            self.call({"cmd": "inject", "name": name, "value": repr(value)})
            return
        methods = [
            m
            for m in dir(value)
            if not m.startswith("_") and callable(getattr(value, m, None))
        ]
        for m in methods:
            self.proxied[f"{name}.{m}"] = getattr(value, m)
        self.call({"cmd": "inject_object_proxy", "name": name, "methods": methods})

    def clone(self, workspace: str | Path | None = None) -> Runtime:
        """Fresh runtime with the same tool registrations.

        The REPL namespace (injected values, variables from executed
        code) does NOT carry over — the clone starts empty.

        ``workspace`` overrides the new runtime's workspace path.
        Useful when forking a branch — the forked engine's child
        runtimes need to point at the forked workspace copy, not the
        parent's. Defaults to this runtime's workspace.

        Override only if your ``__init__`` takes arguments other than
        ``workspace``; the default calls ``self.__class__(workspace=...)``.
        """
        new = self.__class__(workspace=workspace or self.workspace)
        for name, td in self.tools.items():
            new.tools[name] = td
            if td.fn is not None:
                new.inject(name, td.fn)
        return new

    def fork(self, new_workspace: str | Path) -> Runtime:
        """Deep-copy this runtime's workspace and return a clone over it.

        Used by branching/best-of-N: each branch needs an isolated
        copy of the workspace so writes by the divergent tail can't
        clobber the parent's state. Subclasses with non-filesystem
        state (containers, remote sandboxes) should override.
        """
        dst = Path(new_workspace).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        if self.workspace.exists():
            shutil.copytree(self.workspace, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
        return self.clone(workspace=dst)

    def close(self) -> None:
        """Release any external resources held by this runtime.

        Default is a no-op (``LocalRuntime`` has nothing to close).
        Subclasses that spawn subprocesses, hold container handles, or
        keep network connections open should override this to free
        them — leaking subprocess pipes across many tasks is the most
        common way to trip ``OSError: Too many open files``.
        """

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def available_modules(self) -> list[str]:
        """Names of modules the REPL pre-imports, for inclusion in prompts.

        Defaults to the ``DEFAULT_MODULES`` list — every concrete subclass
        ships with the same REPL preamble. Override only if your REPL is
        wired up differently (e.g. a sandbox with a custom whitelist).
        """
        return list(DEFAULT_MODULES)

    # ── tool registration ────────────────────────────────────────────

    def register_tool(
        self,
        fn: Callable,
        description: str | None = None,
        *,
        core: bool = False,
    ) -> None:
        """Register a function as a tool — injects it and makes it discoverable."""
        td = ToolDef.from_fn(fn, description, core=core)
        self.tools[td.name] = td
        self.inject(td.name, fn)

    def register_tools(self, tools: list[Callable]) -> None:
        for tool in tools:
            self.register_tool(tool)

    def tool(self, description: str, *, name: str | None = None, core: bool = False):
        """Decorator that registers a function as a tool on this runtime.

        Usage::

            @runtime.tool("Run a shell command.")
            def shell(cmd: str) -> str:
                ...
        """

        def decorator(fn: Callable) -> Callable:
            fn = tool_decorator(description, name=name)(fn)
            self.register_tool(fn, core=core)
            return fn

        return decorator

    def get_tool_defs(self) -> list[ToolDef]:
        return list(self.tools.values())


def parse_response(resp: dict) -> tuple[bool, object, bool]:
    """Convert a REPL response dict into ``(suspended, payload, errored)``."""
    errored = bool(resp.get("errored"))
    if resp.get("suspended"):
        return (
            True,
            (
                WaitRequest(agent_ids=resp["agent_ids"]),
                resp.get("pre_output", ""),
            ),
            errored,
        )
    return False, resp.get("output", ""), errored
