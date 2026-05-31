"""Runtime — ship a JSON message to a REPL, get the response back.

The REPL itself (code execution, await suspension, tool-call
proxying) lives in :mod:`rlmflow.runtime.repl`.  A :class:`Runtime`
subclass just decides *how* to talk to it: in-process, over a
subprocess pipe, over a container's stdio, over SSH, whatever.
"""

from __future__ import annotations

import ast
import inspect
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from rlmflow.graph import WaitRequest
from rlmflow.runtime.repl import deserialize, serialize
from rlmflow.tools import get_tool_metadata
from rlmflow.tools import tool as tool_decorator
from rlmflow.tools.builtins import DoneSignal
from rlmflow.tools.context import reset_tool_context, set_tool_context
from rlmflow.tools.registry import (
    LAUNCHER_TOOLS,
    SHOW_VARS_NAME,
    partition_tool_defs,
    public_tool_defs,
)
from rlmflow.workspace import BaseWorkspace, Workspace

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
    fn: Callable[..., object] | None = None
    core: bool = False
    hidden: bool = False

    @classmethod
    def from_fn(
        cls,
        fn: Callable[..., object],
        description: str | None = None,
        *,
        core: bool = False,
        hidden: bool = False,
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
            hidden=hidden,
        )


def workspace_path(workspace: BaseWorkspace | str | Path) -> Path:
    """Return the filesystem working tree for a runtime workspace handle."""
    if isinstance(workspace, str | Path):
        return Path(workspace).resolve()
    return Path(workspace.root).resolve()


class Runtime(ABC):
    """Where agent code runs.

    Subclass and implement two methods: :meth:`send` ships a dict to
    a REPL and :meth:`recv` reads the next dict back.  Everything else
    — the proxy loop, code execution, delegation, tool registration,
    cloning — is handled by this base class.

    See :class:`DockerRuntime` for the canonical out-of-process example
    and :class:`LocalRuntime` for the in-process one.
    """

    def __init__(self, workspace: BaseWorkspace | str | Path = ".") -> None:
        self.workspace_obj = (
            workspace
            if isinstance(workspace, BaseWorkspace)
            else Workspace.create(workspace)
        )
        self.workspace = workspace_path(self.workspace_obj)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.tools: dict[str, ToolDef] = {}
        self._installed_tools: set[str] = set()
        self.proxied: dict[str, Callable[..., object]] = {}
        # Per-runtime mutable state shared between the engine and any
        # core tool closures (``done``, ``rlm_delegate``) the engine binds to
        # this runtime. The engine resets this between executions.
        self.env: dict[str, object] = {}
        # ``True`` while the REPL holds a block paused at ``await rlm_wait``.
        # The engine reads this to detect a lost suspension (e.g. after
        # fork or process restart) and trigger replay-of-one to rebuild
        # the coroutine before calling ``resume_code``.
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
            self.handle_proxy_call(resp)

    def handle_proxy_call(self, resp: dict) -> object:
        """Run one proxied host call and send the result back to the REPL."""

        fn = self.proxied[resp["proxy"]]
        args = [deserialize(a) for a in resp.get("args", [])]
        kwargs = {k: deserialize(v) for k, v in resp.get("kwargs", {}).items()}
        token = set_tool_context(partition_tool_defs(self.tools.values()).context())
        try:
            with self._in_workspace():
                result = fn(*args, **kwargs)
        except DoneSignal:
            self.send({"done": True})
            return None
        except Exception as exc:
            self.send({"error": f"{type(exc).__name__}: {exc}"})
            return None
        finally:
            reset_tool_context(token)

        self.send({"value": serialize(result)})
        return result

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
        self.prepare_for_execution()
        return self.call(
            {
                "cmd": "run",
                "code": code,
                "tool_context": partition_tool_defs(self.tools.values()).names(),
            }
        ).get("output", "")

    def start_code(self, code: str) -> tuple[bool, object, bool]:
        """Run code that may await ``rlm_wait``.

        Returns ``(suspended, payload, errored)``. ``payload`` is
        ``(WaitRequest, pre_output)`` when ``suspended`` else captured
        stdout. ``errored`` is ``True`` when user code raised an
        exception (or had a ``SyntaxError``); the traceback is included
        in the captured output.
        """
        self.prepare_for_execution()
        suspended, payload, errored = parse_response(
            self.call(
                {
                    "cmd": "run",
                    "code": code,
                    "tool_context": partition_tool_defs(self.tools.values()).names(),
                }
            )
        )
        self.suspended = suspended
        return suspended, payload, errored

    def resume_code(self, send_value=None) -> tuple[bool, object, bool]:
        """Resume a suspended block. Same return shape as :meth:`start_code`."""
        self.prepare_for_execution()
        suspended, payload, errored = parse_response(
            self.call(
                {
                    "cmd": "resume",
                    "value": send_value,
                    "tool_context": partition_tool_defs(self.tools.values()).names(),
                }
            )
        )
        self.suspended = suspended
        return suspended, payload, errored

    def prepare_for_execution(self) -> None:
        """Prepare runtime state before env injection or code execution."""

        self.install_registered_tools()

    def prepare_for_resume(self) -> None:
        """Prepare runtime state without destroying a suspended coroutine."""

        self.prepare_for_execution()

    def after_execution_transition(self, runtimes: Iterable[Runtime] = ()) -> None:
        """Runtime-owned hook after an exec/resume transition.

        Local runtimes do nothing. Remote runtimes can pull artifacts and mark
        sibling runtime mirrors stale.
        """

    def on_workspace_changed(self) -> None:
        """Observe that another runtime may have updated the shared workspace."""

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        """Run a shell command in the runtime workspace, if supported."""

        raise NotImplementedError(f"{type(self).__name__} does not support shell exec")

    def read(self, name: str) -> object:
        """Return the REPL-namespace value bound to ``name`` (``None`` if missing).

        Symmetric to :meth:`inject`. Used by the engine to read back per-execution
        state (``env``) after running agent code. ``LocalRuntime`` overrides
        this with a direct dict lookup.
        """
        from rlmflow.runtime.repl import deserialize

        return deserialize(self.call({"cmd": "read", "name": name}).get("value"))

    def inject(self, name: str, value: object) -> None:
        """Bind ``name`` to ``value`` in the REPL's namespace.

        Three paths depending on the value:

        - **Callable** — installed as a function proxy inside the REPL.
          Calls round-trip back to the host process.
        - **Literal** (anything whose ``repr`` is a valid Python
          expression — numbers, strings, bools, ``None``, and nested
          lists/tuples/dicts of the same) — shipped as
          ``repr(value)`` and ``eval``'d on the far side.
        - **Object** — installed as an object proxy. Each public method
          round-trips back to the host, so handles like ``CONTEXT`` and
          ``SESSION`` can keep their backing stores local while user code
          runs in a remote REPL.

        Override only if you can bind directly without the round-trip
        (e.g. :class:`LocalRuntime`, which runs in-process).
        """
        if callable(value):
            self.inject_function_proxy(name, value)
            return
        if self._can_inject_literal(value):
            self.inject_literal(name, value)
            return
        self.inject_object_proxy(name, value)

    def inject_literal(self, name: str, value: object) -> None:
        """Copy a literal value into the REPL namespace."""

        self.call({"cmd": "inject", "name": name, "value": repr(value)})

    def inject_function_proxy(self, name: str, fn: Callable[..., object]) -> None:
        """Expose a host callable as a remote REPL function."""

        self.proxied[name] = fn
        self.call({"cmd": "inject_proxy", "name": name})

    def inject_launcher(self, name: str) -> None:
        """Bind a public launcher from already-installed delegate/wait primitives."""

        self.call({"cmd": "inject_launcher", "name": name})

    def inject_object_proxy(self, name: str, value: object) -> None:
        """Expose public host object methods through a remote proxy object."""

        methods = self._proxyable_methods(value)
        for method in methods:
            self.proxied[f"{name}.{method}"] = getattr(value, method)
        self.call({"cmd": "inject_object_proxy", "name": name, "methods": methods})

    @staticmethod
    def _can_inject_literal(value: object) -> bool:
        try:
            ast.parse(repr(value), mode="eval")
        except SyntaxError:
            return False
        return True

    @staticmethod
    def _proxyable_methods(value: object) -> list[str]:
        return [
            method
            for method in dir(value)
            if not method.startswith("_") and callable(getattr(value, method, None))
        ]

    def inject_show_vars(self) -> None:
        """Install the REPL-local SHOW_VARS builtin."""

        self.call({"cmd": "inject_show_vars"})

    def clone(self, workspace: BaseWorkspace | str | Path | None = None) -> Runtime:
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
        new = self.__class__(workspace=workspace or self.workspace_obj)
        for name, td in self.tools.items():
            if td.core:
                continue
            new.tools[name] = td
        return new

    def fork(self, new_workspace: BaseWorkspace | str | Path) -> Runtime:
        """Fork the workspace and return a clone over the fork.

        Workspace implementations own durability and copy semantics. Runtime
        only creates a fresh execution handle over the resulting workspace.
        """
        if isinstance(new_workspace, BaseWorkspace):
            return self.clone(workspace=new_workspace)

        return self.clone(workspace=self.workspace_obj.fork(new_workspace))

    def reset(self) -> None:
        """Wipe REPL namespace + host-side tool/proxy registries.

        Lets a long-lived runtime (test fixture, pooled worker) be
        re-used across independent tasks without paying another
        interpreter cold start. The REPL process itself is preserved —
        only the agent-visible state is cleared.
        """
        self.call({"cmd": "reset"})
        self.tools.clear()
        self._installed_tools.clear()
        self.proxied.clear()
        self.env.clear()
        self.suspended = False

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
        fn: Callable[..., object],
        description: str | None = None,
        *,
        core: bool = False,
        hidden: bool = False,
    ) -> None:
        """Register a function as a tool — injects it and makes it discoverable."""
        td = ToolDef.from_fn(fn, description, core=core, hidden=hidden)
        self.tools[td.name] = td
        self._install_tool(td)

    def install_registered_tools(self) -> None:
        """Install copied tool definitions into this runtime's REPL namespace."""

        for td in list(self.tools.values()):
            self._install_tool(td)

    def _install_tool(self, td: ToolDef) -> None:
        if td.name in self._installed_tools:
            return
        if td.name == SHOW_VARS_NAME:
            self.inject_show_vars()
            self._installed_tools.add(td.name)
            return
        if td.name in LAUNCHER_TOOLS:
            self.inject_launcher(td.name)
            self._installed_tools.add(td.name)
            return
        if td.fn is None:
            return
        self.inject(td.name, td.fn)
        self._installed_tools.add(td.name)

    def register_tools(self, tools: list[Callable[..., object]]) -> None:
        for tool in tools:
            self.register_tool(tool)

    def tool(self, description: str, *, name: str | None = None, core: bool = False):
        """Decorator that registers a function as a tool on this runtime.

        Usage::

            @runtime.tool("Run a shell command.")
            def shell(cmd: str) -> str:
                ...
        """

        def decorator(fn: Callable[..., object]) -> Callable[..., object]:
            fn = tool_decorator(description, name=name)(fn)
            self.register_tool(fn, core=core)
            return fn

        return decorator

    def get_tool_defs(self, *, include_hidden: bool = False) -> list[ToolDef]:
        return public_tool_defs(self.tools.values(), include_hidden=include_hidden)


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
