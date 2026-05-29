"""Engine-bound built-in tools and delegation launchers.

Each tool is a Python closure created per-runtime and bound to a specific
``runtime.env`` dict (and, for ``rlm_delegate``, a ``spawn_child`` callable
that creates new sub-agents). They are registered through the normal
:meth:`Runtime.register_tool` path — ``LocalRuntime`` injects them
straight into the REPL namespace; remote runtimes expose proxy stubs that
round-trip back to the host closure.

The ``env`` dict captured here is the same object the engine reads back
via ``runtime.env`` after each execution to discover ``DONE_RESULT``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlmflow.graph import ChildHandle, WaitRequest
from rlmflow.tools import tool


class DoneSignal(BaseException):
    """Internal control-flow signal raised by ``done()`` to stop execution.

    This intentionally inherits from ``BaseException`` so agent code with a
    broad ``except Exception`` repair block cannot accidentally swallow a
    successful ``done(...)`` call and keep executing.
    """


def make_done(env: dict[str, Any]):
    """Closure that records the final answer and stops the current block."""

    @tool("Return this agent's final answer.")
    def done(answer: str) -> str:
        if env.get("DONE_RESULT") is None:
            env["DONE_RESULT"] = str(answer).strip()
            print(f"[done] {env['DONE_RESULT']}")
        raise DoneSignal(env["DONE_RESULT"])

    return done


def make_wait():
    """Closure that packages :class:`ChildHandle`s into a :class:`WaitRequest`."""

    @tool("Wait for delegated children. Must be called with `await`.")
    def rlm_wait(*handles: ChildHandle) -> WaitRequest:
        if not handles:
            raise ValueError("rlm_wait() requires at least one child handle")
        bad = [(i, h) for i, h in enumerate(handles) if not isinstance(h, ChildHandle)]
        if bad:
            details = "; ".join(
                f"handles[{i}] is {type(h).__name__}: {h!r}" for i, h in bad
            )
            raise TypeError(
                f"rlm_wait() got non-handle arguments — `rlm_delegate()` likely "
                f"refused those calls and returned a refusal string instead of a "
                f"ChildHandle. Read the string(s), fix the cause (e.g. unknown "
                f"`model=` key, max depth reached), and retry. {details}"
            )
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    return rlm_wait


@tool("Show current public REPL variable names and their type names.")
def SHOW_VARS() -> dict[str, str]:
    """Installed specially by Runtime so it can inspect the live REPL namespace."""

    raise RuntimeError("SHOW_VARS must be installed by the runtime")


def make_delegate(
    spawn_child: Callable[..., "ChildHandle | str"],
    env: dict[str, Any],
):
    """Closure that calls ``spawn_child(...)`` and tracks the new id.

    ``spawn_child`` is the engine's child-spawning seam (typically
    :meth:`RLMFlow.spawn_child` bound to a specific engine instance).
    Passing the callable instead of the whole engine keeps this
    module decoupled from :class:`RLMFlow`.

    In *replay mode* (``env['_REPLAY_QUEUE']`` is a list), ``rlm_delegate``
    does not spawn a new child — it pops the next expected agent id
    off the queue and returns a :class:`ChildHandle` to it. This lets
    the engine re-execute action code after a fork or cold start to
    re-create the suspended generator without duplicating children
    that already exist in the graph.
    """

    @tool(
        "Delegate one independent unit of work to a named child agent. "
        "Use for multi-file/component/chunk/trial fanout; the parent should "
        "pass shared requirements/data in context, then integrate and verify "
        "child results."
    )
    def rlm_delegate(
        *,
        name: str,
        query: str,
        context: str | list[str],
        max_iterations: int | None = None,
        model: str = "default",
    ) -> ChildHandle | str:
        replay_queue = env.get("_REPLAY_QUEUE")
        if replay_queue is not None:
            if not replay_queue:
                return (
                    f"[replay error: no expected child for rlm_delegate({name!r}). "
                    "Recorded trajectory diverges from the action code.]"
                )
            return ChildHandle(replay_queue.pop(0))
        context_text = "\n".join(context) if isinstance(context, list) else context
        return spawn_child(
            env["AGENT_ID"],
            env["PARENT_NODE_ID"],
            name,
            query,
            context_text,
            max_iterations=max_iterations,
            model=model,
        )

    return rlm_delegate


def make_launch_subagent(
    rlm_delegate: Callable[..., object], rlm_wait: Callable[..., Any]
):
    """Build the public single-child launcher from bound primitives."""

    @tool("Launch one sub-agent and wait for its finish message. Must be awaited.")
    async def launch_subagent(
        query,
        num_steps=None,
        context="",
        *,
        name="subagent",
        model="default",
    ):
        _handle = rlm_delegate(
            name=name,
            query=query,
            context=context,
            max_iterations=num_steps,
            model=model,
        )
        if isinstance(_handle, str):
            return _handle
        _results = await rlm_wait(_handle)
        return _results[0]

    return launch_subagent


def make_launch_subagents(
    rlm_delegate: Callable[..., object],
    rlm_wait: Callable[..., Any],
):
    """Build the public multi-child launcher from bound primitives."""

    @tool("Launch many sub-agents in parallel and wait for all. Must be awaited.")
    async def launch_subagents(specs, *, context=""):
        """Launch many sub-agents in parallel and wait for all. Must be awaited.

        ``specs`` is a list of dicts (or bare query strings); each dict may set
        ``query`` (required), ``num_steps``, ``context``, ``name``, ``model``.
        Returns finish messages in the same order as ``specs``.
        """
        _results = [None] * len(specs)
        _handles = []
        _positions = []
        for _i, _spec in enumerate(specs):
            if isinstance(_spec, str):
                _spec = {"query": _spec}
            _handle = rlm_delegate(
                name=_spec.get("name", "subagent"),
                query=_spec["query"],
                context=_spec.get("context", context),
                max_iterations=_spec.get("num_steps"),
                model=_spec.get("model", "default"),
            )
            if isinstance(_handle, str):
                _results[_i] = _handle
            else:
                _handles.append(_handle)
                _positions.append(_i)
        if _handles:
            _waited = await rlm_wait(*_handles)
            for _pos, _result in zip(_positions, _waited):
                _results[_pos] = _result
        return _results

    return launch_subagents


__all__ = [
    "DoneSignal",
    "SHOW_VARS",
    "make_delegate",
    "make_done",
    "make_launch_subagent",
    "make_launch_subagents",
    "make_wait",
]
