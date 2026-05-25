"""Engine-bound built-in tools: ``done``, ``rlm_wait``, ``rlm_delegate``.

Each tool is a Python closure created per-runtime and bound to a specific
``runtime.env`` dict (and, for ``rlm_delegate``, a ``spawn_child`` callable
that creates new sub-agents). They are registered through the normal
:meth:`Runtime.register_tool` path — ``LocalRuntime`` injects them
straight into the REPL namespace; remote runtimes expose proxy stubs that
round-trip back to the host closure.

The ``env`` dict captured here is the same object the engine reads back
via ``runtime.env`` after each execution to discover ``DONE_RESULT`` and
``DELEGATED`` agent ids.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlmflow.graph import ChildHandle, WaitRequest
from rlmflow.tools import tool


class DoneSignal(Exception):
    """Internal control-flow signal raised by ``done()`` to stop execution."""


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


def make_llm_query_batched(query_batch: Callable[..., list[str]]):
    """Closure that runs multiple one-shot LLM prompts without child agents."""

    @tool(
        "Run multiple independent one-shot LLM prompts concurrently. "
        "Returns strings in the same order as the prompts."
    )
    def llm_query_batched(
        prompts: list[str],
        *,
        model: str = "default",
    ) -> list[str]:
        if isinstance(prompts, str) or not isinstance(prompts, list):
            raise TypeError("llm_query_batched() requires a list[str] of prompts")
        bad = [
            (i, type(prompt).__name__)
            for i, prompt in enumerate(prompts)
            if not isinstance(prompt, str)
        ]
        if bad:
            details = "; ".join(f"prompts[{i}] is {typ}" for i, typ in bad)
            raise TypeError(f"llm_query_batched() requires list[str]. {details}")
        if not prompts:
            return []
        return query_batch(prompts, model=model)

    return llm_query_batched


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
            child_aid = replay_queue.pop(0)
            env.setdefault("DELEGATED", []).append(child_aid)
            return ChildHandle(child_aid)
        context_text = "\n".join(context) if isinstance(context, list) else context
        handle = spawn_child(
            env["AGENT_ID"],
            env["PARENT_NODE_ID"],
            name,
            query,
            context_text,
            max_iterations=max_iterations,
            model=model,
        )
        if isinstance(handle, str):
            return handle
        env.setdefault("DELEGATED", []).append(handle.agent_id)
        return handle

    return rlm_delegate


__all__ = [
    "DoneSignal",
    "SHOW_VARS",
    "make_delegate",
    "make_done",
    "make_llm_query_batched",
    "make_wait",
]
