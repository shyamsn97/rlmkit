"""Runtime sessions and per-execution environment wiring.

The engine keeps a separate runtime session per agent so each agent's
REPL state, suspended generator, and tool closures are isolated. This
module owns the lifecycle of those sessions:

- :func:`runtime_for` — return a runtime session, lazily restoring it
  after a fork or process restart (clones the root if missing).
- :func:`create_runtime_session` — allocate a new runtime session for
  a freshly-spawned child.
- :func:`inject_env` — clear and re-seed ``runtime.env`` and the REPL
  namespace before each code execution / resume.
- :func:`register_tools` — bind the core ``done`` / ``wait`` /
  ``delegate`` closures to a runtime's ``env`` dict.

Each function takes only the state it actually uses. Callers
(typically :class:`~rlmflow.rlm.RLMFlow` methods) pull the relevant
attributes off the engine and pass them in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from rlmflow.engine.seq import ROOT_RUNTIME_ID
from rlmflow.graph import Graph, Node, RuntimeRef
from rlmflow.runtime import Runtime
from rlmflow.tools.builtins import make_delegate, make_done, make_wait
from rlmflow.utils import OrphanedDelegatesError
from rlmflow.workspace import Context, ContextVariable, Session, SessionVariable


def runtime_for(
    runtime_sessions: dict[str, Runtime],
    ref: RuntimeRef | None,
    *,
    root: Runtime,
    factory: Callable[[], Runtime] | None = None,
    on_create: Callable[[Runtime], None] | None = None,
) -> Runtime:
    """Return the runtime session bound to ``ref``, restoring lazily.

    On a fresh engine attached to a forked or reloaded workspace,
    ``runtime_sessions`` only holds the ``ROOT_RUNTIME_ID`` runtime.
    Any other agent ``RuntimeRef`` would otherwise ``KeyError``.
    Instead, we materialize a fresh runtime via ``factory`` (or by
    cloning ``root``) and call ``on_create`` to register tools
    against it. The REPL namespace and any suspended generator are
    *not* restored — callers that need a paused generator (the
    supervising transition) ask for replay-of-one separately.
    """
    session_id = ref.id if ref is not None else ROOT_RUNTIME_ID
    runtime = runtime_sessions.get(session_id)
    if runtime is None:
        runtime = factory() if factory else root.clone()
        runtime_sessions[session_id] = runtime
        if on_create is not None:
            on_create(runtime)
    return runtime


def create_runtime_session(
    runtime_sessions: dict[str, Runtime],
    parent_runtime: Runtime,
    *,
    agent_id: str,
    factory: Callable[[], Runtime] | None = None,
    on_create: Callable[[Runtime], None] | None = None,
) -> RuntimeRef:
    """Allocate a fresh runtime session for a child agent."""
    session_id = f"{agent_id}:{uuid4().hex[:8]}"
    runtime = factory() if factory else parent_runtime.clone()
    runtime_sessions[session_id] = runtime
    if on_create is not None:
        on_create(runtime)
    return RuntimeRef(id=session_id)


def inject_env(
    runtime: Runtime,
    graph: Graph,
    node: Node,
    *,
    max_depth: int,
    session: Session,
    context: Context,
) -> Runtime:
    """Reset per-execution state on ``runtime`` and inject env-style
    vars into the REPL namespace before running an action or resume.

    ``runtime.env`` is the host-side dict shared with ``done`` /
    ``delegate`` closures (cleared + seeded each call). The same
    per-agent facts plus ``CONTEXT`` / ``SESSION`` are also pushed
    into the REPL namespace so user code can reference them by
    bare name.

    Caller resolves ``runtime`` via :func:`runtime_for` and passes
    it in directly.
    """
    facts: dict[str, Any] = {
        "AGENT_ID": graph.agent_id,
        "DEPTH": graph.depth,
        "MAX_DEPTH": max_depth,
        "PARENT_NODE_ID": node.id,
    }
    runtime.env.clear()
    runtime.env.update({**facts, "DONE_RESULT": None, "DELEGATED": []})

    repl_vars = {
        **facts,
        "OrphanedDelegatesError": OrphanedDelegatesError,
        "SESSION": SessionVariable(
            session,
            agent_id=graph.agent_id,
            node_id=node.id,
            branch_id=graph.branch_id,
        ),
        "CONTEXT": ContextVariable(context, agent_id=graph.agent_id),
    }
    for name, value in repl_vars.items():
        runtime.inject(name, value)
    return runtime


SpawnChildFn = Callable[..., Any]


def register_tools(runtime: Runtime, *, spawn_child: SpawnChildFn) -> None:
    """Bind ``done`` / ``wait`` / ``delegate`` closures to ``runtime.env``.

    The ``delegate`` tool needs a way to spawn child agents — the
    ``spawn_child`` callable provides that. Pass
    :meth:`~rlmflow.rlm.RLMFlow.spawn_child` (bound to a specific
    engine instance) so the tool can call back into engine state.

    Closures live in :mod:`rlmflow.tools.builtins` and capture the
    same ``env`` dict the engine reads back after each execution (so
    ``DONE_RESULT`` / ``DELEGATED`` round-trip cleanly).
    """
    runtime.inject("OrphanedDelegatesError", OrphanedDelegatesError)
    runtime.register_tool(make_done(runtime.env), core=True)
    runtime.register_tool(make_wait(), core=True)
    runtime.register_tool(make_delegate(spawn_child, runtime.env), core=True)


__all__ = [
    "SpawnChildFn",
    "create_runtime_session",
    "inject_env",
    "register_tools",
    "runtime_for",
]
