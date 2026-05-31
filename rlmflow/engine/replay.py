"""Replay-of-one: rebuild a suspended awaitable after a fork or restart.

When the engine is attached to a freshly-loaded workspace (via
``Workspace.fork`` or just opening a saved run later), the durable
graph contains every :class:`~rlmflow.graph.SupervisingOutput` that
was recorded — but the live Python coroutine that awaited inside
the runtime is gone. This module handles re-running the action
code with ``rlm_delegate`` in *replay mode* (returning existing child
handles instead of spawning new ones) so the coroutine pauses again
at the same await. The regular resume path then takes over.

All public functions take only the explicit dependencies they
need; nothing in this module imports :class:`~rlmflow.rlm.RLMFlow`.
"""

from __future__ import annotations

from rlmflow.graph import (
    Graph,
    SupervisingOutput,
    is_done,
    is_exec_action,
    is_llm_action,
    is_llm_output,
    is_resume_action,
    is_supervising,
)
from rlmflow.runtime import Runtime
from rlmflow.runtime.env import clear_replay_queue, set_replay_queue


def can_resume(graph: Graph, supervising: SupervisingOutput) -> bool:
    """True iff every child the supervising node is waiting on is terminal.

    ``graph`` is the supervising agent's sub-graph (children are inside).
    """
    if not supervising.waiting_on:
        # Recovery for older / bad runs that wrote ``await rlm_wait()`` with
        # no handles. New calls are rejected in ``rlm_wait()`` before this
        # node ever exists.
        return True
    for aid in supervising.waiting_on:
        if aid not in graph.agents:
            return False
        if not is_done(graph.agents[aid].current()):
            return False
    return True


def results_for_supervise(
    graph: Graph,
    supervising: SupervisingOutput,
) -> list[str]:
    """Collect terminal results for ``supervising.waiting_on`` children."""
    children = [
        graph.agents[aid] for aid in supervising.waiting_on if aid in graph.agents
    ]
    return [child.result() if is_done(child.current()) else "" for child in children]


def supervise_history(
    graph: Graph,
    current: SupervisingOutput,
) -> tuple[list[SupervisingOutput], str]:
    """Walk back through prior resume/supervise pairs to the originating LLMOutput.

    Returns ``(chain, code)`` where ``chain`` is every
    :class:`SupervisingOutput` in the same await/resume sequence
    (chronological, last element is ``current``) and ``code`` is the
    code from the :class:`LLMOutput` that started the sequence.

    The walk crosses :class:`ResumeAction`, :class:`ExecAction`, and
    :class:`SupervisingOutput` nodes; it stops at the
    :class:`LLMOutput` whose code drove the run.
    """
    states = graph.states
    idx = next(i for i, s in enumerate(states) if s is current)
    chain: list[SupervisingOutput] = [current]
    code = ""
    i = idx - 1
    while i >= 0:
        s = states[i]
        if is_resume_action(s) or is_exec_action(s):
            i -= 1
            continue
        if is_supervising(s):
            chain.append(s)
            i -= 1
            continue
        if is_llm_output(s):
            code = s.code
            break
        if is_llm_action(s):
            # Reached the action without seeing its output — should not
            # happen for a well-formed graph, but stop walking.
            break
        break
    chain.reverse()
    return chain, code


def replay_to_suspension(
    graph: Graph,
    target: SupervisingOutput,
    runtime: Runtime,
) -> None:
    """Re-execute action code so ``runtime`` is paused at the same wait as ``target``.

    Used after a fork or cold start, when the engine has the graph
    but not the live coroutine frame. ``rlm_delegate`` runs in replay
    mode (returns existing child handles instead of spawning), so
    the code reaches each await with the same ``WaitRequest`` the
    original run produced. We verify the match at every await and
    raise on divergence.
    """
    chain, code = supervise_history(graph, target)
    if not code:
        raise RuntimeError(
            f"replay: could not locate originating LLMOutput.code for "
            f"agent {graph.agent_id!r} at supervise seq={target.seq}"
        )
    set_replay_queue(runtime.env, list(chain[0].waiting_on))
    suspended, raw, _ = runtime.start_code(code)
    if not suspended:
        raise RuntimeError(
            f"replay: action code for agent {graph.agent_id!r} finished "
            "without suspending; trajectory expects an await"
        )
    _verify_replay_wait(raw, chain[0], graph.agent_id)

    for prev, nxt in zip(chain[:-1], chain[1:]):
        prev_results = results_for_supervise(graph, prev)
        set_replay_queue(runtime.env, list(nxt.waiting_on))
        suspended, raw, _ = runtime.resume_code(prev_results)
        if not suspended:
            raise RuntimeError(
                f"replay: code for agent {graph.agent_id!r} finished at "
                f"supervise seq={nxt.seq}; expected another await"
            )
        _verify_replay_wait(raw, nxt, graph.agent_id)

    clear_replay_queue(runtime.env)


def _verify_replay_wait(
    raw: object,
    supervising: SupervisingOutput,
    agent_id: str,
) -> None:
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise RuntimeError(f"replay: unexpected runtime payload {raw!r}")
    request, _ = raw
    actual = list(getattr(request, "agent_ids", []))
    expected = list(supervising.waiting_on)
    if actual != expected:
        raise RuntimeError(
            f"replay diverged for agent {agent_id!r} at supervise "
            f"seq={supervising.seq}: expected wait on {expected}, "
            f"got {actual}"
        )


__all__ = [
    "can_resume",
    "replay_to_suspension",
    "results_for_supervise",
    "supervise_history",
]
