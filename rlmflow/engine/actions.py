"""Action types and the pure projection ``Graph -> ActionPlan``.

The engine advances by **one observation-to-observation step** per
``apply_one`` call (see :meth:`rlmflow.rlm.RLMFlow.apply_one`): the
agent rests at an observation, the engine runs ``new_obs = act(obs)``,
and persists ``(action, new_obs)`` atomically. Every leaf of the
persisted graph is therefore an observation; intermediate action nodes
only exist *inside* a single ``apply_one`` call's writes.

This module owns the **pure half** of that loop:

- :class:`CallLLM` / :class:`Exec` / :class:`Resume` — the lite policy
  values (just intent + inputs not recoverable from the graph). The
  persisted :class:`~rlmflow.graph.ActionNode` carries the full
  record of what happened. See :doc:`docs/internal/act_apply.md`.
- :func:`act_one` / :func:`act` — pure projection
  ``Graph -> ActionPlan``: which transition each ready agent should
  take next. No I/O, no writes, no engine state.

The side-effectful half — :meth:`~rlmflow.rlm.RLMFlow.apply_one` and
the three handlers — lives on :class:`~rlmflow.rlm.RLMFlow` so every
piece is overridable.
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

from rlmflow.engine.config import RLMConfig
from rlmflow.engine.seq import iteration_count
from rlmflow.graph import (
    ErrorOutput,
    ExecOutput,
    Graph,
    UserQuery,
    is_done,
    is_llm_output,
    is_supervising,
)

# ── Action union (lite policy values) ───────────────────────────────


@dataclass(frozen=True, slots=True)
class CallLLM:
    """Plan to call the LLM. Materialized as ``LLMAction → LLMOutput``.

    ``force_final`` carries the policy decision to force a terminal
    answer this turn (max iterations exhausted or terminate
    requested). ``model`` is an optional override; ``None`` means
    "use whatever ``graph.config['model']`` says."
    """

    agent_id: str
    force_final: bool = False
    model: str | None = None


@dataclass(frozen=True, slots=True)
class Exec:
    """Plan to run the code from the preceding ``LLMOutput``.

    Materialized as ``ExecAction → CodeObservation``. The code
    itself is recovered from the graph in ``apply_one``; it is
    not duplicated here.
    """

    agent_id: str


@dataclass(frozen=True, slots=True)
class Resume:
    """Plan to resume a paused generator with settled child results.

    Materialized as ``ResumeAction → CodeObservation``. The child
    results are recovered from the graph (the agents named in the
    preceding ``SupervisingOutput.waiting_on``).
    """

    agent_id: str


Action = CallLLM | Exec | Resume
ActionPlan = dict[str, Action]


# ── act: pure projection Graph -> ActionPlan ────────────────────────


def act_one(
    graph: Graph,
    *,
    config: RLMConfig,
    terminate_requested: Set[str] = frozenset(),
) -> Action | None:
    """Project one agent's current observation into the next ``Action``.

    Returns ``None`` if the agent has nothing to do this round
    (terminal, empty trajectory, or stray half-written state).
    Pure: no I/O, no writes.
    """
    if graph.finished:
        return None
    cur = graph.current()
    if cur is None or is_done(cur):
        return None

    if is_supervising(cur):
        # Caller (scheduler) only routes here when children are
        # all settled; ``apply_one`` re-checks via ``can_resume``
        # inside the resume handler.
        return Resume(agent_id=graph.agent_id)

    if is_llm_output(cur):
        return Exec(agent_id=graph.agent_id)

    if isinstance(cur, (UserQuery, ExecOutput, ErrorOutput)):
        iters = iteration_count(graph)
        max_iter = graph.config.get("max_iterations", config.max_iterations)
        force_final = iters >= max_iter or graph.agent_id in terminate_requested
        return CallLLM(agent_id=graph.agent_id, force_final=force_final)

    # Stray ActionNode without its paired observation — half-written
    # state from a crash mid-step. Skip; the next round sees the
    # rewritten observation.
    return None


def act(
    graph: Graph,
    *,
    config: RLMConfig,
    runnable: list[str] | None = None,
    terminate_requested: Set[str] = frozenset(),
) -> ActionPlan:
    """Project every ready agent's next action into an ``ActionPlan``.

    If ``runnable`` is provided, only those agent ids are
    considered (the typical caller is :class:`NodeScheduler`,
    which already filters supervising-with-pending-children).
    Otherwise every agent in ``graph.agents`` is considered.

    Pure: no I/O, no writes, no engine state.
    """
    aids = runnable if runnable is not None else list(graph.agents)
    plan: ActionPlan = {}
    for aid in aids:
        if aid not in graph.agents:
            continue
        action = act_one(
            graph.agents[aid],
            config=config,
            terminate_requested=terminate_requested,
        )
        if action is not None:
            plan[aid] = action
    return plan


__all__ = [
    "Action",
    "ActionPlan",
    "CallLLM",
    "Exec",
    "Resume",
    "act",
    "act_one",
]
