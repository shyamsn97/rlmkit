"""Retrace a fully-loaded :class:`Graph` as a list of intermediate snapshots.

The engine appends one :class:`Node` at a time but only the final
:class:`Graph` is persisted. ``retrace_steps(graph)`` reconstructs the
intermediate snapshots so tools (the viewer slider, exporters, etc.)
can step through the run.

This is *not* the same as :mod:`rlmflow.engine.replay`, which actually
re-executes a paused generator to recover its Python state. Here we
just walk the recorded states and emit truncated copies of the graph
— no runtime, no code re-execution.

Snapshots are produced **one per parallel tick** under unbounded
``max_concurrency`` semantics: every agent whose dependencies are
satisfied advances by one engine step (one ``apply_one`` call) in
lockstep, and the snapshot taken afterwards shows all of them at
the new level simultaneously.

Engine semantics are obs-to-obs: ``new_obs = act(obs)``. Each
``apply_one`` call takes the agent from one observation to the
next, persisting ``(action, new_obs)`` atomically. So every leaf
of every snapshot is an observation. Stable types are exactly
the observations:

* :class:`UserQuery` — initial state on spawn.
* :class:`LLMOutput` — LLM replied; next step runs its code.
* :class:`ExecOutput` / :class:`SupervisingOutput` /
  :class:`ErrorOutput` / :class:`DoneOutput` — code finished.

A typical agent turn shows up as two ticks: one at
:class:`LLMOutput` ("LLM replied with this code"), then one at
the matching :class:`CodeObservation` ("code finished with this
result"). Action nodes (:class:`LLMAction`, :class:`ExecAction`,
:class:`ResumeAction`) never become the latest state — they're
always part of the same tick that ends at their paired
observation.

Ordering rules per tick:

* Within a single agent, states are emitted in their recorded ``seq``
  order.
* A child agent's first state cannot appear before the parent's
  :class:`SupervisingOutput` that spawned it.
* A parent's resume cannot fire until every child the supervising
  was waiting on has reached its final state.
"""

from __future__ import annotations

from rlmflow.graph.graph import Graph

# Types where an agent rests between engine steps. Each tick
# advances forward until it lands on (and includes) a state of
# one of these types, so action / intermediate-output nodes never
# appear as the latest state in a snapshot.
_STABLE_TYPES: frozenset[str] = frozenset(
    {
        "user_query",
        "llm_output",
        "exec_output",
        "supervising_output",
        "error_output",
        "done_output",
    }
)


def retrace_steps(graph: Graph) -> list[Graph]:
    """Return one :class:`Graph` snapshot per parallel tick, in order.

    Every tick advances *all* currently-unblocked agents by one
    obs-to-obs ``apply_one`` call simultaneously. The first
    snapshot is just the root's :class:`UserQuery`; each subsequent
    snapshot adds one more obs-to-obs step (i.e. one
    ``(action, observation)`` pair) per ready agent. The final
    snapshot equals ``graph``.
    """
    ticks = _execution_ticks(graph)
    if not ticks:
        return [graph]

    snapshots: list[Graph] = []
    counts: dict[str, int] = dict.fromkeys(graph.agents, 0)
    for tick in ticks:
        for aid, _ in tick:
            counts[aid] += 1
        snap = graph.copy(deep=True)
        for sub in snap.walk():
            keep = counts.get(sub.agent_id, 0)
            del sub.states[keep:]
        snapshots.append(snap)
    return snapshots


def _execution_ticks(graph: Graph) -> list[list[tuple[str, int]]]:
    """Return execution events grouped into parallel ticks.

    Each tick is a list of ``(agent_id, state_index)`` pairs that
    became ready at the same logical moment. All ready agents
    advance by one ``apply_one``-call's worth of states in the
    same tick — that's the "infinite ``max_concurrency``"
    interleaving where every runnable agent runs its next step in
    parallel.
    """
    states = {aid: list(sub.states) for aid, sub in graph.agents.items()}

    # child_aid -> (parent_aid, idx of the SupervisingOutput that
    # spawned it). First match wins so a child re-listed in a later
    # supervising still attaches to its original spawn point.
    spawn_dep: dict[str, tuple[str, int]] = {}
    for aid, agent_states in states.items():
        for i, s in enumerate(agent_states):
            if s.type != "supervising_output":
                continue
            for child in getattr(s, "waiting_on", []):
                spawn_dep.setdefault(child, (aid, i))

    pos: dict[str, int] = dict.fromkeys(states, 0)
    ticks: list[list[tuple[str, int]]] = []

    def is_ready(aid: str) -> bool:
        i = pos[aid]
        if i >= len(states[aid]):
            return False
        if i == 0:
            # Child agent — wait until the parent has actually emitted
            # its spawning supervising_output. Root has no spawn dep.
            dep = spawn_dep.get(aid)
            if dep is None:
                return True
            parent_aid, parent_idx = dep
            return pos.get(parent_aid, 0) > parent_idx
        prev = states[aid][i - 1]
        if prev.type == "supervising_output":
            # Resume — gated on every waiting_on child reaching its
            # final state.
            for child in getattr(prev, "waiting_on", []):
                if pos.get(child, 0) < len(states.get(child, [])):
                    return False
        return True

    def step_count(aid: str) -> int:
        """How many states this agent advances in one tick.

        Mirrors one ``apply_one`` call's writes — always the
        ``(action, observation)`` pair from the obs-to-obs
        transition. UserQuery (1 state, on spawn) and any other
        observation followed directly by its action+result pair
        (2 states). Consume forward until landing on (and
        including) the next observation.
        """
        i = pos[aid]
        agent_states = states[aid]
        n = len(agent_states)
        j = i
        while j < n:
            if agent_states[j].type in _STABLE_TYPES:
                return j - i + 1
            j += 1
        # Trailing intermediate states with no closing observation
        # (incomplete / mid-write). Emit whatever is left as the
        # final tick for this agent.
        return max(1, n - i)

    while True:
        ready = sorted(aid for aid in pos if is_ready(aid))
        if not ready:
            break
        tick: list[tuple[str, int]] = []
        for aid in ready:
            count = step_count(aid)
            for _ in range(count):
                tick.append((aid, pos[aid]))
                pos[aid] += 1
        ticks.append(tick)

    # Anything still pending (cycles / dangling deps) — flush as a
    # single trailing tick per agent in lexicographic order so the
    # output stays total. Should not happen for a well-formed graph;
    # this is purely defensive.
    for aid in sorted(pos):
        while pos[aid] < len(states[aid]):
            ticks.append([(aid, pos[aid])])
            pos[aid] += 1

    return ticks


__all__ = ["retrace_steps"]
