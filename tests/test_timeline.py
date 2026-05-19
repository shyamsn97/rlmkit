"""Tests for :func:`rlmflow.graph.retrace_steps` and ``Workspace.load_steps``.

Focus: each successive snapshot reflects one round of ``engine.step()``
(i.e. each ready agent advances by one obs-to-obs ``apply_one`` —
1 state on spawn (``UserQuery``) or 2 states per LLM/exec/resume
half-turn ``(action, observation)`` pair), dependency order is
respected, and concurrent siblings advance in lockstep.
"""

from __future__ import annotations

from rlmflow import (
    Graph,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMFlow,
    Workspace,
    retrace_steps,
)
from rlmflow.graph.timeline import _execution_ticks
from rlmflow.runtime.local import LocalRuntime


# ── tiny scripted LLMs ───────────────────────────────────────────────


class _OneChild(LLMClient):
    """root delegates one child, child returns immediately."""

    ROOT = (
        "```repl\n"
        "h = rlm_delegate('child', 'do thing', '')\n"
        "results = yield rlm_wait(h)\n"
        "done('root:' + results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)
        for m in messages:
            if "do thing" in (m.get("content") or ""):
                return self.CHILD
        return self.ROOT


class _ParallelChildren(LLMClient):
    """root delegates 3 children in one supervising step.

    Each child needs *two* LLM turns so we can observe interleaving:
    if siblings were drained one-at-a-time we'd see ``a a b b c c``;
    round-robin produces ``a b c a b c``.
    """

    ROOT = (
        "```repl\n"
        "ha = rlm_delegate('a', 'task a', '')\n"
        "hb = rlm_delegate('b', 'task b', '')\n"
        "hc = rlm_delegate('c', 'task c', '')\n"
        "results = yield rlm_wait(ha, hb, hc)\n"
        "done(' '.join(results))\n"
        "```"
    )
    CHILD_PRINT = "```repl\nprint('thinking')\n```"

    def __init__(self) -> None:
        self.turns: dict[str, int] = {}

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)
        text = "\n".join((m.get("content") or "") for m in messages)
        for tag in ("task a", "task b", "task c"):
            if tag in text:
                self.turns[tag] = self.turns.get(tag, 0) + 1
                if self.turns[tag] == 1:
                    return self.CHILD_PRINT
                return f"```repl\ndone('{tag}-done')\n```"
        return self.ROOT


# ── helpers ──────────────────────────────────────────────────────────


def _final_one_child(tmp_path):
    workspace = Workspace.create(tmp_path / "ws")
    agent = RLMFlow(
        llm_client=_OneChild(),
        workspace=workspace,
        config=RLMConfig(max_depth=2),
    )
    graph = agent.start("kick off")
    while not graph.finished:
        graph = agent.step(graph)
    return workspace, graph


def _final_parallel(tmp_path):
    workspace = Workspace.create(tmp_path / "ws")
    agent = RLMFlow(
        llm_client=_ParallelChildren(),
        runtime=LocalRuntime(workspace=workspace),
        workspace=workspace,
        config=RLMConfig(max_depth=2, max_concurrency=8),
    )
    graph = agent.start("kick off")
    while not graph.finished:
        graph = agent.step(graph)
    return workspace, graph


def _state_count(graph: Graph) -> int:
    return sum(1 for _ in graph.nodes)


# ── retrace_steps ────────────────────────────────────────────────────


def test_retrace_steps_state_counts_are_strictly_increasing(tmp_path):
    """Snapshots correspond to ticks; each adds at least one state."""
    _, graph = _final_one_child(tmp_path)
    steps = retrace_steps(graph)

    counts = [_state_count(s) for s in steps]
    assert counts == sorted(counts)
    for prev, nxt in zip(counts, counts[1:]):
        assert nxt > prev
    assert counts[-1] == _state_count(graph)
    assert steps[-1].result() == graph.result()


def test_retrace_steps_first_snapshot_is_root_user_query(tmp_path):
    _, graph = _final_one_child(tmp_path)
    steps = retrace_steps(graph)
    first = steps[0]

    assert _state_count(first) == 1
    only = next(iter(first.nodes))
    assert only.type == "user_query"
    assert only.agent_id == graph.agent_id


def test_retrace_steps_respects_spawn_dependency(tmp_path):
    _, graph = _final_one_child(tmp_path)
    steps = retrace_steps(graph)

    # Find the first snapshot containing a child state. The parent's
    # supervising_output must already be present in that snapshot.
    child_id = next(iter(graph.children))
    for snap in steps:
        child = snap.children.get(child_id)
        if child and child.states:
            sup_present = any(
                s.type == "supervising_output" for s in snap.states
            )
            assert sup_present, "child appeared before parent supervised"
            return
    raise AssertionError("no snapshot contained a child state")


def test_retrace_steps_resume_waits_for_all_children(tmp_path):
    _, graph = _final_parallel(tmp_path)
    steps = retrace_steps(graph)

    # The state right after the parent's supervising_output is the
    # resume. In every snapshot that contains the resume, all three
    # children must already be terminal.
    for snap in steps:
        states = list(snap.states)
        for i, s in enumerate(states):
            if i == 0 or states[i - 1].type != "supervising_output":
                continue
            sup = states[i - 1]
            for child_aid in sup.waiting_on:
                child = snap.children.get(child_aid)
                assert child is not None
                last = child.states[-1] if child.states else None
                assert last is not None and last.terminal, (
                    f"resume emitted before child {child_aid!r} finished"
                )


def test_retrace_steps_advances_concurrent_siblings_in_lockstep(tmp_path):
    """Parallel-tick semantics: while all three siblings are alive,
    every tick advances all three by the *same* number of states
    (because they're all at the same kind of state-machine position).
    No tick should advance a strict subset of the still-running
    siblings, and no tick should advance two siblings by different
    amounts."""
    _, graph = _final_parallel(tmp_path)
    steps = retrace_steps(graph)

    child_ids = sorted(graph.children)
    assert len(child_ids) == 3
    final_counts = {aid: len(graph.children[aid].states) for aid in child_ids}

    prev = {aid: 0 for aid in child_ids}
    parallel_ticks = 0
    for snap in steps:
        cur = {aid: len(snap.children[aid].states) for aid in child_ids}
        delta = {aid: cur[aid] - prev[aid] for aid in child_ids}
        # Pre-spawn (root running alone) and post-resume ticks add no
        # child states — skip those. We only enforce the lockstep
        # invariant on ticks where children actually advanced.
        if any(delta.values()):
            alive = [
                aid for aid in child_ids if prev[aid] < final_counts[aid]
            ]
            alive_deltas = {aid: delta[aid] for aid in alive}
            # Every alive sibling advanced the same amount (1 / 4 / 2).
            assert len(set(alive_deltas.values())) == 1, (
                f"siblings advanced by uneven amounts in one tick: "
                f"{alive_deltas!r}"
            )
            assert all(d > 0 for d in alive_deltas.values()), (
                f"alive sibling stalled while another advanced: "
                f"{alive_deltas!r}"
            )
            parallel_ticks += 1
        prev = cur

    assert parallel_ticks > 0, "no parallel ticks observed"


def test_retrace_steps_singleton_graph_returns_self():
    g = Graph(agent_id="root")
    steps = retrace_steps(g)
    assert steps == [g]


# ── max-concurrency / "realistic parallelism" guarantees ─────────────


def test_all_siblings_user_query_appears_in_one_tick(tmp_path):
    """When the parent supervises N children, ALL N children's
    ``user_query`` states must appear in the *single* tick immediately
    following the parent's ``supervising_output`` — never spread across
    N separate ticks."""
    _, graph = _final_parallel(tmp_path)
    ticks = _execution_ticks(graph)

    # Find the tick that emitted the root's supervising_output.
    sup_tick_idx = None
    for i, tick in enumerate(ticks):
        for aid, idx in tick:
            s = graph.agents[aid].states[idx]
            if s.type == "supervising_output" and aid == graph.agent_id:
                sup_tick_idx = i
                break
        if sup_tick_idx is not None:
            break
    assert sup_tick_idx is not None, "no supervising_output tick"

    spawn_tick = ticks[sup_tick_idx + 1]
    spawned = {aid for aid, _ in spawn_tick}
    expected = set(graph.children)
    assert spawned == expected, (
        f"expected all {len(expected)} children spawned in one tick, "
        f"got {len(spawned)} this tick: {spawned}"
    )
    # Every event in that tick is each child's first state.
    for aid, idx in spawn_tick:
        assert idx == 0
        assert graph.agents[aid].states[0].type == "user_query"


def test_critical_path_equals_longest_child_not_sum(tmp_path):
    """Under unbounded parallelism, the number of ticks between a
    parent's ``supervising_output`` and its resume equals the LONGEST
    child's tick-count (steps the child took) — not the sum across
    all children. That's the whole point of full concurrency."""
    _, graph = _final_parallel(tmp_path)
    ticks = _execution_ticks(graph)

    sup_tick_idx = next(
        i
        for i, tick in enumerate(ticks)
        for aid, idx in tick
        if graph.agents[aid].states[idx].type == "supervising_output"
        and aid == graph.agent_id
    )
    resume_tick_idx = next(
        i
        for i, tick in enumerate(ticks)
        for aid, idx in tick
        if graph.agents[aid].states[idx].type == "resume_action"
        and aid == graph.agent_id
    )

    intervening = resume_tick_idx - sup_tick_idx - 1
    # Tick-count per child = number of distinct ticks the child
    # appears in (each tick may emit multiple states for one child
    # under the atomic-step model, so dedupe per tick).
    ticks_per_child = {aid: 0 for aid in graph.children}
    for tick in ticks[sup_tick_idx + 1 : resume_tick_idx]:
        agents_this_tick = {aid for aid, _ in tick}
        for aid in agents_this_tick:
            if aid in ticks_per_child:
                ticks_per_child[aid] += 1
    longest = max(ticks_per_child.values())
    summed = sum(ticks_per_child.values())

    assert intervening == longest, (
        f"expected critical path {longest} ticks (longest child); "
        f"got {intervening}"
    )
    assert intervening < summed, (
        "critical path matched the SUM of child tick-counts — "
        "children are running sequentially, not in parallel"
    )


def test_no_idle_ticks_when_agents_are_ready(tmp_path):
    """Every tick must advance at least one agent. No empty ticks,
    no ticks that skip a still-ready agent in favor of waiting."""
    _, graph = _final_parallel(tmp_path)
    ticks = _execution_ticks(graph)

    for i, tick in enumerate(ticks):
        assert len(tick) >= 1, f"tick {i} was empty"


def test_mismatched_sibling_lengths_run_in_parallel():
    """Hand-construct a graph where children take a different number
    of ``apply_one`` turns and confirm the parent doesn't "wait" for
    the long one before the short one starts — both spawn on the same
    tick, the short one finishes early, the long one ticks alone
    afterwards.

    Short = UserQuery + one full turn (5 states, 2 ticks).
    Long  = UserQuery + two full turns (9 states, 3 ticks).
    """
    from rlmflow import (
        DoneOutput,
        ExecAction,
        ExecOutput,
        Graph as G,
        LLMAction,
        LLMOutput,
        SupervisingOutput,
        UserQuery,
    )

    short = G(agent_id="root.short", depth=1)
    short.states = [
        UserQuery(agent_id="root.short", seq=0, content="task short"),
        LLMAction(agent_id="root.short", seq=1, model="x"),
        LLMOutput(agent_id="root.short", seq=2, code="done('ok')"),
        ExecAction(agent_id="root.short", seq=3, code="done('ok')"),
        DoneOutput(agent_id="root.short", seq=4, result="ok"),
    ]

    long = G(agent_id="root.long", depth=1)
    long.states = [
        UserQuery(agent_id="root.long", seq=0, content="task long"),
        # Turn 1 — produces an ExecOutput, agent loops.
        LLMAction(agent_id="root.long", seq=1, model="x"),
        LLMOutput(agent_id="root.long", seq=2, code="x=1"),
        ExecAction(agent_id="root.long", seq=3, code="x=1"),
        ExecOutput(agent_id="root.long", seq=4, output="ok"),
        # Turn 2 — closes with done().
        LLMAction(agent_id="root.long", seq=5, model="x"),
        LLMOutput(agent_id="root.long", seq=6, code="done('lo')"),
        ExecAction(agent_id="root.long", seq=7, code="done('lo')"),
        DoneOutput(agent_id="root.long", seq=8, result="lo"),
    ]

    root = G(agent_id="root", depth=0)
    root.states = [
        UserQuery(agent_id="root", seq=0, content="kick"),
        LLMAction(agent_id="root", seq=1, model="x"),
        LLMOutput(agent_id="root", seq=2, code="..."),
        ExecAction(agent_id="root", seq=3, code="..."),
        SupervisingOutput(
            agent_id="root", seq=4, waiting_on=["root.short", "root.long"]
        ),
    ]
    root.add_child(short)
    root.add_child(long)

    ticks = _execution_ticks(root)

    sup_tick = next(
        i
        for i, tick in enumerate(ticks)
        for aid, idx in tick
        if aid == "root" and root.states[idx].type == "supervising_output"
    )

    # Tick right after supervising should contain BOTH children
    # spawning together — short shouldn't have to wait for long.
    spawn_tick = ticks[sup_tick + 1]
    spawned = {aid for aid, _ in spawn_tick}
    assert spawned == {"root.short", "root.long"}

    # Tick-counts per child under the action-split model:
    #   short = UQ + dispatch + done                     = 3 ticks
    #   long  = UQ + dispatch + result + dispatch + done = 5 ticks
    post_sup_ticks = ticks[sup_tick + 1 :]
    assert len(post_sup_ticks) == 5, (
        f"expected 5 post-supervising ticks (= longest child); "
        f"got {len(post_sup_ticks)}"
    )

    short_ticks = sum(
        1 for tick in post_sup_ticks if any(aid == "root.short" for aid, _ in tick)
    )
    long_ticks = sum(
        1 for tick in post_sup_ticks if any(aid == "root.long" for aid, _ in tick)
    )
    assert short_ticks == 3 and long_ticks == 5

    # Ticks 0–1 have BOTH children advancing in lockstep; tick 2 is
    # long-only (short already terminal).
    for i, tick in enumerate(post_sup_ticks):
        agents_in_tick = {aid for aid, _ in tick}
        if i < short_ticks:
            assert agents_in_tick == {"root.short", "root.long"}, (
                f"tick {i} should have both children advancing, got {agents_in_tick}"
            )
        else:
            assert agents_in_tick == {"root.long"}, (
                f"tick {i} (post-short-finish) should be long-only, "
                f"got {agents_in_tick}"
            )


def test_finished_children_drop_out_of_subsequent_ticks(tmp_path):
    """Once a child reaches a terminal state, it must not appear in
    any later tick. ``alive`` siblings keep ticking; finished ones go
    quiet."""
    _, graph = _final_parallel(tmp_path)
    ticks = _execution_ticks(graph)

    final_lens = {aid: len(c.states) for aid, c in graph.children.items()}
    cumulative: dict[str, int] = {aid: 0 for aid in graph.children}
    for tick in ticks:
        for aid, idx in tick:
            if aid not in cumulative:
                continue
            assert cumulative[aid] < final_lens[aid], (
                f"{aid!r} kept appearing after reaching terminal state"
            )
            cumulative[aid] += 1
        # Once an agent has emitted all its states, it shouldn't be
        # in any future tick.
        for aid in cumulative:
            if cumulative[aid] == final_lens[aid]:
                # No future tick may contain this agent.
                pass  # checked implicitly by the assertion above


# ── Workspace.load_steps ─────────────────────────────────────────────


def test_workspace_load_steps_matches_retrace_of_load_graph(tmp_path):
    workspace, _ = _final_one_child(tmp_path)
    via_method = workspace.load_steps()
    via_function = retrace_steps(workspace.load_graph())

    assert len(via_method) == len(via_function)
    assert [_state_count(s) for s in via_method] == [
        _state_count(s) for s in via_function
    ]
    assert via_method[-1].result() == via_function[-1].result()


def test_workspace_load_steps_reopens_after_persistence(tmp_path):
    workspace, graph = _final_one_child(tmp_path)
    reopened = Workspace.open_path(workspace.root)

    steps = reopened.load_steps()
    assert len(steps) >= 1
    assert steps[-1].result() == graph.result()
    assert _state_count(steps[-1]) == _state_count(graph)
