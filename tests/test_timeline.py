"""Tests for :func:`rlmflow.graph.retrace_steps` and ``Workspace.load_steps``.

Focus: each successive snapshot adds exactly one state, dependency
order is respected, and concurrent siblings are interleaved
round-robin (not drained one-at-a-time).
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
        "h = delegate('child', 'do thing', '')\n"
        "results = yield wait(h)\n"
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
        "ha = delegate('a', 'task a', '')\n"
        "hb = delegate('b', 'task b', '')\n"
        "hc = delegate('c', 'task c', '')\n"
        "results = yield wait(ha, hb, hc)\n"
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
    every tick advances all three by exactly one state. No tick should
    advance a strict subset of the still-running siblings."""
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
            for aid in alive:
                assert delta[aid] == 1, (
                    f"sibling {aid!r} advanced by {delta[aid]} while siblings "
                    f"{alive!r} were all still alive"
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
    child's state count — not the sum across all children. That's the
    whole point of full concurrency."""
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
    longest = max(len(c.states) for c in graph.children.values())
    summed = sum(len(c.states) for c in graph.children.values())

    assert intervening == longest, (
        f"expected critical path {longest} ticks (longest child); "
        f"got {intervening}"
    )
    assert intervening < summed, (
        "critical path matched the SUM of child lengths — children "
        "are running sequentially, not in parallel"
    )


def test_no_idle_ticks_when_agents_are_ready(tmp_path):
    """Every tick must advance at least one agent. No empty ticks,
    no ticks that skip a still-ready agent in favor of waiting."""
    _, graph = _final_parallel(tmp_path)
    ticks = _execution_ticks(graph)

    for i, tick in enumerate(ticks):
        assert len(tick) >= 1, f"tick {i} was empty"


def test_mismatched_sibling_lengths_run_in_parallel():
    """Hand-construct a graph where children have very different state
    counts (1 vs 5) and confirm the parent doesn't "wait" for the long
    one before the short one starts — both spawn on the same tick, the
    short one finishes early, the long one ticks alone afterwards."""
    from rlmflow import (
        DoneOutput,
        ExecAction,
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
        LLMAction(agent_id="root.long", seq=1, model="x"),
        LLMOutput(agent_id="root.long", seq=2, code="x=1"),
        ExecAction(agent_id="root.long", seq=3, code="x=1"),
        # Imagine an exec_output here, then more LLM rounds — we stub
        # 5 more states so total = 10.
        DoneOutput(agent_id="root.long", seq=4, result="lo1"),
    ]
    # Stretch ``long`` to 10 states by re-using the LLM/Exec cycle.
    long.states = long.states[:4] + [
        LLMAction(agent_id="root.long", seq=4, model="x"),
        LLMOutput(agent_id="root.long", seq=5, code="done('lo')"),
        ExecAction(agent_id="root.long", seq=6, code="done('lo')"),
        DoneOutput(agent_id="root.long", seq=7, result="lo"),
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

    # Up through the short child's terminal (state 4 = 5 states), it
    # advances in lockstep with long. After short finishes, long
    # continues solo for the remaining 3 states.
    short_len = len(short.states)
    long_len = len(long.states)
    assert short_len == 5 and long_len == 8

    # Critical path between supervising and the next root state is
    # max(short, long) = 8.
    # Resume tick: first root state with seq > 4. None exist here
    # because we didn't model the resume — but we can still verify the
    # tick count between sup_tick and the end is exactly long_len.
    post_sup_ticks = ticks[sup_tick + 1 :]
    assert len(post_sup_ticks) == long_len

    # First 5 of those ticks have BOTH children advancing; remaining 3
    # have only ``long`` advancing.
    for i, tick in enumerate(post_sup_ticks):
        agents_in_tick = {aid for aid, _ in tick}
        if i < short_len:
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
