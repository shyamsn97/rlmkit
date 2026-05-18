"""End-to-end engine tests: state machine, delegation, edge cases, persistence.

Replaces ``test_rlmflow_core.py`` (engine bits), ``test_step_ordering.py``,
``test_nested_delegation.py``, and ``test_final_answer_exhaustion.py``.

Covers:

- lifecycle (start, step, run, chat, terminate)
- per-agent node sequences (tight pattern, verify pattern, multi-yield same/split)
- recursive depth (1, 3, 5 mixed-branching)
- edge cases (orphan delegate, max depth, max iterations, budget, no code block)
- resume semantics (no child injection, REPL state preserved)
- per-agent isolation (each agent gets its own runtime + ``runtime.env``)
- ``RLMFlow.spawn_child`` override seam
- workspace persistence and trace round-trip
"""

from __future__ import annotations

from rlmflow import (
    ErrorOutput,
    Graph,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMFlow,
    DoneOutput,
    ExecOutput,
    SupervisingOutput,
    Workspace,
    is_done,
    is_errored,
    is_resumed,
    is_supervising,
)
from rlmflow.prompts.messages import FINAL_ANSWER_ACTION
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.trace import load_trace, save_trace


# ── helpers ──────────────────────────────────────────────────────────


def _types(g: Graph) -> list[str]:
    """Trajectory types in order — strict obs/action alternation."""
    return [s.type for s in g.states]


def _run(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def _assert_seqs_monotonic(g: Graph) -> None:
    """Every agent in the subtree has seq 0..n-1 with no gaps or dupes."""
    for sub in g.walk():
        seqs = [s.seq for s in sub.states]
        assert seqs == list(range(len(seqs))), (
            f"{sub.agent_id}: seqs={seqs} (expected 0..{len(seqs) - 1})"
        )


def _assert_spawn_links(g: Graph, spawner_seq: int | None = None) -> None:
    """Every direct child's parent_node_id equals the spawning state's id.

    The spawning state is whichever :class:`ExecAction` /
    :class:`ResumeAction` was running when ``delegate()`` was called.
    If ``spawner_seq`` is None (the default), we look it up by
    matching child IDs back to whichever action node's id they
    record as ``parent_node_id``.
    """
    if not g.children:
        return
    valid_ids = {s.id for s in g.states}
    for child in g.children.values():
        assert child.parent_agent_id == g.agent_id
        assert child.parent_node_id in valid_ids
        if spawner_seq is not None:
            assert child.parent_node_id == g.states[spawner_seq].id


class _StaticLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, *args, **kwargs) -> str:
        return self.reply


def _agent(reply: str = '```repl\ndone("ok")\n```', **config_kwargs) -> RLMFlow:
    config_kwargs.setdefault("max_iterations", 3)
    return RLMFlow(_StaticLLM(reply), runtime=LocalRuntime(), config=RLMConfig(**config_kwargs))


# ── lifecycle ────────────────────────────────────────────────────────


def test_start_records_query_node_at_seq_zero():
    graph = _agent().start("say ok")
    assert isinstance(graph, Graph)
    assert graph.root_agent_id == "root"
    assert graph.query == "say ok"
    assert _types(graph) == ["user_query"]
    assert graph.states[0].seq == 0


def test_step_drives_one_shot_to_result():
    """A trivial one-turn agent reaches ``done`` in two ``step``s
    (the obs→obs split: LLM half, then exec half)."""
    agent = _agent()
    graph = agent.start("say ok")
    graph = agent.step(graph)  # LLM half — writes LLMAction + LLMOutput
    assert graph.current().type == "llm_output"
    graph = agent.step(graph)  # exec half — writes ExecAction + DoneOutput
    assert is_done(graph.current())
    assert graph.result() == "ok"
    assert _types(graph) == ["user_query", "llm_action", "llm_output", "exec_action", "done_output"]


def test_run_returns_result_string():
    assert _agent().run("say ok") == "ok"


def test_chat_uses_last_user_message_as_query():
    out = _agent().chat([{"role": "user", "content": "say ok"}])
    assert out == "ok"


def test_tree_renders_root_query_and_result():
    agent = _agent()
    tree = _run(agent, agent.start("say ok")).tree()
    assert "root" in tree
    assert "query" in tree
    assert "done -> ok" in tree


# ── single-agent state machine ───────────────────────────────────────


def test_single_agent_observation_loop():
    """Two-turn agent: stash a value, then read it and ``done``."""

    class _TwoTurn(LLMClient):
        def chat(self, messages, *args, **kwargs):
            joined = "\n".join(m["content"] for m in messages)
            if "STASH" in joined:
                return '```repl\ndone("got:" + STASH)\n```'
            return "```repl\nSTASH = 'value'\nprint('hello')\n```"

    agent = RLMFlow(_TwoTurn(), runtime=LocalRuntime(), config=RLMConfig(max_depth=0, max_iterations=5))
    g = _run(agent, agent.start("hi"))
    assert _types(g) == [
        "user_query",
        "llm_action", "llm_output", "exec_action", "exec_output",
        "llm_action", "llm_output", "exec_action", "done_output",
    ]
    _assert_seqs_monotonic(g)
    assert g.result() == "got:value"


# ── delegation patterns ──────────────────────────────────────────────


class _TightChildLLM(LLMClient):
    """delegate → wait → done all in one block (tight pattern)."""

    def chat(self, messages, *args, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "child task" in prompt:
            return '```repl\ndone("c")\n```'
        return (
            "```repl\n"
            'h = delegate("child", "child task", "")\n'
            "results = yield wait(h)\n"
            'done("p:" + results[0])\n'
            "```"
        )


def test_tight_pattern_records_resume_before_result():
    agent = RLMFlow(
        _TightChildLLM(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=5)
    )
    g = _run(agent, agent.start("parent"))

    assert _types(g) == [
        "user_query",
        "llm_action", "llm_output", "exec_action", "supervising_output",
        "resume_action", "done_output",
    ]
    assert _types(g["root.child"]) == ["user_query", "llm_action", "llm_output", "exec_action", "done_output"]
    _assert_seqs_monotonic(g)
    _assert_spawn_links(g)

    sup = next(s for s in g.states if is_supervising(s))
    assert set(sup.waiting_on) == {"root.child"}
    assert g.result() == "p:c"


def test_tight_pattern_with_many_siblings_shares_one_supervising():
    n = 4

    class _MultiSibling(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "leaf task" in prompt:
                return '```repl\ndone("leaf:" + AGENT_ID)\n```'
            delegations = "\n".join(
                f'h{i} = delegate("c{i}", "leaf task", "")' for i in range(n)
            )
            handles = ", ".join(f"h{i}" for i in range(n))
            return (
                "```repl\n"
                f"{delegations}\n"
                f"results = yield wait({handles})\n"
                'done(",".join(results))\n'
                "```"
            )

    agent = RLMFlow(_MultiSibling(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=5))
    g = _run(agent, agent.start("fan out"))

    assert _types(g) == [
        "user_query",
        "llm_action", "llm_output", "exec_action", "supervising_output",
        "resume_action", "done_output",
    ]
    assert set(g.children) == {f"root.c{i}" for i in range(n)}
    sup = next(s for s in g.states if is_supervising(s))
    assert set(sup.waiting_on) == set(g.children)
    _assert_seqs_monotonic(g)
    _assert_spawn_links(g)


def test_verify_pattern_records_resume_then_action():
    """Block ends after ``yield wait``; agent calls ``done`` on the next turn."""

    class _VerifyChild(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone("c")\n```'
            prior = "\n".join(m["content"] for m in messages if m.get("role") == "assistant")
            if "yield wait" in prior:
                return '```repl\ndone("p:c-verified")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "yield wait(h)\n"
                "```"
            )

    agent = RLMFlow(_VerifyChild(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=8))
    g = _run(agent, agent.start("parent"))

    assert _types(g) == [
        "user_query",
        "llm_action", "llm_output", "exec_action", "supervising_output",
        "resume_action", "exec_output",
        "llm_action", "llm_output", "exec_action", "done_output",
    ]
    _assert_seqs_monotonic(g)
    assert g.result() == "p:c-verified"


def test_multi_yield_same_block_records_two_supervising_resume_pairs():
    class _Scripted(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone("child-result")\n```'
            if "verify task" in prompt:
                return '```repl\ndone("verified")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "child_results = yield wait(h)\n"
                'v = delegate("verify", "verify task", "")\n'
                "verdict = yield wait(v)\n"
                'done("parent:" + verdict[0])\n'
                "```"
            )

    agent = RLMFlow(_Scripted(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=5))
    g = _run(agent, agent.start("parent task"))

    assert g.result() == "parent:verified"
    assert set(g.children) == {"root.child", "root.verify"}
    assert _types(g) == [
        "user_query",
        "llm_action", "llm_output", "exec_action", "supervising_output",
        "resume_action", "supervising_output",
        "resume_action", "done_output",
    ]
    _assert_seqs_monotonic(g)


def test_multi_yield_split_blocks_records_action_between_each_resume():
    class _Scripted(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "alpha task" in prompt:
                return '```repl\ndone("a")\n```'
            if "beta task" in prompt:
                return '```repl\ndone("b")\n```'

            joined = "\n".join(
                m["content"] for m in messages if m.get("role") == "assistant"
            )
            if "beta task" in joined:
                return '```repl\ndone("p:" + first[0] + "+" + second[0])\n```'
            if "alpha task" in joined:
                return (
                    "```repl\n"
                    'h2 = delegate("beta", "beta task", "")\n'
                    "second = yield wait(h2)\n"
                    "```"
                )
            return (
                "```repl\n"
                'h1 = delegate("alpha", "alpha task", "")\n'
                "first = yield wait(h1)\n"
                "```"
            )

    agent = RLMFlow(_Scripted(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=8))
    g = _run(agent, agent.start("parent"))

    assert g.result() == "p:a+b"
    assert _types(g) == [
        "user_query",
        "llm_action", "llm_output", "exec_action", "supervising_output",
        "resume_action", "exec_output",
        "llm_action", "llm_output", "exec_action", "supervising_output",
        "resume_action", "exec_output",
        "llm_action", "llm_output", "exec_action", "done_output",
    ]
    _assert_seqs_monotonic(g)


def test_intra_agent_loop_then_delegation():
    """observation in the middle of a parent's run does not break delegation."""

    class _Loopy(LLMClient):
        def chat(self, messages, *args, **kwargs):
            joined = "\n".join(m["content"] for m in messages)
            if "child task" in joined.lower() and 'delegate("child"' not in joined:
                return '```repl\ndone("c")\n```'
            if "READY" not in joined:
                return "```repl\nprint('READY')\n```"
            if 'delegate("child"' not in joined:
                return (
                    "```repl\n"
                    'h = delegate("child", "child task", "")\n'
                    "results = yield wait(h)\n"
                    'done("p:" + results[0])\n'
                    "```"
                )
            return '```repl\ndone("p:c")\n```'

    agent = RLMFlow(_Loopy(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=8))
    g = _run(agent, agent.start("parent"))

    assert _types(g) == [
        "user_query",
        "llm_action", "llm_output", "exec_action", "exec_output",
        "llm_action", "llm_output", "exec_action", "supervising_output",
        "resume_action", "done_output",
    ]
    _assert_seqs_monotonic(g)
    assert g["root.child"].parent_node_id == g.states[7].id


# ── recursive depth ──────────────────────────────────────────────────


class _DeepChainLLM(LLMClient):
    """Each non-leaf delegates to one child until ``max_child_depth``."""

    def __init__(self, *, max_child_depth: int) -> None:
        self.max_child_depth = max_child_depth

    def chat(self, messages, *args, **kwargs):
        depth = self._depth(messages)
        if depth < self.max_child_depth:
            return (
                "```repl\n"
                'h = delegate("child", "go deeper", "")\n'
                "results = yield wait(h)\n"
                'done(AGENT_ID + "->" + results[0])\n'
                "```"
            )
        return '```repl\ndone("leaf:" + AGENT_ID)\n```'

    @staticmethod
    def _depth(messages):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        marker = "You are at recursion depth **"
        if marker not in system:
            return 0
        return int(system.split(marker, 1)[1].split("**", 1)[0])


def test_depth_one_delegation():
    agent = RLMFlow(_DeepChainLLM(max_child_depth=1), runtime=LocalRuntime(), config=RLMConfig(max_depth=1))
    g = _run(agent, agent.start("kick"))
    assert g["root.child"].depth == 1
    assert g.result() == "root->leaf:root.child"


def test_depth_three_chain_each_level_records_supervising():
    agent = RLMFlow(_DeepChainLLM(max_child_depth=3), runtime=LocalRuntime(), config=RLMConfig(max_depth=3))
    g = _run(agent, agent.start("kick"))

    chain = ["root", "root.child", "root.child.child", "root.child.child.child"]
    for aid in chain[:-1]:
        sub = g[aid]
        assert _types(sub) == [
            "user_query",
            "llm_action", "llm_output", "exec_action", "supervising_output",
            "resume_action", "done_output",
        ]
        sup = next(s for s in sub.states if is_supervising(s))
        assert set(sup.waiting_on) == {aid + ".child"}
    assert _types(g[chain[-1]]) == ["user_query", "llm_action", "llm_output", "exec_action", "done_output"]
    _assert_seqs_monotonic(g)
    assert g.result() == "root->root.child->root.child.child->leaf:root.child.child.child"


def test_depth_five_mixed_branching_tree():
    """Depth-5 tree with a 3-way fan-out at depth 2 — 12 agents total."""

    class _Branching(LLMClient):
        fanouts = [1, 1, 3, 1, 1]

        def chat(self, messages, *args, **kwargs):
            depth = _DeepChainLLM._depth(messages)
            if depth >= len(self.fanouts) or self.fanouts[depth] == 0:
                return '```repl\ndone("leaf:" + AGENT_ID)\n```'
            n = self.fanouts[depth]
            delegations = "\n".join(
                f'h{i} = delegate("c{i}", "go", "")' for i in range(n)
            )
            handles = ", ".join(f"h{i}" for i in range(n))
            return (
                "```repl\n"
                f"{delegations}\n"
                f"results = yield wait({handles})\n"
                'done(AGENT_ID + "(" + ",".join(results) + ")")\n'
                "```"
            )

    agent = RLMFlow(_Branching(), runtime=LocalRuntime(), config=RLMConfig(max_depth=5))
    g = _run(agent, agent.start("kick"))

    all_ids = sorted(sub.agent_id for sub in g.walk())
    assert len(all_ids) == 12
    leaves = [aid for aid in all_ids if aid.count(".") == 5]
    assert len(leaves) == 3

    for aid in all_ids:
        sub = g[aid]
        if aid in leaves:
            assert _types(sub) == ["user_query", "llm_action", "llm_output", "exec_action", "done_output"], aid
        else:
            assert _types(sub) == [
            "user_query",
            "llm_action", "llm_output", "exec_action", "supervising_output",
            "resume_action", "done_output",
        ], aid
            sup = next(s for s in sub.states if is_supervising(s))
            assert set(sup.waiting_on) == set(sub.children)
        assert sub.depth == aid.count(".")
    _assert_seqs_monotonic(g)


def test_each_step_advances_runnable_agents_once():
    """One step ≠ one agent: every runnable agent advances per ``step()`` call."""

    class _Recursive(LLMClient):
        def chat(self, messages, *args, **kwargs):
            depth = _DeepChainLLM._depth(messages)
            if depth < 2:
                return (
                    "```repl\n"
                    'h = delegate("c", "deeper", "")\n'
                    "results = yield wait(h)\n"
                    'done("d" + str(' + str(depth) + ') + ":" + results[0])\n'
                    "```"
                )
            return '```repl\ndone("leaf")\n```'

    agent = RLMFlow(_Recursive(), runtime=LocalRuntime(), config=RLMConfig(max_depth=2))
    graph = agent.start("kick")
    snapshots = [graph]
    while not graph.finished:
        graph = agent.step(graph)
        snapshots.append(graph)
        assert len(snapshots) < 20
    assert graph.finished
    assert "root.c" in graph
    assert "root.c.c" in graph


# ── edge cases ───────────────────────────────────────────────────────


def test_orphan_delegate_records_error_node():
    """``delegate(...)`` without ``yield wait(...)`` records an ErrorOutput."""

    class _OrphanThenDone(LLMClient):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '```repl\ndelegate("c", "leaf task", "")\n```'  # no yield wait
            return '```repl\ndone("recovered")\n```'

    agent = RLMFlow(
        _OrphanThenDone(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=5)
    )
    g = _run(agent, agent.start("p"))
    errors = [s for s in g.states if is_errored(s)]
    assert errors and any(e.error == "orphaned_delegates" for e in errors)
    assert g.result() == "recovered"


def test_max_depth_refusal_when_child_would_exceed_limit():
    class _AlwaysDelegate(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "refused" in prompt:
                return '```repl\ndone("done")\n```'
            return (
                "```repl\n"
                'r = delegate("c", "go", "")\n'
                'done(r if isinstance(r, str) else "ok")\n'
                "```"
            )

    agent = RLMFlow(_AlwaysDelegate(), runtime=LocalRuntime(), config=RLMConfig(max_depth=0))
    g = _run(agent, agent.start("p"))
    # max_depth=0 means even the root can't delegate
    assert "refused" in g.result()


def test_max_iterations_forces_final_answer_turn():
    class _Stalling(LLMClient):
        def __init__(self):
            self.calls = 0
            self.last_messages = []
            self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)

        def chat(self, messages, *args, **kwargs):
            self.calls += 1
            self.last_messages = list(messages)
            if any("full iteration budget" in m.get("content", "") for m in messages):
                return '```repl\ndone("final answer")\n```'
            return "```repl\nx = 1\n```"

    llm = _Stalling()
    agent = RLMFlow(llm, runtime=LocalRuntime(), config=RLMConfig(max_iterations=1, max_depth=0))
    final = _run(agent, agent.start("answer"))

    assert final.result() == "final answer"
    assert llm.calls == 2
    user = [m["content"] for m in llm.last_messages if m.get("role") == "user"]
    assert user[-1] == FINAL_ANSWER_ACTION


def test_terminate_marks_every_running_agent():
    class _DelegatingThenStalling(LLMClient):
        def __init__(self):
            self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)

        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "full iteration budget" in prompt:
                return '```repl\ndone("forced")\n```'
            if "child" in prompt:
                return "```repl\ny = 2\n```"
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "r = yield wait(h)\n"
                "done(r[0])\n"
                "```"
            )

    agent = RLMFlow(_DelegatingThenStalling(), runtime=LocalRuntime(), config=RLMConfig(max_depth=2, max_iterations=10))
    # Obs→obs split: LLM half, then exec half (which spawns the
    # child and yields, producing the SupervisingOutput).
    g = agent.start("kickoff")
    g = agent.step(g)
    g = agent.step(g)
    assert is_supervising(g.current())
    g = agent.terminate(g)
    assert {"root", "root.child"} <= agent.terminate_requested
    final = _run(agent, g)
    assert final["root.child"].result() == "forced"


def test_budget_exceeded_records_result_node():
    class _BudgetBuster(LLMClient):
        def __init__(self):
            self.last_usage = LLMUsage(input_tokens=1_000_000, output_tokens=0)

        def chat(self, messages, *args, **kwargs):
            return "```repl\nx = 1\n```"

    agent = RLMFlow(_BudgetBuster(), runtime=LocalRuntime(), config=RLMConfig(max_budget=1, max_depth=0))
    g = _run(agent, agent.start("p"))
    assert "budget exceeded" in g.result()


def test_no_code_block_records_error_node():
    class _Mute(LLMClient):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "I forgot the code block."
            return '```repl\ndone("ok")\n```'

    agent = RLMFlow(_Mute(), runtime=LocalRuntime(), config=RLMConfig(max_iterations=5, max_depth=0))
    g = _run(agent, agent.start("p"))
    assert any(is_errored(s) and s.error == "no_code_block" for s in g.states)


def test_no_code_block_keeps_action_in_trajectory():
    """The malformed LLM reply is recorded as an ActionNode with empty
    code, then a separate ErrorOutput tells the model to retry. The run
    log stays honest about what happened (the model was called, it
    replied, the reply was malformed)."""

    class _MuteThenFix(LLMClient):
        def __init__(self):
            self.calls = 0
            self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)

        def chat(self, messages, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "Sure! Here's how I'll start: ..."  # no code block
            return '```repl\ndone("ok")\n```'

    agent = RLMFlow(_MuteThenFix(), runtime=LocalRuntime(), config=RLMConfig(max_iterations=5, max_depth=0))
    g = _run(agent, agent.start("p"))

    assert _types(g) == [
        "user_query",
        "llm_action", "llm_output", "exec_action", "error_output",
        "llm_action", "llm_output", "exec_action", "done_output",
    ]
    bad_output = g.states[2]
    assert bad_output.code == ""
    assert "Sure!" in bad_output.reply
    empty_exec = g.states[3]
    assert empty_exec.type == "exec_action" and empty_exec.code == ""
    err = g.states[4]
    assert is_errored(err) and err.error == "no_code_block"
    _assert_seqs_monotonic(g)


# ── resume semantics ─────────────────────────────────────────────────


def test_resume_node_does_not_inject_child_result_into_prompt():
    secret = "SECRET_CHILD_RESULT"
    stdout_marker = "RESUME_STDOUT_MARKER"

    class _Scripted(LLMClient):
        def __init__(self):
            self.resume_messages = None

        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return f'```repl\ndone("{secret}")\n```'
            prior = "\n".join(m["content"] for m in messages if m.get("role") == "assistant")
            if "yield wait" in prior:
                self.resume_messages = messages
                return '```repl\ndone("parent:" + results[0])\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "results = yield wait(h)\n"
                'marker = "RESUME_" + "STDOUT_MARKER"\n'
                "print(marker)\n"
                "```"
            )

    llm = _Scripted()
    agent = RLMFlow(llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=5))
    g = _run(agent, agent.start("parent task"))

    assert g.result() == f"parent:{secret}"
    resume = next(s for s in g.states if is_resumed(s))
    assert resume.resumed_from == ["root.child"]
    assert resume.output == stdout_marker
    assert stdout_marker in resume.content
    # Critical invariant: child result must not appear anywhere downstream.
    assert "root.child" not in resume.content
    resume_prompt = "\n".join(m["content"] for m in llm.resume_messages)
    assert secret not in resume_prompt
    assert "root.child" not in resume_prompt


def test_repl_state_persists_across_resume():
    """Variables assigned before/after wait survive into the next block."""

    class _Stateful(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone("c")\n```'
            prior = "\n".join(m["content"] for m in messages if m.get("role") == "assistant")
            if "yield wait" in prior:
                # Resume turn — ``stash`` should still be in scope from prior block.
                return '```repl\ndone("p:" + stash)\n```'
            return (
                "```repl\n"
                'stash = "remembered"\n'
                'h = delegate("c", "child task", "")\n'
                "yield wait(h)\n"
                "```"
            )

    agent = RLMFlow(_Stateful(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=5))
    g = _run(agent, agent.start("p"))
    assert g.result() == "p:remembered"


# ── tools / closures / overrides ─────────────────────────────────────


def test_each_child_gets_its_own_runtime_session():
    """Verifies ``runtime.env`` isolation between agents (regression for env-dict design)."""

    class _Scripted(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\nprint(AGENT_ID, DEPTH)\ndone("child")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "results = yield wait(h)\n"
                "done(results[0])\n"
                "```"
            )

    agent = RLMFlow(_Scripted(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=5))
    g = _run(agent, agent.start("parent"))

    child = g["root.child"]
    parent_runtime = agent.runtime_for(g.runtime)
    child_runtime = agent.runtime_for(child.runtime)

    assert parent_runtime is not child_runtime
    assert parent_runtime.env is not child_runtime.env
    assert parent_runtime.env.get("AGENT_ID") == "root"
    assert child_runtime.env.get("AGENT_ID") == "root.child"
    assert child.result() == "child"


def test_spawn_child_is_overridable():
    """User-facing seam: subclass ``RLMFlow`` and override ``spawn_child``."""
    seen: list[tuple[str, str]] = []

    class _CustomFlow(RLMFlow):
        def spawn_child(self, parent_agent_id, parent_node_id, name, query, context, **opts):
            seen.append((parent_agent_id, name))
            return super().spawn_child(parent_agent_id, parent_node_id, name, query, context, **opts)

    class _Scripted(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child" in prompt:
                return '```repl\ndone("c")\n```'
            return (
                "```repl\n"
                'h = delegate("c", "child task", "")\n'
                "r = yield wait(h)\n"
                'done("p:" + r[0])\n'
                "```"
            )

    agent = _CustomFlow(_Scripted(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=5))
    g = _run(agent, agent.start("parent"))
    assert g.result() == "p:c"
    assert seen == [("root", "c")]


def test_spawn_child_can_refuse_via_returning_string():
    """Returning a string from ``spawn_child`` is the documented refusal protocol."""

    class _RefusingFlow(RLMFlow):
        def spawn_child(self, *a, **k):
            return "[refused: testing]"

    class _OneTry(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "refused" in prompt:
                return '```repl\ndone("ok")\n```'
            return (
                "```repl\n"
                'r = delegate("c", "go", "")\n'
                "done(r)\n"
                "```"
            )

    agent = _RefusingFlow(_OneTry(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1, max_iterations=3))
    g = _run(agent, agent.start("p"))
    assert g.result() == "[refused: testing]"
    assert not g.children


def test_delegate_passes_child_context_payload(tmp_path):
    class _Scripted(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone(CONTEXT.read())\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "child payload")\n'
                "r = yield wait(h)\n"
                "done(r[0])\n"
                "```"
            )

    workspace = Workspace.create(tmp_path / "ws")
    agent = RLMFlow(_Scripted(), workspace=workspace, config=RLMConfig(max_depth=1, max_iterations=5))
    g = _run(agent, agent.start("parent", context="root payload"))
    assert g.result() == "child payload"
    assert workspace.context.read("context", agent_id="root.child") == "child payload"


def test_model_routing_is_stored_on_child_agent_meta():
    class _Scripted(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child" in prompt:
                return '```repl\ndone("c")\n```'
            return (
                "```repl\n"
                'h = delegate("c", "child", "", model="fast")\n'
                "r = yield wait(h)\n"
                'done(r[0])\n'
                "```"
            )

    fast_llm = _Scripted()
    agent = RLMFlow(
        _Scripted(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
        llm_clients={"fast": {"model": fast_llm, "description": "fast model"}},
    )
    g = _run(agent, agent.start("p"))
    assert g["root.c"].config["model"] == "fast"


# ── persistence ──────────────────────────────────────────────────────


def test_workspace_round_trips_states_and_context(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    agent = RLMFlow(_StaticLLM('```repl\ndone("ok")\n```'), workspace=ws, config=RLMConfig(max_iterations=2))
    g = _run(agent, agent.start("p", context="hello"))

    assert g.result() == "ok"
    assert ws.context.read("context") == "hello"
    reloaded = ws.session.load_graph()
    assert _types(reloaded) == ["user_query", "llm_action", "llm_output", "exec_action", "done_output"]
    assert ws.load_graph().tree() == reloaded.tree()


def test_trace_save_load_round_trip(tmp_path):
    agent = _agent()
    graphs = [agent.start("p")]
    while not graphs[-1].finished:
        graphs.append(agent.step(graphs[-1]))

    path = save_trace(graphs, tmp_path / "trace")
    trace = load_trace(path)
    assert len(trace.graphs) == len(graphs)
    assert trace.graphs[-1].result() == "ok"
    assert isinstance(trace.graphs[0], Graph)


# ── graph node filters (small smoke tests) ───────────────────────────


def test_graph_node_filters_separate_errors_results_and_predicates():
    class _Stumbling(LLMClient):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "no code block here"
            return '```repl\ndone("ok")\n```'

    agent = RLMFlow(_Stumbling(), runtime=LocalRuntime(), config=RLMConfig(max_iterations=5, max_depth=0))
    g = _run(agent, agent.start("p"))

    errors = g.nodes.errors()
    results = g.nodes.results()
    llm_actions = g.nodes.where(type="llm_action")
    llm_outputs = g.nodes.where(type="llm_output")
    on_root = g.nodes.where(lambda e: e.agent_id == "root")

    assert errors and all(is_errored(e) for e in errors)
    assert results and all(is_done(r) for r in results)
    assert llm_actions and all(a.type == "llm_action" for a in llm_actions)
    assert llm_outputs and all(a.type == "llm_output" for a in llm_outputs)
    assert on_root and all(e.agent_id == "root" for e in on_root)
