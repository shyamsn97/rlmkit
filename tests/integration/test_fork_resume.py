"""Fork-and-resume from a ``SupervisingOutput``.

Covers the scenario where a parent agent has yielded inside ``wait(...)``,
its children have run to completion, the workspace is forked (or the
process restarted), and a fresh ``RLMFlow`` is asked to continue. The
new engine has no live generator on its runtime — it has to replay the
parent's action code with ``delegate`` in replay mode to rebuild the
suspended generator at the right yield, then drop into the normal
resume path.
"""

from __future__ import annotations

from pathlib import Path

from rlmflow import (
    Graph,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMFlow,
    SupervisingOutput,
    Workspace,
    is_supervising,
)


class _ScriptedLLM(LLMClient):
    """Returns a fixed reply per ``(prompt-substring → reply)`` rule."""

    def __init__(self, rules: list[tuple[str, str]]) -> None:
        self.rules = list(rules)

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        prompt = messages[-1]["content"]
        for needle, reply in self.rules:
            if needle in prompt:
                return reply
        raise AssertionError(
            f"No scripted reply matched the prompt:\n{prompt[:200]}"
        )


def _step_until(agent: RLMFlow, graph: Graph, predicate) -> Graph:
    """Step the engine until ``predicate(graph)`` is true. Safety bound."""
    for _ in range(50):
        if predicate(graph):
            return graph
        graph = agent.step(graph)
    raise AssertionError("predicate never became true within 50 steps")


def _parent_supervising_with_terminal_children(graph: Graph) -> bool:
    parent = graph.agents.get("root")
    if parent is None:
        return False
    cur = parent.current()
    if not is_supervising(cur):
        return False
    waiting = [graph.agents[aid] for aid in cur.waiting_on if aid in graph.agents]
    return bool(waiting) and all(c.finished for c in waiting)


# ── single-yield fork-resume ─────────────────────────────────────────


PARENT_REPLY_SINGLE = (
    "```repl\n"
    'h = delegate("worker", "do thing", "")\n'
    "results = yield wait(h)\n"
    'done("got: " + results[0])\n'
    "```"
)
WORKER_REPLY = '```repl\ndone("hello")\n```'


def _scripted() -> _ScriptedLLM:
    return _ScriptedLLM(
        [
            ("do thing", WORKER_REPLY),
            ("", PARENT_REPLY_SINGLE),  # default for the parent's first turn
        ]
    )


def test_fork_resumes_supervising_with_terminal_children(tmp_path: Path):
    source = Workspace.create(tmp_path / "main", branch_id="main")
    src_engine = RLMFlow(
        llm_client=_scripted(),
        workspace=source,
        config=RLMConfig(max_depth=2, max_iterations=5),
    )

    graph = src_engine.start("parent task")
    graph = _step_until(src_engine, graph, _parent_supervising_with_terminal_children)

    # Sanity: parent is sitting at SupervisingOutput, child finished.
    assert is_supervising(graph.agents["root"].current())
    assert graph.agents["root.worker"].finished
    assert not graph.finished

    # Fork the workspace. New engine: brand-new runtime, no live generator,
    # no REPL namespace. The forked engine must rebuild via replay-of-one.
    forked = source.fork(new_branch_id="b2", new_dir=tmp_path / "b2")
    new_engine = RLMFlow(
        llm_client=_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=2, max_iterations=5),
    )

    forked_graph = forked.session.load_graph()
    assert is_supervising(forked_graph.agents["root"].current())

    # First step on the fresh engine must replay-of-one and resume.
    forked_graph = new_engine.step(forked_graph)
    while not forked_graph.finished:
        forked_graph = new_engine.step(forked_graph)

    assert forked_graph.result() == "got: hello"

    # Source workspace was not touched by the fork's resume.
    src_graph = source.session.load_graph()
    assert is_supervising(src_graph.agents["root"].current())


def test_fork_lets_us_swap_a_child_result_and_re_resume(tmp_path: Path):
    """The headline use case: branch a run, replace a child's result, continue."""
    source = Workspace.create(tmp_path / "main", branch_id="main")
    src_engine = RLMFlow(
        llm_client=_scripted(),
        workspace=source,
        config=RLMConfig(max_depth=2, max_iterations=5),
    )

    graph = src_engine.start("parent task")
    graph = _step_until(src_engine, graph, _parent_supervising_with_terminal_children)

    forked = source.fork(new_branch_id="alt", new_dir=tmp_path / "alt")

    # Manually rewrite the child's terminal DoneOutput in the forked
    # session so that resume sees a different result.
    child_session_path = (
        forked.root / "session" / "root.worker" / "session.jsonl"
    )
    lines = child_session_path.read_text().splitlines()
    import json

    rewritten = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("type") == "done_output":
            rec["result"] = "swapped"
            rec["content"] = "swapped"
        rewritten.append(json.dumps(rec))
    child_session_path.write_text("\n".join(rewritten) + "\n")

    new_engine = RLMFlow(
        llm_client=_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=2, max_iterations=5),
    )

    forked_graph = forked.session.load_graph()
    while not forked_graph.finished:
        forked_graph = new_engine.step(forked_graph)

    assert forked_graph.result() == "got: swapped"


# ── multi-yield fork-resume ──────────────────────────────────────────


PARENT_REPLY_MULTI = (
    "```repl\n"
    'h = delegate("a", "step a", "")\n'
    "first = yield wait(h)\n"
    'v = delegate("b", "step b", "")\n'
    "second = yield wait(v)\n"
    'done("p:" + first[0] + "+" + second[0])\n'
    "```"
)


def _multi_scripted() -> _ScriptedLLM:
    return _ScriptedLLM(
        [
            ("step a", '```repl\ndone("A")\n```'),
            ("step b", '```repl\ndone("B")\n```'),
            ("", PARENT_REPLY_MULTI),
        ]
    )


def _parent_at_second_supervise(graph: Graph) -> bool:
    parent = graph.agents.get("root")
    if parent is None:
        return False
    supervises = [s for s in parent.states if is_supervising(s)]
    if len(supervises) < 2:
        return False
    cur = parent.current()
    if not is_supervising(cur):
        return False
    if cur is not supervises[-1]:
        return False
    waiting = [graph.agents[aid] for aid in cur.waiting_on if aid in graph.agents]
    return bool(waiting) and all(c.finished for c in waiting)


def test_fork_resume_replays_through_multiple_yields(tmp_path: Path):
    source = Workspace.create(tmp_path / "main", branch_id="main")
    src_engine = RLMFlow(
        llm_client=_multi_scripted(),
        workspace=source,
        config=RLMConfig(max_depth=2, max_iterations=8),
    )

    graph = src_engine.start("multi yield")
    graph = _step_until(src_engine, graph, _parent_at_second_supervise)

    forked = source.fork(new_branch_id="b2", new_dir=tmp_path / "b2")
    new_engine = RLMFlow(
        llm_client=_multi_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=2, max_iterations=8),
    )

    forked_graph = forked.session.load_graph()
    while not forked_graph.finished:
        forked_graph = new_engine.step(forked_graph)

    assert forked_graph.result() == "p:A+B"
