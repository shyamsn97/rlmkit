"""Integration tests for run observability hooks."""

from __future__ import annotations

from pathlib import Path

from rlmflow import Graph, LLMClient, LLMUsage, RLMConfig, RLMFlow, is_done
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.trace import Trace, load_trace, save_trace


class DelegatingLLM(LLMClient):
    ROOT = (
        "```repl\n"
        "h = rlm_delegate(name='child', query='do the thing', context='')\n"
        "results = await rlm_wait(h)\n"
        "done(results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        for message in messages:
            if "do the thing" in (message.get("content") or ""):
                return self.CHILD
        return self.ROOT


def _run_to_completion(agent: RLMFlow, query: str) -> list[Graph]:
    graph = agent.start(query)
    graphs = [graph]
    while not graph.finished:
        graph = agent.step(graph)
        graphs.append(graph)
        assert len(graphs) < 50
    return graphs


def _agent() -> RLMFlow:
    return RLMFlow(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )


def test_tree_render_contains_every_agent():
    final = _run_to_completion(_agent(), "obs-tree")[-1]
    rendered = final.tree()

    assert "root" in rendered
    assert "root.child" in rendered
    assert "done -> " in rendered


def test_tokens_sum_children_into_root():
    final = _run_to_completion(_agent(), "obs-usage")[-1]

    root_in, root_out = final.tokens(recursive=False)
    child_in, child_out = final["root.child"].tokens(recursive=False)
    tree_in, tree_out = final.tokens()

    assert tree_in == root_in + child_in
    assert tree_out == root_out + child_out
    assert child_in > 0 and child_out > 0
    assert final.total_tokens() == tree_in + tree_out


def test_step_snapshots_are_graph_instances():
    graphs = _run_to_completion(_agent(), "obs-graphs")
    assert all(isinstance(g, Graph) for g in graphs)
    # The trajectory passes through a seed (start), at least one
    # yielded supervise (mid-run), and a terminal done (end).
    from rlmflow import is_done, is_user_query, is_supervising

    currents = [g.current() for g in graphs]
    assert any(is_user_query(c) for c in currents)
    assert any(is_supervising(c) for c in currents)
    assert any(is_done(c) for c in currents)


def test_trace_save_and_load_round_trip(tmp_path: Path):
    graphs = _run_to_completion(_agent(), "obs-trace")
    out_dir = tmp_path / "trace"

    save_trace(graphs, out_dir, metadata={"kind": "test"})
    loaded = load_trace(out_dir)

    assert isinstance(loaded, Trace)
    assert loaded.metadata == {"kind": "test"}
    assert len(loaded.graphs) == len(graphs)
    assert loaded.graphs[0].root_agent_id == "root"
    assert is_done(loaded.graphs[-1].current())
    assert "root.child" in loaded.graphs[-1]
    assert loaded.graphs[-1].tree() == graphs[-1].tree()


def test_graph_save_load_round_trip(tmp_path: Path):
    final = _run_to_completion(_agent(), "obs-save")[-1]
    ckpt = tmp_path / "graph.json"

    final.save(ckpt)
    restored = Graph.load(ckpt)

    assert restored.tree() == final.tree()
    assert restored.tokens() == final.tokens()
    assert restored.result() == final.result()
