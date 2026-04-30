"""Integration tests for static exports and swimlane visualizations."""

from __future__ import annotations

from rlmflow import LLMClient, LLMUsage, Node, RLMConfig, RLMFlow
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.export import to_dot, to_mermaid
from rlmflow.utils.viz import gantt_html, gantt_matrix


class DelegatingLLM(LLMClient):
    ROOT = (
        "```repl\n"
        "h = delegate('child', 'do the thing')\n"
        "results = yield wait(h)\n"
        "done('root:' + results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        for message in messages:
            if "do the thing" in (message.get("content") or ""):
                return self.CHILD
        return self.ROOT


def _run(agent: RLMFlow, query: str) -> list[Node]:
    node = agent.start(query)
    states = [node]
    while not node.finished:
        node = agent.step(node)
        states.append(node)
        assert len(states) < 50
    return states


def _agent() -> RLMFlow:
    return RLMFlow(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )


def test_mermaid_export_contains_every_agent_and_result():
    final = _run(_agent(), "mermaid-test")[-1]
    mmd = to_mermaid(final)

    assert mmd.startswith("stateDiagram-v2")
    assert "root (result)" in mmd
    assert "root.child (result)" in mmd
    assert "[*] -->" in mmd
    assert "root:child-answer" in mmd


def test_mermaid_respects_include_results_flag():
    final = _run(_agent(), "mermaid-noresults")[-1]
    mmd = to_mermaid(final, include_results=False)

    assert "[*] -->" in mmd
    assert "--> [*] :" not in mmd


def test_dot_export_has_edges_and_type_labels():
    final = _run(_agent(), "dot-test")[-1]
    dot = to_dot(final)

    assert dot.startswith("digraph rlmflow {")
    assert dot.rstrip().endswith("}")
    assert "->" in dot
    assert "result" in dot


def test_gantt_matrix_row_per_agent_column_per_step():
    states = _run(_agent(), "gantt-test")
    agents, rows = gantt_matrix(states)

    assert agents[0] == "root"
    assert "root.child" in agents
    assert len(rows) == len(agents)
    assert all(len(row) == len(states) for row in rows)

    child_idx = agents.index("root.child")
    assert rows[child_idx][-1].startswith("result")
    assert rows[child_idx][0] is None


def test_gantt_html_is_self_contained():
    states = _run(_agent(), "gantt-html-test")
    html = gantt_html(states, title="test run")

    assert html.strip().startswith("<!doctype html>")
    assert "test run" in html
    assert "root" in html
    assert "root.child" in html
    for node_type in ("query", "action", "supervising", "result"):
        assert node_type in html
