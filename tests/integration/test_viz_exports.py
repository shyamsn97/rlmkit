"""Integration tests for the static viz exports.

Checks:
- ``to_mermaid`` produces a valid ``stateDiagram-v2`` with every agent and the
  final result.
- ``to_dot`` produces a valid ``digraph`` with parent→child edges for every
  delegation.
- ``gantt_matrix`` / ``gantt_html`` build a row per agent, column per step,
  marking the correct status at each step boundary.
"""

from __future__ import annotations

from rlmflow import (
    RLM,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMNode,
    Status,
)
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
        for m in messages:
            if "do the thing" in (m.get("content") or ""):
                return self.CHILD
        return self.ROOT


def _run(agent: RLM, query: str) -> list[RLMNode]:
    state = agent.start(query)
    states = [state]
    while not state.finished:
        state = agent.step(state)
        states.append(state)
        assert len(states) < 50
    return states


def _agent() -> RLM:
    return RLM(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )


def test_mermaid_export_contains_every_agent_and_result():
    states = _run(_agent(), "mermaid-test")
    mmd = to_mermaid(states[-1])

    assert mmd.startswith("stateDiagram-v2")
    assert "as root" in mmd
    assert "as root_child" in mmd
    assert "[*] --> root" in mmd
    assert "root --> root_child : delegate" in mmd
    assert "root --> [*]" in mmd
    assert "root:child-answer" in mmd


def test_mermaid_respects_include_results_flag():
    states = _run(_agent(), "mermaid-noresults")
    mmd = to_mermaid(states[-1], include_results=False)

    assert "[*] --> root" in mmd
    assert "--> [*] :" not in mmd


def test_dot_export_has_edges_and_status_labels():
    states = _run(_agent(), "dot-test")
    dot = to_dot(states[-1])

    assert dot.startswith("digraph rlmflow {")
    assert dot.rstrip().endswith("}")
    assert "root -> root_child [label=\"delegate\"];" in dot
    assert "finished" in dot


def test_gantt_matrix_row_per_agent_column_per_step():
    states = _run(_agent(), "gantt-test")
    agents, rows = gantt_matrix(states)

    assert agents[0] == "root"
    assert "root.child" in agents
    assert len(rows) == len(agents)
    assert all(len(r) == len(states) for r in rows)

    child_idx = agents.index("root.child")
    # child must exist at the last step, and be finished by then.
    assert rows[child_idx][-1] == Status.FINISHED
    # child did not exist at step 0.
    assert rows[child_idx][0] is None


def test_gantt_html_is_self_contained():
    states = _run(_agent(), "gantt-html-test")
    html = gantt_html(states, title="test run")

    assert html.strip().startswith("<!doctype html>")
    assert "test run" in html
    assert "root" in html
    assert "root.child" in html
    for status in ("ready", "executing", "supervising", "finished"):
        assert status in html
