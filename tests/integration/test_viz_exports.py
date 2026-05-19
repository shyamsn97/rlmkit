"""Integration tests for static exports and swimlane visualizations."""

from __future__ import annotations

from rlmflow import (
    ErrorOutput,
    Graph,
    LLMClient,
    LLMUsage,
    UserQuery,
    RLMConfig,
    RLMFlow,
)
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.export import (
    to_d2,
    to_dot,
    to_mermaid,
    to_mermaid_flowchart,
    to_mermaid_sequence,
)
from rlmflow.utils.tracing import json_logs
from rlmflow.utils.viz import (
    bench_table,
    budget_burndown,
    code_log,
    error_summary,
    gantt_html,
    report_md,
    tee,
    token_sparkline,
)


class DelegatingLLM(LLMClient):
    ROOT = (
        "```repl\n"
        "h = rlm_delegate('child', 'do the thing', '')\n"
        "results = yield rlm_wait(h)\n"
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


def _run(agent: RLMFlow, query: str) -> list[Graph]:
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


def _error_graph() -> Graph:
    root_query = UserQuery(agent_id="root", seq=0, content="boom")
    child = Graph(
        agent_id="root.child",
        depth=1,
        parent_agent_id="root",
        parent_node_id=root_query.id,
        states=(
            ErrorOutput(
                agent_id="root.child",
                seq=0,
                error="orphaned_delegates",
                content="oops",
            ),
        ),
    )
    return Graph(
        agent_id="root",
        query="boom",
        states=(root_query,),
        children={"root.child": child},
    )


# ── mermaid ──────────────────────────────────────────────────────────


def test_mermaid_export_contains_every_agent_and_result():
    final = _run(_agent(), "mermaid-test")[-1]
    mmd = to_mermaid(final)

    assert mmd.startswith("stateDiagram-v2")
    assert "root (done)" in mmd
    assert "root.child (done)" in mmd
    assert "[*] -->" in mmd
    assert "root:child-answer" in mmd


def test_mermaid_respects_include_results_flag():
    final = _run(_agent(), "mermaid-noresults")[-1]
    mmd = to_mermaid(final, include_results=False)

    assert "[*] -->" in mmd
    assert "--> [*] :" not in mmd


def test_mermaid_flowchart_includes_classes_and_edges():
    final = _run(_agent(), "flow-test")[-1]
    out = to_mermaid_flowchart(final)
    assert out.startswith("flowchart TD")
    assert "-->" in out
    assert "classDef done" in out
    assert "root" in out


def test_mermaid_sequence_has_participant_per_agent():
    final = _run(_agent(), "seq-test")[-1]
    out = to_mermaid_sequence(final)
    assert out.startswith("sequenceDiagram")
    assert "participant root" in out
    assert "root_child" in out
    assert "->>+" in out
    assert "-->>-" in out


# ── dot ──────────────────────────────────────────────────────────────


def test_dot_export_has_edges_and_type_labels():
    final = _run(_agent(), "dot-test")[-1]
    dot = to_dot(final)

    assert dot.startswith("digraph rlmflow {")
    assert dot.rstrip().endswith("}")
    assert "->" in dot
    assert "done" in dot


# ── d2 ───────────────────────────────────────────────────────────────


def test_d2_export_has_arrows_and_styles():
    final = _run(_agent(), "d2-test")[-1]
    out = to_d2(final)
    assert " -> " in out
    assert "style" in out


# ── gantt ────────────────────────────────────────────────────────────


def test_gantt_html_is_self_contained():
    graphs = _run(_agent(), "gantt-html-test")
    html = gantt_html(graphs, title="test run")

    assert html.strip().startswith("<!doctype html>")
    assert "test run" in html
    assert "root" in html
    assert "root.child" in html
    for node_type in ("query", "llm", "supervising", "done"):
        assert node_type in html


# ── reports ──────────────────────────────────────────────────────────


def test_error_summary_groups_by_kind():
    out = error_summary(_error_graph())
    assert "orphaned_delegates" in out
    assert "1" in out


def test_error_summary_no_errors():
    final = _run(_agent(), "no-err")[-1]
    assert "(no errors)" in error_summary(final)


def test_code_log_contains_action_and_observation():
    graphs = _run(_agent(), "code-test")
    log = code_log(graphs)
    assert "rlm_delegate('child', 'do the thing', '')" in log
    assert "[root]" in log


def test_token_sparkline_has_summary():
    graphs = _run(_agent(), "spark-test")
    out = token_sparkline(graphs)
    assert "tok over" in out
    assert "step" in out


def test_budget_burndown_with_and_without_budget():
    graphs = _run(_agent(), "burn-test")
    no_budget = budget_burndown(graphs)
    with_budget = budget_burndown(graphs, max_budget=1_000_000)
    assert "%" in no_budget and "tok" in no_budget
    assert "1000000" in with_budget


def test_report_md_contains_tree_and_outcome():
    graphs = _run(_agent(), "report-test")
    md = report_md(graphs, title="bench-1")
    assert "# bench-1" in md
    assert "## Tree" in md
    assert "## Result" in md
    assert "root" in md


def test_bench_table_aggregates_traces():
    a = _run(_agent(), "a")
    b = _run(_agent(), "b")
    table = bench_table({"a": a, "b": b})
    assert "label" in table.splitlines()[0]
    assert table.splitlines()[2].startswith("a")
    assert "b" in table


def test_tee_yields_every_graph_and_calls_sinks():
    graphs = _run(_agent(), "tee-test")
    seen: list[Graph] = []
    out = list(tee(graphs, seen.append))
    assert len(out) == len(graphs)
    assert seen == out


def test_json_logs_writes_one_line_per_state(tmp_path):
    final = _run(_agent(), "jsonl-test")[-1]
    p = json_logs(final, tmp_path / "states.jsonl")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(final.nodes)
    for line in lines:
        assert '"id":' in line
        assert '"type":' in line
