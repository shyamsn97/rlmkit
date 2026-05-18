"""Branch-local workspace/session integration tests."""

from __future__ import annotations

from pathlib import Path

from rlmflow import Graph, LLMClient, RLMConfig, RLMFlow, Workspace


class StaticLLM(LLMClient):
    def chat(self, messages, *args, **kwargs):
        return '```repl\ndone("ok")\n```'


def _run(engine: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = engine.step(graph)
    return graph


def test_workspace_session_records_states_for_branch(tmp_path: Path):
    workspace = Workspace.create(tmp_path / "b1", branch_id="b1")
    engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=workspace,
        config=RLMConfig(max_iterations=2),
    )

    final = _run(engine, engine.start("test query"))
    reloaded = workspace.session.load_graph()

    assert final.result() == "ok"
    assert reloaded.branch_id == "b1"
    assert [s.type for s in reloaded.states] == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert reloaded.states[-1].type == "done_output"


def test_workspace_fork_copies_user_files_session_and_context(tmp_path: Path):
    source = Workspace.create(tmp_path / "b1", branch_id="b1")
    source.path("marker.txt").write_text("copied")
    engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=source,
        config=RLMConfig(max_iterations=2),
    )
    _run(engine, engine.start("test query", context="payload"))

    forked = source.fork(new_branch_id="b2", new_dir=tmp_path / "b2")

    assert forked.path("marker.txt").read_text() == "copied"
    assert forked.context.read("context") == "payload"
    src_aids = list(source.session.load_graph().agents)
    dst_aids = list(forked.session.load_graph().agents)
    assert src_aids == dst_aids
    assert forked.branch_id == "b2"
    assert (tmp_path / "b2" / "graph.json").exists()
    assert (tmp_path / "b2" / "session" / "root" / "session.jsonl").exists()
    assert (tmp_path / "b2" / "context" / "root" / "context.txt").exists()


def test_workspace_fork_isolates_subsequent_session_writes(tmp_path: Path):
    source = Workspace.create(tmp_path / "b1", branch_id="b1")
    source_engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=source,
        config=RLMConfig(max_iterations=2),
    )
    _run(source_engine, source_engine.start("source"))
    source_state_count = sum(
        len(agent.states) for agent in source.session.load_graph().agents.values()
    )

    forked = source.fork(new_branch_id="b2", new_dir=tmp_path / "b2")
    fork_engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=forked,
        config=RLMConfig(max_iterations=2),
    )
    _run(fork_engine, fork_engine.start("fork"))

    src_after = sum(
        len(g.states) for g in source.session.load_graph().agents.values()
    )
    dst_after = sum(
        len(g.states) for g in forked.session.load_graph().agents.values()
    )
    assert src_after == source_state_count
    assert dst_after > source_state_count
