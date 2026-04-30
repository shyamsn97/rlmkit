"""Branch-local workspace/session integration tests."""

from __future__ import annotations

from pathlib import Path

from rlmflow import LLMClient, RLMConfig, RLMFlow, Workspace


class StaticLLM(LLMClient):
    def chat(self, messages, *args, **kwargs):
        return '```repl\ndone("ok")\n```'


def _run(engine: RLMFlow, node):
    while not node.finished:
        node = engine.step(node)
    return node


def test_workspace_session_records_nodes_for_branch(tmp_path: Path):
    workspace = Workspace.create(tmp_path / "b1", branch_id="b1")
    engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=workspace,
        config=RLMConfig(max_iterations=2),
    )

    final = _run(engine, engine.start("test query"))
    nodes = workspace.session.load()

    assert final.result == "ok"
    assert len(nodes) == 3
    assert {node.branch_id for node in nodes.values()} == {"b1"}


def test_workspace_fork_copies_user_files_session_and_context(tmp_path: Path):
    source = Workspace.create(tmp_path / "b1", branch_id="b1")
    source.path("marker.txt").write_text("copied")
    source.context.write("context", "payload")
    engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=source,
        config=RLMConfig(max_iterations=2),
    )
    _run(engine, engine.start("test query"))

    forked = source.fork(new_branch_id="b2", new_dir=tmp_path / "b2")

    assert forked.path("marker.txt").read_text() == "copied"
    assert forked.context.read("context") == "payload"
    assert len(forked.session.load()) == len(source.session.load())
    assert forked.branch_id == "b2"


def test_workspace_fork_isolates_subsequent_session_writes(tmp_path: Path):
    source = Workspace.create(tmp_path / "b1", branch_id="b1")
    source_engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=source,
        config=RLMConfig(max_iterations=2),
    )
    _run(source_engine, source_engine.start("source"))
    source_node_ids = set(source.session.load())

    forked = source.fork(new_branch_id="b2", new_dir=tmp_path / "b2")
    fork_engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=forked,
        config=RLMConfig(max_iterations=2),
    )
    _run(fork_engine, fork_engine.start("fork"))

    assert set(source.session.load()) == source_node_ids
    assert set(forked.session.load()) > source_node_ids
