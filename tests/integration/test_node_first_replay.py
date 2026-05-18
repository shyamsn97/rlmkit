"""Workspace-attached engine state survives forks and reloads."""

from __future__ import annotations

from pathlib import Path

from rlmflow import Graph, RLMConfig, RLMFlow, Workspace
from rlmflow.llm import LLMClient


class StaticLLM(LLMClient):
    def chat(self, messages, *args, **kwargs):
        return '```repl\ndone("ok")\n```'


def _run(engine: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = engine.step(graph)
    return graph


def test_workspace_fork_carries_flat_runtime_tree(tmp_path: Path):
    source_workspace = Workspace.create(tmp_path / "b1", branch_id="b1")
    source_workspace.path("marker.txt").write_text("copied")
    source_workspace.path("nested").mkdir()
    source_workspace.path("nested", "data.txt").write_text("nested")
    source_workspace.context.write("context", "payload stays in context store")
    source_workspace.path("trace").mkdir(parents=True, exist_ok=True)
    source_workspace.path("trace", "trace.json").write_text("{}")

    fork_workspace = source_workspace.fork(
        new_branch_id="b2", new_dir=tmp_path / "b2"
    )

    assert fork_workspace.path("marker.txt").read_text() == "copied"
    assert fork_workspace.path("nested", "data.txt").read_text() == "nested"
    assert not fork_workspace.path("trace", "trace.json").exists()
    assert (
        fork_workspace.context.read("context")
        == "payload stays in context store"
    )


def test_rlmflow_records_workspace_ref_on_root_agent(tmp_path: Path):
    workspace = Workspace.create(tmp_path / "workspace")
    engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=workspace,
        config=RLMConfig(max_iterations=2),
    )

    final = _run(engine, engine.start("test query"))

    assert __import__("rlmflow").is_done(final.current())
    workspace_ref = final.workspace
    assert workspace_ref is not None
    assert Path(workspace_ref.root) == workspace.root
