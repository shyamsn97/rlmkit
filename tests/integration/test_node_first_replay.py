from __future__ import annotations

from pathlib import Path

from rlmkit import RLMConfig, RLMFlow, Workspace
from rlmkit.llm import LLMClient


class StaticLLM(LLMClient):
    def chat(self, messages, *args, **kwargs):
        return '```repl\ndone("ok")\n```'


def _run(engine: RLMFlow, node):
    while not node.finished:
        node = engine.step(node)
    return node


def test_workspace_fork_carries_flat_runtime_tree(tmp_path: Path):
    source_workspace = Workspace.create(tmp_path / "b1", branch_id="b1")
    source_workspace.path("marker.txt").write_text("copied")
    source_workspace.path("nested").mkdir()
    source_workspace.path("nested", "data.txt").write_text("nested")
    source_workspace.context.write("context", "payload stays in context store")
    source_workspace.trace_dir.joinpath("trace.json").write_text("{}")

    fork_workspace = source_workspace.fork(new_branch_id="b2", new_dir=tmp_path / "b2")

    assert fork_workspace.path("marker.txt").read_text() == "copied"
    assert fork_workspace.path("nested", "data.txt").read_text() == "nested"
    assert not fork_workspace.path("trace", "trace.json").exists()
    assert fork_workspace.context.read("context") == "payload stays in context store"


def test_rlmflow_workspace_ref_points_at_workspace_root(tmp_path: Path):
    workspace = Workspace.create(tmp_path / "workspace")

    engine = RLMFlow(
        llm_client=StaticLLM(),
        workspace=workspace,
        config=RLMConfig(max_iterations=2),
    )

    result = _run(engine, engine.start("test query"))

    assert result.result == "ok"
    assert result.workspace is not None
    assert Path(result.workspace.root) == workspace.root
