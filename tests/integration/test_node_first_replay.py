from __future__ import annotations

from pathlib import Path

from rlmkit import RLM, RLMConfig, Workspace
from rlmkit.llm import LLMClient, LLMUsage


ROOT_REPLY = """```repl
h1 = delegate("worker_a", "task A")
h2 = delegate("worker_b", "task B")
results = yield wait(h1, h2)
done(results[0] + "+" + results[1])
```"""


class SourceLLM(LLMClient):
    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        if "task A" in first_user:
            return '```repl\ndone("b1-A")\n```'
        if "task B" in first_user:
            return '```repl\ndone("b1-B")\n```'
        return ROOT_REPLY


class FailIfCalledLLM(LLMClient):
    def chat(self, messages, *args, **kwargs):  # pragma: no cover
        raise AssertionError("unexpected live LLM call")


def _run(engine: RLM, node):
    while not node.finished:
        node = engine.step(node)
    return node


def _replay(engine: RLM, node):
    while not node.finished:
        node = engine.step(node, use_cache=True)
    return node


def test_node_fork_carries_workspace_and_override(tmp_path: Path):
    source_workspace = Workspace.create(tmp_path / "b1", branch_id="b1")
    (source_workspace.files / "marker.txt").write_text("copied")

    engine = RLM(
        llm_client=SourceLLM(),
        workspace=source_workspace,
        config=RLMConfig(max_depth=1, max_iterations=10),
    )
    source = _run(engine, engine.start("test query"))
    assert source.result == "b1-A+b1-B"
    assert source.workspace is not None

    forked = source.fork(
        "root.worker_a",
        branch_id="b2",
        workspace=tmp_path / "b2",
        override_reply='```repl\ndone("b2-A")\n```',
    )

    assert forked.workspace is not None
    assert Path(forked.workspace.root) == (tmp_path / "b2").resolve()
    assert (forked.workspace.files / "marker.txt").read_text() == "copied"

    fork_workspace = Workspace.open(forked.workspace)
    fork_engine = RLM(
        llm_client=FailIfCalledLLM(),
        workspace=fork_workspace,
        config=RLMConfig(max_depth=1, max_iterations=10),
    )
    result = _replay(fork_engine, forked)

    assert result.result == "b2-A+b1-B"
