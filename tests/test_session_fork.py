"""Session and context stores stay separate across forks."""

from __future__ import annotations

import json
from pathlib import Path

from rlmflow import QueryNode, RLMConfig, RLMFlow, Workspace
from rlmflow.llm import LLMClient
from rlmflow.workspace import ContextVariable, FileContext, FileSession


class DummyLLM(LLMClient):
    def chat(self, messages, **kwargs):
        return '```repl\ndone("ok")\n```'


def test_session_fork_copies_existing_nodes(tmp_path: Path):
    src_dir = tmp_path / "b1"
    src = FileSession(src_dir)
    root = QueryNode(agent_id="root", content="hello")
    child = QueryNode(agent_id="root.child", content="delegated")
    src.write(root)
    src.write(child)

    dst = src.fork(tmp_path / "b2")
    nodes = dst.load()
    assert nodes[root.id].content == "hello"
    assert nodes[child.id].content == "delegated"


def test_agent_view_records_depth(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    session.write(QueryNode(agent_id="root", depth=0))
    session.write(QueryNode(agent_id="root.html", depth=1))

    root_view = json.loads((tmp_path / "session" / "agents" / "root.json").read_text())
    child_view = json.loads(
        (tmp_path / "session" / "agents" / "root.html.json").read_text()
    )

    assert root_view["depth"] == 0
    assert child_view["depth"] == 1


def test_fork_overwrites_existing_destination(tmp_path: Path):
    src_dir = tmp_path / "b1"
    src = FileSession(src_dir)
    root = QueryNode(agent_id="root", content="from-b1")
    src.write(root)

    dst_dir = tmp_path / "b2"
    pre = FileSession(dst_dir)
    stale = QueryNode(agent_id="root.stale", content="leftover")
    pre.write(stale)

    dst = src.fork(dst_dir)
    nodes = dst.load()
    assert root.id in nodes
    assert stale.id not in nodes


def test_fork_isolates_subsequent_writes(tmp_path: Path):
    src = FileSession(tmp_path / "b1")
    original = QueryNode(agent_id="root", content="shared")
    src.write(original)

    dst = src.fork(tmp_path / "b2")
    divergent = QueryNode(agent_id="root", content="diverged")
    dst.write(divergent)

    assert divergent.id not in src.load()
    assert divergent.id in dst.load()


def test_fork_handles_empty_source(tmp_path: Path):
    src = FileSession(tmp_path / "empty")
    dst = src.fork(tmp_path / "b2")
    assert dst.load() == {}
    assert (tmp_path / "b2").exists()


def test_context_variable_read_lines_and_grep(tmp_path: Path):
    store = FileContext(tmp_path / "context")
    store.write("context", "alpha\nbeta user 123\ngamma user 456\n")

    context = ContextVariable(store)

    assert context.info()["lines"] == 3
    assert context.read(0, 5) == "alpha"
    assert context.lines(1, 2).strip() == "beta user 123"
    assert context.line_count() == 3
    assert context.grep(r"user\s+456").strip() == "3:gamma user 456"


def test_context_falls_back_to_root_for_child_agents(tmp_path: Path):
    store = FileContext(tmp_path / "context")
    store.write("context", "shared root context")

    context = ContextVariable(store, agent_id="root.child")

    assert context.read(0, 6) == "shared"


def test_fork_copies_contexts(tmp_path: Path):
    src = FileContext(tmp_path / "b1")
    src.write("context", "shared context")

    dst = src.fork(tmp_path / "b2")

    assert dst.read("context") == "shared context"
    dst.write("context", "diverged")
    assert src.read("context") == "shared context"
    assert dst.read("context") == "diverged"


def test_rlm_start_seeds_context_and_injects_context_variable(tmp_path: Path):
    workspace = Workspace.create(tmp_path / "workspace")
    engine = RLMFlow(
        llm_client=DummyLLM(),
        workspace=workspace,
        config=RLMConfig(max_iterations=2),
    )

    engine.step(engine.start("Use the seeded context.", context="one\ntwo\nthree\n"))

    assert workspace.context.list_contexts(agent_id="root") == ["context"]
    assert "CONTEXT" in engine.runtime.repl.namespace
    assert engine.runtime.repl.namespace["CONTEXT"].lines(1, 2).strip() == "two"
