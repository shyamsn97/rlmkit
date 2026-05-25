"""Session and context stores stay separate across forks."""

from __future__ import annotations

import json
from pathlib import Path

from rlmflow import (
    Graph,
    UserQuery,
    RLMConfig,
    RLMFlow,
    DoneOutput,
    Workspace,
)
from rlmflow.llm import LLMClient
from rlmflow.workspace import ContextVariable, FileContext, FileSession, FileStore


class DummyLLM(LLMClient):
    def chat(self, messages, **kwargs):
        return '```repl\ndone("ok")\n```'


def _seed_root(session: FileSession) -> UserQuery:
    session.write_agent(Graph(agent_id="root", query="hello"))
    state = UserQuery(agent_id="root", seq=0, content="hello")
    session.write_state(state)
    return state


# ── FileSession persistence ──────────────────────────────────────────


def test_file_session_writes_flat_layout(tmp_path: Path):
    store = FileStore(tmp_path / "workspace")
    session = FileSession(store)
    _seed_root(session)

    manifest = json.loads((tmp_path / "workspace" / "graph.json").read_text())
    assert manifest["root_agent_id"] == "root"
    assert manifest["agents"] == ["root"]
    assert "edges" not in manifest
    assert (
        tmp_path / "workspace" / "session" / "root" / "agent.json"
    ).exists()
    assert (
        tmp_path / "workspace" / "session" / "root" / "session.jsonl"
    ).exists()


def test_file_session_latest_view_tracks_terminal_result(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    session.write_agent(Graph(agent_id="root.worker", depth=1))
    session.write_state(UserQuery(agent_id="root.worker", seq=0))
    session.write_state(DoneOutput(agent_id="root.worker", seq=1, result="ok"))

    latest = json.loads(
        (
            tmp_path / "workspace" / "session" / "root.worker" / "latest.json"
        ).read_text()
    )
    assert latest["agent_id"] == "root.worker"
    assert latest["type"] == "done_output"
    assert latest["terminal"] is True
    assert latest["result"] == "ok"


def test_child_agent_persists_parent_link(tmp_path: Path):
    """The child's ``parent_agent_id`` + ``parent_node_id`` capture the spawn edge."""
    session = FileSession(tmp_path / "workspace")
    root_state = _seed_root(session)
    session.write_agent(
        Graph(
            agent_id="root.child",
            parent_agent_id="root",
            parent_node_id=root_state.id,
        )
    )
    child_state = UserQuery(agent_id="root.child", seq=0)
    session.write_state(child_state)

    manifest = json.loads((tmp_path / "workspace" / "graph.json").read_text())
    assert manifest["agents"] == ["root", "root.child"]

    graph = session.load_graph()
    assert "root.child" in graph
    assert graph["root.child"].parent_id == "root"
    assert graph["root.child"].parent_node_id == root_state.id
    spawns = graph.edges.spawns()
    assert spawns == [spawns[0].__class__(from_=root_state.id, to=child_state.id, kind="spawns")]


# ── fork semantics ───────────────────────────────────────────────────


def test_file_session_fork_copies_existing_state(tmp_path: Path):
    src = FileSession(tmp_path / "b1")
    root_state = _seed_root(src)

    dst = src.fork(tmp_path / "b2")
    graph = dst.load_graph()

    assert "root" in graph
    assert graph.states[0].id == root_state.id
    assert (tmp_path / "b2" / "graph.json").exists()


def test_file_session_fork_overwrites_existing_destination(tmp_path: Path):
    src = FileSession(tmp_path / "b1")
    _seed_root(src)

    pre = FileSession(tmp_path / "b2")
    pre.write_agent(Graph(agent_id="root.stale", depth=1))
    pre.write_state(UserQuery(agent_id="root.stale", seq=0, content="leftover"))

    dst = src.fork(tmp_path / "b2")
    graph = dst.load_graph()
    assert list(graph.agents) == ["root"]


def test_file_session_fork_isolates_subsequent_writes(tmp_path: Path):
    src = FileSession(tmp_path / "b1")
    _seed_root(src)
    dst = src.fork(tmp_path / "b2")

    dst.write_agent(
        Graph(agent_id="root.diverged", depth=1, parent_agent_id="root")
    )
    dst.write_state(UserQuery(agent_id="root.diverged", seq=0, content="dst-only"))

    src_graph = src.load_graph()
    dst_graph = dst.load_graph()
    assert "root.diverged" not in src_graph
    assert "root.diverged" in dst_graph


def test_file_session_fork_handles_empty_source(tmp_path: Path):
    src = FileSession(tmp_path / "empty")
    dst = src.fork(tmp_path / "b2")
    graph = dst.load_graph()
    assert not graph.states
    assert not graph.children
    assert (tmp_path / "b2").exists()


# ── context store still works ───────────────────────────────────────


def test_context_variable_read_lines_and_grep(tmp_path: Path):
    store = FileContext(tmp_path / "workspace")
    store.write("context", "alpha\nbeta user 123\ngamma user 456\n")

    context = ContextVariable(store)

    assert context.info()["lines"] == 3
    assert context.read(0, 5) == "alpha"
    assert context.lines(1, 2) == ["beta user 123"]
    assert context.line_count() == 3
    assert context.grep(r"user\s+456").strip() == "3:gamma user 456"


def test_store_backed_context_writes_flat_layout(tmp_path: Path):
    store = FileStore(tmp_path / "workspace")
    context = FileContext(store)
    context.write("context", "child payload", agent_id="root.child")

    child_dir = tmp_path / "workspace" / "context" / "root.child"
    assert (child_dir / "context.txt").read_text() == "child payload"
    metadata = json.loads((child_dir / "context_metadata.json").read_text())
    assert metadata["agent_id"] == "root.child"
    assert metadata["key"] == "context"


def test_context_falls_back_to_root_for_child_agents(tmp_path: Path):
    store = FileContext(tmp_path / "workspace")
    store.write("context", "shared root context")
    context = ContextVariable(store, agent_id="root.child")
    assert context.read(0, 6) == "shared"


def test_context_fork_isolates_subsequent_writes(tmp_path: Path):
    src = FileContext(tmp_path / "b1")
    src.write("context", "shared context")

    dst = src.fork(tmp_path / "b2")

    assert dst.read("context") == "shared context"
    dst.write("context", "diverged")
    assert src.read("context") == "shared context"
    assert dst.read("context") == "diverged"


# ── RLMFlow start round-trip ────────────────────────────────────────


def test_rlm_start_seeds_context_and_persists_files(tmp_path: Path):
    workspace = Workspace.create(tmp_path / "workspace")
    engine = RLMFlow(
        llm_client=DummyLLM(),
        workspace=workspace,
        config=RLMConfig(max_iterations=2),
    )
    # Engine semantics are obs-to-obs — one ``step`` advances by a
    # single obs→obs transition, so the LLM half writes
    # ``LLMAction → LLMOutput``; we need a second step for the exec
    # half (``ExecAction → CodeObservation``) to actually inject
    # ``CONTEXT`` into the runtime's REPL namespace.
    graph = engine.start("Use the seeded context.", context="one\ntwo\nthree\n")
    graph = engine.step(graph)  # LLM half
    engine.step(graph)          # exec half — runs code, injects CONTEXT

    assert workspace.context.list_contexts(agent_id="root") == ["context"]
    assert "CONTEXT" in engine.runtime.repl.namespace
    text = engine.runtime.repl.namespace["CONTEXT"].read()
    assert text == "one\ntwo\nthree\n"
    root = tmp_path / "workspace"
    assert (root / "graph.json").exists()
    assert (root / "session" / "root" / "session.jsonl").exists()
    assert (root / "context" / "root" / "context.txt").read_text() == "one\ntwo\nthree\n"


def test_runtime_workspace_path_auto_attaches_workspace(tmp_path: Path):
    from rlmflow.runtime.local import LocalRuntime

    root = tmp_path / "workspace"
    engine = RLMFlow(
        llm_client=DummyLLM(),
        runtime=LocalRuntime(workspace=root),
        config=RLMConfig(max_iterations=2),
    )
    engine.step(engine.start("persist graph automatically"))

    assert engine.workspace is not None
    assert engine.workspace.root == root.resolve()
    assert (root / "graph.json").exists()
    assert (root / "session" / "root" / "session.jsonl").exists()
