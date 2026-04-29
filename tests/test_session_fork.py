"""Session.fork: deep-copies the message store to a new location."""

from __future__ import annotations

from pathlib import Path

from rlmkit import RLM, Workspace
from rlmkit.llm import LLMClient
from rlmkit.workspace import ContextTools, FileSession


class DummyLLM(LLMClient):
    def chat(self, messages, **kwargs):
        return '```repl\ndone("ok")\n```'


def test_fork_copies_existing_messages(tmp_path: Path):
    src_dir = tmp_path / "b1"
    src = FileSession(src_dir)
    src.write("root", [{"role": "user", "content": "hello"}])
    src.write("root.child", [{"role": "user", "content": "delegated"}])

    dst = src.fork(tmp_path / "b2")
    assert dst.read("root") == [{"role": "user", "content": "hello"}]
    assert dst.read("root.child") == [
        {"role": "user", "content": "delegated"},
    ]


def test_fork_overwrites_existing_destination(tmp_path: Path):
    src_dir = tmp_path / "b1"
    src = FileSession(src_dir)
    src.write("root", [{"role": "user", "content": "from-b1"}])

    dst_dir = tmp_path / "b2"
    pre = FileSession(dst_dir)
    pre.write("root", [{"role": "user", "content": "stale"}])
    pre.write("root.stale", [{"role": "user", "content": "leftover"}])

    dst = src.fork(dst_dir)
    assert dst.read("root") == [{"role": "user", "content": "from-b1"}]
    # Stale child should be wiped
    assert "root.stale" not in dst.list_agents()


def test_fork_isolates_subsequent_writes(tmp_path: Path):
    src = FileSession(tmp_path / "b1")
    src.write("root", [{"role": "user", "content": "shared"}])

    dst = src.fork(tmp_path / "b2")
    dst.write("root", [{"role": "user", "content": "diverged"}])

    assert src.read("root") == [{"role": "user", "content": "shared"}]
    assert dst.read("root") == [{"role": "user", "content": "diverged"}]


def test_fork_handles_empty_source(tmp_path: Path):
    src = FileSession(tmp_path / "empty")
    dst = src.fork(tmp_path / "b2")
    assert dst.list_agents() == []
    assert (tmp_path / "b2").exists()


def test_context_tools_read_lines_and_grep(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    session.write_context("context", "alpha\nbeta user 123\ngamma user 456\n")

    context = ContextTools(session)

    assert context.info()["lines"] == 3
    assert context.read(0, 5) == "alpha"
    assert context.lines(1, 2).strip() == "beta user 123"
    assert context.line_count() == 3
    assert context.grep(r"user\s+456").strip() == "3:gamma user 456"


def test_context_tools_append_updates_default_context(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    session.write_context("context", "alpha")

    context = ContextTools(session)
    context.append("\nbeta")

    assert context.read() == "alpha\nbeta"
    assert context.info()["lines"] == 2


def test_context_falls_back_to_root_for_child_agents(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    session.write_context("context", "shared root context")

    context = ContextTools(session, agent_id="root.child")

    assert context.read(0, 6) == "shared"


def test_fork_copies_contexts(tmp_path: Path):
    src = FileSession(tmp_path / "b1")
    src.write_context("context", "shared context")

    dst = src.fork(tmp_path / "b2")

    assert dst.read_context("context") == "shared context"
    dst.write_context("context", "diverged")
    assert src.read_context("context") == "shared context"
    assert dst.read_context("context") == "diverged"


def test_rlm_start_seeds_context_and_injects_ctx(tmp_path: Path):
    workspace = Workspace.create(tmp_path / "workspace")
    engine = RLM(llm_client=DummyLLM(), workspace=workspace)

    engine.start("Use the seeded context.", context="one\ntwo\nthree\n")

    assert engine.session is not None
    assert engine.session.list_contexts(agent_id="root") == ["context"]
    assert "CONTEXT" in engine.runtime.repl.namespace
    assert engine.runtime.repl.namespace["CONTEXT"].lines(1, 2).strip() == "two"
