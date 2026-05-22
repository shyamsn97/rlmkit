"""SessionVariable — cross-agent transcript access from inside the REPL."""

from __future__ import annotations

from pathlib import Path

from rlmflow.graph import (
    DoneOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMAction,
    LLMOutput,
    UserQuery,
)
from rlmflow.workspace import FileSession, InMemorySession, SessionVariable


def _seed_agent(
    session,
    *,
    agent_id: str,
    system: str = "",
    query: str = "",
    code: str = "",
    observation: str = "",
    result: str = "",
    depth: int = 0,
    parent_agent_id: str | None = None,
    parent_node_id: str | None = None,
) -> tuple[str, str]:
    """Seed a minimal query → action → observation → result chain for one agent.

    Returns ``(query_node_id, result_node_id)`` so the caller can wire
    parent → child spawn edges.
    """
    graph = Graph(
        agent_id=agent_id,
        depth=depth,
        query=query,
        system_prompt=system,
        parent_agent_id=parent_agent_id,
        parent_node_id=parent_node_id,
    )
    session.write_agent(graph)

    q = UserQuery(agent_id=agent_id, seq=0, content=query)
    la = LLMAction(agent_id=agent_id, seq=1)
    lo = LLMOutput(agent_id=agent_id, seq=2, reply=code, code=code)
    ea = ExecAction(agent_id=agent_id, seq=3, code=code)
    o = ExecOutput(agent_id=agent_id, seq=4, content=observation, output=observation)
    r = DoneOutput(agent_id=agent_id, seq=5, result=result)
    for state in (q, la, lo, ea, o, r):
        session.write_state(state)

    return q.id, r.id


def test_list_agents_excludes_self_and_lists_siblings(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    root_q, _ = _seed_agent(
        session,
        agent_id="root",
        system="root prompt",
        query="build a thing",
        code='rlm_delegate(name="html", query="...", context="")',
        observation="ok",
        result="all done",
    )
    _seed_agent(
        session,
        agent_id="root.html",
        depth=1,
        system="child prompt",
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
        parent_agent_id="root",
        parent_node_id=root_q,
    )

    var = SessionVariable(session, agent_id="root")
    assert var.list_agents() == ["root.html"]


def test_summarize_agent_reports_latest_state(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    root_q, _ = _seed_agent(
        session,
        agent_id="root",
        system="root prompt",
        query="build a thing",
        code='rlm_delegate(name="html", query="...", context="")',
        observation="ok",
        result="all done",
    )
    _seed_agent(
        session,
        agent_id="root.html",
        depth=1,
        system="child prompt",
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
        parent_agent_id="root",
        parent_node_id=root_q,
    )

    var = SessionVariable(session, agent_id="root")
    child = var.summarize_agent("root.html")
    assert child is not None
    assert child["type"] == "done_output"
    assert child["terminal"] is True
    assert "wrote index.html" in child["result_preview"]
    assert var.summarize_agent("root.missing") is None


def test_read_renders_full_chain_for_a_sibling(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    _seed_agent(
        session,
        agent_id="root.html",
        depth=1,
        system="child prompt",
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
    )

    var = SessionVariable(session, agent_id="root")
    text = var.read("root.html")

    assert "--- system ---\nchild prompt" in text
    assert "--- query ---\nwrite index.html" in text
    assert "--- assistant ---\nwrite_file" in text
    assert "--- observation ---\nREPL output:" in text
    assert "--- result ---\nwrote index.html" in text


def test_read_unknown_agent_returns_clear_message():
    var = SessionVariable(InMemorySession(), agent_id="root")
    assert var.read("root.missing") == "(no nodes for agent 'root.missing')"


def test_messages_projects_graph_states_when_no_transcript(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    _seed_agent(
        session,
        agent_id="root.html",
        depth=1,
        system="child prompt",
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
    )

    var = SessionVariable(session, agent_id="root")
    msgs = var.messages("root.html")

    assert msgs[0] == {"role": "system", "content": "child prompt"}
    assert msgs[1] == {"role": "user", "content": "write index.html"}
    assert msgs[2]["role"] == "assistant"
    assert "write_file" in msgs[2]["content"]
    assert msgs[3] == {"role": "user", "content": "REPL output:\nNone"}
    assert var.messages("root.missing") == []


def test_messages_prefers_stored_transcript(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    _seed_agent(
        session,
        agent_id="root.html",
        depth=1,
        system="child prompt",
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
    )
    session.write_transcript(
        "root.html",
        {
            "agent_id": "root.html",
            "messages": [
                {"role": "system", "content": "stored system"},
                {"role": "user", "content": "stored user"},
                {"role": "assistant", "content": "stored assistant"},
            ],
            "metadata": [{}, {}, {}],
        },
    )

    var = SessionVariable(session, agent_id="root")
    assert var.messages("root.html") == [
        {"role": "system", "content": "stored system"},
        {"role": "user", "content": "stored user"},
        {"role": "assistant", "content": "stored assistant"},
    ]


def test_recent_returns_tail_of_messages(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    _seed_agent(
        session,
        agent_id="root.html",
        depth=1,
        system="child prompt",
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
    )

    var = SessionVariable(session, agent_id="root")
    recent = var.recent("root.html", n=2)

    assert recent == [
        {"role": "assistant", "content": 'write_file("index.html", "<!DOCTYPE html>")'},
        {"role": "user", "content": "REPL output:\nNone"},
    ]
    assert var.recent("root.html", n=0) == []
    assert var.recent("root.missing", n=3) == []


def test_messages_can_read_self(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    _seed_agent(
        session,
        agent_id="root",
        system="root prompt",
        query="build a thing",
        code='print("hello")',
        observation="REPL output:\nhello",
        result="done",
    )

    var = SessionVariable(session, agent_id="root")
    assert var.messages("root")[1]["content"] == "build a thing"


def test_grep_searches_across_agents_but_skips_self(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    _seed_agent(
        session,
        agent_id="root",
        query="self should be skipped",
        code='rlm_delegate(name="html", query="...", context="")',
        observation="self_only_marker",
        result="self_only_marker",
    )
    _seed_agent(
        session,
        agent_id="root.html",
        depth=1,
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
    )

    var = SessionVariable(session, agent_id="root")
    hits = var.grep("DOCTYPE").splitlines()
    assert hits, "expected at least one match"
    assert all(h.startswith("root.html:") for h in hits)
    assert (
        not var.grep("self_only_marker")
    ), "matches in self-agent must be filtered out"


def test_grep_respects_max_results(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    for i in range(5):
        _seed_agent(
            session,
            agent_id=f"root.child_{i}",
            depth=1,
            query="needle here",
            code='print("needle")',
            observation="needle observed",
            result="needle done",
        )

    var = SessionVariable(session, agent_id="root")
    assert len(var.grep("needle", max_results=3).splitlines()) == 3


def test_parent_and_ancestors_use_graph_edges_not_dot_parsing(tmp_path: Path):
    """Agent names with dots (``script.js``) must not fool tree navigation."""
    session = FileSession(tmp_path / "workspace")
    root_q, _ = _seed_agent(session, agent_id="root", result="root done")
    _seed_agent(
        session,
        agent_id="root.script.js",
        depth=1,
        result="ok",
        parent_agent_id="root",
        parent_node_id=root_q,
    )
    _seed_agent(
        session,
        agent_id="root.index.html",
        depth=1,
        result="ok",
        parent_agent_id="root",
        parent_node_id=root_q,
    )

    var = SessionVariable(session, agent_id="root.script.js")
    assert var.parent() == "root"
    assert var.parent("root.index.html") == "root"
    assert var.parent("root") is None
    assert var.ancestors() == ["root"]
    assert var.ancestors("root") == []


def test_children_and_subtree_only_return_real_agents(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    root_q, _ = _seed_agent(session, agent_id="root", result="root done")
    html_q, _ = _seed_agent(
        session,
        agent_id="root.html",
        depth=1,
        result="html done",
        parent_agent_id="root",
        parent_node_id=root_q,
    )
    _seed_agent(
        session,
        agent_id="root.css",
        depth=1,
        result="css done",
        parent_agent_id="root",
        parent_node_id=root_q,
    )
    _seed_agent(
        session,
        agent_id="root.html.physics",
        depth=2,
        result="physics done",
        parent_agent_id="root.html",
        parent_node_id=html_q,
    )

    var = SessionVariable(session, agent_id="root")

    assert var.children("root") == ["root.css", "root.html"]
    assert var.children("root.html") == ["root.html.physics"]
    assert var.children("root.html.physics") == []

    subtree_ids = sorted(a["agent_id"] for a in var.subtree("root"))
    assert subtree_ids == ["root.css", "root.html", "root.html.physics"]


def test_tree_renders_full_recursive_hierarchy(tmp_path: Path):
    session = FileSession(tmp_path / "workspace")
    root_q, _ = _seed_agent(session, agent_id="root", result="root done")
    _seed_agent(
        session, agent_id="root.a", depth=1,
        result="a", parent_agent_id="root", parent_node_id=root_q,
    )
    b_q, _ = _seed_agent(
        session, agent_id="root.b", depth=1,
        result="b", parent_agent_id="root", parent_node_id=root_q,
    )
    _seed_agent(
        session,
        agent_id="root.b.deep",
        depth=2,
        result="deep",
        parent_agent_id="root.b",
        parent_node_id=b_q,
    )

    var = SessionVariable(session, agent_id="root")
    rendered = var.tree()

    assert "root" in rendered
    assert "root.a" in rendered
    assert "root.b" in rendered
    assert "root.b.deep" in rendered


def test_session_variable_injected_via_inject_env(tmp_path: Path):
    """End-to-end: SessionVariable lands in the REPL namespace, not a dict."""
    from rlmflow import RLMConfig, RLMFlow, Workspace
    from rlmflow.llm import LLMClient

    class _StubLLM(LLMClient):
        def chat(self, messages, **kwargs):
            return '```repl\ndone("ok")\n```'

    workspace = Workspace.create(tmp_path / "ws")
    flow = RLMFlow(
        llm_client=_StubLLM(),
        config=RLMConfig(max_depth=1),
        workspace=workspace,
    )
    graph = flow.start("hi")
    runtime = flow.inject_env(graph, graph.states[0])

    ns = runtime.repl.namespace
    assert "SESSION" in ns
    assert "RLM_SESSION" not in ns
    handle = ns["SESSION"]
    assert isinstance(handle, SessionVariable)
    assert handle.agent_id == "root"
    assert callable(handle.list_agents)
    assert callable(handle.summarize_agent)
    assert callable(handle.read)
    assert callable(handle.messages)
    assert callable(handle.recent)
    assert callable(handle.grep)
