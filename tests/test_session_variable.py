"""SessionVariable — cross-agent transcript access from inside the REPL."""

from __future__ import annotations

from pathlib import Path

from rlmflow.node import (
    ActionNode,
    ErrorNode,
    ObservationNode,
    QueryNode,
    ResultNode,
)
from rlmflow.workspace import FileSession, InMemorySession, SessionVariable


def _seed_session(
    session,
    *,
    agent_id: str,
    system: str = "",
    query: str = "",
    code: str = "",
    observation: str = "",
    result: str = "",
    delegated_by: str | None = None,
) -> str:
    """Append a minimal query → action → observation → result chain for one agent.

    rlmflow stores forward links: ``prev.children`` lists successor ids. We
    seed bottom-up so each parent can record its child's id at construction.
    Returns the id of the leaf result node.

    If ``delegated_by`` is set, also write a parent-side ``SupervisingNode``
    whose ``children`` references this agent's query — encoding the
    cross-agent delegation edge the same way the real engine does.
    """
    r = ResultNode(agent_id=agent_id, result=result)
    o = ObservationNode(agent_id=agent_id, content=observation, children=[r.id])
    a = ActionNode(agent_id=agent_id, reply=code, code=code, children=[o.id])
    q = QueryNode(
        agent_id=agent_id,
        query=query,
        content=query,
        system_prompt=system,
        children=[a.id],
    )
    for n in (q, a, o, r):
        session.write(n)
    if delegated_by is not None:
        from rlmflow.node import SupervisingNode

        bridge = SupervisingNode(
            agent_id=delegated_by,
            code="",
            output="",
            waiting_on=[agent_id],
            children=[q.id],
        )
        session.write(bridge)
    return r.id


def test_list_agents_excludes_self_and_summarizes_siblings(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    _seed_session(
        session,
        agent_id="root",
        system="root prompt",
        query="build a thing",
        code='delegate("html", "...")',
        observation="ok",
        result="all done",
    )
    _seed_session(
        session,
        agent_id="root.html",
        system="child prompt",
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
    )

    var = SessionVariable(session, agent_id="root")
    rows = var.list_agents()
    ids = [r["agent_id"] for r in rows]
    assert ids == ["root.html"], "self-agent should be excluded"
    [child] = rows
    assert child["type"] == "result"
    assert child["terminal"] is True
    assert "wrote index.html" in child["result_preview"]


def test_read_renders_full_chain_for_a_sibling(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    _seed_session(
        session,
        agent_id="root.html",
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


def test_read_unknown_agent_returns_clear_message(tmp_path: Path):
    var = SessionVariable(InMemorySession(), agent_id="root")
    assert var.read("root.missing") == "(no nodes for agent 'root.missing')"


def test_grep_searches_across_agents_but_skips_self(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    _seed_session(
        session,
        agent_id="root",
        system="root",
        query="self should be skipped",
        code='delegate("html", "...")',
        observation="self_only_marker",
        result="self_only_marker",
    )
    _seed_session(
        session,
        agent_id="root.html",
        system="child",
        query="write index.html",
        code='write_file("index.html", "<!DOCTYPE html>")',
        observation="REPL output:\nNone",
        result="wrote index.html",
    )

    var = SessionVariable(session, agent_id="root")
    hits = var.grep("DOCTYPE").splitlines()
    assert hits, "expected at least one match"
    assert all(h.startswith("root.html:") for h in hits)
    assert not var.grep("self_only_marker"), "matches in self-agent must be filtered out"


def test_grep_respects_max_results(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    for i in range(5):
        _seed_session(
            session,
            agent_id=f"root.child_{i}",
            system="s",
            query="needle here",
            code='print("needle")',
            observation="needle observed",
            result="needle done",
        )

    var = SessionVariable(session, agent_id="root")
    assert len(var.grep("needle", max_results=3).splitlines()) == 3


def test_parent_and_ancestors_use_graph_edges_not_dot_parsing(tmp_path: Path):
    """Agent names with dots (``script.js``) must not fool tree navigation."""
    session = FileSession(tmp_path / "session")
    _seed_session(session, agent_id="root", result="root done")
    _seed_session(
        session, agent_id="root.script.js", result="ok", delegated_by="root"
    )
    _seed_session(
        session, agent_id="root.index.html", result="ok", delegated_by="root"
    )

    var = SessionVariable(session, agent_id="root.script.js")

    # Despite the dot in 'script.js', parent is root — not root.script.
    assert var.parent() == "root"
    assert var.parent("root.index.html") == "root"
    assert var.parent("root") is None
    assert var.ancestors() == ["root"]
    assert var.ancestors("root") == []


def test_children_and_subtree_only_return_real_agents(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    _seed_session(session, agent_id="root", result="root done")
    _seed_session(
        session, agent_id="root.html", result="html done", delegated_by="root"
    )
    _seed_session(
        session, agent_id="root.css", result="css done", delegated_by="root"
    )
    _seed_session(
        session,
        agent_id="root.html.physics",
        result="physics done",
        delegated_by="root.html",
    )

    var = SessionVariable(session, agent_id="root")

    # Direct children of root: just html and css (NOT html.physics).
    assert var.children("root") == ["root.css", "root.html"]
    assert var.children("root.html") == ["root.html.physics"]
    assert var.children("root.html.physics") == []

    # subtree includes every descendant, regardless of depth.
    subtree_ids = sorted(a["agent_id"] for a in var.subtree("root"))
    assert subtree_ids == ["root.css", "root.html", "root.html.physics"]


def test_tree_renders_full_recursive_hierarchy(tmp_path: Path):
    session = FileSession(tmp_path / "session")
    _seed_session(session, agent_id="root", result="root done")
    _seed_session(session, agent_id="root.a", result="a", delegated_by="root")
    _seed_session(session, agent_id="root.b", result="b", delegated_by="root")
    _seed_session(
        session, agent_id="root.b.deep", result="deep", delegated_by="root.b"
    )

    var = SessionVariable(session, agent_id="root")
    rendered = var.tree()

    assert rendered.splitlines()[0].startswith("root [")
    assert "├── root.a" in rendered
    assert "└── root.b" in rendered
    assert "└── root.b.deep" in rendered
    # Indentation under root.b uses spaces (last sibling), not pipes.
    assert "    └── root.b.deep" in rendered


def test_session_variable_injected_via_prepare_runtime(tmp_path: Path):
    """End-to-end: SessionVariable lands in the REPL namespace, not a dict."""
    from rlmflow import QueryNode, RLMConfig, RLMFlow, Workspace
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
    node = QueryNode(agent_id="root.test", query="hi", content="hi")
    runtime = flow.prepare_runtime(node)

    ns = runtime.repl.namespace
    assert "SESSION" in ns, "SESSION must be injected"
    assert "RLM_SESSION" not in ns, "old placeholder dict should be gone"
    handle = ns["SESSION"]
    assert isinstance(handle, SessionVariable)
    assert handle.agent_id == "root.test"
    assert callable(handle.list_agents)
    assert callable(handle.read)
    assert callable(handle.grep)
