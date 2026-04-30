"""Session-store tests replacing the removed replay-log contract."""

from __future__ import annotations

from rlmflow import FileSession, InMemorySession, QueryNode, ResultNode, SupervisingNode


def test_file_session_appends_and_loads_typed_nodes(tmp_path):
    session = FileSession(tmp_path / "session")
    root = QueryNode(agent_id="root", content="start")
    result = ResultNode(agent_id="root", result="done")

    session.write(root)
    session.write(result)

    loaded = session.load()
    assert loaded[root.id].content == "start"
    assert loaded[result.id].result == "done"


def test_file_session_latest_agent_view_tracks_terminal_result(tmp_path):
    session = FileSession(tmp_path / "session")
    session.write(QueryNode(agent_id="root.worker", depth=1))
    session.write(ResultNode(agent_id="root.worker", depth=1, result="ok"))

    view = (tmp_path / "session" / "agents" / "root.worker.json").read_text()

    assert '"depth": 1' in view
    assert '"type": "result"' in view
    assert '"terminal": true' in view
    assert '"result": "ok"' in view


def test_in_memory_session_fork_isolates_subsequent_writes():
    source = InMemorySession()
    root = QueryNode(agent_id="root", content="source")
    source.write(root)

    forked = source.fork(None)
    divergent = QueryNode(agent_id="root", content="forked")
    forked.write(divergent)

    assert divergent.id not in source.load()
    assert divergent.id in forked.load()


def test_session_chain_to_follows_same_agent_successors():
    session = InMemorySession()
    first = QueryNode(agent_id="root", content="first")
    second = first.successor(QueryNode, content="second")
    third = second.successor(ResultNode, result="done")

    session.write(first.update(children=[second.id]))
    session.write(second.update(children=[third.id]))
    session.write(third)

    assert [node.id for node in session.chain_to(third)] == [
        first.id,
        second.id,
        third.id,
    ]


def test_session_chain_stops_at_nested_agent_boundary():
    session = InMemorySession()
    child = QueryNode(agent_id="root.child", depth=1, content="child")
    root = SupervisingNode(agent_id="root", waiting_on=["root.child"], children=[child])

    session.write(root)
    session.write(child)

    assert session.chain_to(child) == [child]
