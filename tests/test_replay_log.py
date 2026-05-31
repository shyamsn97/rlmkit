"""Session protocol: persist per-agent invariants and per-turn states."""

from __future__ import annotations

import json

from rlmflow import (
    FileSession,
    Graph,
    InMemorySession,
    UserQuery,
    DoneOutput,
    SupervisingOutput,
)


def test_file_session_round_trips_states_through_load_graph(tmp_path):
    session = FileSession(tmp_path / "workspace")
    session.write_agent(Graph(agent_id="root", query="hello"))
    q = UserQuery(agent_id="root", seq=0, content="start")
    r = DoneOutput(agent_id="root", seq=1, result="done")
    session.write_state(q)
    session.write_state(r)

    graph = session.load_graph()
    assert "root" in graph
    states = graph.states
    assert [s.id for s in states] == [q.id, r.id]
    assert states[-1].result == "done"


def test_file_session_writes_plain_node_rows(tmp_path):
    session = FileSession(tmp_path / "workspace")
    session.write_agent(Graph(agent_id="root", query="hello"))

    session.write_state(UserQuery(agent_id="root", seq=0, content="start"))
    row = json.loads(
        (tmp_path / "workspace" / "session" / "root" / "session.jsonl").read_text()
    )

    assert row["type"] == "user_query"
    assert row["content"] == "start"


def test_in_memory_session_fork_isolates_subsequent_writes():
    source = InMemorySession()
    source.write_agent(Graph(agent_id="root", query="src"))
    source.write_state(UserQuery(agent_id="root", seq=0, content="source"))

    forked = source.fork(None)
    forked.write_state(UserQuery(agent_id="root", seq=1, content="forked"))

    src_contents = [s.content for s in source.load_graph().states]
    dst_contents = [s.content for s in forked.load_graph().states]
    assert src_contents == ["source"]
    assert dst_contents == ["source", "forked"]


def test_load_graph_derives_flows_to_edges_from_seq_order():
    session = InMemorySession()
    session.write_agent(Graph(agent_id="root", query="q"))
    a = UserQuery(agent_id="root", seq=0)
    b = UserQuery(agent_id="root", seq=1)
    session.write_state(a)
    session.write_state(b)

    edges = list(session.load_graph().edges)
    assert any(
        e.from_ == a.id and e.to == b.id and e.kind == "flows_to" for e in edges
    )


def test_child_agent_links_to_parent_via_parent_node_id():
    """Spawn edges are derived from the child's ``parent_node_id``."""
    session = InMemorySession()
    session.write_agent(Graph(agent_id="root", query="q"))
    parent = SupervisingOutput(agent_id="root", seq=1, waiting_on=["root.child"])
    session.write_state(parent)

    session.write_agent(
        Graph(
            agent_id="root.child",
            query="child task",
            parent_agent_id="root",
            parent_node_id=parent.id,
        )
    )
    child = UserQuery(agent_id="root.child", seq=0)
    session.write_state(child)

    graph = session.load_graph()
    spawn_edges = graph.edges.spawns()
    assert len(spawn_edges) == 1
    e = spawn_edges[0]
    assert e.from_ == parent.id
    assert e.to == child.id
    assert e.kind == "spawns"
    assert graph["root.child"].parent_id == "root"
    assert list(graph["root"].children) == ["root.child"]


def test_load_graph_skips_agents_without_meta(tmp_path):
    """If a state log exists but agent.json is missing, the agent is dropped."""
    session = FileSession(tmp_path / "workspace")
    session.write_agent(Graph(agent_id="root"))
    session.write_state(UserQuery(agent_id="root", seq=0))
    graph = session.load_graph()
    assert list(graph.agents) == ["root"]
