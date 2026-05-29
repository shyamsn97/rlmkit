"""Querying a Graph: flat views, filters, find(), tokens.

Builds a small recursive Graph by hand and walks every read-side surface:

- ``graph.nodes`` — flat NodesView over every state in the subtree
- ``graph.agents`` — Mapping[agent_id, sub-Graph] over the subtree
- ``graph.edges`` — derived flow + spawn edges
- ``.where(...)`` / ``.queries()`` / ``.actions()`` / ``.errors()`` filters
- ``graph.find(node_id)`` — by-id lookup
- ``graph.tokens()`` / ``graph.total_tokens()`` — recursive rollups
- ``graph.result()`` / ``graph.current()``

Run:
    python examples/graph-features/01_query.py
"""

from __future__ import annotations

from rlmflow.graph import (
    ActionNode,
    ErrorNode,
    Graph,
    QueryNode,
    ResultNode,
    SupervisingNode,
)


def build_graph() -> Graph:
    """A root with two children, one of which errors before succeeding."""
    root_q = QueryNode(agent_id="root", seq=0, content="ship a tiny package")
    root_action = ActionNode(
        agent_id="root",
        seq=1,
        reply="splitting into two children",
        code='await launch_subagents([{"name": "write", "query": "..."}, {"name": "test", "query": "..."}])',
        input_tokens=120,
        output_tokens=40,
    )
    root_sup = SupervisingNode(
        agent_id="root",
        seq=2,
        waiting_on=["root.write", "root.test"],
        input_tokens=0,
        output_tokens=0,
    )
    root_done = ResultNode(agent_id="root", seq=3, result="package shipped")

    write_q = QueryNode(agent_id="root.write", seq=0, content="write the module")
    write_done = ResultNode(agent_id="root.write", seq=1, result="wrote pkg/__init__.py")

    test_q = QueryNode(agent_id="root.test", seq=0, content="run pytest")
    test_err = ErrorNode(
        agent_id="root.test",
        seq=1,
        error="exec_error",
        content="ModuleNotFoundError: pkg",
    )
    test_action = ActionNode(
        agent_id="root.test",
        seq=2,
        reply="retrying after install",
        code="install_and_run()",
        input_tokens=80,
        output_tokens=20,
    )
    test_done = ResultNode(agent_id="root.test", seq=3, result="3 passed")

    write = Graph.from_meta_dict(
        {
            "agent_id": "root.write",
            "depth": 1,
            "parent_agent_id": "root",
            "parent_node_id": root_action.id,
        },
        states=[write_q, write_done],
    )
    test = Graph.from_meta_dict(
        {
            "agent_id": "root.test",
            "depth": 1,
            "parent_agent_id": "root",
            "parent_node_id": root_action.id,
        },
        states=[test_q, test_err, test_action, test_done],
    )
    return Graph.from_meta_dict(
        {"agent_id": "root", "depth": 0, "query": "ship a tiny package"},
        states=[root_q, root_action, root_sup, root_done],
        children={"root.write": write, "root.test": test},
    )


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    g = build_graph()

    banner("flat views over the whole subtree")
    print(f"agents : {list(g.agents)}")
    print(f"nodes  : {len(g.nodes)} states")
    print(f"edges  : {len(g.edges)}  ({len(g.edges.flows_to())} flows_to, "
          f"{len(g.edges.spawns())} spawns)")

    banner("filters on graph.nodes")
    print(f"queries     : {len(g.nodes.queries())}")
    print(f"actions     : {len(g.nodes.actions())}")
    print(f"errors      : {[n.error for n in g.nodes.errors()]}")
    print(f"results     : {[n.result for n in g.nodes.results()]}")

    long_replies = g.nodes.where(lambda n: getattr(n, "reply", "") and len(n.reply) > 20)
    print(f"long replies: {len(long_replies)}")

    banner("graph.find by id")
    sup = g.nodes.where(type="supervising")[0]
    found = g.find(sup.id)
    print(f"find({sup.id[:10]}…) -> {type(found).__name__} agent={found.agent_id}")

    banner("token rollups")
    inp, out = g.tokens()
    print(f"input tokens : {inp}")
    print(f"output tokens: {out}")
    print(f"total        : {g.total_tokens()}")

    banner("terminal helpers")
    print(f"finished : {g.finished}")
    print(f"current  : {g.current().type}")
    print(f"result   : {g.result()!r}")


if __name__ == "__main__":
    main()
