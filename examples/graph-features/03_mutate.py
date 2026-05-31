"""Editing a Graph in place.

A Graph is mutable. The engine writes through ``add_state`` /
``add_child`` during a run, but you can use the same helpers offline to
edit a persisted graph: rewrite a result, drop an agent, swap a node.

Demonstrates:

- ``graph.add_state(node)``                   — append a state
- ``graph.update_state(node_id, **changes)``  — copy-with-changes by id
- ``graph.replace_state(node_id, new_node)``  — full swap by id
- ``graph.remove_state(node_id)``             — drop a state
- ``graph.add_child(child)`` / ``graph.remove_child(aid)``
- ``graph.update(**fields)``                  — bulk top-level edit
- ``graph.nodes.replace / update / remove``   — by id, anywhere in subtree
- ``graph.copy(deep=True)``                   — clone before mutating

Run:
    python examples/graph-features/03_mutate.py
"""

from __future__ import annotations

from rlmflow.graph import DoneOutput, Graph, UserQuery


def base_graph() -> Graph:
    root_q = UserQuery(agent_id="root", seq=0, content="hello")
    root_done = DoneOutput(agent_id="root", seq=1, result="ok")
    child_q = UserQuery(agent_id="root.child", seq=0, content="sub")
    child_done = DoneOutput(agent_id="root.child", seq=1, result="sub ok")
    child = Graph.from_meta_dict(
        {"agent_id": "root.child", "depth": 1, "parent_agent_id": "root"},
        states=[child_q, child_done],
    )
    return Graph.from_meta_dict(
        {"agent_id": "root", "depth": 0, "query": "hello"},
        states=[root_q, root_done],
        children={"root.child": child},
    )


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def summary(g: Graph) -> str:
    return (
        f"agents={list(g.agents)} states={len(g.nodes)} "
        f"result={g.result()!r} model={g.model_label}"
    )


def main() -> None:
    g = base_graph()
    banner("baseline")
    print(summary(g))

    banner("graph.copy(deep=True) — clone before mutating")
    twin = g.copy(deep=True)
    twin.update(model="gpt-5", config={"temperature": 0.0})
    print(f"original: {summary(g)}")
    print(f"twin    : {summary(twin)}")

    banner("update_state — copy-with-changes by id")
    result_id = g.nodes.results()[0].id
    g.update_state(result_id, result="ok (rewritten)")
    print(summary(g))

    banner("replace_state — swap a state object")
    g.replace_state(result_id, DoneOutput(
        agent_id="root", seq=1, result="ok (full swap)", id=result_id,
    ))
    print(summary(g))

    banner("nodes.update — same edit, but addressed via the flat view")
    child_result_id = g["root.child"].nodes.results()[0].id
    g.nodes.update(child_result_id, result="sub ok (via subtree view)")
    print(f"root.child result -> {g['root.child'].result()!r}")

    banner("add_state — append onto a sub-Graph")
    g["root.child"].add_state(UserQuery(agent_id="root.child", seq=2, content="follow-up"))
    print(f"root.child states: {[n.type for n in g['root.child'].states]}")

    banner("add_child / remove_child — attach + detach sub-agents")
    sibling = Graph.from_meta_dict(
        {"agent_id": "root.sibling", "depth": 1, "parent_agent_id": "root"},
        states=[UserQuery(agent_id="root.sibling", seq=0, content="hi")],
    )
    g.add_child(sibling)
    print(f"after add_child : {list(g.agents)}")
    g.remove_child("root.child")
    print(f"after remove    : {list(g.agents)}")

    banner("graph.update — bulk top-level field edit")
    g.update(query="hello (updated)", config={"max_depth": 2})
    print(f"query={g.query!r} config={g.config}")


if __name__ == "__main__":
    main()
