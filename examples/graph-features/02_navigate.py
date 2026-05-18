"""Navigating a Graph: indexing, dotted paths, walk(), parent/child links.

A Graph is one agent — recursion lives in ``graph.children``. You can
descend by id, by dotted path, or just iterate.

Demonstrates:

- ``graph[aid]`` — sub-Graph for an agent (raises KeyError if missing)
- dotted paths like ``graph["root.write.linter"]``
- ``graph.walk()`` / ``graph.subtree()`` — depth-first agent iteration
- ``len(graph)`` and ``for aid in graph`` — agent count + ids
- ``parent_id`` / ``parent_node_id`` — the spawn link back upward

Run:
    python examples/graph-features/02_navigate.py
"""

from __future__ import annotations

from rlmflow.graph import ActionNode, Graph, QueryNode, ResultNode


def build_graph() -> Graph:
    root_q = QueryNode(agent_id="root", seq=0, content="ship pkg")
    root_action = ActionNode(agent_id="root", seq=1, reply="split", code="...")

    writer_q = QueryNode(agent_id="root.write", seq=0, content="write code")
    writer_action = ActionNode(agent_id="root.write", seq=1, reply="lint", code="...")
    linter_q = QueryNode(agent_id="root.write.linter", seq=0, content="lint pass")
    linter_done = ResultNode(agent_id="root.write.linter", seq=1, result="clean")

    test_q = QueryNode(agent_id="root.test", seq=0, content="run pytest")
    test_done = ResultNode(agent_id="root.test", seq=1, result="3 passed")

    linter = Graph.from_meta_dict(
        {
            "agent_id": "root.write.linter",
            "depth": 2,
            "parent_agent_id": "root.write",
            "parent_node_id": writer_action.id,
        },
        states=[linter_q, linter_done],
    )
    writer = Graph.from_meta_dict(
        {
            "agent_id": "root.write",
            "depth": 1,
            "parent_agent_id": "root",
            "parent_node_id": root_action.id,
        },
        states=[writer_q, writer_action],
        children={"root.write.linter": linter},
    )
    tester = Graph.from_meta_dict(
        {
            "agent_id": "root.test",
            "depth": 1,
            "parent_agent_id": "root",
            "parent_node_id": root_action.id,
        },
        states=[test_q, test_done],
    )
    return Graph.from_meta_dict(
        {"agent_id": "root", "depth": 0, "query": "ship pkg"},
        states=[root_q, root_action],
        children={"root.write": writer, "root.test": tester},
    )


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    g = build_graph()

    banner("size + iteration")
    print(f"len(graph)          : {len(g)} agents")
    print(f"list(graph)         : {list(g)}")
    print(f"list(graph.agents)  : {list(g.agents)}")

    banner("indexing by id")
    writer = g["root.write"]
    print(f"g['root.write']        -> depth={writer.depth} states={len(writer.states)}")
    linter = g["root.write.linter"]
    print(f"g['root.write.linter'] -> depth={linter.depth} states={len(linter.states)}")
    print(f"'root.test' in g       -> {'root.test' in g}")
    print(f"'root.missing' in g    -> {'root.missing' in g}")

    banner("dotted-path descent")
    same_linter = g["root"]["root.write"]["root.write.linter"]
    print(f"g['root']['root.write']['root.write.linter'] is g[full] -> "
          f"{same_linter is linter}")

    banner("walk() / subtree() depth-first")
    for sub in g.walk():
        indent = "  " * sub.depth
        tip = sub.current()
        print(f"{indent}{sub.agent_id}  ({len(sub.states)} states, tip="
              f"{tip.type if tip else 'empty'})")

    banner("parent links")
    for sub in g.walk():
        print(f"{sub.agent_id:<24} parent_id={sub.parent_id!s:<12} "
              f"parent_node_id={(sub.parent_node_id or '-')[:14]}")


if __name__ == "__main__":
    main()
