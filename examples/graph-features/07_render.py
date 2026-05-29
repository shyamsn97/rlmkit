"""Rendering a Graph: text trees, transcripts, HTML viewer.

Several read-only renderers ship with rlmflow:

- ``graph.tree()``           — ASCII tree of agents + states
- ``graph.session(...)``     — full chat-style transcript across the run
- ``graph.transcript(aid)``  — one agent's transcript only
- ``graph.save_html(path)``  — interactive viewer page over the snapshots

This script writes ``examples/graph-features/out/viewer.html`` (creating
``out/`` if needed) and prints the text renderers to stdout.

Run:
    python examples/graph-features/07_render.py
    open examples/graph-features/out/viewer.html
"""

from __future__ import annotations

from pathlib import Path

from rlmflow.graph import (
    ActionNode,
    Graph,
    QueryNode,
    ResultNode,
    SupervisingNode,
)


def build_graph() -> Graph:
    root_q = QueryNode(agent_id="root", seq=0, content="write hello world", iteration=1)
    root_action = ActionNode(
        agent_id="root",
        seq=1,
        reply="I'll delegate the file write.",
        code='await launch_subagent("write hello.py", name="hello")',
        iteration=2,
        input_tokens=120,
        output_tokens=30,
    )
    root_sup = SupervisingNode(
        agent_id="root", seq=2, waiting_on=["root.hello"], iteration=2,
    )
    root_done = ResultNode(agent_id="root", seq=3, result="hello.py created", iteration=4)

    hello_q = QueryNode(agent_id="root.hello", seq=0, content="write hello.py", iteration=2)
    hello_action = ActionNode(
        agent_id="root.hello",
        seq=1,
        reply="writing the file",
        code='write_file("hello.py", "print(\\"hello\\")\\n")',
        iteration=3,
        input_tokens=80,
        output_tokens=20,
    )
    hello_done = ResultNode(
        agent_id="root.hello", seq=2, result="wrote hello.py", iteration=3,
    )

    hello = Graph.from_meta_dict(
        {
            "agent_id": "root.hello",
            "depth": 1,
            "parent_agent_id": "root",
            "parent_node_id": root_action.id,
            "query": "write hello.py",
        },
        states=[hello_q, hello_action, hello_done],
    )
    return Graph.from_meta_dict(
        {"agent_id": "root", "depth": 0, "query": "write hello world"},
        states=[root_q, root_action, root_sup, root_done],
        children={"root.hello": hello},
    )


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    g = build_graph()

    banner("graph.tree() — ASCII summary")
    print(g.tree())

    banner("graph.transcript('root.hello') — single-agent transcript")
    print(g.transcript("root.hello", include_system=False))

    banner("graph.session() — full chat-style transcript")
    print(g.session(include_system=False))

    banner("graph.save_html(...) — interactive viewer over the history")
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = g.save_html(out_dir / "viewer.html")
    print(f"wrote {html_path} ({html_path.stat().st_size:,} bytes)")
    print(f"\nopen with: open {html_path}")


if __name__ == "__main__":
    main()
