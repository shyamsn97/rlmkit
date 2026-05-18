"""Persisting a Graph: JSON dump/load and Workspace round-trip.

Two ways to put a Graph on disk:

1. ``graph.save(path)`` / ``Graph.load(path)`` — single JSON blob.
   Good for sharing one snapshot, or pickling a curated graph.
2. ``Workspace.create(...)`` + ``workspace.load_graph()`` — the engine's
   normal append-only run directory. Reading replays
   ``session/<aid>/session.jsonl`` for every agent.

This script does both with a tiny mock LLM (no API key) so you can see
that the persisted graph round-trips identically.

Run:
    python examples/graph-features/04_save_load.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rlmflow import (
    Graph,
    LLMClient,
    LLMUsage,
    QueryNode,
    RLMConfig,
    RLMFlow,
    ResultNode,
    Workspace,
)


class DummyLLM(LLMClient):
    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        return '```repl\ndone("ok")\n```'


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp).resolve()

        banner("1. Graph.save / Graph.load — single-file JSON")
        g = Graph.from_meta_dict(
            {"agent_id": "root", "depth": 0, "query": "hi"},
            states=[
                QueryNode(agent_id="root", seq=0, content="hi"),
                ResultNode(agent_id="root", seq=1, result="hello"),
            ],
        )
        path = g.save(tmp_path / "graph.json")
        print(f"wrote {path.relative_to(tmp_path)} "
              f"({path.stat().st_size} bytes)")
        loaded = Graph.load(path)
        print(f"reloaded: agents={list(loaded.agents)} result={loaded.result()!r}")
        print(f"identical roundtrip: {g.to_dict() == loaded.to_dict()}")

        # Peek at the JSON shape so you know what's persisted.
        print("\nfirst-state JSON keys:")
        first = json.loads(path.read_text())["states"][0]
        for k, v in first.items():
            print(f"  {k:<10} {v!r}")

        banner("2. Workspace round-trip — append-only run directory")
        workspace = Workspace.create(tmp_path / "ws")
        engine = RLMFlow(
            llm_client=DummyLLM(),
            workspace=workspace,
            config=RLMConfig(max_iterations=2),
        )
        graph = engine.start("hello workspace")
        while not graph.finished:
            graph = engine.step(graph)

        print(f"workspace root: {workspace.root.relative_to(tmp_path)}/")
        for p in sorted(workspace.root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(workspace.root)
                print(f"  {rel}  ({p.stat().st_size} bytes)")

        reloaded = workspace.load_graph()
        print(f"\nload_graph(): agents={list(reloaded.agents)} "
              f"result={reloaded.result()!r}")
        print(f"states match in-memory: "
              f"{[n.type for n in reloaded.nodes] == [n.type for n in graph.nodes]}")


if __name__ == "__main__":
    main()
