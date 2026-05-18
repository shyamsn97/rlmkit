"""Replaying a real engine run with ``graph.history()``.

The replay system is genuinely driven by the engine. Every call to
``RLMFlow.step()`` bumps an internal counter; every node it appends is
stamped with that counter as its ``Node.iteration`` field. ``history()``
just buckets states by iteration and snapshots the cumulative graph
after each bucket.

This script:
  1. runs the real engine end-to-end with a tiny scripted LLM,
  2. prints the iteration stamps the engine wrote,
  3. calls ``graph.history()`` and shows the snapshots line up with the
     engine's actual scheduling rounds.

Two children advance in parallel during the same step, which shows up as
both children's results landing in the same iteration bucket.

Run:
    python examples/graph-features/05_replay.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rlmflow import LLMClient, LLMUsage, RLMConfig, RLMFlow, Workspace


ROOT_SPLIT = (
    "```repl\n"
    'h1 = delegate("a", "do A", "")\n'
    'h2 = delegate("b", "do B", "")\n'
    "results = yield wait(h1, h2)\n"
    'done("/".join(results))\n'
    "```"
)


class ScriptedLLM(LLMClient):
    """Tiny deterministic LLM: root delegates, each child returns its name."""

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        last = messages[-1]["content"]
        if "do A" in last:
            return '```repl\ndone("A done")\n```'
        if "do B" in last:
            return '```repl\ndone("B done")\n```'
        return ROOT_SPLIT


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace.create(Path(tmp).resolve() / "ws")
        engine = RLMFlow(
            llm_client=ScriptedLLM(),
            workspace=workspace,
            config=RLMConfig(max_depth=1, max_iterations=5, max_concurrency=2),
        )

        banner("running the engine — one tick per step()")
        graph = engine.start("split into A and B")
        tick = 0
        while not graph.finished:
            tick += 1
            graph = engine.step(graph)
            print(f"step {tick}: agents=" + ", ".join(
                f"{aid}:{sub.current().type}"
                for aid, sub in graph.agents.items()
            ))
        print(f"\nfinal result: {graph.result()!r}")

        banner("iteration stamps written by the engine")
        for n in graph.nodes:
            tag = (
                getattr(n, "result", None)
                or getattr(n, "content", None)
                or getattr(n, "reply", None)
                or ""
            )
            preview = tag.splitlines()[0][:50] if tag else ""
            print(f"  iter={n.iteration}  {n.agent_id:<7} {n.type:<12} {preview!r}")

        banner("graph.history() — one snapshot per engine step()")
        snapshots = graph.history()
        print(f"{len(snapshots)} snapshots ({tick} engine ticks)\n")
        for i, snap in enumerate(snapshots, start=1):
            agents = ", ".join(
                f"{aid}:{sub.current().type}" for aid, sub in snap.agents.items()
            )
            print(f"snapshot {i}  states={len(snap.nodes)}  ({agents})")

        banner("the parallel snapshot — both children advance together")
        for snap in snapshots:
            kids = [s for aid, s in snap.agents.items() if aid != "root"]
            if kids and all(k.finished for k in kids):
                print(snap.tree())
                break


if __name__ == "__main__":
    main()
