"""Forking a Workspace: branch a run, diverge, compare outcomes.

``Workspace.fork(new_branch_id, new_dir)`` copies the working tree,
session log, and context store into a new directory and returns a fresh
Workspace handle. Subsequent writes go into the new branch only — the
source workspace stays untouched.

Use it for:

- repair branches (try fix A vs fix B from the same starting point)
- best-of-N exploration (fan out a partial run multiple ways)
- speculative edits without disturbing the canonical run

This script:
  1. seeds a workspace with a deterministic mock LLM,
  2. forks twice (one branch keeps going, the other is replayed with a
     different LLM),
  3. shows that the two diverged graphs are independent.

Run:
    python examples/graph-features/06_fork.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from rlmflow import LLMClient, LLMUsage, RLMConfig, RLMFlow, Workspace


class ScriptedLLM(LLMClient):
    """Returns scripted REPL blocks one per call so the engine terminates."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.idx = 0

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        reply = self.replies[min(self.idx, len(self.replies) - 1)]
        self.idx += 1
        return reply


def run(workspace: Workspace, llm: LLMClient, query: str) -> str:
    engine = RLMFlow(
        llm_client=llm,
        workspace=workspace,
        config=RLMConfig(max_iterations=3),
    )
    graph = engine.start(query)
    while not graph.finished:
        graph = engine.step(graph)
    return graph.result()


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()

        banner("seed: a fresh main branch")
        main_ws = Workspace.create(root / "main", branch_id="main")
        # Pre-seed a side file in the working tree so we can show it copies on fork.
        main_ws.path("notes.md").write_text("# starting notes\n")
        seeded_result = run(
            main_ws,
            ScriptedLLM(['```repl\ndone("seeded result")\n```']),
            "do the thing",
        )
        print(f"main branch result: {seeded_result!r}")
        print(f"main branch files : {sorted(p.name for p in main_ws.root.iterdir())}")

        banner("fork twice — each branch gets its own copy")
        # The fork helper deletes the destination if it exists, so make sure
        # we hand it a fresh path.
        for branch in ("retry_a", "retry_b"):
            shutil.rmtree(root / branch, ignore_errors=True)
        a = main_ws.fork(new_branch_id="retry_a", new_dir=root / "retry_a")
        b = main_ws.fork(new_branch_id="retry_b", new_dir=root / "retry_b")
        print(f"main session.jsonl   : {(main_ws.root / 'session/root/session.jsonl').stat().st_size}b")
        print(f"retry_a session.jsonl: {(a.root / 'session/root/session.jsonl').stat().st_size}b "
              f"(copied from main)")
        print(f"retry_b session.jsonl: {(b.root / 'session/root/session.jsonl').stat().st_size}b "
              f"(copied from main)")
        print(f"working tree carried over: {(a.root / 'notes.md').read_text().strip()!r}")

        banner("diverge: each branch keeps running with its own LLM")
        # Append a *new* run into each branch — the seeded states stay,
        # and the new ones get appended after them.
        a_result = run(
            a,
            ScriptedLLM(['```repl\ndone("retry_a took the careful path")\n```']),
            "redo with caution",
        )
        b_result = run(
            b,
            ScriptedLLM(['```repl\ndone("retry_b took the bold path")\n```']),
            "redo aggressively",
        )

        print(f"retry_a: {a_result!r}")
        print(f"retry_b: {b_result!r}")

        banner("the source workspace is unchanged")
        main_loaded = Workspace.open_path(main_ws.root).load_graph()
        print(f"main result still : {main_loaded.result()!r}")
        print(f"main state count  : {len(main_loaded.nodes)}")

        banner("compare branches by result")
        for ws in (main_ws, a, b):
            g = Workspace.open_path(ws.root).load_graph()
            print(f"  branch={ws.branch_id:<8} states={len(g.nodes):>2} "
                  f"result={g.result()!r}")


if __name__ == "__main__":
    main()
