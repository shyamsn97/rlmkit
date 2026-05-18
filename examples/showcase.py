"""Showcase the Graph-centric RLMFlow API.

This walks through the pieces that matter after the engine refactor:

1. Step-by-step execution returning :class:`~rlmflow.graph.Graph` snapshots.
2. Workspace persistence through ``workspace.load_graph()``.
3. Session layout and latest-state inspection.
4. Optional in-process history by keeping graph snapshots.
5. Graph summary helpers (``graph.tree()``, ``graph.tokens()``).
6. Gym-style stepping with a scalar reward.

Usage:
    python examples/showcase.py
    python examples/showcase.py --no-viz
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rlmflow.graph import Graph
from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.rlm import RLMConfig, RLMFlow
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import FILE_TOOLS
from rlmflow.workspace import Workspace

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class DemoLLM(LLMClient):
    """Deterministic LLM for an offline showcase."""

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = LLMUsage(input_tokens=80, output_tokens=20)
        prompt = messages[-1]["content"].lower()
        if "hello.py" in prompt and "goodbye.py" in prompt:
            return (
                "```repl\n"
                'h1 = delegate("hello", "Create hello.py", "")\n'
                'h2 = delegate("goodbye", "Create goodbye.py", "")\n'
                "results = yield wait(h1, h2)\n"
                'done("\\n".join(results))\n'
                "```"
            )
        if "hello.py" in prompt:
            return '```repl\nwrite_file("hello.py", "print(\\"hello\\")\\n")\ndone("hello.py")\n```'
        if "goodbye.py" in prompt:
            return '```repl\nwrite_file("goodbye.py", "print(\\"goodbye\\")\\n")\ndone("goodbye.py")\n```'
        if "haiku" in prompt:
            return '```repl\nwrite_file("haiku.txt", "Calls fold into calls\\nNodes branch, wait, and then resume\\nFlow returns a leaf\\n")\ndone("wrote haiku.txt")\n```'
        return '```repl\ndone("ok")\n```'


def banner(msg: str) -> None:
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}\n")


def make_agent(workspace: Workspace, *, max_depth: int, max_iterations: int) -> RLMFlow:
    runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools(FILE_TOOLS)
    return RLMFlow(
        llm_client=DemoLLM(),
        runtime=runtime,
        workspace=workspace,
        config=RLMConfig(max_depth=max_depth, max_iterations=max_iterations),
    )


def run(agent: RLMFlow, graph: Graph, no_viz: bool) -> list[Graph]:
    if no_viz:
        history = [graph]
        step = 0
        while not graph.finished:
            graph = agent.step(graph)
            step += 1
            history.append(graph)
            print(f"-- step {step} --")
            print(graph.tree())
        return history
    from rlmflow.utils.viz import live

    return live(agent, graph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace.create(Path(tmpdir) / "workspace")
        agent = make_agent(
            workspace,
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
        )

        banner("1. Step-by-step execution")
        graph = agent.start("Create hello.py and goodbye.py. Delegate each file.")
        history = run(agent, graph, args.no_viz)
        final = history[-1]
        print(f"\n{GREEN}Result:{RESET} {final.result()}")

        banner("2. Workspace persistence")
        loaded = workspace.load_graph()
        print(
            f"Loaded graph with {len(loaded.agents)} agents and "
            f"{len(loaded.nodes)} states from {workspace.root}"
        )
        print(loaded.tree())

        banner("3. Session layout")
        reloaded = workspace.load_graph()
        print(
            f"Persisted {len(reloaded.nodes)} states across "
            f"{len(reloaded.agents)} agents in {workspace.root / 'session'}"
        )
        print("Latest state per agent:")
        for aid, sub in reloaded.agents.items():
            current = sub.current()
            label = current.type if current else "(empty)"
            print(f"  {aid}: {label}")

        banner("4. Time travel")
        for idx, snapshot in enumerate(history):
            current = snapshot.current()
            kind = current.type if current else "empty"
            print(
                f"{CYAN}step {idx}{RESET}: root [{kind}]  "
                f"agents={len(snapshot.agents)}"
            )

        banner("5. Graph summary")
        inp, out = final.tokens()
        print(f"Agents:  {len(final.agents)}")
        print(f"States:  {len(final.nodes)}")
        print(f"Tokens:  {inp + out:,} ({inp:,} in / {out:,} out)")
        print(f"Final:   {final.current().type if final.current() else '(empty)'}")

        banner("6. Gym-style loop")
        agent3 = make_agent(workspace, max_depth=0, max_iterations=args.max_iterations)
        graph3 = agent3.start("Write a haiku about recursion to haiku.txt")
        rewards: list[float] = []
        step = 0
        while not graph3.finished:
            graph3 = agent3.step(graph3)
            step += 1
            current = graph3.current()
            reward = 1.0 if graph3.finished else 0.0
            rewards.append(reward)
            kind = current.type if current else "empty"
            print(f"step {step}: state={kind} reward={reward}")
        print(f"{GREEN}Result:{RESET} {graph3.result()}")
        print(f"Total reward: {sum(rewards):.1f}")

        banner("Done")


if __name__ == "__main__":
    main()
