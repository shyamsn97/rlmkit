"""Showcase the current node-first RLMFlow API.

This walks through the pieces that still matter after the refactor:

1. Step-by-step execution over typed nodes.
2. Checkpoint round-trip with `Node.save/load`.
3. Session graph persistence through `Workspace.session`.
4. Time travel by keeping a list of node snapshots.
5. Manual intervention by replacing a child leaf.
6. Gym-style stepping with a scalar reward.

Usage:
    python examples/showcase.py
    python examples/showcase.py --no-viz
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rlmkit.llm import LLMClient, LLMUsage
from rlmkit.node import Node, ResultNode, SupervisingNode
from rlmkit.rlm import RLMConfig, RLMFlow
from rlmkit.runtime.local import LocalRuntime
from rlmkit.tools import FILE_TOOLS
from rlmkit.workspace import Workspace

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
                'h1 = delegate("hello", "Create hello.py")\n'
                'h2 = delegate("goodbye", "Create goodbye.py")\n'
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


def run(agent: RLMFlow, state: Node, no_viz: bool) -> list[Node]:
    if no_viz:
        history = [state]
        step = 0
        while not state.finished:
            state = agent.step(state)
            step += 1
            history.append(state)
            print(f"-- step {step} --")
            print(state.tree())
        return history
    from rlmkit.utils.viz import live

    return live(agent, state)


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
        state = agent.start("Create hello.py and goodbye.py. Delegate each file.")
        history = run(agent, state, args.no_viz)
        final = history[-1]
        print(f"\n{GREEN}Result:{RESET} {final.result}")

        banner("2. Checkpoint round-trip")
        ckpt = workspace.checkpoint_path
        final.save(ckpt)
        loaded = Node.load(ckpt)
        print(f"Loaded {loaded.type} checkpoint with {len(loaded.children)} child refs")
        print(loaded.tree())

        banner("3. Session graph persistence")
        nodes = workspace.session.load()
        print(f"Persisted {len(nodes)} node events in {workspace.root / 'session'}")
        print("Latest agents:")
        for node in sorted(nodes.values(), key=lambda item: item.agent_id):
            print(f"  {node.agent_id}: {node.type}")

        banner("4. Time travel")
        for idx, item in enumerate(history):
            print(f"{CYAN}step {idx}{RESET}: {item.agent_id} [{item.type}]")

        banner("5. Manual intervention")
        agent2 = make_agent(
            workspace,
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
        )
        state2 = agent2.start("Create hello.py and goodbye.py. Delegate each file.")
        intervened = False
        while not state2.finished:
            state2 = agent2.step(state2)
            if isinstance(state2, SupervisingNode) and not intervened:
                print(f"{YELLOW}Replacing goodbye child with a manual ResultNode{RESET}")
                new_children = []
                for child in state2.children:
                    if "goodbye" in child.agent_id:
                        new_children.append(
                            child.successor(ResultNode, result="goodbye.py skipped manually")
                        )
                    else:
                        new_children.append(child)
                state2 = state2.update(children=new_children)
                intervened = True
        print(f"{GREEN}Intervention result:{RESET} {state2.result}")

        banner("6. Gym-style loop")
        agent3 = make_agent(workspace, max_depth=0, max_iterations=args.max_iterations)
        state3 = agent3.start("Write a haiku about recursion to haiku.txt")
        rewards: list[float] = []
        step = 0
        while not state3.finished:
            state3 = agent3.step(state3)
            step += 1
            reward = 1.0 if state3.finished else 0.0
            rewards.append(reward)
            print(f"step {step}: node={state3.type} reward={reward}")
        print(f"{GREEN}Result:{RESET} {state3.result}")
        print(f"Total reward: {sum(rewards):.1f}")

        banner("Done")


if __name__ == "__main__":
    main()
