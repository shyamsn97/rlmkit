"""Needle in a massive in-memory CONTEXT.

Inspired by alexzhang13/rlm-minimal's million-line magic-number demo. This
version stores the haystack in RLMFlow's `CONTEXT` instead of writing many
files, so the agent must use `CONTEXT.lines(...)` and parallel child agents.

Usage:
    python examples/needle_haystack.py
    python examples/needle_haystack.py --num-lines 1000000 --no-viz
    python examples/needle_haystack.py --viewer
    python examples/needle_haystack.py --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

from rlmflow.llm import AnthropicClient, OpenAIClient
from rlmflow.rlm import RLMConfig, RLMFlow
from rlmflow.runtime.docker import DockerRuntime
from rlmflow.runtime.local import LocalRuntime


def generate_massive_context(
    num_lines: int = 1_000_000,
    *,
    answer: str | None = None,
) -> tuple[str, str, int]:
    print(f"Generating massive context with {num_lines:,} lines...")

    words = ["blah", "random", "text", "data", "content", "information", "sample"]
    answer = answer or "".join(random.choices(string.digits, k=7))

    lines = []
    for _ in range(num_lines):
        n = random.randint(3, 8)
        lines.append(" ".join(random.choice(words) for _ in range(n)))

    if num_lines <= 0:
        raise ValueError("--num-lines must be positive")

    low = min(num_lines - 1, max(0, int(num_lines * 0.4)))
    high = min(num_lines - 1, max(low, int(num_lines * 0.6)))
    needle_line = random.randint(low, high)
    lines[needle_line] = f"The magic number is {answer}"

    print(f"Magic number inserted at line {needle_line}")
    return "\n".join(lines), answer, needle_line


def main():
    parser = argparse.ArgumentParser(description="Needle in a massive CONTEXT")
    parser.add_argument("--num-lines", type=int, default=1_000_000)
    parser.add_argument(
        "--viewer", action="store_true", help="Open the state viewer after finishing"
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fast-model", default="gpt-5-nano")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="If set, run agent code inside this Docker image (e.g. rlmflow:local).",
    )
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    else:
        print(">>> LOCAL RUNTIME")

    workspace = Path("example-workspaces/needle-haystack-context").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    context, answer, needle_line = generate_massive_context(num_lines=args.num_lines)

    def make_runtime():
        if args.docker_image:
            return DockerRuntime(
                args.docker_image,
                workspace=workspace,
                mounts={str(workspace): "/workspace"},
                workdir="/workspace",
            )
        return LocalRuntime(workspace=workspace)

    llm = (
        AnthropicClient(args.model)
        if args.model.startswith("claude")
        else OpenAIClient(args.model)
    )
    llm_clients = None
    if args.fast_model:
        fast = (
            AnthropicClient(args.fast_model)
            if args.fast_model.startswith("claude")
            else OpenAIClient(args.fast_model)
        )
        llm_clients = {
            "fast": {
                "model": fast,
                "description": "Cheaper model for independent context chunks.",
            },
        }

    agent = RLMFlow(
        llm_client=llm,
        runtime=make_runtime(),
        config=RLMConfig(max_depth=args.max_depth, max_iterations=args.max_iterations),
        llm_clients=llm_clients,
        runtime_factory=make_runtime,
    )

    graph = agent.start(
        "I'm looking for a magic number. What is it? You cannot use CONTEXT methods, you have to just use CONTEXT.read()!",
        context=context,
        context_metadata={"num_lines": args.num_lines, "needle_line": needle_line},
    )

    if args.no_viz:
        while not graph.finished:
            graph = agent.step(graph)
            print(graph.tree())
    else:
        from rlmflow.utils.viz import live

        graphs = live(agent, graph)
        graph = graphs[-1]

    print(f"\n{'=' * 40}")
    print(f"Actual answer:  {answer}")
    print(f"Correct:        {answer in graph.result()}")
    print(f"Workspace saved to {workspace}")

    if args.viewer:
        from rlmflow.utils.viewer import open_viewer

        open_viewer(workspace)


if __name__ == "__main__":
    main()
