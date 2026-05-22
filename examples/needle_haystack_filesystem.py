"""Needle in a haystack across many files.

Generates many files of random noise. One file contains a magic string. The
agent uses the standard file tools to find it, delegating the search in
parallel across batches.

Usage:
    python examples/needle_haystack_filesystem.py
    python examples/needle_haystack_filesystem.py --no-viz
    python examples/needle_haystack_filesystem.py --viewer
    python examples/needle_haystack_filesystem.py --docker-image rlmflow:local
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
from rlmflow.tools import FILE_TOOLS


def generate_haystack(
    directory: Path, num_files: int = 500, lines_per_file: int = 200
) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    answer = "".join(random.choices(string.digits, k=7))
    needle_file = random.randint(0, num_files - 1)
    needle_line = random.randint(0, lines_per_file - 1)

    for i in range(num_files):
        lines = []
        for j in range(lines_per_file):
            if i == needle_file and j == needle_line:
                lines.append(f"The magic number is {answer}")
            else:
                n = random.randint(3, 8)
                lines.append(" ".join(random.choice(words) for _ in range(n)))
        (directory / f"file_{i:04d}.txt").write_text("\n".join(lines))

    print(f"Needle in file_{needle_file:04d}.txt line {needle_line}")
    return answer


def main():
    parser = argparse.ArgumentParser(description="Needle in a haystack across many files")
    parser.add_argument("--num-files", type=int, default=500)
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
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    else:
        print(">>> LOCAL RUNTIME")

    workspace = Path("example-workspaces/needle-haystack-filesystem").resolve()
    haystack_path = workspace / "haystack"
    workspace.mkdir(parents=True, exist_ok=True)
    haystack_path.mkdir(parents=True, exist_ok=True)
    for stale in haystack_path.glob("*.txt"):
        stale.unlink()
    answer = generate_haystack(haystack_path, num_files=args.num_files)
    print(f"Generated {args.num_files} files in {haystack_path}")

    def make_runtime():
        if args.docker_image:
            rt = DockerRuntime(
                args.docker_image,
                workspace=workspace,
                mounts={str(workspace): "/workspace"},
                workdir="/workspace",
            )
        else:
            rt = LocalRuntime(workspace=workspace)
        rt.register_tools(FILE_TOOLS)
        return rt

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
            "fast": {"model": fast, "description": "Cheaper model for small sub-tasks."},
        }

    agent = RLMFlow(
        llm_client=llm,
        runtime=make_runtime(),
        config=RLMConfig(max_depth=args.max_depth, max_iterations=args.max_iterations),
        llm_clients=llm_clients,
        runtime_factory=make_runtime,
    )

    graph = agent.start(
        f"There are {args.num_files} text files in haystack/. "
        f"Exactly one line in one file matches the pattern `The magic number is <number>`. Find and return the number. There are too many files to search manually, so you should split the work into batches and delegate."
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
