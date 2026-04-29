"""Interactive coding agent.

A REPL interface to an RLMFlow coding agent. Talk to it, give it tasks,
it writes and edits files in your workspace using delegation.

Usage:
    python agent.py --workspace ./myproject
    python agent.py --workspace ./myproject --no-viz
    python agent.py --workspace ./myproject --docker-image rlmkit:local
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlmkit.node import Node
from rlmkit.llm import AnthropicClient, OpenAIClient
from rlmkit.rlm import RLMConfig, RLMFlow
from rlmkit.runtime.docker import DockerRuntime
from rlmkit.runtime.local import LocalRuntime
from rlmkit.tools import FILE_TOOLS
from rlmkit.utils.trace import save_trace
from rlmkit.workspace import Workspace


def main():
    parser = argparse.ArgumentParser(description="Interactive coding agent")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument("--docker-image", default=None,
                        help="If set, run agent code inside this Docker image (e.g. rlmkit:local).")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="Maximum number of concurrent tasks to run.")
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    else:
        print(">>> LOCAL RUNTIME")

    workspace_root = Path(args.workspace).resolve()
    workspace = Workspace.create(workspace_root)
    print(f"Workspace: {workspace.root}")

    if args.docker_image:
        runtime = DockerRuntime(
            args.docker_image,
            workspace=workspace,
        )
    else:
        runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools(FILE_TOOLS)

    llm = (
        AnthropicClient(args.model)
        if args.model.startswith("claude")
        else OpenAIClient(args.model)
    )
    fast = (
        AnthropicClient(args.fast_model)
        if args.fast_model.startswith("claude")
        else OpenAIClient(args.fast_model)
    )

    agent = RLMFlow(
        llm_client=llm,
        runtime=runtime,
        workspace=workspace,
        config=RLMConfig(max_depth=args.max_depth, max_iterations=args.max_iterations),
        llm_clients={
            "fast": {"model": fast, "description": "Cheap model for smaller subtasks"},
        },
    )

    ckpt = workspace.checkpoint_path
    trace_dir = workspace.trace_dir
    trace: list[Node] = []

    print("Agent ready. Type a query, or 'quit' to exit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        state = agent.start(query)
        trace.append(state)

        if args.no_viz:
            while not state.finished:
                state = agent.step(state)
                trace.append(state)
        else:
            from rlmkit.utils.viz import live
            for s in live(agent, state):
                state = s
                trace.append(state)

        print(f"\n{state.result or '(no result)'}\n")
        state.save(ckpt)
        save_trace(trace, trace_dir)
        print(f"Saved checkpoint → {ckpt}  |  trace → {trace_dir}/")


if __name__ == "__main__":
    main()
