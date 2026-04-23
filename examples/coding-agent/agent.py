"""Interactive coding agent with resume support.

A REPL interface to an RLM coding agent. Talk to it, give it tasks,
it writes and edits files in your workspace using delegation.

Usage:
    python agent.py --workspace ./myproject
    python agent.py --workspace ./myproject --resume checkpoint.json
    python agent.py --workspace ./myproject --no-viz
    python agent.py --workspace ./myproject --docker-image rlmkit:local
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlmkit.llm import AnthropicClient, OpenAIClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.docker import DockerRuntime
from rlmkit.runtime.local import LocalRuntime
from rlmkit.session import FileSession
from rlmkit.state import RLMState
from rlmkit.tools import FILE_TOOLS


def main():
    parser = argparse.ArgumentParser(description="Interactive coding agent")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to saved state JSON")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument("--docker-image", default=None,
                        help="If set, run agent code inside this Docker image (e.g. rlmkit:local).")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    else:
        print(">>> LOCAL RUNTIME")

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {workspace}")

    session = FileSession(workspace / "context")

    if args.docker_image:
        runtime = DockerRuntime(
            args.docker_image,
            workspace=workspace,
            mounts={str(workspace): "/workspace"},
            workdir="/workspace",
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

    agent = RLM(
        llm_client=llm,
        runtime=runtime,
        config=RLMConfig(
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
            session=session,
        ),
        llm_clients={
            "fast": {"model": fast, "description": "Cheap model for smaller subtasks"},
        },
    )

    def run_and_save(state: RLMState) -> None:
        if args.no_viz:
            while not state.finished:
                state = agent.step(state)
        else:
            from rlmkit.utils.viz import live
            state = live(agent, state)[-1]

        print(f"\n{state.result or '(no result)'}\n")
        ckpt = workspace / "checkpoint.json"
        ckpt.write_text(state.model_dump_json())
        print(f"State saved to {ckpt}")

    if args.resume:
        saved = RLMState.model_validate_json(Path(args.resume).read_text())
        state = agent.restore(saved)
        print(f"Resumed from {args.resume}")
        run_and_save(state)
        return

    print("Agent ready. Type a query, or 'quit' to exit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        run_and_save(agent.start(query))


if __name__ == "__main__":
    main()
