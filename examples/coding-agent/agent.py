"""Interactive coding agent.

A REPL interface to an RLMFlow coding agent. Talk to it, give it tasks,
it writes and edits files in your workspace using delegation.

Usage:
    python agent.py --workspace ./myproject
    python agent.py --workspace ./myproject --no-viz
    python agent.py --workspace ./myproject --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from rlmflow.llm import AnthropicClient, OpenAIClient
from rlmflow.rlm import RLMConfig, RLMFlow
from rlmflow.runtime.docker import DockerRuntime
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import FILE_TOOLS
from rlmflow.workspace import Workspace
    
from rlmflow.prompts.default import DEFAULT_BUILDER

CODING_INSTRUCTIONS = """
- **Plan ownership before writing.** For non-trivial coding tasks, first make a compact manifest: file/component owners, shared interfaces, dependencies, and acceptance checks. The parent plans boundaries; children own implementation details.
- **Keep plans lightweight.** Give children enough direction to own their piece without pre-writing the whole file for them.
- **Bias toward delegation for separable work.** Do small, local tasks directly; when files, components, chunks, or checks can be owned independently, delegate them before implementing inline.
- **Honor the requested artifact.** If the user asks for a component, app, CLI, test, script, library, config, migration, or data file, produce that artifact's expected files and behavior. Do not substitute a generic page, placeholder, README, template, or unrelated scaffold; preserve the requested runtime/API/contract in verification.
- **Delegate focused artifact work.** Give each child one bounded file/component/chunk/check with a clear expected output. File-writing children must call `write_file(...)`, verify that file from disk, and return a short status.
- **Pass only needed context.** Give children a spec, a relevant `CONTEXT.lines(...)` slice, or `""`; don't dump your whole view unless necessary.
- **Combine from disk.** After children finish, read the files they wrote and verify the shared contract.
- **Run real checks.** Syntax-check, run tests, or smoke-test the entry point before `done()` when the runtime can.
- **Repair surgically.** After an exception, `ls`/`read_file` first; fix the broken file instead of rewriting or re-delegating everything.
""".strip()


CODING_BUILDER = DEFAULT_BUILDER.section(
    "coding",
    CODING_INSTRUCTIONS,
    title="Coding",
    after="builtins",
)


__all__ = ["CODING_BUILDER", "CODING_INSTRUCTIONS"]

def main():
    parser = argparse.ArgumentParser(description="Interactive coding agent")
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(Path(__file__).parent.parent / "runs" / "coding-agent"),
        help="workspace dir (default: examples/runs/coding-agent/)",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument("--docker-image", default=None,
                        help="If set, run agent code inside this Docker image (e.g. rlmflow:local).")
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
        prompt_builder=CODING_BUILDER,
    )

    print("Agent ready. Type a query, or 'quit' to exit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        graph = agent.start(query)

        if args.no_viz:
            while not graph.finished:
                graph = agent.step(graph)
        else:
            from rlmflow.utils.viz import live_view
            with live_view() as view:
                view(graph)
                while not graph.finished:
                    graph = agent.step(graph)
                    view(graph)

        print(f"\n{graph.result() or '(no result)'}\n")
        print(f"Workspace saved to {workspace.root}")


if __name__ == "__main__":
    main()
