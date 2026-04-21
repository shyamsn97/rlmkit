"""Interactive coding agent with resume support.

A REPL interface to an RLM coding agent. Talk to it, give it tasks,
it writes and edits files in your workspace using delegation.

Usage:
    python agent.py --workspace ./myproject
    python agent.py --workspace ./myproject --resume checkpoint.json
    python agent.py --workspace ./myproject --no-viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlmkit.llm import OpenAIClient
from rlmkit.prompts import make_default_builder
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import RLMState
from rlmkit.session import FileSession
from rlmkit.tools import FILE_TOOLS


CODING_ROLE = """You are a coding agent. You edit real source files on the user's disk, so be deliberate.

**Work in four phases: orient → plan → split → verify.**

1. **Orient** — `ls`, `grep`, `line_count`, and targeted `read_file` BEFORE changing anything. Never edit a file you haven't sized up.
2. **Plan** — write out (as REPL comments or a short print) what files you'll touch and what each change is. Re-plan if scope grows.
3. **Delegate by default.** If the task produces or touches multiple files, you should delegate — one sub-agent per file or logical unit. Doing everything in a single code block wastes your context window. The only exception is if you are already a leaf sub-agent or at max depth.
4. **Split** — delegate one child per independent unit of work:
    - Multi-file change → one child per file.
    - Distinct phases (explore → implement → test) → one child per phase.
    - Pure exploration → one child reads/reports, you make the edits.
5. **Verify** — before `done()`, read modified files back or run a check. Call `done()` with a short summary: what changed, where, why."""


def build_prompt():
    return (
        make_default_builder()
        .section("role", CODING_ROLE, title="Role")
    )


def main():
    strong = OpenAIClient("gpt-5")
    fast = OpenAIClient("gpt-5-mini")

    parser = argparse.ArgumentParser(description="Interactive coding agent")
    parser.add_argument("--workspace", type=str, default="workspace", help="Workspace directory")
    parser.add_argument("--max-iterations", type=int, default=30, help="Max steps per task")
    parser.add_argument("--resume", type=str, default=None, help="Path to saved state JSON")
    parser.add_argument("--no-viz", action="store_true", help="Disable live tree visualization")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {workspace}")

    session = FileSession(workspace / "context")
    runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools(FILE_TOOLS)

    agent = RLM(
        llm_client=strong,
        runtime=runtime,
        config=RLMConfig(
            max_depth=3,
            max_iterations=args.max_iterations,
            session=session,
        ),
        llm_clients={
            "fast": {"model": fast, "description": "Cheap model for smaller subtasks"},
        },
        prompt_builder=build_prompt(),
    )

    if args.resume:
        saved = RLMState.model_validate_json(Path(args.resume).read_text())
        state = agent.restore(saved)
        print(f"Resumed from {args.resume}")

        if args.no_viz:
            while not state.finished:
                state = agent.step(state)
        else:
            from rlmkit.utils.viz import live
            states = live(agent, state)
            state = states[-1]

        print(f"\n{state.result or '(no result)'}\n")

        ckpt = workspace / "checkpoint.json"
        ckpt.write_text(state.model_dump_json())
        print(f"State saved to {ckpt}")
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

        state = agent.start(query)
        if args.no_viz:
            while not state.finished:
                state = agent.step(state)
        else:
            from rlmkit.utils.viz import live
            states = live(agent, state)
            state = states[-1]

        print(f"\n{state.result or '(no result)'}\n")

        ckpt = workspace / "checkpoint.json"
        ckpt.write_text(state.model_dump_json())
        print(f"State saved to {ckpt}")


if __name__ == "__main__":
    main()
