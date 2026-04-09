"""Interactive coding agent.

A REPL interface to an RLM coding agent. Talk to it, give it tasks,
it writes and edits files in your workspace using delegation.

Usage:
    python agent.py --workspace ./myproject
    python agent.py --workspace ./myproject --no-viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlmkit.llm import OpenAIClient
from rlmkit.prompts import make_default_builder
from rlmkit.prompts.default import ROLE_TEXT
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime


ROLE = f"""\
You are a coding agent. You write, edit, and organize code.

When given a task:
- Read existing files to understand the codebase before making changes.
- Write clean, idiomatic code. Split logic across files when it makes sense.
- For large tasks, delegate subtasks to child agents — one per file or component.
- After writing code, verify it works (run tests, check imports, etc.).
- When done, call done() with a short summary of what you did.

{ROLE_TEXT}"""


def main():
    strong = OpenAIClient("gpt-5")
    fast = OpenAIClient("gpt-5-mini")

    parser = argparse.ArgumentParser(description="Interactive coding agent")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    parser.add_argument("--max-iterations", type=int, default=30, help="Max steps per task")
    parser.add_argument("--no-viz", action="store_true", help="Disable live tree visualization")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True)
    print(f"Workspace: {workspace}")

    runtime = LocalRuntime(workspace=workspace)

    builder = (
        make_default_builder()
        .section("role", ROLE, title="Role")
    )

    agent = RLM(
        llm_client=strong,
        runtime=runtime,
        config=RLMConfig(
            max_depth=3,
            max_iterations=args.max_iterations,
            session="context",
        ),
        prompt_builder=builder,
        llm_clients={
            "fast": {"model": fast, "description": "Cheap model for smaller subtasks"},
        },
    )

    print("Agent ready. Type a task, or 'quit' to exit.\n")

    while True:
        try:
            task = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task or task.lower() in ("quit", "exit", "q"):
            break

        state = agent.start(task)
        if args.no_viz:
            while not state.finished:
                state = agent.step(state)
        else:
            from rlmkit.utils.viz import live
            states = live(agent, state)
            state = states[-1]

        print(f"\n{state.result or '(no result)'}\n")


if __name__ == "__main__":
    main()
