"""Interactive coding agent.

A REPL interface to an RLM coding agent. Talk to it, give it tasks,
it writes and edits files in your workspace using delegation.

Usage:
    python agent.py                        # workspace = current dir
    python agent.py --workspace ./myproject
    python agent.py --model gpt-5-mini
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

from utils import StepLogger

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
    parser = argparse.ArgumentParser(description="Interactive coding agent")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace directory")
    parser.add_argument("--model", type=str, default="gpt-5", help="OpenAI model name")
    parser.add_argument("--max-iterations", type=int, default=30, help="Max steps per task")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {workspace}")

    runtime = LocalRuntime(workspace=workspace)
    logger = StepLogger(workspace / ".agent_log.md", live=True)

    builder = (
        make_default_builder()
        .section("role", ROLE, title="Role")
    )

    agent = RLM(
        llm_client=OpenAIClient(args.model),
        runtime=runtime,
        config=RLMConfig(
            max_depth=3,
            max_iterations=args.max_iterations,
            session="context",
        ),
        prompt_builder=builder,
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
        step = 0
        while not state.finished:
            logger.status(state)
            state = agent.step(state)
            step += 1
            logger.log(step, state)

        print(f"\n{state.result or '(no result)'}\n")


if __name__ == "__main__":
    main()
