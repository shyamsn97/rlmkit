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
import os
import re
from pathlib import Path

from rlmkit.llm import OpenAIClient
from rlmkit.prompts import make_default_builder
from rlmkit.prompts.default import ROLE_TEXT
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import RLMState
from rlmkit.utils import tool
from rlmkit.session import FileSession

def register_file_tools(runtime: LocalRuntime, workspace: Path) -> None:
    ws = workspace.resolve()

    def _resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else ws / p

    @tool("Read a file and return its contents.")
    def read_file(path: str) -> str:
        return _resolve(path).read_text()

    @tool("Write content to a file, creating directories if needed.")
    def write_file(path: str, content: str) -> str:
        resolved = _resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    @tool("Append content to a file.")
    def append_file(path: str, content: str) -> str:
        resolved = _resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a") as f:
            f.write(content)
        return f"Appended {len(content)} bytes to {path}"

    @tool("Find-and-replace edits. Each edit is (old, new).")
    def edit_file(path: str, *edits: tuple[str, str]) -> str:
        resolved = _resolve(path)
        text = resolved.read_text()
        count = 0
        for old, new in edits:
            if old in text:
                text = text.replace(old, new, 1)
                count += 1
        resolved.write_text(text)
        return f"Applied {count}/{len(edits)} edits to {path}"

    @tool("List files and directories.")
    def ls(path: str = ".") -> list[str]:
        resolved = _resolve(path)
        if resolved.is_file():
            return [resolved.name]
        return sorted(p.name for p in resolved.iterdir())

    @tool("Search for lines matching a regex pattern.")
    def grep(pattern: str, path: str = ".", *, max_results: int = 50) -> str:
        resolved = _resolve(path)
        regex = re.compile(pattern)
        matches: list[str] = []
        files = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if regex.search(line):
                        rel = f.relative_to(ws)
                        matches.append(f"{rel}:{i}: {line}")
                        if len(matches) >= max_results:
                            return "\n".join(matches)
            except (UnicodeDecodeError, PermissionError):
                continue
        return "\n".join(matches)

    for fn in (read_file, write_file, append_file, edit_file, ls, grep):
        runtime.register_tool(fn)


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
    runtime = LocalRuntime()
    os.chdir(workspace)
    register_file_tools(runtime, workspace)

    builder = (
        make_default_builder()
        .section("role", ROLE_TEXT, title="Role")
    )

    agent = RLM(
        llm_client=strong,
        runtime=runtime,
        config=RLMConfig(
            max_depth=3,
            max_iterations=args.max_iterations,
            session=session,
        ),
        prompt_builder=builder,
        llm_clients={
            "fast": {"model": fast, "description": "Cheap model for smaller subtasks"},
        },
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

        ckpt = workspace / "checkpoint.json"
        ckpt.write_text(state.model_dump_json())
        print(f"State saved to {ckpt}")


if __name__ == "__main__":
    main()
