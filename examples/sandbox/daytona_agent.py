"""Run a platformer-building RLMFlow task inside a Daytona Sandbox.

Setup:
    pip install -e ".[openai,daytona]"
    export OPENAI_API_KEY=...
    export DAYTONA_API_KEY=...

Run:
    python examples/sandbox/daytona_agent.py --model gpt-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlmflow import OpenAIClient, RLMConfig, RLMFlow, Workspace  # noqa: E402
from rlmflow.runtime.sandbox.daytona import DaytonaRuntime  # noqa: E402
from rlmflow.tools import FILE_TOOLS  # noqa: E402

PLATFORMER_QUERY = """\
Build a simple 2D side-scrolling platformer in plain HTML/CSS/JS under output/.
No build tools, no libraries, no ES modules.

Files:
- output/index.html
- output/styles.css
- output/scripts/engine.js   — state, input, physics
- output/scripts/main.js     — level, render, requestAnimationFrame loop

index.html loads engine.js then main.js. Canvas with left/right movement, jump,
gravity, platform collision, scrolling camera, and restart.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLMFlow inside Daytona.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--fast-model",
        default="gpt-5-mini",
        help="Cheaper model exposed to delegates as `model='fast'`.",
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--snapshot", help="Daytona snapshot name or ID.")
    parser.add_argument("--create-timeout", type=float, default=60)
    parser.add_argument("--repl-timeout", type=float, default=30)
    parser.add_argument("--remote-workdir", default="/workspace")
    parser.add_argument(
        "--setup-command",
        action="append",
        help=(
            "Command to run before starting the REPL. Repeat for multiple commands. "
            "Defaults to installing rlmflow from PyPI."
        ),
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip setup commands, useful for snapshots with rlmflow preinstalled.",
    )
    return parser.parse_args()


def run_platformer_task(
    runtime: DaytonaRuntime,
    *,
    model: str,
    fast_model: str,
    max_iterations: int,
) -> None:
    agent = RLMFlow(
        llm_client=OpenAIClient(model=model),
        runtime=runtime,
        runtime_factory=runtime.clone,
        config=RLMConfig(max_iterations=max_iterations, max_depth=1),
        llm_clients={
            "fast": {
                "model": OpenAIClient(model=fast_model),
                "description": f"cheaper/faster model ({fast_model}); prefer for delegated subtasks.",
            },
        },
    )
    print(agent.run(PLATFORMER_QUERY))


def create_params(snapshot: str | None) -> Any:
    if snapshot is None:
        return None
    from daytona import CreateSandboxFromSnapshotParams

    return CreateSandboxFromSnapshotParams(snapshot=snapshot, language="python")


def main() -> None:
    args = parse_args()
    workspace = Workspace.create(REPO_ROOT / "examples" / "example-workspaces" / "sandbox-daytona")
    setup_commands = [] if args.skip_setup else args.setup_command
    runtime = DaytonaRuntime(
        workspace=workspace,
        create_params=create_params(args.snapshot),
        create_timeout=args.create_timeout,
        remote_workdir=args.remote_workdir,
        repl_timeout=args.repl_timeout,
        setup_commands=setup_commands,
    )
    runtime.register_tools(FILE_TOOLS)
    try:
        run_platformer_task(
            runtime,
            model=args.model,
            fast_model=args.fast_model,
            max_iterations=args.max_iterations,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
