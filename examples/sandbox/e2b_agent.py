"""Run a platformer-building RLMFlow task inside an E2B Sandbox.

Setup:
    pip install -e ".[openai,e2b]"
    export OPENAI_API_KEY=...
    export E2B_API_KEY=...

Run:
    python examples/sandbox/e2b_agent.py --model gpt-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlmflow import OpenAIClient, RLMConfig, RLMFlow, Workspace  # noqa: E402
from rlmflow.runtime.sandbox.e2b import E2BRuntime  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Run RLMFlow inside E2B.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--fast-model",
        default="gpt-5-mini",
        help="Cheaper model exposed to delegates as `model='fast'`.",
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Recursive sub-agent depth. Defaults to 1 so delegation is enabled.",
    )
    parser.add_argument("--template", help="E2B template name or ID.")
    parser.add_argument("--sandbox-timeout", type=int, default=300)
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
        help="Skip setup commands, useful for templates with rlmflow preinstalled.",
    )
    return parser.parse_args()


def run_platformer_task(
    runtime: E2BRuntime,
    *,
    model: str,
    fast_model: str,
    max_iterations: int,
    max_depth: int,
) -> None:
    agent = RLMFlow(
        llm_client=OpenAIClient(model=model),
        runtime=runtime,
        runtime_factory=runtime.clone,
        config=RLMConfig(max_iterations=max_iterations, max_depth=max_depth),
        llm_clients={
            "fast": {
                "model": OpenAIClient(model=fast_model),
                "description": f"cheaper/faster model ({fast_model}); prefer for delegated subtasks.",
            },
        },
    )
    print(agent.run(PLATFORMER_QUERY))


def main() -> None:
    args = parse_args()
    workspace = Workspace.create(REPO_ROOT / "examples" / "example-workspaces" / "sandbox-e2b")
    setup_commands = [] if args.skip_setup else args.setup_command
    runtime = E2BRuntime(
        workspace=workspace,
        template=args.template,
        timeout=args.sandbox_timeout,
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
            max_depth=args.max_depth,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
