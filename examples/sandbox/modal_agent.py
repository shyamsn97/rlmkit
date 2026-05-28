"""Run a platformer-building RLMFlow task inside a Modal Sandbox.

Setup:
    pip install -e ".[openai,modal]"
    export OPENAI_API_KEY=...
    modal setup

Run:
    python examples/sandbox/modal_agent.py --model gpt-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlmflow import OpenAIClient, RLMConfig, RLMFlow, Workspace  # noqa: E402
from rlmflow.runtime.sandbox.modal import ModalRuntime  # noqa: E402
from rlmflow.tools import FILE_TOOLS  # noqa: E402
from rlmflow.utils.viz import live  # noqa: E402

REMOTE_REPO = "/opt/rlmflow"

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


def log(message: str) -> None:
    print(f"[modal-agent] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLMFlow inside Modal.")
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
    parser.add_argument("--app-name", default="rlmflow")
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=3600,
        help="Modal sandbox lifetime in seconds. Multi-agent LLM turns can exceed 5 minutes.",
    )
    parser.add_argument("--repl-timeout", type=float, default=30)
    parser.add_argument("--remote-workdir", default="/workspace")
    parser.add_argument(
        "--quiet-runtime",
        action="store_true",
        help="Disable ModalRuntime lifecycle logs.",
    )
    parser.add_argument(
        "--trace-runtime",
        action="store_true",
        help="Print low-level Modal exec and REPL transport logs.",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable the live terminal graph view.",
    )
    return parser.parse_args()


def run_turn(agent: RLMFlow, query: str, *, use_live: bool) -> str:
    graph = agent.start(query)
    if use_live:
        graphs = live(agent, graph)
        return graphs[-1].result()
    while not graph.finished:
        graph = agent.step(graph)
    return graph.result()


def run_platformer_task(
    runtime: ModalRuntime,
    *,
    model: str,
    fast_model: str,
    max_iterations: int,
    max_depth: int,
    use_live: bool,
) -> None:
    log(
        "creating RLMFlow agent with "
        f"model={model}, fast_model={fast_model}, "
        f"max_iterations={max_iterations}, max_depth={max_depth}"
    )
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
    log("running platformer task; first run may build/start Modal sandbox")
    print(run_turn(agent, PLATFORMER_QUERY, use_live=use_live))


def local_rlmflow_image() -> modal.Image:
    log(f"preparing Modal image from local checkout: {REPO_ROOT} -> {REMOTE_REPO}")
    return (
        modal.Image.debian_slim()
        .add_local_dir(
            REPO_ROOT,
            remote_path=REMOTE_REPO,
            copy=True,
            ignore=[
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "media",
                "docs",
                "examples/example-workspaces",
                "examples/notebooks/boids-sim-workspace",
            ],
        )
        .run_commands(f"python -m pip install -e {REMOTE_REPO}")
    )


def main() -> None:
    args = parse_args()
    workspace = Workspace.create(REPO_ROOT / "examples" / "example-workspaces" / "sandbox-modal")
    log(f"workspace: {workspace.root}")
    runtime = ModalRuntime(
        app_name=args.app_name,
        workspace=workspace,
        remote_workdir=args.remote_workdir,
        image=local_rlmflow_image(),
        timeout=args.sandbox_timeout,
        repl_timeout=args.repl_timeout,
        verbose=not args.quiet_runtime,
        trace=args.trace_runtime,
    )
    runtime.register_tools(FILE_TOOLS)
    log("Modal runtime configured; sandbox starts on first agent/runtime call")
    try:
        run_platformer_task(
            runtime,
            model=args.model,
            fast_model=args.fast_model,
            max_iterations=args.max_iterations,
            max_depth=args.max_depth,
            use_live=not args.no_live,
        )
    finally:
        log("closing Modal runtime")
        runtime.close()


if __name__ == "__main__":
    main()
