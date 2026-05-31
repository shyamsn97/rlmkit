"""Run the example suite as smoke tests.

Default usage runs deterministic/offline examples only:

    python examples/run_examples.py

Opt into heavier examples when you have the dependencies/credentials:

    python examples/run_examples.py --include-optional --include-notebooks
    python examples/run_examples.py --include-live
    python examples/run_examples.py --all
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Category = Literal["offline", "optional", "live", "sandbox", "notebook", "manual"]

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Example:
    name: str
    path: str
    category: Category = "offline"
    args: tuple[str, ...] = ()
    env: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    timeout: int = 120
    stdin: str | None = None
    note: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)

    def command(self, tmpdir: Path) -> list[str]:
        if self.path.endswith(".ipynb"):
            out = tmpdir / (Path(self.path).stem + ".executed.ipynb")
            return [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                f"--ExecutePreprocessor.timeout={self.timeout}",
                "--output",
                str(out),
                str(REPO_ROOT / self.path),
            ]
        return [
            sys.executable,
            str(REPO_ROOT / self.path),
            *[arg.format(tmp=tmpdir) for arg in self.args],
        ]


EXAMPLES: list[Example] = [
    Example("injections", "examples/injections.py"),
    Example("eager-children", "examples/eager_children.py"),
    Example("llm-query-batched", "examples/llm_query_batched.py"),
    Example("best-of-n", "examples/best_of_n.py", args=("--n", "4", "--root-dir", "{tmp}/best_of_n")),
    Example(
        "fork-repair",
        "examples/fork_repair.py",
        args=("--root-dir", "{tmp}/fork_repair"),
        modules=("pytest",),
    ),
    Example("showcase", "examples/showcase.py", args=("--no-viz",)),
    Example("graph-query", "examples/graph-features/01_query.py"),
    Example("graph-navigate", "examples/graph-features/02_navigate.py"),
    Example("graph-mutate", "examples/graph-features/03_mutate.py"),
    Example("graph-save-load", "examples/graph-features/04_save_load.py"),
    Example("graph-replay", "examples/graph-features/05_replay.py"),
    Example("graph-fork", "examples/graph-features/06_fork.py"),
    Example("graph-render", "examples/graph-features/07_render.py"),
    Example(
        "circle-packing-evaluate",
        "examples/autoresearch/circle_packing/evaluate.py",
        args=("examples/autoresearch/circle_packing/solution.py",),
        category="optional",
        modules=("numpy",),
    ),
    Example(
        "circle-packing-plot",
        "examples/autoresearch/circle_packing/plot_circles.py",
        args=(
            "examples/autoresearch/circle_packing/solution.py",
            "--out",
            "{tmp}/circle_packing.png",
        ),
        category="optional",
        modules=("matplotlib", "numpy"),
    ),
    Example(
        "view-demo",
        "examples/view_demo.py",
        category="manual",
        modules=("gradio", "plotly"),
        note="opens the interactive viewer",
    ),
    Example(
        "drop-in-llm",
        "examples/drop_in_llm.py",
        category="live",
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "summarizer",
        "examples/summarizer.py",
        category="live",
        args=("--sections", "6", "--no-viz", "--max-iterations", "8"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "needle-haystack",
        "examples/needle_haystack.py",
        category="live",
        args=("--num-lines", "2000", "--no-viz", "--max-iterations", "8"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "needle-haystack-filesystem",
        "examples/needle_haystack_filesystem.py",
        category="live",
        args=("--num-files", "50", "--no-viz", "--max-iterations", "8"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "dspy-drop-in",
        "examples/dspy_drop_in.py",
        category="live",
        env=("OPENAI_API_KEY",),
        modules=("dspy", "openai"),
        timeout=300,
    ),
    Example(
        "autoresearch",
        "examples/autoresearch/autoresearch.py",
        category="live",
        args=(
            "--target",
            "examples/autoresearch/circle_packing",
            "--workspace",
            "{tmp}/autoresearch",
            "--max-submissions",
            "0",
            "--max-iterations",
            "3",
            "--no-ui",
        ),
        env=("OPENAI_API_KEY",),
        modules=("openai", "numpy"),
        timeout=300,
        note="smoke mode: baseline + no child submissions",
    ),
    Example(
        "sandbox-e2b",
        "examples/sandbox/e2b_agent.py",
        category="sandbox",
        args=("--max-iterations", "2", "--skip-setup"),
        env=("OPENAI_API_KEY", "E2B_API_KEY"),
        modules=("e2b", "openai"),
        timeout=600,
    ),
    Example(
        "sandbox-daytona",
        "examples/sandbox/daytona_agent.py",
        category="sandbox",
        args=("--max-iterations", "2", "--skip-setup"),
        env=("OPENAI_API_KEY", "DAYTONA_API_KEY"),
        modules=("daytona", "openai"),
        timeout=600,
    ),
    Example(
        "sandbox-modal",
        "examples/sandbox/modal_agent.py",
        category="sandbox",
        args=("--max-iterations", "2", "--no-live", "--quiet-runtime"),
        env=("OPENAI_API_KEY",),
        modules=("modal", "openai"),
        timeout=900,
    ),
    Example(
        "notebook-node-basics",
        "examples/notebooks/node_basics.ipynb",
        category="notebook",
        modules=("jupyter", "nbconvert", "ipykernel"),
        timeout=180,
    ),
    Example(
        "notebook-viz-walkthrough",
        "examples/notebooks/viz_walkthrough.ipynb",
        category="notebook",
        modules=("jupyter", "nbconvert", "ipykernel"),
        timeout=180,
    ),
    Example(
        "notebook-coding-agent",
        "examples/notebooks/coding_agent.ipynb",
        category="notebook",
        env=("OPENAI_API_KEY",),
        modules=("jupyter", "nbconvert", "ipykernel", "openai"),
        timeout=600,
        note="live notebook: executes real LLM cells",
    ),
    Example(
        "coding-agent-interactive",
        "examples/coding-agent/agent.py",
        category="manual",
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        stdin="quit\n",
        note="interactive shell; smoke only starts and exits",
    ),
]


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def should_include(example: Example, args: argparse.Namespace) -> bool:
    if args.pattern and args.pattern.lower() not in example.name.lower() and args.pattern not in example.path:
        return False
    return (
        example.category == "offline"
        or (example.category == "optional" and args.include_optional)
        or (example.category == "live" and args.include_live)
        or (example.category == "sandbox" and args.include_sandbox)
        or (example.category == "notebook" and args.include_notebooks)
        or (example.category == "manual" and args.include_manual)
    )


def skip_reason(example: Example) -> str | None:
    missing_env = [name for name in example.env if not os.environ.get(name)]
    if missing_env:
        return "missing env: " + ", ".join(missing_env)
    missing_modules = [name for name in example.modules if not module_exists(name)]
    if missing_modules:
        return "missing modules: " + ", ".join(missing_modules)
    if not (REPO_ROOT / example.path).exists():
        return "missing path"
    return None


def run_example(example: Example, tmpdir: Path, *, verbose: bool) -> tuple[str, float]:
    command = example.command(tmpdir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(example.extra_env)

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        input=example.stdin,
        text=True,
        capture_output=True,
        timeout=example.timeout,
    )
    elapsed = time.perf_counter() - start
    output = (proc.stdout + proc.stderr).strip()
    if verbose and output:
        print(output)
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-40:])
        raise RuntimeError(f"exit code {proc.returncode}\n{tail}")
    return output, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true", help="run optional-dependency examples")
    parser.add_argument("--include-live", action="store_true", help="run examples that call live LLM APIs")
    parser.add_argument("--include-sandbox", action="store_true", help="run Modal/E2B/Daytona examples")
    parser.add_argument("--include-notebooks", action="store_true", help="execute notebooks with nbconvert")
    parser.add_argument("--include-manual", action="store_true", help="include interactive/manual smoke checks")
    parser.add_argument("--all", action="store_true", help="enable every include flag")
    parser.add_argument("--list", action="store_true", help="list selected examples without running them")
    parser.add_argument("--pattern", help="only include examples whose name/path contains this text")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failure")
    parser.add_argument("--strict-skips", action="store_true", help="treat missing env/deps as failure")
    parser.add_argument("--verbose", action="store_true", help="print full output for successful examples")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        args.include_optional = True
        args.include_live = True
        args.include_sandbox = True
        args.include_notebooks = True
        args.include_manual = True

    selected = [example for example in EXAMPLES if should_include(example, args)]
    if args.list:
        for example in selected:
            reason = skip_reason(example)
            status = "skip: " + reason if reason else "ready"
            note = f" ({example.note})" if example.note else ""
            print(f"{example.name:<28} {example.category:<8} {status}{note}")
        return 0

    failures: list[str] = []
    skipped: list[str] = []
    passed = 0

    with tempfile.TemporaryDirectory(prefix="rlmflow-example-runs-") as raw_tmpdir:
        tmpdir = Path(raw_tmpdir)
        print(f"Running {len(selected)} selected examples. temp outputs: {tmpdir}")
        for example in selected:
            reason = skip_reason(example)
            label = f"{example.name} ({example.path})"
            if reason:
                print(f"SKIP {label}: {reason}")
                skipped.append(f"{example.name}: {reason}")
                if args.strict_skips:
                    failures.append(f"{example.name}: {reason}")
                continue

            print(f"RUN  {label}")
            try:
                _output, elapsed = run_example(example, tmpdir, verbose=args.verbose)
            except subprocess.TimeoutExpired:
                message = f"{example.name}: timed out after {example.timeout}s"
                print(f"FAIL {message}")
                failures.append(message)
                if args.fail_fast:
                    break
            except Exception as exc:  # noqa: BLE001
                message = f"{example.name}: {exc}"
                print(f"FAIL {message}")
                failures.append(message)
                if args.fail_fast:
                    break
            else:
                passed += 1
                note = f" [{example.note}]" if example.note else ""
                print(f"PASS {example.name} ({elapsed:.1f}s){note}")

    print("\nSummary")
    print(f"  passed : {passed}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed : {len(failures)}")

    if skipped:
        print("\nSkipped")
        for item in skipped:
            print(f"  - {item}")
    if failures:
        print("\nFailures")
        for item in failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
