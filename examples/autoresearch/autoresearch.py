"""Karpathy-style autoresearch on top of RLMFlow.

The agent edits `train.py`, runs it via `run_experiment(source)`, looks
at `val_bpb`, and decides what to try next. To run trials in parallel
the agent uses RLMFlow's normal `delegate` / `wait`. There is no
"trial dir" concept — every call to `run_experiment(source)` writes the
source to `history/<n>_train.py` and runs it. That's the entire driver.

Usage:
    python examples/autoresearch/autoresearch.py --target examples/autoresearch/tinker
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from rlmflow import RLMConfig, RLMFlow, Workspace
from rlmflow.llm import AnthropicClient, OpenAIClient
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import FILE_TOOLS, tool
from rlmflow.utils.viz import LiveView

VAL_BPB_RE = re.compile(r"val_bpb\s*[:=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE)
SKIP_NAMES = {".git", ".DS_Store", "__pycache__", ".ipynb_checkpoints", "runs"}


def make_run_experiment(workspace_root: Path):
    """Build `run_experiment(source)` and `list_runs()`.

    `run_experiment(source: str)` writes `source` to `history/<n>_train.py`,
    runs `python <that file>` from `history/` (where `data/` and any
    other harness file is symlinked), parses `val_bpb` from stdout, and
    appends a row to `history/ledger.jsonl`. Concurrent callers each get
    a unique `n` via one shared lock.
    """
    root = workspace_root.resolve()
    history = root / "history"
    ledger = history / "ledger.jsonl"
    lock = threading.RLock()

    # train.py uses `Path(__file__).parent / "data"`, so every archived
    # copy needs `data/` (and any other harness file) sitting next to it.
    # Symlink everything except bookkeeping into `history/` once.
    HARNESS_SKIP = {"train.py", "history", "session", "context",
                    "graph.json", "viewer.html", "runs"}

    harness_done = [False]  # mutable flag, guarded by `lock`

    def _ensure_harness() -> None:
        with lock:
            if harness_done[0]:
                return
            history.mkdir(parents=True, exist_ok=True)
            for entry in root.iterdir():
                if entry.name in HARNESS_SKIP:
                    continue
                dst = history / entry.name
                if dst.exists() or dst.is_symlink():
                    continue
                try:
                    dst.symlink_to(entry.resolve())
                except OSError:
                    if entry.is_dir():
                        shutil.copytree(entry, dst, symlinks=True, dirs_exist_ok=True)
                    else:
                        shutil.copy2(entry, dst)
            harness_done[0] = True

    @tool(
        "Run a train.py source string under `budget_s` seconds. `source` "
        "is the full Python text of a train.py — the tool writes it to "
        "`history/<n>_train.py` and runs it. Returns a dict {n, val_bpb, "
        "returncode, stdout_tail, stderr_tail, elapsed_s, train_py_path}. "
        "`val_bpb` is parsed from a `val_bpb: <float>` line in stdout "
        "(None on crash; lower is better). If `returncode != 0`, the "
        "real error is in `stderr_tail`. Typical use: "
        "`src = read_file('train.py'); new = src.replace('LR: float = 3e-4', "
        "'LR: float = 1.5e-4'); r = run_experiment(new)`."
    )
    def run_experiment(source: str, budget_s: int = 300) -> dict:
        if not isinstance(source, str) or not source.strip():
            return {
                "n": -1, "val_bpb": None, "returncode": -2,
                "stdout_tail": "",
                "stderr_tail": "run_experiment(source) expected a non-empty "
                               "train.py source string. Did you pass a path? "
                               "Use `src = read_file('train.py')` first.",
                "elapsed_s": 0.0, "train_py_path": None,
            }
        budget_s = max(10, min(int(budget_s), 3600))
        _ensure_harness()

        with lock:
            n = len(list(history.glob("*_train.py")))
            archive = history / f"{n}_train.py"
            while archive.exists():
                n += 1
                archive = history / f"{n}_train.py"
            archive.write_text(source)

        t0 = time.time()
        # `-u` + PYTHONUNBUFFERED=1 force the child to flush prints as
        # they happen. Without this, piped stdout is block-buffered and
        # a timeout-kill loses every print the run made — the ledger
        # then shows `stdout_tail=""` and you can't tell whether the run
        # was making progress or actually hung.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-u", archive.name],
                cwd=str(history), capture_output=True, text=True,
                timeout=budget_s, env=env,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
            stderr += f"\n[timed out after {budget_s}s]"
            rc = -1
        elapsed = round(time.time() - t0, 2)

        m = VAL_BPB_RE.search(stdout or "")
        val_bpb = float(m.group(1)) if m else None

        row = {
            "n": n, "ts": time.time(),
            "val_bpb": val_bpb, "returncode": rc, "elapsed_s": elapsed,
            "train_py_path": str(archive.relative_to(root)),
            "stdout_tail": (stdout or "")[-2000:],
            "stderr_tail": (stderr or "")[-1000:],
        }
        with lock, ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    @tool(
        "Return every recorded run, best-first (successful runs sorted "
        "by val_bpb ascending; crashes go last). Each row is `{n, "
        "val_bpb, returncode, elapsed_s, ts, train_py_path}`. Survives "
        "REPL crashes; iterate directly: `for r in list_runs(): ...`."
    )
    def list_runs() -> list:
        if not ledger.exists():
            return []
        rows = []
        for line in ledger.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        rows.sort(key=lambda r: (
            0 if (r.get("returncode") == 0 and r.get("val_bpb") is not None) else 1,
            r.get("val_bpb") if r.get("val_bpb") is not None else float("inf"),
            r.get("ts", 0),
        ))
        return [
            {k: r.get(k) for k in
             ("n", "val_bpb", "returncode", "elapsed_s", "ts", "train_py_path")}
            for r in rows
        ]

    return [run_experiment, list_runs]


def make_llm(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)



def _node_one_liner(node) -> str:
    if node.type == "query":
        text = (getattr(node, "content", "") or "").strip()
    elif node.type == "action":
        text = (getattr(node, "code", "") or "").strip()
    elif node.type == "observation":
        text = (getattr(node, "output", "") or "").strip()
    elif node.type == "result":
        text = (getattr(node, "result", "") or "").strip()
    elif node.type == "supervising":
        waiting = ", ".join(getattr(node, "waiting_on", []) or [])
        return f"supervising  waiting on [{waiting}]"
    else:
        text = ""
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    return f"{node.type:11s} {first[:140]}"


def _print_event(node, t0: float) -> None:
    elapsed = time.monotonic() - t0
    head = f"[{elapsed:6.1f}s] {node.agent_id:24s}"
    print(head, _node_one_liner(node), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", type=Path, required=True,
                   help="Source dir with train.py + program.md (e.g. examples/autoresearch/tinker).")
    p.add_argument("--budget-s", type=int, default=600,
                   help="Default wall-clock budget per run_experiment call. "
                        "Solo runs take ~210s; bump higher for parallel "
                        "trials since the Tinker API serializes them.")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--workspace", type=Path, default=Path("./runs/autoresearch"))
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--no-ui", action="store_true",
                   help="Disable the live rich dashboard; just stream events.")
    args = p.parse_args()

    target = args.target.resolve()
    if not (target / "train.py").exists() or not (target / "program.md").exists():
        raise SystemExit(f"autoresearch: {target} must contain train.py and program.md")

    workspace = Workspace.create(args.workspace)
    for entry in target.iterdir():
        if entry.name in SKIP_NAMES:
            continue
        dst = workspace.root / entry.name
        if dst.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, dst, symlinks=True,
                            ignore=lambda _, n: {x for x in n if x in SKIP_NAMES})
        else:
            shutil.copy2(entry, dst)

    runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools([*FILE_TOOLS, *make_run_experiment(workspace.root)])

    flow = RLMFlow(
        llm_client=make_llm(args.model),
        runtime=runtime,
        workspace=workspace,
        config=RLMConfig(
            max_iterations=args.rounds,
            max_depth=args.max_depth,
            max_concurrency=args.max_concurrency,
        ),
    )

    program = (workspace.root / "program.md").read_text()
    query = (
        f"{program}\n\n"
        f"## Run parameters\n"
        f"- pass `budget_s={args.budget_s}` to `run_experiment`.\n"
        f"- you have {args.rounds} iterations total.\n"
        f"- finish with `done(<short summary including final val_bpb>)`.\n"
    )

    print(f"[autoresearch] target={target}", flush=True)
    print(f"[autoresearch] workspace={workspace.root}", flush=True)
    print(f"[autoresearch] model={args.model}  rounds={args.rounds}  budget_s={args.budget_s}", flush=True)

    if args.no_ui:
        t0 = time.monotonic()
        seen: set[str] = set()
        graph = flow.start(query)
        for node in graph.nodes:
            if node.id not in seen:
                _print_event(node, t0); seen.add(node.id)
        while not graph.finished:
            graph = flow.step(graph)
            for node in graph.nodes:
                if node.id not in seen:
                    _print_event(node, t0); seen.add(node.id)
    else:
        with LiveView() as live:
            graph = flow.start(query)
            live(graph)
            while not graph.finished:
                graph = flow.step(graph)
                live(graph)

    print("=" * 80, flush=True)
    print(graph.result() or "(no result)", flush=True)
    print(f"\nWorkspace: {workspace.root}", flush=True)

    try:
        from rlmflow.utils.viewer import save_html
        viewer = args.workspace / "viewer.html"
        save_html(workspace, viewer)
        print(f"Viewer:    {viewer}", flush=True)
    except ImportError as exc:
        print(f"Viewer not saved: {exc}", flush=True)


if __name__ == "__main__":
    main()
