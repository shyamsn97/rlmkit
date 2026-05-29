"""Recursive autoresearch on top of RLMFlow.

Inspired by https://github.com/karpathy/autoresearch . The agent
searches over one solution file (default ``solution.py``), while
RLMFlow supplies recursion: a parent agent plans hypotheses and
delegates child agents to run independent trials. Each child passes
a complete source string to ``run_experiment(source, description)``.
A separate ``evaluate.py`` imports each archived candidate, runs it,
and prints ``score: <float>``.

Per-target ``program.md`` carries the contract + worked examples.
This file carries the recursive agent wiring and task-agnostic
parent/child guidance.

Usage:
    python examples/autoresearch/autoresearch.py \\
        --target examples/autoresearch/circle_packing
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
from rlmflow.prompts import DEFAULT_BUILDER
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import FILE_TOOLS, tool
from rlmflow.utils.viz import LiveView

AUTORESEARCH_RECURSION_TEXT = """
You are running recursive autoresearch.

**Parent agent:** your query points at `program.md`. Read it, run
`run_baseline()` once, inspect `get_runs()`, choose idea-named
slugs, then launch one child per hypothesis in parallel with
`await launch_subagents([{"name": slug, "query": query, "context": context}, ...])`.
The parent does not write or run candidate code directly. End the block exactly
at that `await launch_subagents([...])` call. On the resumed turn, inspect the
ledger, then either spawn another small batch of children with fresh
slugs or `done(...)`. Child agents have a small turn cap set by the
engine; do not try to simulate your own budget system.
Check `submission_status()` before spawning children. If
`remaining_submissions` is zero, summarize the best ledger entry and
`done(...)`. When `remaining_submissions` is not `None`, never spawn
more children than that number.

When creating child queries, keep them narrow. Do not ask children to
infer the task type or build generic adapters. For circle packing, say
explicitly: `N = 26`, exactly `def solve():`, return `(centers, radii)`,
maximize `sum(radii)`, no `program` imports, no env vars, no CLI/stdout,
no TSP/permutation/bitstring/generic-vector fallback code.

The parent MUST tell children to retry quick crashes. Do not group
crashes together with timeouts. Only timeouts stop immediately.
Syntax errors, wrong shapes, missing imports, invalid geometry, and
exceptions in `solve()` should be fixed and rerun as `<slug>_fix1`,
then `_fix2`, then `_fix3`.

Retries must be thinking turns, not blind loops. Children should make
one `run_experiment(...)` attempt per REPL block. If it crashes, print
the exact `stderr_tail` and stop the block without `done(...)`; on the
next turn, patch the source at the failing line/name and submit one
`_fixN`. Do not write a Python `for fix_idx in range(...)` loop or a
generic heuristic patcher that can resubmit unchanged source.

```repl
program = read_file("program.md")
base = run_baseline()
prior = get_runs()
status = submission_status()
context = (
    program
    + "\\n\\nCurrent ledger summary:\\n"
    + "\\n".join(
        f"{k}: score={v.get('score')} rc={v.get('returncode')}"
        for k, v in prior.items()
    )
)
ideas = [
    ("hex_lattice", "try a hexagonal lattice seed plus local polish"),
    ("sa_anneal", "try simulated annealing over centers with feasible radii"),
    ("greedy_maximin", "place centers greedily by largest empty-space clearance"),
]
remaining = status["remaining_submissions"]
if remaining == 0:
    done(f"submission cap reached; runs={len(prior)}")
ideas = ideas[:remaining] if remaining is not None else ideas

def child_query(slug, hyp):
    return f'''slug={slug!r}. Circle packing only.
Idea: {hyp}
Hard contract:
- exactly def solve():
- hard-code N = 26
- return centers, radii with shapes (26, 2), (26,)
- valid non-overlapping circles in [0, 1]^2; maximize sum(radii)
- no program imports, env vars, CLI/main/stdout, or generic adapters
- include docstrings that explain the strategy, helper invariants, and
  non-obvious constants
Run run_experiment(source, description=slug, budget_s=<numeric budget
from kickoff, for example 45; do not leave a placeholder in code>).
If it returns a numeric score, done.
If it times out, do not retry that slow idea; report timeout and done.
If it crashes quickly, inspect e.row or latest_run(), patch that exact
error, and retry as slug + "_fix1", then "_fix2", then "_fix3".
Do not run all fixes in one Python loop. After a crash, print the exact
stderr_tail and stop the block so the next turn can patch the failing
line/name. Each `_fixN` must visibly change the exact failed code before
calling run_experiment again.
Do not stop after a quick crash until you have either produced a score
or exhausted the three targeted fixes.'''

results = await launch_subagents([
    {"name": slug, "query": child_query(slug, hyp), "context": context}
    for slug, hyp in ideas
    if slug not in prior
])
```

```repl
# Resumed parent turn: ledger is the source of truth.
runs = get_runs()
scored = [r for r in runs.values() if r.get("score") is not None]
best = max(scored, key=lambda r: r["score"], default=None)
print("best:", best)
# Either spawn another small batch with fresh slugs, or finish:
done(f"best={best['score'] if best else None}; runs={len(runs)}")
```

**Child trial agent:** your query starts with `slug='<name>'.`.
Read `CONTEXT` and `program.md`, then build one complete source
string for `solution.py` and call
`run_experiment(source, description=slug)`. Do not write
`solution.py`; children run in parallel and would clobber each
other.

**Only delegate if children would explore a different approach
than yours.** If you'd just be re-asking your own query and returning
the answer, do the work yourself. Spawn children only when each
one is doing something genuinely different from your assigned idea
(different hyperparameters, an algorithmic variant, a sub-question)
that you'd then aggregate.

Candidate sources are archived research notes, not throwaway snippets.
Write docstrings as part of the experiment result. A good candidate has:
a module docstring naming the slug/strategy and expected score band;
a docstring on the target function explaining the algorithm steps,
termination condition, and validity invariants; docstrings on helpers
covering inputs, outputs, units, and invariants they preserve; and short
comments for magic constants, seeds, schedules, or thresholds. On
`_fixN` retries, update stale docstrings/comments when the algorithm
changes.

For circle packing children: do not generalize. Hard-code `N = 26`.
Do not infer `N`, import `program`, read environment variables, write
`if __name__ == "__main__"`, print/serialize a CLI answer, or support
other problem families. One child = one experiment idea.

```repl
import json
slug = "hex_lattice"  # copy exactly from your query
print(CONTEXT.read())
print(read_file("program.md"))
base_src = read_file("solution.py")

source = '''
\"\"\"Hexagonal lattice trial for 26-circle packing.

Strategy: place candidate centers on a compact grid, then assign
feasible radii. Expected score band: baseline-quality; intended as a
clear starting point for local polish.
\"\"\"

def solve():
    \"\"\"Return valid `(centers, radii)` arrays for the 26-circle task.

    The implementation initializes all centers/radii, enforces the
    required shapes, and leaves every radius non-negative and in-bounds.
    Replace this skeleton with the full strategy before running.
    \"\"\"
    import numpy as np
    # full implementation here; obey program.md exactly
    centers = np.zeros((26, 2))
    radii = np.zeros(26)
    return centers, radii
'''
r = run_experiment(source, description=slug)
done(json.dumps({"slug": slug, "score": r["score"], "path": r["solution_path"]}))
```

If `run_experiment` raises `ExperimentCrashed`, the row is already
on the ledger and the exception has `e.row`. Prefer:

```repl
try:
    r = run_experiment(source, description=slug)
    done(json.dumps({"slug": slug, "score": r["score"], "path": r["solution_path"]}))
except ExperimentCrashed as e:
    row = e.row
    print(row["stderr_tail"][-1000:])
```

If the exception class name is unavailable in your REPL, use
`except Exception as e:` and then call `latest_run()`. Do not use
`list_runs()[0]` as "latest"; `list_runs()` is best-first, not
chronological. If `returncode == -1` or `stderr_tail` says timed out,
do NOT retry the same slow idea. If a run returns a numeric score,
even a terrible one, it is complete: `done(...)` and let the parent
decide. Only retry quick crashes, and each `_fixN` must address the
specific `stderr_tail`; no generic compatibility fixes. Never perform
all `_fixN` attempts inside one Python loop; a quick crash should be
printed, then repaired in the next LLM turn.
"""


def build_prompt_builder():
    return DEFAULT_BUILDER.section(
        "autoresearch_recursion",
        AUTORESEARCH_RECURSION_TEXT,
        title="Autoresearch",
        after="examples",
    )


SCORE_RE = re.compile(
    r"score\s*[:=]\s*(-?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)",
    re.IGNORECASE,
)
SKIP_NAMES = {".git", ".DS_Store", "__pycache__", ".ipynb_checkpoints", "runs"}


class ExperimentCrashed(RuntimeError):
    """Raised by ``run_experiment`` on non-zero exit; ``self.row``
    carries the full ledger row (including ``stderr_tail``)."""

    def __init__(self, row: dict):
        self.row = row
        tail = (row.get("stderr_tail") or "").rstrip()
        super().__init__(
            f"returncode={row.get('returncode')} "
            f"description={row.get('description')!r}\n{tail}"
        )


class SubmissionError(RuntimeError):
    """Raised when the run has exhausted its experiment submission cap."""


def _run_evaluator(
    workspace_root: Path, evaluator: str,
    solution_path_rel: str, budget_s: int,
) -> tuple[str, str, int, float]:
    """``python -u <evaluator> <solution>`` from the workspace root."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-u", evaluator, solution_path_rel],
            cwd=str(workspace_root),
            capture_output=True, text=True, timeout=budget_s, env=env,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        stderr += f"\n[timed out after {budget_s}s]"
        rc = -1
    return stdout, stderr, rc, round(time.time() - t0, 2)


def _check_syntax(source: str) -> str | None:
    """Return a concise syntax error, or None if source compiles."""
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError as e:
        bad = ""
        try:
            bad = source.splitlines()[(e.lineno or 1) - 1].strip()
        except IndexError:
            pass
        return (
            f"{type(e).__name__} at line {e.lineno}:{e.offset}: "
            f"{e.msg}. Source: {bad!r}"
        )
    return None


def make_run_experiment(
    workspace_root: Path, solution: str,
    evaluator: str, lower_is_better: bool,
    max_submissions: int | None = None,
):
    """Build the agent-facing experiment tools."""
    root = workspace_root.resolve()
    history = root / "history"
    ledger = history / "ledger.jsonl"
    history.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    pending_submissions = 0
    stem = Path(solution).stem
    path_key = f"{stem}_path"

    _SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

    def _safe_slug(s: str) -> str:
        return (_SLUG_RE.sub("_", s).strip("_") or "trial")[:60]

    def _read_ledger() -> list[dict]:
        if not ledger.exists():
            return []
        out: list[dict] = []
        for line in ledger.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def _append(row: dict) -> None:
        with lock, ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def _next_n() -> int:
        # Count from the ledger (not the history/ filesystem) so archived
        # filenames track the research log, including pending parallel trials.
        with lock:
            return len(_read_ledger()) + pending_submissions

    def _submission_count() -> int:
        return sum(
            1
            for row in _read_ledger()
            if row.get("description") != "baseline"
        )

    def _remaining_submissions() -> int | None:
        if max_submissions is None:
            return None
        return max(0, max_submissions - _submission_count() - pending_submissions)

    def _reserve_submission_slot() -> int:
        nonlocal pending_submissions
        used = _submission_count() + pending_submissions
        if max_submissions is not None and used >= max_submissions:
            raise SubmissionError(
                f"too many submissions: max_submissions={max_submissions}"
            )
        n = _next_n()
        pending_submissions += 1
        return n

    def _release_submission_slot() -> None:
        nonlocal pending_submissions
        pending_submissions = max(0, pending_submissions - 1)

    def _append_submission(row: dict) -> None:
        with lock:
            _append(row)
            _release_submission_slot()

    def _parse_score(stdout: str) -> float | None:
        m = SCORE_RE.search(stdout or "")
        return float(m.group(1)) if m else None

    direction = "lower is better" if lower_is_better else "higher is better"

    @tool(
        f"Run a complete `{solution}` source string. Archives the "
        f"source as `history/<n>_<description>.py`, runs the evaluator, "
        f"and appends a row to `history/ledger.jsonl`. `description` "
        f"is a required idea slug. Returns {{n, score, returncode, "
        f"stdout_tail, stderr_tail, elapsed_s, {path_key}, "
        f"description}} ({direction}). Raises `ExperimentCrashed` on "
        f"syntax errors or non-zero exit; the failure row is still on "
        f"the ledger. Children should call this with their own source "
        f"string instead of writing `{solution}`, so parallel trials "
        f"do not clobber each other. Archived sources should include "
        f"module/function/helper docstrings explaining the strategy, "
        f"invariants, and non-obvious constants. "
        + (
            f"This run has a hard cap of {max_submissions} experiment "
            f"submissions; once exhausted, this raises SubmissionError "
            f"without writing a new ledger row."
            if max_submissions is not None
            else "This run has no explicit experiment submission cap."
        )
    )
    def run_experiment(
        source: str, description: str, budget_s: int = 300,
    ) -> dict:
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"run_experiment(source, ...) requires a non-empty "
                f"{solution} source string. Read `program.md`, build "
                f"the full file contents, and pass that string."
            )
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                "run_experiment(source, description=...) requires a non-empty "
                "slug like 'hex_grid' or 'sa_anneal'. The slug names "
                "the archived file (history/<n>_<slug>.py)."
            )
        description = description.strip()[:500]
        budget_s = max(10, min(int(budget_s), 3600))

        reserved = False
        try:
            with lock:
                n = _reserve_submission_slot()
                reserved = True
                archive = history / f"{n}_{_safe_slug(description)}.py"
                suffix = 1
                while archive.exists():
                    archive = history / f"{n}_{_safe_slug(description)}_{suffix}.py"
                    suffix += 1
                archive.write_text(source)
        except Exception:
            if reserved:
                with lock:
                    _release_submission_slot()
            raise
        archive_rel = str(archive.relative_to(root))

        row_recorded = False
        try:
            syntax = _check_syntax(source)
            if syntax is not None:
                row = {
                    "n": n, "ts": time.time(),
                    "score": None, "returncode": 1,
                    "elapsed_s": 0.0, path_key: archive_rel,
                    "description": description,
                    "stdout_tail": "",
                    "stderr_tail": f"INVALID: {syntax}\n",
                }
                _append_submission(row)
                row_recorded = True
                raise ExperimentCrashed(row)

            stdout, stderr, rc, elapsed = _run_evaluator(
                root, evaluator, archive_rel, budget_s,
            )
            row = {
                "n": n, "ts": time.time(),
                "score": _parse_score(stdout), "returncode": rc,
                "elapsed_s": elapsed, path_key: archive_rel,
                "description": description,
                "stdout_tail": (stdout or "")[-2000:],
                "stderr_tail": (stderr or "")[-1000:],
            }
            _append_submission(row)
            row_recorded = True
            if rc != 0:
                raise ExperimentCrashed(row)
            return row
        except Exception:
            if not row_recorded:
                with lock:
                    _release_submission_slot()
            raise

    @tool(
        f"Run the original baseline `{solution}` once and record it "
        f"as `description='baseline'`. Idempotent: if a baseline row "
        f"exists, that row is returned. Call once at the start; "
        f"every other trial goes through `run_experiment(...)`."
    )
    def run_baseline(budget_s: int = 600) -> dict:
        with lock:
            for row in _read_ledger():
                if row.get("description") == "baseline":
                    return row
        budget_s = max(10, min(int(budget_s), 3600))
        stdout, stderr, rc, elapsed = _run_evaluator(
            root, evaluator, solution, budget_s,
        )
        row = {
            "n": _next_n(), "ts": time.time(),
            "score": _parse_score(stdout), "returncode": rc,
            "elapsed_s": elapsed, path_key: solution,
            "description": "baseline",
            "stdout_tail": (stdout or "")[-2000:],
            "stderr_tail": (stderr or "")[-1000:],
        }
        _append(row)
        if rc != 0:
            raise ExperimentCrashed(row)
        return row

    sign = 1 if lower_is_better else -1
    miss = float("inf") if lower_is_better else float("-inf")

    def _rank_key(r):
        ok = r.get("returncode") == 0 and r.get("score") is not None
        return (0 if ok else 1,
                sign * r["score"] if ok else sign * miss,
                r.get("ts", 0))

    @tool(
        f"Every recorded trial, best-first by score "
        f"({'ascending' if lower_is_better else 'descending'}; "
        f"crashes last). Use `get_run(n)` for full stdout/stderr."
    )
    def list_runs() -> list:
        rows = sorted(_read_ledger(), key=_rank_key)
        keys = ("n", "score", "returncode", "elapsed_s", "ts",
                path_key, "description")
        return [{k: r.get(k) for k in keys} for r in rows]

    @tool("Full ledger row for trial #n, including stdout/stderr_tail. None if missing.")
    def get_run(n: int) -> dict | None:
        for row in _read_ledger():
            if row.get("n") == n:
                return row
        return None

    @tool(
        "Most recent ledger row by trial number, including stdout_tail "
        "and stderr_tail. Use this after a crash if you caught a broad "
        "Exception. Do NOT use list_runs()[0] for latest because "
        "list_runs() is sorted best-first."
    )
    def latest_run() -> dict | None:
        rows = _read_ledger()
        return max(rows, key=lambda r: r.get("n", -1), default=None)

    @tool(
        "Every recorded trial as a dict keyed by `description`. "
        "Latest wins on collision. Crashes included — check "
        "`r['returncode']` and `r['score']`."
    )
    def get_runs() -> dict:
        rows = sorted(_read_ledger(), key=lambda r: r.get("ts", 0))
        return {r.get("description") or f"n_{r.get('n')}": r for r in rows}

    @tool(
        "Submission budget for this autoresearch run. `max_submissions` "
        "and `remaining_submissions` count only run_experiment(...) trials, "
        "not the idempotent baseline."
    )
    def submission_status() -> dict:
        with lock:
            return {
                "max_submissions": max_submissions,
                "submissions": _submission_count(),
                "pending_submissions": pending_submissions,
                "remaining_submissions": _remaining_submissions(),
            }

    return [
        run_experiment, run_baseline, list_runs, get_run, latest_run, get_runs,
        submission_status,
    ]


def make_llm(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def _node_one_liner(node) -> str:
    if node.type == "supervising":
        waiting = ", ".join(getattr(node, "waiting_on", []) or [])
        return f"supervising  waiting on [{waiting}]"
    text = (
        getattr(node, "content", None) or getattr(node, "code", None)
        or getattr(node, "output", None) or getattr(node, "result", None) or ""
    ).strip()
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    return f"{node.type:11s} {first[:140]}"


def _print_event(node, t0: float) -> None:
    elapsed = time.monotonic() - t0
    print(f"[{elapsed:6.1f}s] {node.agent_id:24s} {_node_one_liner(node)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", type=Path, required=True,
                   help="Source dir with <solution>, <evaluator>, program.md.")
    p.add_argument("--solution", default="solution.py",
                   help="Baseline solution filename in --target.")
    p.add_argument("--evaluator", default="evaluate.py",
                   help="Filename of the harness; the agent never sees it.")
    p.add_argument("--lower-is-better", action="store_true")
    p.add_argument("--budget-s", type=int, default=600,
                   help="Default per-trial wall-clock cap.")
    p.add_argument("--max-submissions", type=int, default=None,
                   help="Hard cap on run_experiment submissions; baseline excluded.")
    p.add_argument("--max-iterations", type=int, default=40,
                   help="Engine max_iterations cap.")
    p.add_argument("--branches-per-turn", type=int, default=4,
                   help="How many children the parent should spawn at a time.")
    p.add_argument("--child-iterations", type=int, default=6,
                   help="Max turns per child agent. Keeps slow/broken trial "
                        "agents from retrying forever.")
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--workspace", type=Path, default=Path("./runs/autoresearch"))
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--no-ui", action="store_true")
    args = p.parse_args()
    if args.max_submissions is not None and args.max_submissions < 0:
        raise SystemExit("--max-submissions must be >= 0")

    target = args.target.resolve()
    missing = [n for n in (args.solution, args.evaluator, "program.md")
               if not (target / n).exists()]
    if missing:
        raise SystemExit(f"autoresearch: {target} is missing {missing}.")

    workspace = Workspace.create(args.workspace)
    # Bootstrap target files into the workspace. CRITICAL: the
    # immutable inputs (solution.py = the baseline, evaluate.py = the
    # harness, program.md = the brief) are *always* overwritten on
    # each run. Otherwise an agent that does `write_file('solution.py',
    # ...)` permanently corrupts the baseline for every future run.
    # Other files (e.g. README.md) are bootstrapped only on first run.
    ALWAYS_REFRESH = {args.solution, args.evaluator, "program.md"}
    for entry in target.iterdir():
        if entry.name in SKIP_NAMES:
            continue
        dst = workspace.root / entry.name
        if entry.is_dir():
            if not dst.exists():
                shutil.copytree(
                    entry, dst, symlinks=True,
                    ignore=lambda _, n: {x for x in n if x in SKIP_NAMES},
                )
            continue
        if entry.name in ALWAYS_REFRESH or not dst.exists():
            shutil.copy2(entry, dst)

    runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools([
        *FILE_TOOLS,
        *make_run_experiment(
            workspace.root, args.solution, args.evaluator,
            args.lower_is_better, args.max_submissions,
        ),
    ])

    flow = RLMFlow(
        llm_client=make_llm(args.model),
        runtime=runtime, workspace=workspace,
        config=RLMConfig(
            max_iterations=args.max_iterations,
            child_max_iterations=args.child_iterations,
            max_depth=args.max_depth,
            max_concurrency=args.max_concurrency,
        ),
        prompt_builder=build_prompt_builder(),
    )

    # Minimal kickoff query. The task contract lives in program.md;
    # recursive parent/child mechanics live in the system prompt.
    query = (
        f"Read `program.md`, run the baseline, then fan out "
        f"up to {args.branches_per_turn} child trials with `launch_subagents`, "
        f"but first call `submission_status()`; when `remaining_submissions` "
        f"is not None, never spawn more children than that number. "
        f"The hard submission cap "
        f"is {args.max_submissions if args.max_submissions is not None else 'unlimited'} "
        f"run_experiment calls, excluding the baseline. "
        f"Run MULTIPLE ROUNDS: after each batch resumes, re-read the ledger "
        f"and `submission_status()` and launch another batch while "
        f"`remaining_submissions > 0`. Keep exploration DIVERSE — each round "
        f"should mix refinements of the top 2-3 DISTINCT performers with "
        f"several brand-new, qualitatively different families (different "
        f"optimizer/seeding/geometry). Hard limit: at most ~2 variants of "
        f"any single idea family per round; never fill a batch with a dozen "
        f"micro-tweaks of the same idea. Give every child a UNIQUE slug not "
        f"already in `get_runs()` (e.g. `<best>_tighten`, or a `_r2`/`_r3` "
        f"round suffix). If the best plateaus while budget remains, switch "
        f"to an untried family rather than tweaking the winner again. Do "
        f"NOT call `done(...)` while submissions remain just because your "
        f"first idea list is used up — invent new slugs. Only summarize the "
        f"best ledger entry and call `done(...)` when no submissions remain, "
        f"or after a round yields no improvement and several families have "
        f"been tried. "
        f"Each child is capped at {args.child_iterations} turns by "
        f"the engine. Per-trial budget is {args.budget_s}s; put "
        f"`budget_s={args.budget_s}` in each child query. If a "
        f"child times out, do not retry that idea. For any quick "
        f"crash, explicitly tell the child to inspect e.row or "
        f"`latest_run()` and retry targeted fixes as `_fix1`, "
        f"`_fix2`, then `_fix3`. Never tell a child not to retry "
        f"non-timeout crashes. Do not let children run all fixes in "
        f"one Python loop; after each quick crash they should print "
        f"the exact stderr_tail, stop that block, and patch the "
        f"failing line/name on the next turn before one `_fixN` run."
    )

    print(f"[autoresearch] target={target}", flush=True)
    print(f"[autoresearch] workspace={workspace.root}", flush=True)
    print(
        f"[autoresearch] model={args.model}  max_iterations={args.max_iterations}  "
        f"branches_per_turn={args.branches_per_turn}  "
        f"child_iterations={args.child_iterations}  budget_s={args.budget_s}  "
        f"max_submissions={args.max_submissions}",
        flush=True,
    )

    if args.no_ui:
        t0 = time.monotonic()
        seen: set[str] = set()
        graph = flow.start(query)
        for node in graph.nodes:
            if node.id not in seen:
                _print_event(node, t0)
                seen.add(node.id)
        while not graph.finished:
            graph = flow.step(graph)
            for node in graph.nodes:
                if node.id not in seen:
                    _print_event(node, t0)
                    seen.add(node.id)
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
