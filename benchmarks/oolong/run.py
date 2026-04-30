"""OOLONG runner — rlmflow port of Prime Intellect's ``oolong-rlm`` env.

Mirrors the layout in
[`PrimeIntellect-ai/verifiers`](https://github.com/PrimeIntellect-ai/verifiers/tree/sebastian/experiment/rlm/environments/oolong)
so results can be compared head-to-head:

- Three **modes**: ``standard`` (single-call baseline, ``\\boxed{}``
  extraction), ``rlm`` (rlmflow recursive scaffold, file-backed context),
  ``rlm_tips`` (same scaffold + the verbatim ``<env_tips>`` strategy
  block PI uses for SFT data generation).
- Three **subsets**: ``synth`` and ``synth_with_labels`` (from
  ``oolongbench/oolong-synth``) and ``real`` (from
  ``oolongbench/oolong-real``, ``dnd`` config).
- Output layout PI's ``aggregate_results.py`` understands:
  ``outputs/evals/<run_id>/{metadata.json, results.jsonl}``.

Where we **diverge** from PI on purpose: scoring. PI uses ``gpt-5-mini``
as a judge (weight 1.0). We use the OOLONG paper's deterministic
template-slot extractor in :mod:`scoring`, which gives a
reproducible per-row ``score`` (numeric partial credit + normalised
exact match). PI-style ``exact_match_reward`` and
``contains_answer_reward`` are recorded as 0-weight diagnostics so the
PI aggregator's columns still line up.

Usage::

    python benchmarks/oolong/run.py \\
        --mode rlm --subset synth --split validation \\
        --limit 50 --shuffle --seed 42 \\
        --model claude-sonnet-4-20250514 \\
        --max-iterations 30 --max-depth 3

    python benchmarks/oolong/run.py --mode standard --subset real --limit 20

The Prime Intellect ``run_ablations.sh`` is ported as
``benchmarks/oolong/run_ablations.sh`` for the full mode × subset
sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

# Make ``scoring`` importable when this script is run as
# ``python benchmarks/oolong/run.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import (  # noqa: E402
    instantiate_template,
    normalize,
    normalize_answer_type,
    parse_gold,
    render_gold_templated,
    score_one,
    template_from_question,
)

from rlmflow.llm import AnthropicClient, LLMClient, LLMUsage, OpenAIClient  # noqa: E402
from rlmflow.node import Node  # noqa: E402
from rlmflow.rlm import RLMConfig, RLMFlow  # noqa: E402
from rlmflow.runtime.docker import DockerRuntime  # noqa: E402
from rlmflow.runtime.local import LocalRuntime  # noqa: E402
from rlmflow.tools import FILE_TOOLS  # noqa: E402
from rlmflow.utils.trace import save_trace  # noqa: E402
from rlmflow.workspace import Workspace  # noqa: E402


# ── Prompts (PI-aligned) ──────────────────────────────────────────────

# Verbatim from Prime Intellect's `oolong.py`. Kept in <env_tips> tags
# so the same strip-tip preprocessing as PI's harness works.
ENV_TIPS = """
<env_tips>
Strategy for long-context information retrieval:

1. Split the context into chunks (e.g., by paragraphs or fixed character windows with some overlap)
2. Write a prompt describing what to look for, then append it to each chunk to create a list of prompts
3. Call delegate(name, query) once per chunk, then `yield wait(*handles)` to scan chunks in parallel
4. Aggregate the relevant findings from the responses
</env_tips>"""

# RLM prompt — the agent gets the question only; the long context is
# durable on disk as ``task_NNNN.txt`` and reachable via file tools and
# delegation. This matches the file-backed-context discipline from the
# RLM paper (arXiv:2512.24601 §2) and the rlmflow harness doc
# (`docs/internal/oolong_harness.md`).
RLM_PROMPT_TEMPLATE = """You are answering an OOLONG long-context aggregation question over a passage
stored in {context_file} ({tokens} approx tokens / {bytes} bytes).

Task metadata:
- subset: {subset}
- task_group: {task_group}
- task: {task}
- answer_type: {answer_type}
- dataset: {dataset}

Tools:
- read_file, read_lines, line_count, list_files, grep, ls
- delegate(name, query) and yield wait(*handles)

Question:
{question}

Required method:
1. Inspect the format first: sample the start, middle, and end of {context_file}.
   Identify record boundaries and fields like Date, User, and Instance.
2. Important: in the normal `synth` subset, labels are usually NOT stored as an
   explicit field. You must infer/classify each Instance when the task asks for
   label statistics. Do not answer zero or "same frequency as" just because no
   `label:` field exists. If subset is `synth_with_labels`, then use the explicit
   labels.
3. For large contexts, split independent line ranges and delegate chunk work.
   Children should return compact JSON counts or facts only.
4. Aggregate child JSON/counts in Python. If a child output is malformed, inspect
   or repair once.
5. Verify the final relation/label/number/user and match the exact answer
   template from the question.

Useful label hints:
- TREC coarse labels: abbreviation, entity, human being, numeric value,
  location, description and abstract concept. Classify by the answer type the
  question is asking for (who -> human being, where -> location, how many/how
  much -> numeric value, what does X stand for -> abbreviation, why/how/what
  explanation -> description, named object/concept -> entity).
- Spam labels: classify each message/instance as spam or ham from its content
  if no label field is present.

Call done(message) with only the final answer string in the exact requested
form. No reasoning in done()."""

STANDARD_PROMPT_TEMPLATE = """{question}

<context>
{context}
</context>

Provide your answer inside \\boxed{{}}."""

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


# ── Subset / dataset wiring ───────────────────────────────────────────


def resolve_subset(subset: str) -> tuple[str, str | None, str]:
    """Return ``(hf_dataset, hf_config, context_column)`` for a subset.

    Mirrors PI's mapping in ``oolong.py``::

        synth              -> oolong-synth, context_window_text
        synth_with_labels  -> oolong-synth, context_window_text_with_labels
        real               -> oolong-real(dnd), context_window_text
    """
    if subset == "synth":
        return "oolongbench/oolong-synth", None, "context_window_text"
    if subset == "synth_with_labels":
        return "oolongbench/oolong-synth", None, "context_window_text_with_labels"
    if subset == "real":
        return "oolongbench/oolong-real", "dnd", "context_window_text"
    raise ValueError(
        f"unknown subset {subset!r}; expected one of: synth, synth_with_labels, real"
    )


def load_examples(args) -> list[dict[str, Any]]:
    """Load the HuggingFace dataset and return a list of plain dicts.

    We materialise to plain dicts up-front so worker threads don't share
    a HuggingFace ``Dataset`` (its lazy decoding isn't thread-friendly
    on huge string columns) and so the ``--limit`` slice is cheap.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "OOLONG requires `pip install datasets`. Install it and re-run."
        ) from exc

    hf_name, hf_config, context_col = resolve_subset(args.subset)
    print(
        f">>> loading {hf_name} "
        f"(config={hf_config or 'default'}, split={args.split})"
    )
    ds = load_dataset(hf_name, hf_config, split=args.split)

    rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        rows.append(
            {
                "example_id": idx,
                "question": ex.get("question", ""),
                "context": ex.get(context_col, ""),
                "answer": ex.get("answer", ""),
                "answer_type": ex.get("answer_type"),
                "task_group": ex.get("task_group"),
                "task": ex.get("task"),
                "context_len": ex.get("context_len"),
                "dataset": ex.get("dataset"),
                "num_labels": ex.get("num_labels"),
            }
        )

    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(rows)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    return rows


# ── Model wiring ──────────────────────────────────────────────────────


def make_llm(model: str, *, api_key_var: str | None = None) -> LLMClient:
    """Build an LLM client by family. ``claude*`` → Anthropic, else OpenAI."""
    api_key = os.environ.get(api_key_var) if api_key_var else None
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if model.startswith("claude"):
        return AnthropicClient(model=model, **kwargs)
    return OpenAIClient(model=model, **kwargs)


def make_runtime(workspace: Path | str | Workspace, *, docker_image: str | None):
    """Build a runtime over *workspace* and register OOLONG file tools."""
    if docker_image:
        rt = DockerRuntime(docker_image, workspace=workspace)
    else:
        rt = LocalRuntime(workspace=workspace)
    rt.register_tools(FILE_TOOLS)
    return rt


# ── Mode runners ──────────────────────────────────────────────────────


def run_standard_task(
    *,
    row: dict[str, Any],
    llm: LLMClient,
) -> dict[str, Any]:
    """One-shot baseline: inline the context, ask for ``\\boxed{...}``."""
    prompt = STANDARD_PROMPT_TEMPLATE.format(
        question=row["question"], context=row["context"]
    )
    t0 = time.time()
    try:
        reply = llm.chat([{"role": "user", "content": prompt}])
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "prediction": "",
            "raw_reply": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": round(elapsed, 3),
            "incomplete": True,
            "tree": None,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": None,
        }
    elapsed = time.time() - t0
    usage = llm.last_usage or LLMUsage()

    boxed = BOXED_RE.findall(reply or "")
    prediction = boxed[-1].strip() if boxed else (reply or "").strip()

    return {
        "prediction": prediction,
        "raw_reply": reply or "",
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "elapsed_s": round(elapsed, 3),
        "incomplete": not boxed,
        "tree": None,
        "error": None,
        "trace": None,
    }


def run_rlm_task(
    *,
    row: dict[str, Any],
    llm: LLMClient,
    args,
    workspace_root: Path,
    trace_root: Path,
    task_id: str,
    on_step,
) -> dict[str, Any]:
    """RLM scaffold: per-task workspace, file-backed context, recursive REPL."""
    workspace = Workspace.create(workspace_root / task_id, branch_id=task_id)
    context_file = f"task_{row['example_id']:04d}.txt"
    workspace.path(context_file).write_text(row["context"])

    runtime = make_runtime(workspace, docker_image=args.docker_image)

    def runtime_factory():
        return make_runtime(workspace, docker_image=args.docker_image)

    config = RLMConfig(
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        max_output_length=args.max_output_length,
        max_concurrency=args.max_concurrency,
    )

    agent = RLMFlow(
        llm_client=llm,
        runtime=runtime,
        config=config,
        runtime_factory=runtime_factory,
        workspace=workspace,
    )

    question = row["question"]
    if args.include_env_tips:
        question = question + ENV_TIPS

    prompt = RLM_PROMPT_TEMPLATE.format(
        context_file=context_file,
        tokens=max(1, len(row["context"]) // 4),
        bytes=len(row["context"].encode("utf-8")),
        subset=args.subset,
        task_group=row.get("task_group") or "?",
        task=row.get("task") or "?",
        answer_type=row.get("answer_type") or "?",
        dataset=row.get("dataset") or "?",
        question=question,
    )

    t0 = time.time()
    error: str | None = None
    states: list[Node] = []
    state: Node | None = None
    prediction = ""

    try:
        state = agent.start(prompt)
        states.append(state)
        on_step(state)

        while not state.finished:
            state = agent.step(state)
            states.append(state)
            on_step(state)

        prediction = state.result or ""
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        try:
            runtime.close()
        except Exception:
            pass

    elapsed = time.time() - t0

    if state is not None:
        tree_in, tree_out = state.tree_usage()
        all_nodes = state.walk()
        max_branching = max((len(n.children) for n in all_nodes), default=0)
        tree_summary = {
            "depth": _tree_depth(state),
            "nodes": len(all_nodes),
            "total_iterations": sum(n.type == "action" for n in all_nodes),
            "root_iterations": agent.iteration_of(state),
            "max_branching": max_branching,
        }
    else:
        tree_in = tree_out = 0
        tree_summary = None

    incomplete = (state is None or not state.finished) or error is not None

    trace_path: str | None = None
    if state is not None:
        trace_dir = trace_root / task_id
        try:
            save_trace(states, trace_dir, metadata={"task_id": task_id})
            trace_path = str(trace_dir / "trace.json")
        except Exception:
            trace_path = None

    return {
        "prediction": prediction or "",
        "raw_reply": getattr(state, "reply", "") if state is not None else "",
        "input_tokens": tree_in,
        "output_tokens": tree_out,
        "elapsed_s": round(elapsed, 3),
        "incomplete": incomplete,
        "tree": tree_summary,
        "error": error,
        "trace": trace_path,
    }


def _tree_depth(node: Node) -> int:
    children = node.child_nodes()
    if not children:
        return 0
    return 1 + max(_tree_depth(child) for child in children)


# ── Per-row scoring ───────────────────────────────────────────────────


def score_row(row: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """Compute paper-style ``score`` plus PI-shape diagnostics.

    Diagnostics line up with PI's reward functions in ``oolong.py`` so
    ``aggregate.py`` (or PI's own ``aggregate_results.py``) reads our
    rows without a schema delta:

    - ``exact_match_reward`` — full prediction equals gold (case-insensitive,
      whitespace-collapsed). PI's definition.
    - ``contains_answer_reward`` — gold appears as a substring of the
      prediction (case-insensitive). PI's definition.
    - ``score`` / ``answer_type`` / ``extracted`` — paper methodology
      (template-slot extraction + per-type dispatch from
      :mod:`scoring`). This is the primary metric.
    """
    prediction = outcome["prediction"] or ""
    gold_raw = row["answer"]
    gold_list = parse_gold(gold_raw)

    paper = score_one(
        prediction=prediction,
        gold=gold_raw,
        question=row["question"],
        answer_type=row.get("answer_type"),
    )

    norm_pred = normalize(prediction)
    em = 1.0 if any(norm_pred == normalize(g) for g in gold_list) else 0.0
    contains = 1.0 if any(normalize(g) and normalize(g) in norm_pred for g in gold_list) else 0.0

    return {
        "score": float(paper["score"]),
        "answer_type": paper["answer_type"],
        "extracted": paper["extracted"],
        "exact_match": float(paper["exact_match"]),
        "exact_match_reward": em,
        "contains_answer_reward": contains,
        "judge_reward": float(paper["score"]),
    }


# ── Manifest / IO helpers ─────────────────────────────────────────────


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def rlmflow_version() -> str | None:
    try:
        import rlmflow  # noqa: F401

        from importlib.metadata import version

        return version("rlmflow")
    except Exception:
        return None


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    h = sha256()
    for row in rows:
        h.update(str(row["example_id"]).encode())
        h.update(b"\x00")
        h.update((row.get("dataset") or "").encode())
        h.update(b"\x00")
        h.update(str(row.get("context_len") or "").encode())
        h.update(b"\x00")
        h.update((row["question"] or "").encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


def build_metadata(args, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Self-contained snapshot of run config (PI-shape ``env_args`` block)."""
    use_rlm = args.mode in ("rlm", "rlm_tips")
    return {
        "model": args.model,
        "argv": sys.argv,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rlmflow_version": rlmflow_version(),
        "git_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint(rows),
        "n": len(rows),
        "env_args": {
            "subset": args.subset,
            "split": args.split,
            "use_rlm": use_rlm,
            "include_env_tips": args.include_env_tips,
            "max_iterations": args.max_iterations,
            "max_output_length": args.max_output_length,
            "max_depth": args.max_depth,
            "max_concurrency": args.max_concurrency,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "docker_image": args.docker_image,
            "mode": args.mode,
        },
    }


def aggregate_summary(
    metadata: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Roll up per-task rows into the ``summary.json`` shape."""
    n_total = len(rows)
    errored = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
        s = sorted(values)
        m = len(s)
        return {
            "n": m,
            "mean": sum(s) / m,
            "median": s[m // 2],
            "min": s[0],
            "max": s[-1],
            "sum": sum(s),
        }

    def by(key: str) -> dict[str, dict[str, float]]:
        groups: dict[str, list[float]] = {}
        for r in ok:
            k = str(r.get(key) or "?")
            groups.setdefault(k, []).append(float(r.get("score", 0.0)))
        return {k: {"n": len(v), "score": sum(v) / len(v)} for k, v in groups.items()}

    return {
        "manifest": metadata,
        "n_total": n_total,
        "n_ok": len(ok),
        "n_error": len(errored),
        "scores": {
            "score": stats([float(r.get("score", 0.0)) for r in ok]),
            "exact_match": stats([float(r.get("exact_match", 0.0)) for r in ok]),
            "exact_match_reward": stats(
                [float(r.get("exact_match_reward", 0.0)) for r in ok]
            ),
            "contains_answer_reward": stats([float(r.get("contains_answer_reward", 0.0)) for r in ok]),
        },
        "tokens": {
            "input": stats([float(r.get("input_tokens", 0)) for r in ok]),
            "output": stats([float(r.get("output_tokens", 0)) for r in ok]),
            "total": stats([float(r.get("input_tokens", 0) + r.get("output_tokens", 0)) for r in ok]),
        },
        "latency_s": stats([float(r.get("elapsed_s", 0.0)) for r in ok]),
        "by_answer_type": by("answer_type"),
        "by_task_group": by("task_group"),
        "by_dataset": by("dataset"),
        "incomplete": sum(1 for r in ok if r.get("incomplete")),
    }


def render_summary_md(summary: dict[str, Any]) -> str:
    m = summary["manifest"]
    env = m["env_args"]
    lines = [
        "# OOLONG run summary",
        "",
        f"- Model: `{m['model']}`",
        f"- Mode: `{env['mode']}`  (use_rlm={env['use_rlm']}, include_env_tips={env['include_env_tips']})",
        f"- Subset / split: `{env['subset']}` / `{env['split']}`",
        f"- N: {summary['n_total']}  (ok={summary['n_ok']}, error={summary['n_error']})",
        f"- Incomplete: {summary['incomplete']}",
        f"- Git: `{m.get('git_sha') or 'n/a'}`  ·  rlmflow `{m.get('rlmflow_version') or 'n/a'}`",
        f"- Dataset fingerprint: `{m['dataset_fingerprint']}`",
        "",
        "## Scores",
        f"- score (paper): **{summary['scores']['score']['mean']:.4f}**",
        f"- exact_match (after extraction): {summary['scores']['exact_match']['mean']:.4f}",
        f"- exact_match_reward (raw prediction vs raw gold): {summary['scores']['exact_match_reward']['mean']:.4f}",
        f"- contains_answer_reward: {summary['scores']['contains_answer_reward']['mean']:.4f}",
        "",
        "## Tokens (per task, mean)",
        f"- input: {summary['tokens']['input']['mean']:.0f}",
        f"- output: {summary['tokens']['output']['mean']:.0f}",
        f"- total: {summary['tokens']['total']['mean']:.0f}",
        "",
    ]
    if summary["by_answer_type"]:
        lines.append("## By answer_type")
        for k, v in sorted(summary["by_answer_type"].items()):
            lines.append(f"- `{k}`: n={v['n']}, score={v['score']:.4f}")
        lines.append("")
    if summary["by_task_group"]:
        lines.append("## By task_group")
        for k, v in sorted(summary["by_task_group"].items()):
            lines.append(f"- `{k}`: n={v['n']}, score={v['score']:.4f}")
        lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "example_id",
        "mode",
        "subset",
        "dataset",
        "task_group",
        "task",
        "answer_type",
        "score",
        "exact_match",
        "prediction",
        "extracted",
        "gold_templated",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "elapsed_s",
        "incomplete",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_errors_md(rows: list[dict[str, Any]]) -> str:
    bad = [r for r in rows if r.get("error") or r.get("incomplete")]
    lines = ["# OOLONG Errors And Incomplete Tasks", ""]
    if not bad:
        lines.append("No errors or incomplete tasks.")
        return "\n".join(lines) + "\n"

    for row in bad:
        lines.extend(
            [
                f"## {row['task_id']}",
                "",
                f"- score: `{row.get('score', 0):.4f}`",
                f"- incomplete: `{row.get('incomplete')}`",
                f"- error: `{row.get('error') or ''}`",
                f"- trace: `{row.get('trace_path') or ''}`",
                "",
                "**Question**",
                "",
                (row.get("question") or "")[:1200],
                "",
                "**Prediction**",
                "",
                (row.get("prediction") or "")[:1200],
                "",
                "**Gold**",
                "",
                str(row.get("gold_templated") or row.get("gold") or ""),
                "",
            ]
        )
    return "\n".join(lines)


def render_run_readme(metadata: dict[str, Any]) -> str:
    env = metadata["env_args"]
    return f"""# OOLONG Run

Model: `{metadata['model']}`

Mode: `{env['mode']}`  
Subset: `{env['subset']}` / `{env['split']}`  
Examples: `{metadata['n']}`  
Dataset fingerprint: `{metadata['dataset_fingerprint']}`  
Git SHA: `{metadata.get('git_sha') or 'n/a'}`

## Files

- `config/manifest.json` — full run metadata.
- `config/env_args.json` — benchmark/runtime config only.
- `data/tasks.jsonl` — task ids and dataset metadata, without full contexts.
- `results/results.jsonl` — one scored row per task.
- `results/predictions.csv` — compact spreadsheet-friendly predictions.
- `results/summary.json` — aggregate metrics.
- `reports/report.md` — human-readable summary.
- `reports/errors.md` — errors and incomplete tasks with trace links.
- `traces/<task_id>/trace.json` — full RLM trace per task.
- `workspaces/<task_id>/` — task-local runtime workspace and metadata.

Root-level `metadata.json`, `results.jsonl`, `summary.json`, and `summary.md`
are compatibility copies for existing aggregation scripts.
"""


def write_run_artifacts(
    *,
    out_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    write_result_rows: bool = True,
) -> dict[str, Path]:
    """Write the final experiment directory in both rich and compat layouts."""
    config_dir = out_dir / "config"
    data_dir = out_dir / "data"
    results_dir = out_dir / "results"
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_md = render_summary_md(summary)
    errors_md = render_errors_md(rows)
    task_index = [
        {
            "task_id": r["task_id"],
            "example_id": r["example_id"],
            "subset": r["subset"],
            "split": r["split"],
            "dataset": r.get("dataset"),
            "task_group": r.get("task_group"),
            "task": r.get("task"),
            "answer_type": r.get("answer_type"),
            "context_len": r.get("context_len"),
            "context_chars": r.get("context_chars"),
        }
        for r in rows
    ]

    write_json(config_dir / "manifest.json", metadata)
    write_json(config_dir / "env_args.json", metadata["env_args"])
    write_json(
        config_dir / "run_config.json",
        {"model": metadata["model"], **metadata["env_args"]},
    )
    write_jsonl(data_dir / "tasks.jsonl", task_index)
    if write_result_rows:
        write_jsonl(results_dir / "results.jsonl", rows)
    write_predictions_csv(results_dir / "predictions.csv", rows)
    write_json(results_dir / "summary.json", summary)
    (reports_dir / "report.md").write_text(report_md)
    (reports_dir / "errors.md").write_text(errors_md)
    (out_dir / "README.md").write_text(render_run_readme(metadata))

    # Compatibility copies for PI-style and older local scripts.
    write_json(out_dir / "metadata.json", metadata)
    if write_result_rows:
        write_jsonl(out_dir / "results.jsonl", rows)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(report_md)

    return {
        "report": reports_dir / "report.md",
        "errors": reports_dir / "errors.md",
        "results": results_dir / "results.jsonl",
        "summary": results_dir / "summary.json",
        "manifest": config_dir / "manifest.json",
    }


# ── Reporter ──────────────────────────────────────────────────────────


def make_reporter(args, total: int):
    """Best-effort live viz; falls back to plain prints if rich is missing."""
    if not args.viz:
        return None
    try:
        from viz import Reporter

        return Reporter(
            viz=True,
            total=total,
            mode=args.mode,
            model=args.model,
            fast_model=None,
            workers=args.workers,
            max_concurrency=args.max_concurrency or 1,
        )
    except Exception:
        return None


# ── Main loop ─────────────────────────────────────────────────────────


def run_one(
    *,
    row: dict[str, Any],
    args,
    workspace_root: Path,
    trace_root: Path,
    task_id: str,
    reporter,
) -> dict[str, Any]:
    use_rlm = args.mode in ("rlm", "rlm_tips")
    llm = make_llm(args.model, api_key_var=args.api_key_var)

    if reporter is not None:
        reporter.start(task_id)

    def on_step(state: Node) -> None:
        if reporter is not None:
            reporter.update(task_id, state)

    if use_rlm:
        outcome = run_rlm_task(
            row=row,
            llm=llm,
            args=args,
            workspace_root=workspace_root,
            trace_root=trace_root,
            task_id=task_id,
            on_step=on_step,
        )
    else:
        outcome = run_standard_task(row=row, llm=llm)

    scoring = score_row(row, outcome)

    template = template_from_question(row["question"])
    gold_list = parse_gold(row["answer"])
    gold_templated = render_gold_templated(row["question"], gold_list)
    if gold_templated is None and template and gold_list:
        gold_templated = instantiate_template(template, str(gold_list[0]))

    final_row = {
        "task_id": task_id,
        "example_id": row["example_id"],
        "mode": args.mode,
        "subset": args.subset,
        "split": args.split,
        "task_group": row.get("task_group"),
        "task": row.get("task"),
        "dataset": row.get("dataset"),
        "context_len": row.get("context_len"),
        "context_chars": len(row["context"]),
        "context_tokens_approx": max(1, len(row["context"]) // 4),
        "info": {"context_length": len(row["context"])},
        "question": row["question"],
        "gold": gold_list,
        "gold_templated": gold_templated,
        "prediction": outcome["prediction"],
        "final_answer": outcome["prediction"],
        "raw_reply": outcome["raw_reply"][:2000],
        "input_tokens": outcome["input_tokens"],
        "output_tokens": outcome["output_tokens"],
        "total_tokens": outcome["input_tokens"] + outcome["output_tokens"],
        "prompt_tokens": outcome["input_tokens"],
        "completion_tokens": outcome["output_tokens"],
        "turns": (outcome["tree"] or {}).get("total_iterations") if outcome["tree"] else 1,
        "elapsed_s": outcome["elapsed_s"],
        "generation_ms": int(outcome["elapsed_s"] * 1000),
        "total_ms": int(outcome["elapsed_s"] * 1000),
        "tree": outcome["tree"],
        "trace_path": outcome["trace"],
        "incomplete": outcome["incomplete"],
        "error": outcome["error"],
        "answer_type_normalized": normalize_answer_type(row.get("answer_type")),
        **scoring,
    }

    if reporter is not None:
        reporter.complete(task_id, final_row)
    return final_row


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OOLONG benchmark runner — rlmflow port of Prime Intellect's "
            "oolong-rlm environment. Three modes (standard / rlm / rlm_tips) "
            "× three subsets (synth / synth_with_labels / real)."
        )
    )

    parser.add_argument(
        "--mode",
        choices=("standard", "rlm", "rlm_tips"),
        default="rlm",
        help="standard=single-call \\boxed{} baseline; rlm=rlmflow recursive scaffold; rlm_tips=rlm + verbatim PI <env_tips> block.",
    )
    parser.add_argument(
        "--subset",
        choices=("synth", "synth_with_labels", "real"),
        default="synth",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=50)

    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--api-key-var",
        default=None,
        help="Environment variable to read the LLM API key from. Defaults to provider standard.",
    )

    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-output-length", type=int, default=8192)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Per-task delegation parallelism (rlm only). Default sequential.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Task-level threads. Bumps API concurrency, not delegation fan-out.",
    )

    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--out", default=None,
                        help="Output dir. Defaults to outputs/evals/<run_id>/.")
    parser.set_defaults(viz=True)
    parser.add_argument("--viz", dest="viz", action="store_true",
                        help="Live Rich board (default; requires rich).")
    parser.add_argument("--no-viz", dest="viz", action="store_false",
                        help="Disable the live board and print one line per task.")
    parser.add_argument(
        "--include-env-tips",
        action="store_true",
        help="Force-on the <env_tips> block. Implied by --mode rlm_tips.",
    )

    args = parser.parse_args()

    if args.mode == "rlm_tips":
        args.include_env_tips = True
    if args.mode == "standard" and args.include_env_tips:
        print(
            ">>> warning: --include-env-tips ignored in standard mode (no REPL).",
            file=sys.stderr,
        )
        args.include_env_tips = False

    rows = load_examples(args)
    if not rows:
        raise SystemExit("no examples to run after subset/split/limit filtering")

    run_id = args.out or _default_run_id(args)
    out_dir = Path(run_id)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace_root = out_dir / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    trace_root = out_dir / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(args, rows)
    initial_summary = aggregate_summary(metadata, [])
    artifact_paths = write_run_artifacts(
        out_dir=out_dir,
        metadata=metadata,
        rows=[],
        summary=initial_summary,
        write_result_rows=True,
    )

    print(f">>> writing results to {out_dir}")
    print(f">>> manifest: {artifact_paths['manifest']}")
    print(f">>> live report: {artifact_paths['report']}")
    print(f">>> workspaces: {workspace_root}")
    print(f">>> traces: {trace_root}")
    print(f">>> mode={args.mode}  subset={args.subset}/{args.split}  n={len(rows)}  "
          f"model={args.model}  workers={args.workers}")

    reporter = make_reporter(args, total=len(rows))
    completed: list[dict[str, Any]] = []

    results_path = out_dir / "results" / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_fp = results_path.open("w")

    cm = reporter if reporter is not None else nullcontext()
    try:
        with cm:
            if args.workers <= 1:
                for i, row in enumerate(rows):
                    task_id = f"task_{i:04d}"
                    final = run_one(
                        row=row,
                        args=args,
                        workspace_root=workspace_root,
                        trace_root=trace_root,
                        task_id=task_id,
                        reporter=reporter,
                    )
                    completed.append(final)
                    results_fp.write(json.dumps(final, default=str) + "\n")
                    results_fp.flush()
                    live_rows = sorted(completed, key=lambda r: r["task_id"])
                    write_run_artifacts(
                        out_dir=out_dir,
                        metadata=metadata,
                        rows=live_rows,
                        summary=aggregate_summary(metadata, live_rows),
                        write_result_rows=False,
                    )
                    if reporter is None:
                        _print_progress(i + 1, len(rows), final)
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(
                            run_one,
                            row=row,
                            args=args,
                            workspace_root=workspace_root,
                            trace_root=trace_root,
                            task_id=f"task_{i:04d}",
                            reporter=reporter,
                        ): i
                        for i, row in enumerate(rows)
                    }
                    done = 0
                    for fut in as_completed(futures):
                        final = fut.result()
                        completed.append(final)
                        results_fp.write(json.dumps(final, default=str) + "\n")
                        results_fp.flush()
                        done += 1
                        live_rows = sorted(completed, key=lambda r: r["task_id"])
                        write_run_artifacts(
                            out_dir=out_dir,
                            metadata=metadata,
                            rows=live_rows,
                            summary=aggregate_summary(metadata, live_rows),
                            write_result_rows=False,
                        )
                        if reporter is None:
                            _print_progress(done, len(rows), final)
    finally:
        results_fp.close()

    completed.sort(key=lambda r: r["task_id"])
    summary = aggregate_summary(metadata, completed)
    artifact_paths = write_run_artifacts(
        out_dir=out_dir,
        metadata=metadata,
        rows=completed,
        summary=summary,
        write_result_rows=True,
    )

    s = summary["scores"]
    print()
    print("=" * 60)
    print(f"score (paper):           {s['score']['mean']:.4f}  "
          f"(n={s['score']['n']})")
    print(f"exact_match:             {s['exact_match']['mean']:.4f}")
    print(f"exact_match_reward:      {s['exact_match_reward']['mean']:.4f}  (raw diagnostic)")
    print(f"contains_answer_reward:  {s['contains_answer_reward']['mean']:.4f}")
    print(f"errors / incomplete:     {summary['n_error']} / {summary['incomplete']}")
    print(f"results: {artifact_paths['results']}")
    print(f"report:  {artifact_paths['report']}")
    print(f"traces:  {trace_root}")


def _default_run_id(args) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^a-zA-Z0-9]+", "-", args.model).strip("-")
    return str(Path("outputs") / "evals" / f"{ts}_{args.mode}_{args.subset}_{safe_model}")


def _print_progress(done: int, total: int, row: dict[str, Any]) -> None:
    err = row.get("error")
    flag = "ERR" if err else f"{row.get('score', 0):.2f}"
    extra = f" [in={row.get('input_tokens', 0)} out={row.get('output_tokens', 0)}]"
    tree = row.get("tree")
    if tree:
        extra += f" [d{tree['depth']} n{tree['nodes']} i{tree['total_iterations']}]"
    print(f">>> [{done}/{total}] {row['task_id']} score={flag}  {row.get('elapsed_s', 0):.1f}s{extra}")
    if err:
        print(f"    {err.splitlines()[0] if err else ''}")


if __name__ == "__main__":
    main()
