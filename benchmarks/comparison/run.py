"""Compare rlmflow against upstream ``alexzhang13/rlm`` on one task family.

The task is a deterministic synthetic needle-in-haystack benchmark: a large
context contains many records, exactly one record has the requested marker, and
the model must return that record's secret. It is intentionally small enough to
run during development but large/structured enough to exercise RLM-style context
inspection and optional delegation.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import platform
import random
import re
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlmflow.graph import Graph  # noqa: E402
from rlmflow.llm import AnthropicClient, LLMClient, OpenAIClient  # noqa: E402
from rlmflow.rlm import RLMConfig, RLMFlow  # noqa: E402
from rlmflow.runtime.local import LocalRuntime  # noqa: E402
from rlmflow.workspace import Workspace  # noqa: E402


@dataclass(frozen=True)
class Task:
    task_id: str
    query: str
    context: str
    expected: str
    marker: str
    records: int
    needle_index: int


def make_task(*, seed: int, records: int, filler_words: int) -> Task:
    rng = random.Random(seed)
    needle_index = rng.randrange(records)
    marker = f"cmp-marker-{seed:04d}-{rng.randrange(10**8):08d}"
    expected = f"SECRET-{seed:04d}-{rng.randrange(10**10):010d}"
    vocabulary = [
        "harbor",
        "lantern",
        "quartz",
        "meadow",
        "orbit",
        "cobalt",
        "archive",
        "cedar",
        "signal",
        "vector",
        "delta",
        "ember",
        "garden",
        "matrix",
        "notebook",
        "prairie",
    ]

    blocks: list[str] = []
    for i in range(records):
        local_rng = random.Random((seed * 1_000_003) + i)
        words = [local_rng.choice(vocabulary) for _ in range(filler_words)]
        record_marker = marker if i == needle_index else f"decoy-{seed:04d}-{i:04d}"
        record_secret = expected if i == needle_index else f"DECOY-{seed:04d}-{i:04d}"
        blocks.append(
            "\n".join(
                [
                    f"RECORD {i:04d}",
                    f"title: synthetic comparison record {i}",
                    f"marker: {record_marker}",
                    f"body: {' '.join(words)}",
                    f"secret: {record_secret}",
                    "END_RECORD",
                ]
            )
        )

    # Put the records in a non-sorted order so line position cannot be inferred
    # from the marker itself.
    rng.shuffle(blocks)
    context = "\n\n".join(blocks)
    query = (
        "The CONTEXT contains many records. Exactly one record has "
        f"`marker: {marker}`. Return exactly that record's `secret` value, "
        "with no explanation and no extra text.\n\n"
        "If the CONTEXT is large, prefer splitting it into independent chunks "
        "and using the recursive or batched query tools available in your "
        "environment to inspect chunks in parallel, then combine the results."
    )
    return Task(
        task_id=f"needle_{seed:04d}",
        query=query,
        context=context,
        expected=expected,
        marker=marker,
        records=records,
        needle_index=needle_index,
    )


def make_tasks(args: argparse.Namespace) -> list[Task]:
    return [
        make_task(seed=args.seed + i, records=args.records, filler_words=args.filler_words)
        for i in range(args.n)
    ]


def infer_backend(model: str, backend: str) -> str:
    if backend != "auto":
        return backend
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


def make_rlmflow_llm(model: str, backend: str) -> LLMClient:
    backend = infer_backend(model, backend)
    if backend == "anthropic":
        return AnthropicClient(model=model)
    if backend == "openai":
        return OpenAIClient(model=model)
    raise ValueError("rlmflow runner currently supports --backend auto|openai|anthropic")


def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().strip("`'\"").strip()


def score_prediction(prediction: str, expected: str) -> dict[str, Any]:
    pred = normalize(prediction)
    exp = normalize(expected)
    return {
        "normalized_prediction": pred,
        "exact": pred == exp,
        "contains": exp in pred,
    }


def graph_usage(graph: Graph | None) -> dict[str, Any]:
    if graph is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "agents": 0,
            "nodes": 0,
            "llm_turns": 0,
            "max_depth": 0,
            "max_branching": 0,
        }

    agents = list(graph.walk())
    nodes = [node for agent in agents for node in agent.states]
    input_tokens = sum(int(getattr(node, "input_tokens", 0) or 0) for node in nodes)
    output_tokens = sum(int(getattr(node, "output_tokens", 0) or 0) for node in nodes)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "agents": len(agents),
        "nodes": len(nodes),
        "llm_turns": sum(node.type == "llm_output" for node in nodes),
        "max_depth": max((agent.depth for agent in agents), default=0),
        "max_branching": max((len(agent.children) for agent in agents), default=0),
    }


def count_python_calls(code: str, names: tuple[str, ...]) -> dict[str, int]:
    counts = {name: 0 for name in names}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        for name in names:
            counts[name] += len(re.findall(rf"(?<![\w.]){re.escape(name)}\s*\(", code))
        return counts

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> Any:
            func = node.func
            if isinstance(func, ast.Name) and func.id in counts:
                counts[func.id] += 1
            elif isinstance(func, ast.Attribute) and func.attr in counts:
                counts[func.attr] += 1
            self.generic_visit(node)

    Visitor().visit(tree)
    return counts


def merge_counts(*items: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            out[key] = out.get(key, 0) + int(value or 0)
    return out


def flatten_counts(counts: dict[str, int]) -> dict[str, int]:
    return {f"{key}_calls": value for key, value in counts.items()}


def iter_code_strings(value: Any) -> list[str]:
    snippets: list[str] = []
    if isinstance(value, dict):
        code = value.get("code")
        if isinstance(code, str) and code.strip():
            snippets.append(code)
        for child in value.values():
            snippets.extend(iter_code_strings(child))
    elif isinstance(value, list):
        for child in value:
            snippets.extend(iter_code_strings(child))
    return snippets


class CountingRLMFlow(RLMFlow):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.method_counts = {"rlm_delegate": 0, "llm_query_batched": 0}

    def llm_query_batched(self, prompts: list[str], *, model: str = "default") -> list[str]:
        self.method_counts["llm_query_batched"] += 1
        return super().llm_query_batched(prompts, model=model)

    def spawn_child(self, *args: Any, **kwargs: Any) -> Any:
        self.method_counts["rlm_delegate"] += 1
        return super().spawn_child(*args, **kwargs)


def run_rlmflow_task(task: Task, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    workspace = Workspace.create(
        run_dir / "workspaces" / "rlmflow" / task.task_id,
        branch_id=task.task_id,
    )
    runtime = LocalRuntime(workspace=workspace)
    llm = make_rlmflow_llm(args.model, args.backend)
    config = RLMConfig(
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        max_output_length=args.max_output_length,
        max_concurrency=args.max_concurrency,
    )
    agent = CountingRLMFlow(llm_client=llm, runtime=runtime, config=config, workspace=workspace)

    started = time.time()
    graph: Graph | None = None
    error: str | None = None
    prediction = ""
    try:
        graph = agent.start(
            task.query,
            context=task.context,
            context_metadata={
                "task_family": "synthetic_needle",
                "records": task.records,
                "bytes": len(task.context.encode("utf-8")),
                "marker": task.marker,
            },
        )
        while not graph.finished:
            graph = agent.step(graph)
        prediction = graph.result()
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        if hasattr(runtime, "close"):
            runtime.close()

    elapsed = time.time() - started
    usage = graph_usage(graph)
    method_counts = dict(agent.method_counts)
    return {
        "framework": "rlmflow",
        "task_id": task.task_id,
        "prediction": prediction,
        "elapsed_s": round(elapsed, 3),
        "error": error,
        "method_counts": method_counts,
        **flatten_counts(method_counts),
        **usage,
        **score_prediction(prediction, task.expected),
    }


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    result: dict[str, Any] = {}
    if hasattr(usage, "to_dict"):
        try:
            result.update(dict(usage.to_dict()))
        except Exception:
            pass
    if isinstance(usage, dict):
        result.update(usage)
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "total_input_tokens",
        "total_output_tokens",
    ):
        if hasattr(usage, key):
            result[key] = getattr(usage, key)
    return result


def make_upstream_logger() -> Any | None:
    try:
        return importlib.import_module("rlm.logger.rlm_logger").RLMLogger()
    except ImportError:
        return None


def completion_to_dict(completion: Any) -> dict[str, Any]:
    if hasattr(completion, "to_dict"):
        try:
            return dict(completion.to_dict())
        except Exception:
            pass

    usage = usage_to_dict(getattr(completion, "usage_summary", None))
    payload = {
        "root_model": getattr(completion, "root_model", None),
        "prompt": getattr(completion, "prompt", None),
        "response": getattr(completion, "response", None),
        "usage_summary": usage,
        "execution_time": getattr(completion, "execution_time", None),
    }
    metadata = getattr(completion, "metadata", None)
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def write_upstream_transcript(
    run_dir: Path,
    task: Task,
    args: argparse.Namespace,
    *,
    completion: Any | None,
    logger: Any | None,
    elapsed_s: float,
    error: str | None,
    method_counts: dict[str, int] | None = None,
) -> Path:
    transcript = {
        "framework": "alexzhang13/rlm",
        "task_id": task.task_id,
        "model": args.model,
        "backend": infer_backend(args.model, args.backend),
        "root_prompt": task.query,
        "task": {
            "expected": task.expected,
            "marker": task.marker,
            "records": task.records,
            "needle_index": task.needle_index,
            "context_bytes": len(task.context.encode("utf-8")),
        },
        "elapsed_s": round(elapsed_s, 3),
        "error": error,
        "method_counts": method_counts or {},
        "completion": completion_to_dict(completion) if completion is not None else None,
    }

    logger_trajectory = None
    if logger is not None and hasattr(logger, "get_trajectory"):
        try:
            logger_trajectory = logger.get_trajectory()
        except Exception:
            logger_trajectory = None
    if logger_trajectory is not None:
        transcript["logger_trajectory"] = logger_trajectory

    path = run_dir / "transcripts" / "alexzhang13_rlm" / f"{task.task_id}.json"
    write_json(path, transcript)
    return path


def upstream_method_counts(completion: Any | None, logger: Any | None) -> dict[str, int]:
    names = ("rlm_query", "llm_query_batched", "llm_query")
    payload: dict[str, Any] = {}
    if completion is not None:
        payload = completion_to_dict(completion)

    if not payload.get("metadata") and logger is not None and hasattr(logger, "get_trajectory"):
        try:
            payload["metadata"] = logger.get_trajectory()
        except Exception:
            pass

    counts = {name: 0 for name in names}
    for code in iter_code_strings(payload):
        counts = merge_counts(counts, count_python_calls(code, names))
    return counts


def run_upstream_task(task: Task, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    try:
        RLM = importlib.import_module("rlm").RLM
    except ImportError as exc:
        raise SystemExit(
            "Could not import upstream `rlm`. Run this with the environment where "
            "`rlms` is installed, e.g. `conda run -n py312 python benchmarks/comparison/run.py`."
        ) from exc

    backend = infer_backend(args.model, args.backend)
    counters = {"subcall_starts": 0, "subcall_completes": 0}
    logger = make_upstream_logger()

    def on_subcall_start(_depth: int, _model: str, _prompt_preview: str) -> None:
        counters["subcall_starts"] += 1

    def on_subcall_complete(
        _depth: int,
        _model: str,
        _elapsed: float,
        _error: str | None,
    ) -> None:
        counters["subcall_completes"] += 1

    rlm = RLM(
        backend=backend,
        backend_kwargs={"model_name": args.model},
        environment="local",
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        logger=logger,
        on_subcall_start=on_subcall_start,
        on_subcall_complete=on_subcall_complete,
    )

    started = time.time()
    error: str | None = None
    prediction = ""
    usage: dict[str, Any] = {}
    completion: Any | None = None
    transcript_path: Path | None = None
    try:
        completion = rlm.completion(task.context, root_prompt=task.query)
        prediction = completion.response or ""
        usage = usage_to_dict(getattr(completion, "usage_summary", None))
        (run_dir / "upstream_raw").mkdir(parents=True, exist_ok=True)
        (run_dir / "upstream_raw" / f"{task.task_id}.json").write_text(
            json.dumps(
                {
                    "response": prediction,
                    "execution_time": getattr(completion, "execution_time", None),
                    "usage_summary": usage,
                },
                indent=2,
                default=str,
            )
        )
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()

    elapsed = time.time() - started
    method_counts = upstream_method_counts(completion, logger)
    transcript_path = write_upstream_transcript(
        run_dir,
        task,
        args,
        completion=completion,
        logger=logger,
        elapsed_s=elapsed,
        error=error,
        method_counts=method_counts,
    )
    input_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("total_input_tokens")
        or 0
    )
    output_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("total_output_tokens")
        or 0
    )
    return {
        "framework": "alexzhang13/rlm",
        "task_id": task.task_id,
        "prediction": prediction,
        "elapsed_s": round(elapsed, 3),
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "agents": 1 + counters["subcall_starts"],
        "nodes": None,
        "llm_turns": None,
        "max_depth": None,
        "max_branching": counters["subcall_starts"],
        "subcall_starts": counters["subcall_starts"],
        "subcall_completes": counters["subcall_completes"],
        "raw_usage": usage,
        "transcript_path": str(transcript_path.relative_to(run_dir)),
        "method_counts": method_counts,
        **flatten_counts(method_counts),
        **score_prediction(prediction, task.expected),
    }


def selected_frameworks(framework: str) -> list[str]:
    if framework == "both":
        return ["rlmflow", "alexzhang13/rlm"]
    return [framework]


def framework_runner(framework: str):
    if framework == "rlmflow":
        return run_rlmflow_task
    if framework == "alexzhang13/rlm":
        return run_upstream_task
    raise ValueError(f"unknown framework: {framework}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def fingerprint_tasks(tasks: list[Task]) -> str:
    h = sha256()
    for task in tasks:
        h.update(task.task_id.encode())
        h.update(task.expected.encode())
        h.update(task.context.encode())
    return h.hexdigest()[:16]


def summarize(rows: list[dict[str, Any]], tasks: list[Task], args: argparse.Namespace) -> dict[str, Any]:
    by_framework: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_framework.setdefault(row["framework"], []).append(row)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def method_names(group: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for row in group:
            counts = row.get("method_counts") or {}
            if isinstance(counts, dict):
                for name in counts:
                    method = str(name)
                    if method not in names:
                        names.append(method)
        return names

    def method_totals(group: list[dict[str, Any]]) -> dict[str, int]:
        return {
            method: sum(int((row.get("method_counts") or {}).get(method) or 0) for row in group)
            for method in method_names(group)
        }

    def method_means(group: list[dict[str, Any]]) -> dict[str, float]:
        return {
            method: mean(
                [float((row.get("method_counts") or {}).get(method) or 0.0) for row in group]
            )
            for method in method_names(group)
        }

    return {
        "benchmark": "synthetic_needle_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "argv": sys.argv,
        "model": args.model,
        "backend": infer_backend(args.model, args.backend),
        "task_fingerprint": fingerprint_tasks(tasks),
        "task": {
            "n": len(tasks),
            "records": args.records,
            "filler_words": args.filler_words,
            "seed": args.seed,
        },
        "budget": {
            "max_depth": args.max_depth,
            "max_iterations": args.max_iterations,
            "max_output_length": args.max_output_length,
            "max_concurrency": args.max_concurrency,
        },
        "frameworks": {
            name: {
                "n": len(group),
                "errors": sum(bool(row.get("error")) for row in group),
                "exact": mean([1.0 if row.get("exact") else 0.0 for row in group]),
                "contains": mean([1.0 if row.get("contains") else 0.0 for row in group]),
                "elapsed_s_mean": mean([float(row.get("elapsed_s") or 0.0) for row in group]),
                "input_tokens_mean": mean([float(row.get("input_tokens") or 0.0) for row in group]),
                "output_tokens_mean": mean([float(row.get("output_tokens") or 0.0) for row in group]),
                "agents_mean": mean([float(row.get("agents") or 0.0) for row in group]),
                "method_counts_total": method_totals(group),
                "method_counts_mean": method_means(group),
            }
            for name, group in sorted(by_framework.items())
        },
    }


def render_summary_md(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RLM Comparison Summary",
        "",
        f"- Model: `{summary['model']}` via `{summary['backend']}`",
        f"- Tasks: {summary['task']['n']} synthetic needle tasks "
        f"({summary['task']['records']} records each, seed={summary['task']['seed']})",
        f"- Budget: max_depth={summary['budget']['max_depth']}, "
        f"max_iterations={summary['budget']['max_iterations']}",
        f"- Fingerprint: `{summary['task_fingerprint']}`",
        "",
        "## Framework Results",
    ]
    for name, stats in summary["frameworks"].items():
        lines.extend(
            [
                "",
                f"### {name}",
                f"- exact: {stats['exact']:.3f}",
                f"- contains: {stats['contains']:.3f}",
                f"- errors: {stats['errors']} / {stats['n']}",
                f"- mean latency: {stats['elapsed_s_mean']:.2f}s",
                f"- mean agents/subcalls: {stats['agents_mean']:.2f}",
                f"- mean tokens: input={stats['input_tokens_mean']:.0f}, "
                f"output={stats['output_tokens_mean']:.0f}",
            ]
        )
        method_totals = stats.get("method_counts_total") or {}
        if method_totals:
            lines.append(
                "- method calls total: "
                + ", ".join(f"{name}={count}" for name, count in method_totals.items())
            )

    misses = [row for row in rows if row.get("error") or not row.get("exact")]
    if misses:
        lines.extend(["", "## Misses / Errors"])
        for row in misses:
            lines.append(
                f"- `{row['framework']}` `{row['task_id']}` exact={row.get('exact')} "
                f"error={row.get('error') or ''} prediction={row.get('normalized_prediction')!r}"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework",
        choices=["both", "rlmflow", "alexzhang13/rlm"],
        default="both",
        help="Which implementation(s) to run.",
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--backend", default="auto", help="auto, openai, anthropic, etc.")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--records", type=int, default=120)
    parser.add_argument("--filler-words", type=int, default=70)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-output-length", type=int, default=8192)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/comparison/runs/<timestamp>",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable upstream rlm verbose output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir or (Path(__file__).resolve().parent / "runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = make_tasks(args)
    write_json(run_dir / "tasks.json", [asdict(task) for task in tasks])

    rows: list[dict[str, Any]] = []
    for task in tasks:
        for framework in selected_frameworks(args.framework):
            print(f">>> {framework} {task.task_id}", flush=True)
            runner = framework_runner(framework)
            row = {
                "task_id": task.task_id,
                "expected": task.expected,
                "marker": task.marker,
                "context_bytes": len(task.context.encode("utf-8")),
                "records": task.records,
                "needle_index": task.needle_index,
            }
            row.update(runner(task, args, run_dir))
            rows.append(row)
            write_jsonl(run_dir / "results.jsonl", rows)

    summary = summarize(rows, tasks, args)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_summary_md(summary, rows))
    print(f">>> wrote {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
