#!/usr/bin/env python3
"""Aggregate OOLONG run directories into a comparison table.

Port of Prime Intellect's
[`aggregate_results.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/sebastian/experiment/rlm/environments/oolong/aggregate_results.py)
adapted to read the rlmkit runner's output layout. Treats every
``<dir>/{metadata.json, results.jsonl}`` pair as one configuration and
groups summary statistics by ``(model, mode, subset)``.

Usage::

    python benchmarks/oolong/aggregate.py
    python benchmarks/oolong/aggregate.py --outputs-dir benchmarks/oolong/outputs
    python benchmarks/oolong/aggregate.py --output runs_summary.csv

Pandas is optional — without it we still print the ranking table; the
CSV outputs only get written when pandas is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def load_all_results(outputs_dir: Path) -> list[dict]:
    """Walk *outputs_dir*, attaching each row to its run's metadata."""
    results: list[dict] = []
    metadata_files = sorted(
        set(outputs_dir.rglob("metadata.json"))
        | set(outputs_dir.rglob("config/manifest.json"))
    )
    if not metadata_files:
        print(f"!!! no metadata.json found under {outputs_dir}")
        return results

    seen_run_dirs: set[Path] = set()
    for meta_path in metadata_files:
        run_dir = meta_path.parent.parent if meta_path.parent.name == "config" else meta_path.parent
        if run_dir in seen_run_dirs:
            continue
        seen_run_dirs.add(run_dir)
        results_path = run_dir / "results.jsonl"
        if not results_path.exists():
            results_path = run_dir / "results" / "results.jsonl"
        if not results_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception as exc:
            print(f"!!! could not read {meta_path}: {exc}")
            continue
        env_args = metadata.get("env_args", {})
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            row["_model"] = metadata.get("model", "unknown")
            row["_mode"] = env_args.get("mode") or _mode_from_flags(env_args)
            row["_subset"] = env_args.get("subset", "synth")
            row["_use_rlm"] = env_args.get("use_rlm", False)
            row["_include_env_tips"] = env_args.get("include_env_tips", False)
            row["_run_dir"] = str(run_dir.relative_to(outputs_dir))
            results.append(row)
    return results


def _mode_from_flags(env_args: dict) -> str:
    if not env_args.get("use_rlm"):
        return "standard"
    return "rlm_tips" if env_args.get("include_env_tips") else "rlm"


def normalize_model(model: str) -> str:
    if model.startswith("openrouter/") or model.startswith("openrouter:"):
        parts = model.replace("openrouter:", "openrouter/").split("/")
        return parts[-1] if len(parts) >= 2 else model
    if "/" in model:
        return model.split("/")[-1]
    return model


def _stats(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return 0.0, 0.0, 0
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0), len(values)


def group_by_config(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        key = (r.get("_model", "?"), r.get("_mode", "?"), r.get("_subset", "?"))
        groups.setdefault(key, []).append(r)
    return groups


def print_table(groups: dict[tuple[str, str, str], list[dict]]) -> None:
    mode_order = {"standard": 0, "rlm": 1, "rlm_tips": 2}
    subset_order = {"synth": 0, "synth_with_labels": 1, "real": 2}

    sorted_keys = sorted(
        groups,
        key=lambda k: (k[0], mode_order.get(k[1], 99), subset_order.get(k[2], 99)),
    )

    print("\n" + "=" * 120)
    print("OOLONG RESULTS SUMMARY")
    print("=" * 120)
    print(
        f"{'Model':<24} {'Mode':<10} {'Subset':<20} "
        f"{'Score (paper)':>16} {'EM':>10} {'Contains':>10} {'N':>6}"
    )
    print("-" * 120)

    for key in sorted_keys:
        model, mode, subset = key
        rows = groups[key]
        score_m, score_s, n = _stats([float(r.get("score", 0.0)) for r in rows])
        em_m, em_s, _ = _stats([float(r.get("exact_match_reward", 0.0)) for r in rows])
        ca_m, ca_s, _ = _stats([float(r.get("contains_answer_reward", 0.0)) for r in rows])

        print(
            f"{normalize_model(model):<24} {mode:<10} {subset:<20} "
            f"{score_m:.3f}±{score_s:.3f}".ljust(70)
            + f" {em_m:.3f}".rjust(8)
            + f" {ca_m:.3f}".rjust(11)
            + f" {n:>6}"
        )
    print("-" * 120)

    print("\n" + "=" * 120)
    print("TOKEN USAGE (mean values)")
    print("=" * 120)
    print(
        f"{'Model':<24} {'Mode':<10} {'Subset':<20} "
        f"{'Turns':>8} {'Prompt Tok':>12} {'Compl Tok':>12} {'Total':>12}"
    )
    print("-" * 120)
    for key in sorted_keys:
        model, mode, subset = key
        rows = groups[key]
        turns_m, *_ = _stats([float(r.get("turns", 0)) for r in rows])
        in_m, *_ = _stats([float(r.get("input_tokens", 0)) for r in rows])
        out_m, *_ = _stats([float(r.get("output_tokens", 0)) for r in rows])
        tot_m, *_ = _stats(
            [float(r.get("input_tokens", 0) + r.get("output_tokens", 0)) for r in rows]
        )
        print(
            f"{normalize_model(model):<24} {mode:<10} {subset:<20} "
            f"{turns_m:>8.1f} {in_m:>12.0f} {out_m:>12.0f} {tot_m:>12.0f}"
        )
    print("-" * 120)


def maybe_save_csv(rows: list[dict], summary_path: Path, raw_path: Path) -> None:
    """If pandas is available, write summary + raw CSVs. Otherwise skip silently."""
    try:
        import pandas as pd
    except ImportError:
        print("!!! pandas not installed; skipping CSV output. `pip install pandas` to enable.")
        return

    if not rows:
        return

    raw = pd.DataFrame(
        [
            {
                "model": r.get("_model"),
                "mode": r.get("_mode"),
                "subset": r.get("_subset"),
                "task_id": r.get("task_id"),
                "score": r.get("score"),
                "exact_match_reward": r.get("exact_match_reward"),
                "contains_answer_reward": r.get("contains_answer_reward"),
                "input_tokens": r.get("input_tokens"),
                "output_tokens": r.get("output_tokens"),
                "elapsed_s": r.get("elapsed_s"),
                "turns": r.get("turns"),
                "incomplete": r.get("incomplete"),
                "answer_type": r.get("answer_type"),
                "task_group": r.get("task_group"),
                "dataset": r.get("dataset"),
                "context_len": r.get("context_len"),
                "context_chars": r.get("context_chars"),
                "run_dir": r.get("_run_dir"),
            }
            for r in rows
        ]
    )

    metric_cols = [
        "score",
        "exact_match_reward",
        "contains_answer_reward",
        "input_tokens",
        "output_tokens",
        "elapsed_s",
        "turns",
    ]
    summary = (
        raw.groupby(["model", "mode", "subset"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["_".join(c).strip() for c in summary.columns.values]
    summary = summary.reset_index()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nRaw rows -> {raw_path}")
    print(f"Summary  -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate OOLONG run directories.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Root directory containing OOLONG runs (looks for metadata.json recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path for the per-config summary. Defaults to <outputs-dir>/aggregate.csv.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="CSV path for the raw per-row dump. Defaults to <outputs-dir>/raw_results.csv.",
    )
    args = parser.parse_args()

    rows = load_all_results(args.outputs_dir)
    if not rows:
        return

    print(f"Loaded {len(rows)} rollouts from {args.outputs_dir}")
    groups = group_by_config(rows)
    print_table(groups)

    summary_path = args.output or args.outputs_dir / "aggregate.csv"
    raw_path = args.raw_output or args.outputs_dir / "raw_results.csv"
    maybe_save_csv(rows, summary_path, raw_path)


if __name__ == "__main__":
    main()
