# benchmarks/

Runnable benchmark harnesses for rlmkit. Each subdirectory is a
self-contained driver for one public benchmark, with its own README,
runner, and scoring script.

See [`docs/internal/benchmarks.md`](../docs/internal/benchmarks.md)
for the broader landscape and rationale for the picks in this directory.

## Conventions (shared across all benchmarks)

- **Runtime.** Default to `LocalRuntime` for dev; pass `--docker-image
  rlmkit:local` for any serious run. Never run third-party benchmark
  prompts under `LocalRuntime` on a trusted machine.
- **Budget.** Every task gets a fixed `max_depth × max_iterations` and
  optional `max_budget` (total tokens). These are declared in the CLI and
  written to the run manifest — do not tune per-task.
- **Manifest.** Every run writes a `manifest.json` with:
  `{model, fast_model, max_depth, max_iterations, max_budget, split, n,
   seed, dataset_sha, runtime, timestamp, rlmkit_version}`.
- **Results.** Per-task rows go to `results.jsonl`; aggregate metrics to
  `summary.json`; full traces to `traces/<task_id>/` via
  `rlmkit.utils.trace.save_trace`.
- **Seeds.** Any sampling from a dataset is deterministic given `--seed`
  so partial reruns are reproducible.

## Layout

```
benchmarks/
  README.md              # this file
  oolong/                # RLM paper: long-context aggregation
  ...                    # more suites added over time
```

## Running any benchmark

```
python benchmarks/<name>/run.py --help
```

Each driver shares the example-style flags (`--model`, `--fast-model`,
`--docker-image`, `--max-depth`, `--max-iterations`) plus its own
dataset flags (`--split`, `--n`, `--seed`, ...).

## Why these and not others

The explicit target set is in
[`docs/internal/benchmarks.md` §13](../docs/internal/benchmarks.md).
Short version: reproduce the RLM paper's quartet first (OOLONG /
OOLONG-Pairs / LongBench-v2 CodeQA / BrowseComp-Plus), then add one
coding and one reasoning anchor for industry credibility. Everything
else is skipped until that story is solid.
