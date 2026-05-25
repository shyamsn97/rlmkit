# rlmflow vs alexzhang13/rlm

This folder contains a small head-to-head harness for comparing this repo's
`RLMFlow` engine with upstream [`alexzhang13/rlm`](https://github.com/alexzhang13/rlm/tree/main)
as installed from the `rlms` package.

The default task is a deterministic synthetic needle-in-haystack benchmark. Each
example builds a large `CONTEXT` made of many records. Exactly one record has the
requested marker, and the model must return only that record's `secret` value.
The task is intentionally simple to score but useful for checking whether each
framework can use its REPL/context machinery without stuffing the full prompt
into the root model message.

## Run

Use the Python environment that has `rlms` installed. From the repo root:

```bash
conda run -n py312 python benchmarks/comparison/run.py \
  --model gpt-5-nano \
  --n 3 \
  --records 120 \
  --max-depth 2 \
  --max-iterations 12
```

To run only one implementation:

```bash
conda run -n py312 python benchmarks/comparison/run.py --framework rlmflow
conda run -n py312 python benchmarks/comparison/run.py --framework alexzhang13/rlm
```

If the model is Anthropic, pass it normally; `--backend auto` treats model names
starting with `claude` as Anthropic and everything else as OpenAI-compatible:

```bash
conda run -n py312 python benchmarks/comparison/run.py \
  --model claude-sonnet-4-20250514 \
  --backend anthropic
```

## Outputs

Each run writes to `benchmarks/comparison/runs/<timestamp>/` by default:

- `tasks.json`: generated tasks, contexts, markers, and gold answers.
- `results.jsonl`: one row per framework per task.
- `summary.json`: aggregate exact-match, latency, token, agent/subcall, and method-call stats.
- `summary.md`: short human-readable summary.
- `workspaces/rlmflow/...`: per-task `rlmflow` workspaces and sessions.
- `transcripts/alexzhang13_rlm/...`: per-task upstream `rlm` trajectories, including nested RLM calls when exposed by `rlms`.
- `upstream_raw/...`: raw upstream `rlm` response and usage payloads.

## What This Compares

This is a framework smoke comparison, not a paper-quality benchmark. It keeps the
task, model, depth, and iteration budget aligned, then measures:

- exact match and contains-gold rate;
- latency;
- token usage when exposed by the implementation;
- number of agents/subcalls used;
- RLM method-call frequency (`rlmflow`: `rlm_delegate`, `llm_query_batched`;
  upstream `rlm`: `rlm_query`, `llm_query_batched`, `llm_query`).

The frameworks still differ in their prompt text, runtime implementation,
logging shape, and available inspection APIs. Treat misses and tree/subcall
behavior as debugging signals before treating aggregate numbers as claims.
