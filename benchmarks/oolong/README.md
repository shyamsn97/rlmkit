# OOLONG

Runnable rlmkit harness for the OOLONG long-context aggregation benchmark.

This is adapted from Prime Intellect's
[`oolong-rlm` environment](https://github.com/PrimeIntellect-ai/verifiers/tree/sebastian/experiment/rlm/environments/oolong),
but uses rlmkit directly instead of `verifiers`.

## What It Runs

Three modes, matching the Prime Intellect environment:

- `standard` — one LLM call with the full context in `<context>...</context>`;
  the answer is extracted from the last `\boxed{...}`.
- `rlm` — rlmkit recursive scaffold. The long context is written to a
  task-local file and the agent uses file tools plus delegation.
- `rlm_tips` — same as `rlm`, plus the Prime Intellect `<env_tips>` chunking
  strategy block.

For the normal `synth` subset, the context usually contains raw instances, not
gold labels. The RLM prompt tells agents to classify each instance when label
statistics are requested; only `synth_with_labels` should be treated as a
field-counting/debug run.

Three subsets:

| subset | HF dataset | config | context column |
| --- | --- | --- | --- |
| `synth` | `oolongbench/oolong-synth` | default | `context_window_text` |
| `synth_with_labels` | `oolongbench/oolong-synth` | default | `context_window_text_with_labels` |
| `real` | `oolongbench/oolong-real` | `dnd` | `context_window_text` |

## Scoring

We do **not** use an LLM judge. OOLONG gold answers are already present in the
HF rows as stringified Python lists, and `scoring.py` implements the paper's
deterministic method:

- parse `answer` with `ast.literal_eval`;
- extract the answer slot using the question's `"in the form '...'"` template;
- fall back to last substring by answer type;
- score by answer type (`NUMERIC` partial credit, exact match for labels/users/dates/comparisons, set-F1 for list answers).

The primary metric is `score`. `exact_match_reward`, `contains_answer_reward`,
and `judge_reward` are kept as compatibility columns for Prime Intellect-style
aggregation; `judge_reward` is just an alias for `score`.

## Final Answer Recovery

This is now baked into `RLM`: if a node hits `max_iterations` before calling
`done()`, the engine performs one private final-answer turn and records it in
the normal trace. The benchmark does not inspect or import that action prompt;
it just runs `while not state.finished: state = agent.step(state)` and scores
`state.result`.

`incomplete: true` means the agent still did not call `done()` even after the
engine's recovery turn, or the task errored.

## Usage

```bash
# RLM mode, synthetic split
python benchmarks/oolong/run.py --mode rlm --subset synth \
  --split validation --limit 20 --shuffle --seed 42 \
  --model gpt-4.1-mini

# Standard baseline
python benchmarks/oolong/run.py --mode standard --subset synth \
  --split validation --limit 20 --shuffle --seed 42 \
  --model gpt-4.1-mini

# Prime Intellect tips mode
python benchmarks/oolong/run.py --mode rlm_tips --subset real \
  --split validation --limit 20 --shuffle --seed 42 \
  --model claude-sonnet-4-20250514

# Full mode x subset sweep
N=20 MODELS_FULL="gpt-4.1-mini" bash benchmarks/oolong/run_ablations.sh

# Aggregate all local runs
python benchmarks/oolong/aggregate.py
```

Make targets:

```bash
make oolong-rlm OOLONG_MODEL=gpt-4.1-mini OOLONG_N=20
make oolong-rlm-tips
make oolong-standard
make oolong-real
make oolong-ablations
make oolong-aggregate
```

## Output Layout

Runs default to:

```text
benchmarks/oolong/outputs/evals/<timestamp>_<mode>_<subset>_<model>/
  metadata.json
  results.jsonl
  summary.json
  summary.md
  workspaces/
    task_0000/
      context/
      trace/
      task_0000.txt
    traces/
```

The output shape intentionally resembles Prime Intellect's eval layout:
`metadata.json` has an `env_args` block with `subset`, `use_rlm`,
`include_env_tips`, and `mode`; `results.jsonl` stores one row per task.

Important row fields:

- `score` — primary paper-style OOLONG score.
- `exact_match` — exact match after slot extraction.
- `prediction` / `extracted` / `gold` / `gold_templated` — answer audit fields.
- `input_tokens`, `output_tokens`, `total_tokens` — full tree usage.
- `tree` — RLM tree shape (`depth`, `nodes`, iterations, branching).
- `incomplete` — no `done()` after the final-answer turn or an error occurred.

## Parallelism

- `--workers N` runs N benchmark tasks concurrently.
- `--max-concurrency K` controls child-branch parallelism inside a single RLM
  task.

Keep these separate: `--workers` is API throughput; `--max-concurrency` changes
the shape of each recursive run.

## Files

- `run.py` — benchmark runner.
- `scoring.py` — standalone paper scorer.
- `aggregate.py` — result table / CSV aggregation.
- `run_ablations.sh` — mode x subset sweep.
- `viz.py` — optional Rich live board for active runs.

## References

- [OOLONG paper](https://arxiv.org/abs/2511.02817)
- [Official OOLONG repo](https://github.com/abertsch72/oolong)
- [Prime Intellect OOLONG environment](https://github.com/PrimeIntellect-ai/verifiers/tree/sebastian/experiment/rlm/environments/oolong)
- [`docs/internal/oolong_scoring.md`](../../docs/internal/oolong_scoring.md)
- [`docs/internal/oolong_harness.md`](../../docs/internal/oolong_harness.md)
# OOLONG / OOLONG-Pairs

Long-context reasoning and aggregation — see
[Bertsch et al., 2025 (arXiv:2511.02817)](https://arxiv.org/abs/2511.02817)
and the upstream repo [`abertsch72/oolong`](https://github.com/abertsch72/oolong).

Used as one of the four main benchmarks in the **Recursive Language
Models** paper ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)).
Frontier models (GPT-5 / Claude-Sonnet-4 / Gemini-2.5-Pro) score
**<50%** at 128K on both splits, and **<0.1% F1** on OOLONG-Pairs —
which makes it the clearest place to show that recursive delegation
actually helps.

## Datasets

Hosted on Hugging Face under the `oolongbench` org:

- [`oolongbench/oolong-synth`](https://huggingface.co/datasets/oolongbench/oolong-synth)
  — synthetic, ~131K-token contexts, ablatable. **Default split here.**
- [`oolongbench/oolong-real`](https://huggingface.co/datasets/oolongbench/oolong-real)
  — real conversational data.

Each row exposes (see `docs/internal/oolong_scoring.md` for the full
schema):

- `context_window_text` — the long input
- `question` — the query, including `"Give your final answer in the
  form '...'"` template
- `answer` — gold answer, **stringified Python list** (e.g. `"[7]"`,
  `"['positive']"`); we `ast.literal_eval` it
- `answer_type` — dispatch key: `ANSWER_TYPE.LABEL`,
  `ANSWER_TYPE.COMPARISON`, `ANSWER_TYPE.NUMERIC`, `ANSWER_TYPE.USER`,
  `ANSWER_TYPE.DATE`, `ANSWER_TYPE.MONTH_YEAR`
- `task_group` — `counting` / `user` / `timeline`
- `task` — fine-grained task id (e.g. `TASK_TYPE.MOST_FREQ`)
- `context_len` — target context length bucket (1K–4M)
- `dataset` — source ICL dataset (`metaphors`, `agnews`, `spam`, …)

Note: the RLM paper's `trec_coarse` setting is in the public
`oolong-synth` **validation** split, not the default `test` split. Use
`--hf-split validation --dataset trec_coarse --context-len 131072` for the
closest paper-style 128K run.

The runner keeps the Prime Intellect OOLONG trick without depending on
`verifiers`: the root prompt only sees the question and file metadata, the
long context is written to `task_XXXX.txt`, and benchmark-local env tips push
the agent toward chunking, parallel sub-calls, and Python aggregation. For a debug
upper bound, run with `--context-column context_window_text_with_labels` so the
agent can aggregate provided labels instead of doing semantic classification.

## Modes

The runner supports two modes so you can quantify how much the
recursive scaffold actually buys you:

- **`--mode rlm`** (default) — the rlmkit recursive scaffold. Each task
  gets a branch `Workspace`: the workspace root is the runtime working
  tree, while node history lives under `context/`. The context is written
  to `task_XXXX.txt` at the workspace root. The runtime (`LocalRuntime`
  or, with `--docker-image`, `DockerRuntime`) is materialized over the
  workspace itself. The agent gets only `ls`, `read_file`, and recursive
  delegation, so it has to read, chunk, classify/extract, and aggregate.

- **`--mode flat`** — baseline. A single LLM call with the entire
  passage inlined in the prompt. No tools, no recursion. Useful for A/B
  runs: same model, same tasks, same scoring, very different token and
  accuracy profiles.

A typical experiment is one run per mode, same model, same `--seed`,
different `--out` directories.

### How the RLM runner works

The RLM paper treats the long prompt as a **variable in a REPL**. This
runner follows that setup with a file-backed task input and a small tool
surface:

1. Create a per-task branch `Workspace`.
2. Write `context_window_text` to `task_XXXX.txt` in the workspace root.
3. Materialize the runtime over the `Workspace`; node history is stored
   in `Workspace.context`.
4. Give the agent a prompt telling it where the task file lives, how
   big it is (approx. tokens + bytes), and that it can `read_file`,
   chunk with Python, and delegate chunk work.
5. Run the agent with `max_depth` / `max_iterations` from the CLI.
6. Compare `state.result` to the gold `answer`.

This follows the same interface split Anthropic describes in
[Managed Agents](https://www.anthropic.com/engineering/managed-agents):
the harness/brain (`RLM`), durable session/context, and sandbox/hands
(`Runtime`) are separate pieces.

This is deliberately simple. For **paper-equivalent** numbers you need
the [official harness](https://github.com/abertsch72/oolong) and its
scorer; for **rlmkit scaffold** signal, this driver is enough.

## Metric

We implement the scoring methodology from the OOLONG paper §2.3 and
§3.2. See [`docs/internal/oolong_scoring.md`](../../docs/internal/oolong_scoring.md)
for the derivation and sources. The scorer lives in
[`scoring.py`](./scoring.py) as a standalone module with zero rlmkit
dependencies, so other harnesses (e.g. a future lighteval task) can
import `score_one`, `extract_answer`, `parse_gold` directly without
pulling in the agent runtime.

1. **Parse** `answer` with `ast.literal_eval` (it's a Python-repr of a
   list in the HF dataset).
2. **Extract** the slot from the prediction:
   - first try the template embedded in the question (`"Give your
     final answer in the form '…'"`);
   - fall back to the last substring matching the expected answer type
     (last number / comparison phrase / date / user ID).
3. **Dispatch** on `answer_type`:
   - `NUMERIC`: partial credit `score = 0.75 ** abs(gold − pred)`
   - `LABEL` / `COMPARISON` / `USER` / `DATE` / `MONTH_YEAR`:
     case-insensitive exact match (0 or 1)
   - `list` (real-only): set-overlap F1

Per-task row fields:

- `score` — primary metric in `[0, 1]`, following the paper
- `exact_match` — 1 iff extracted slot matches gold exactly (equals
  `score` for string types, strict equality for numerics)
- `extracted` — the slot our extractor pulled from the prediction (so
  you can eyeball what was actually scored)
- `answer_type` — dispatch key that was used

We aggregate as the mean `score` over non-errored tasks, and break
down by `answer_type`, `task_group`, and source `dataset` in
`summary.json` / `summary.md`.

The official harness at
[`abertsch72/oolong`](https://github.com/abertsch72/oolong) has not
yet released scoring code ("coming soon"). When it does, re-pin our
numbers against it on a small sample and resolve any drift.

## Usage

The local OOLONG runner has been removed from this repo. The remaining files in
this directory are scoring/reporting utilities and notes for rebuilding or
integrating a runner.

To run OOLONG again, restore or replace a harness that:

1. Loads `oolongbench/oolong-synth` or `oolongbench/oolong-real`.
2. Writes each task context to a task-local file or context object.
3. Runs either a flat baseline or an `RLM` scaffold.
4. Scores predictions with [`scoring.py`](./scoring.py).

Until that runner exists again, the `run-rlm-oolong-*` Makefile targets fail
with an explanatory message instead of pointing at a missing script.

## Parallelism and live viz

Two independent dials, kept apart on purpose:

- `--workers N` parallelizes **tasks**. Each task runs in its own
  thread so N tasks hit the LLM API concurrently. Network-bound; scale
  up until you saturate provider RPM/TPM. Default `1`.
- `--max-concurrency K` controls **per-task delegation** — how many
  child branches the rlmkit pool runs in parallel inside a single
  task. Default `1` (sequential children) so token usage and trace
  shape stay deterministic when you crank up `--workers`. Bump only
  if you actually want intra-agent fan-out.

The removed runner previously had a Rich live board for active workers and
recent completions. If a new runner is added, keep that optional: plain JSONL
output should remain enough for CI and batch runs.

## Output layout

```
results/oolong/<run>/
    manifest.json    # frozen config: argv, env, git SHA, rlmkit version,
                     #   dataset fingerprint, host, rlm_config
    results.jsonl    # one row per task
    summary.json     # aggregates (scores, token stats, latency,
                     #   tree shape) + embedded manifest
    summary.md       # human-readable report of the same
    traces/          # full RLM node trees per task (rlm mode only)
        task_0000/trace.json
        ...
```

### What each per-task row contains

```jsonc
{
  "task_id": "task_0000", "mode": "rlm",
  "dataset": "metaphors", "task_group": "counting",
  "task": "TASK_TYPE.MOST_FREQ", "context_len": 131072,
  "question": "...",
  "gold":           ["positive"],          // raw slot list, as scored
  "gold_templated": "Label: positive",     // gold rendered into the question's
                                           //   "in the form '...'" template,
                                           //   for eyeballing against prediction
  "prediction":     "Label: positive",
  "context_chars": 524288, "context_tokens_approx": 131072,
  "input_tokens": 182341, "output_tokens": 412, "total_tokens": 182753,
  "elapsed_s": 34.12,
  "tree": {              // null in flat mode
    "depth": 2, "nodes": 5,
    "total_iterations": 17, "root_iterations": 8,
    "max_branching": 3
  },
  "answer_type": "label", "extracted": "positive",
  "score": 1.0, "exact_match": 1.0,
  "incomplete": false,    // agent didn't finalise via done() under its budget
  "error": null
}
```

`gold` is the raw slot list from the dataset (this is what scoring
actually compares against). `gold_templated` is **purely cosmetic** —
it renders the slot back into the question's required output form so
you can read a row and see at a glance whether the model's prediction
matches the truth. Use `extracted` vs `gold[0]` to confirm what was
scored.

### Final-answer recovery

When the agent burns its `--max-iterations` budget without calling
`done()`, `rlmkit` performs the official RLM final-answer recovery inside the
engine. The runner does not know the prompt text and does not special-case this
path. The recovered answer becomes `state.result`, token usage is counted in
the tree totals, and the trace contains the normal engine-produced messages for
debugging.

### What `summary.json` rolls up

- **scores:** mean `score` (primary) and `exact_match` over non-errored
  tasks.
- **by_answer_type / by_task_group / by_dataset:** per-group `n`,
  `score`, `exact_match`.
- **tokens (input/output/total):** mean, median, p95, min, max, sum.
- **latency_s:** same stats, whole-run.
- **tree** (rlm only): mean/median/p95/max of depth, nodes,
  total_iterations, root_iterations, max_branching.
- **manifest:** the full frozen config, embedded so `summary.json` is
  self-contained.

Cost is intentionally **not** computed — prices change constantly and
go stale the moment you commit them. Every provider invoice ties back
to tokens, so use `summary.tokens.{input,output}.sum` plus whatever
price sheet is current the day you read it.

## OOLONG-Pairs

The **OOLONG paper itself does not define a Pairs split.** The name
originates in the Recursive Language Models paper
([arXiv:2512.04388](https://arxiv.org/abs/2512.04388)), which
constructs pairwise comparison questions over OOLONG. The community
[`zircote/oolong-pairs`](https://github.com/zircote/oolong-pairs)
tracks that variant. Until we wire up the RLM-paper construction code,
run the standard `synth` split and treat Pairs as a follow-up.
