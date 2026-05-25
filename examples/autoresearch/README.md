# Autoresearch

A [Karpathy-style autoresearch](https://github.com/karpathy/autoresearch)
hill-climb on top of RLMFlow. **The agent is the researcher**: it
rewrites a single function (e.g. `solve()` in `solution.py`),
passes the new source to `run_experiment(source)`, and decides
what to try next. A separate `evaluate.py` — which the agent does
not see — imports that function and prints `score: <float>`. To
run trials in parallel, the agent uses RLMFlow's normal
`rlm_delegate` / `rlm_wait`.

```
turn 0:
 → run_baseline()                         # one-shot, idempotent
turn 1+:
 → fresh = pick_slugs(get_runs())
 → handles = [rlm_delegate(name=slug, query=hyp, context=ctx) for slug, hyp, ctx in fresh]
 → results = await rlm_wait(*handles)     # children call run_experiment
 → ...
 → done(<summary>)
```

There is **no trial dir** to manage. Every call to
`run_experiment(source, description=<slug>)` writes the source to
`history/<n>_<slug>.py` and runs `python evaluate.py
history/<n>_<slug>.py` from the workspace root.
`history/ledger.jsonl` records every run; `list_runs()` /
`get_runs()` read it back. The baseline never goes through
`run_experiment`: it gets its own one-shot `run_baseline()` tool
which evaluates the original `solution.py` once and records
`description='baseline'`.

Use `--max-submissions N` to put a hard cap on experiment submissions.
The cap counts `run_experiment(...)` calls only, not the baseline. Agents
can inspect `submission_status()` and `run_experiment(...)` raises
`SubmissionError("too many submissions...")` without writing a new ledger row
once the cap is exhausted.

Archived candidates are the research record. Prompt children to include
module, target-function, and helper docstrings that explain the strategy,
validity invariants, inputs/outputs, units, and non-obvious constants.
This makes `history/<n>_<slug>.py` useful when the parent resumes or when
a human reviews why one idea beat another.

## Layout

```
examples/autoresearch/
├── README.md
├── autoresearch.py          # driver — target-agnostic
└── circle_packing/          # example target: pack 26 circles in [0,1]^2
    ├── README.md
    ├── program.md           # agent-facing brief
    ├── solution.py          # the file the agent rewrites — just solve()
    └── evaluate.py          # the harness — imports + scores a candidate
```

## Quick start

```bash
pip install -e .
pip install numpy

export OPENAI_API_KEY=...

python examples/autoresearch/circle_packing/evaluate.py \
    examples/autoresearch/circle_packing/solution.py   # optional sanity

python examples/autoresearch/autoresearch.py \
    --target examples/autoresearch/circle_packing \
    --budget-s 120 --max-iterations 20 --max-submissions 64 --model gpt-5-mini
```

Output goes to `runs/autoresearch/`:

- `solution.py` — the original baseline (immutable; the agent edits
  a string in memory and passes it to `run_experiment`).
- `evaluate.py` — the harness (immutable; the agent doesn't see it).
- `history/<n>_<slug>.py` — every source string that was ever run.
- `history/ledger.jsonl` — `{n, ts, score, returncode, ...}` per run.
- `session/` — the agent's per-step transcript (root + each child).
- `viewer.html` — static replay.

## Adding your own target

A target is a directory containing exactly three files:

- `<solution>.py` — defines one top-level function (the unit of
  search; the agent rewrites this).
- `<evaluator>.py` — takes a path to a candidate `<solution>.py` on
  argv, imports it, runs the function, and prints `score: <float>`
  on the last line of stdout. Exit non-zero on crash. **The agent
  never sees this file.** Declare ``FN_NAME = "..."`` at module
  scope so the driver can preflight-check candidate sources for
  the right function name and signature.
- `program.md` — the agent's brief (task description + how to use
  `run_experiment` / `run_baseline` / `rlm_delegate`). Include a short
  docstring rubric for candidate sources: module docstring for the
  strategy, target-function docstring for the algorithm and invariants,
  helper docstrings for inputs/outputs/units, and comments for magic
  constants.

```bash
python examples/autoresearch/autoresearch.py \
    --target path/to/my_target \
    --solution train.py \
    --evaluator score.py \
    --lower-is-better        # use if smaller is better (e.g. val loss)
```

Defaults: `--solution solution.py`, `--evaluator evaluate.py`,
higher score is better.
