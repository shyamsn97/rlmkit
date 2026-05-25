# Circle packing autoresearch (n=26)

A small, fast autoresearch target where the agent hill-climbs a
classic geometry problem: pack 26 non-overlapping circles in the
unit square, maximize the sum of radii.

The agent edits exactly one function — `solve()` in `solution.py`.
A separate `evaluate.py` (which the agent does not see or touch)
imports that function, validates the geometry, and prints
`score: <float>` to stdout (higher is better).

```
examples/autoresearch/circle_packing/
├── README.md
├── solution.py    # the file the agent rewrites — just solve()
├── evaluate.py    # the harness — imports a candidate solution, prints score
└── program.md     # the agent's operating manual
```

The seed scaffold (two concentric rings + greedy radius scaling)
scores ~1.5. The known best for n=26 is ~2.635 — so there's a lot
of room and many viable strategies, all hand-rolled in numpy
(gradient descent, projected gradient, simulated annealing,
physics relaxation, custom LP solvers, hard-coded near-optimal
seeds, …). **Only `numpy` + the stdlib are allowed** — no scipy,
no cvxpy, no third-party optimizers.

Adapted from [SakanaAI/ShinkaEvolve circle_packing](https://github.com/SakanaAI/ShinkaEvolve/blob/main/examples/circle_packing/initial.py).

## One-time setup

```bash
pip install numpy
```

Optional sanity run:

```bash
python examples/autoresearch/circle_packing/evaluate.py \
    examples/autoresearch/circle_packing/solution.py
# n=26  sum_radii=1.525315  min_r=0.010000  max_r=0.093455
# score: 1.525315
```

## Running the agent

```bash
export OPENAI_API_KEY=...
python examples/autoresearch/autoresearch.py \
    --target examples/autoresearch/circle_packing \
    --budget-s 120 \
    --max-iterations 20
```

No API costs (pure-CPU numpy). The agent edits a copy of
`solution.py` inside a disposable workspace under
`runs/autoresearch/`, and every version that ran is archived to
`history/<n>_<slug>.py` for rollback.

## What the agent can mutate

The entire body of `solution.py` — but the file must always end up
defining a single top-level function `solve()` that returns
`(centers, radii)` with shapes `(26, 2)` and `(26,)`. All imports
and helpers must live INSIDE that function (or in another top-level
helper it calls); no module-level state, no
`if __name__ == "__main__":` block.

Strategies the agent can explore:

- Change the initial layout (rings, hex, jittered grid).
- Replace the greedy radius solver with a hand-rolled LP / KKT /
  iterative scheme (no scipy).
- Run a local-search / physics-relaxation loop over centers.
- Hard-code a known-good seed and polish it.

Hard rule: only `numpy` and the Python stdlib. Importing scipy /
cvxpy / shapely / any other third-party package will fail at
runtime and the trial will be recorded as a crash.

## Score contract

`evaluate.py` prints exactly one line `score: <float>` on success
(exit 0). Anything else — `solve()` raised, wrong return shape,
overlap, out-of-square, import error — exits 1 with
`INVALID: <why>` on stderr (and a traceback when applicable). No
`score:` line is printed for invalid trials. The driver records
`score=None` and raises `ExperimentCrashed` to the agent, so every
ledger row with a numeric `score` corresponds to a *valid* packing.
