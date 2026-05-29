# Circle Packing Autoresearch (n=26)

## Goal

Pack **exactly 26** non-overlapping circles inside the unit square
`[0, 1]^2`. Maximize the **sum of radii**. Higher is better.

Baseline score is about `1.525`. Known optimum is about `2.635`.
Allowed dependencies: `numpy` and the Python standard library only.

## Do Not Generalize

This target is **not** a generic optimization harness. Do not infer
the problem type. Do not support TSP, permutations, bitstrings,
generic vectors, stdin/stdout protocols, CLI scripts, environment
variables, or alternate evaluator APIs.

The task is exactly circle packing with `N = 26`.

Hard-code `N = 26` inside `solve()`. Do not import `program`, read
`N` from env vars, parse baseline text to infer `N`, or default to
`20`, `25`, or `30`.

## Candidate Contract

Every trial is a complete `solution.py` source string defining exactly:

```python
def solve():
    import numpy as np
    N = 26
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    # your circle-packing strategy here
    return centers, radii
```

Hard requirements:

- `solve()` takes **no arguments**.
- `solve()` **returns** `(centers, radii)`. The evaluator ignores
  stdout. Do not solve by printing.
- `centers` has shape `(26, 2)` with values in `[0, 1]`.
- `radii` has shape `(26,)`, all values `>= 0`.
- Circles must be inside the unit square and mutually non-overlapping.
- Imports live inside `solve()` or inside helpers called by `solve()`.
- Use only `numpy` + stdlib. No `scipy`, `cvxpy`, `shapely`, etc.
- Do not add `if __name__ == "__main__"`, `sys.stdout.write`,
  `__all__`, CLI handling, or generic fallback task adapters.

## Tools

- `run_baseline()` runs the original `solution.py` once and records
  `description='baseline'`. The parent should call this first.
- `run_experiment(source, description, budget_s=...)` archives the
  complete source string as `history/<n>_<description>.py`, runs it,
  appends a row to `history/ledger.jsonl`, and returns that row.
  It raises `ExperimentCrashed` on syntax errors, import errors,
  invalid geometry, exceptions in `solve()`, or timeout. The failure
  row is still on the ledger.
- `get_runs()` returns the ledger keyed by description.
- `list_runs()` returns compact rows sorted best-first by score.
- `get_run(n)` returns a full row by trial number.
- `latest_run()` returns the most recent full ledger row by trial
  number. Use this after broad exception catches. Do **not** use
  `list_runs()[0]` as latest; it is best-first, not chronological.
- `read_file(path)` reads baseline or archived sources.
- `done(summary)` finishes the current agent.

## Recursive Workflow

### Parent Agent

The parent runs **multiple rounds** of research, hill-climbing on the
ledger until the submission budget is spent. It should:

1. Read this file.
2. Run `run_baseline()` once.
3. Call `submission_status()` and read `get_runs()`; identify the
   current best scored trial.
4. **Round loop — keep going while `remaining_submissions > 0`.** Each
   round:
   a. Choose several independent hypotheses spanning **diverse
      algorithmic families** — keep exploring, do not collapse onto one
      winner. A good round is roughly half **refinements of the top 2–3
      _distinct_ performers** (tighten tolerances, more anneal steps,
      better seeding) and half **brand-new, qualitatively different
      approaches** (a different optimizer, seeding scheme, or geometry).
      Hard limit: **at most ~2 variants of any single idea family per
      round** — never fill a batch with a dozen micro-tweaks of the same
      idea. If you have already spent a round refining the current best,
      prefer an untried family over yet another tweak of it.
   b. Give every child a **unique** slug. Children write
      `history/<n>_<slug>.py`, so a repeated slug just overwrites and
      wastes a submission. Derive fresh names from the best trial, e.g.
      `blue_noise_anneal_v2`, `greedy_maximin_tighten`, `sa_anneal_hot`,
      or append a round suffix (`_r2`, `_r3`). **Never reuse a slug that
      already appears in `get_runs()`.**
   c. Size the batch to `min(<branches_per_turn>, remaining_submissions)`.
   d. Spawn one child per hypothesis in a single
      `results = await launch_subagents([{"name": slug, "query": query, "context": context}, ...])`.
   e. On resume, re-read the ledger and `submission_status()`. If
      `remaining_submissions > 0` and the best score is still improving
      (or you have untried promising directions), start another round.
5. Only call `done(...)` when `remaining_submissions == 0`, or after a
   round produced **no improvement over the previous best** AND you have
   already tried several distinct families. If the best plateaus while
   budget remains, switch to an **untried family** rather than finishing
   or tweaking the same idea again. Do **not** finish with budget left
   just because your first idea list is exhausted — invent new slugs.

The parent does **not** call `run_experiment` for child ideas. It
delegates them.

Child queries should be narrow. Example:

```text
slug='hex_lattice'. Circle packing only.
Idea: try a hexagonal lattice seed plus local polish.
Hard contract:
- exactly def solve():
- N = 26
- return centers, radii with shapes (26,2), (26,)
- centers/radii must be valid; maximize sum(radii)
- no program imports, env vars, CLI/main/stdout, or alternate task adapters
Run one experiment with run_experiment(source, description=slug,
budget_s=BUDGET). If it scores, done. If it crashes, use only the
specific stderr_tail to fix.
```

### Child Trial Agent

Each child gets a slug and one hypothesis. It should:

1. Read `CONTEXT` and this file.
2. Read `solution.py` and, if useful, the best archived source from
   the ledger.
3. Build one complete source string for `solution.py`.
4. Call `run_experiment(source, description=slug)`.
5. If it returns a numeric score, even a terrible one, stop and
   `done(...)`. Do **not** run `_fixN` trials after a valid score.
6. If it raises `ExperimentCrashed`, inspect `e.row["stderr_tail"]`.
   If you used broad `except Exception`, call `latest_run()` and use
   that row. Do not use `list_runs()[0]` for latest.
7. If the row has `returncode == -1` or says it timed out, do **not**
   retry the same slow idea; report timeout and stop.
8. For quick bugs (syntax, wrong shape, missing import, overlap), make
   at most one targeted fix per turn under `slug + "_fix1"`, then
   `_fix2`, then `_fix3`. Each fix must address the exact error.
   Do not add generic compatibility fixes.
9. `done(...)` with JSON containing at least the slug, score,
   returncode, and archived source path.

Children must write **thorough docstrings** in every candidate source.
Future readers (and the parent on resume) only see the archived `.py`
file, so the code must explain itself. Documentation quality is part of
the trial: if two ideas score similarly, prefer the one whose archived
source makes the strategy easiest to audit. Required:

- A module-level docstring naming the strategy, the high-level idea,
  any reference (e.g. "Packomania n=26 seed"), and the expected score
  band.
- A docstring on `solve()` describing the algorithm step by step:
  initialization, the main loop, the termination condition, and how
  `centers` and `radii` end up valid.
- A docstring on every helper function explaining inputs, outputs,
  units (e.g. "radius in unit-square coordinates"), and any invariant
  it preserves (non-overlap, in-bounds, monotone improvement, etc.).
- Inline comments for any non-obvious constant (ring counts, learning
  rates, annealing schedules, magic offsets). A bare number with no
  comment is a bug.

If the function name does not make the behavior obvious from one
glance, the docstring is too short. Prefer a few extra lines of prose
over clever one-liners. On `_fixN` retries, keep docstrings accurate:
do not leave a stale description from the previous algorithm.

Children must **not** write `solution.py` directly. Children run in
parallel and share one workspace; direct writes would clobber other
children. Always pass your candidate as a source string to
`run_experiment(source, description=slug)`.

## Trial Source Skeleton

```python
source = '''
"""Hexagonal lattice seed + radius inflation for n=26 circle packing.

Strategy
--------
1. Place 26 centers on a hexagonal lattice clipped to the unit square.
2. For each center, set its radius to the maximum value that keeps it
   inside [0, 1]^2 and non-overlapping with every previously sized
   circle (greedy feasible radius).
3. Return the resulting (centers, radii).

Expected score band: ~1.8-2.1. Intended as a clean baseline to beat
with local polish in a follow-up trial.
"""

def _hex_lattice(n, rows, cols):
    """Return the first `n` points of a `rows` x `cols` hex lattice,
    rescaled into the unit square with a small margin.

    Points are ordered row-major. Odd rows are offset by half a column
    width, which is the defining property of a hex lattice and gives
    tighter packing than a square grid.
    """
    ...

def _feasible_radius(p, placed_centers, placed_radii):
    """Largest radius for a circle at `p` that stays inside [0,1]^2
    and does not overlap any already-placed circle.

    Returns 0.0 if `p` is already inside another disk.
    """
    ...

def solve():
    """Build a valid (centers, radii) for n=26.

    Steps:
      - generate hex lattice candidates
      - assign each a greedy feasible radius (see `_feasible_radius`)
      - return arrays with shapes (26, 2) and (26,)

    Postconditions enforced by the asserts below: correct shapes,
    radii non-negative, centers in [0, 1]^2.
    """
    import numpy as np
    N = 26
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    # implement one complete circle-packing strategy
    assert centers.shape == (26, 2)
    assert radii.shape == (26,)
    return centers, radii
'''
r = run_experiment(source, description="short_idea_slug")
```

The skeleton above is the **minimum** docstring density. A real trial
should have more, not less.

## Good Retry Pattern

```python
try:
    r = run_experiment(source, description=slug, budget_s=BUDGET)
    done(json.dumps({
        "slug": slug,
        "score": r["score"],
        "returncode": r["returncode"],
        "path": r["solution_path"],
    }))
except ExperimentCrashed as e:
    row = e.row
    print(row["stderr_tail"][-1000:])
    # next turn: patch the exact issue shown above
```

If `ExperimentCrashed` is not in scope, use:

```python
except Exception as e:
    row = latest_run()
    print(row["stderr_tail"][-1000:])
```

## Bad Patterns

Do not do these:

```python
N = int(os.environ.get("N", "30"))       # wrong; N is 26
from program import N                    # wrong; no program module
def solve(seed=0): ...                   # wrong; solve takes no args
if __name__ == "__main__": ...           # wrong; evaluator imports solve()
sys.stdout.write(...)                    # wrong; evaluator reads return value
last = list_runs()[0]                    # wrong; best-first, not latest
```

Do not write alternate task support:

```python
if problem == "tsp": ...
elif problem == "bitstring": ...
else: generic_continuous_solver()
```

## Useful Strategy Families

- Hexagonal / triangular lattice seeds with radius scaling.
- Concentric rings, e.g. center + 8-ring + 16-ring + refinements.
- Greedy maximin placement: repeatedly place the next center in the
  largest empty-space clearance.
- Local search on centers with radii recomputed from feasibility.
- Simulated annealing over centers, with radii set to feasible minima.
- Force-directed relaxation: push close circles apart, clamp to the
  square, inflate feasible radii, repeat.
- Hard-code a known-good Packomania-style seed, then polish.

Score bands: above `2.0` is solid, above `2.4` is good, above
`2.55` is excellent, and `2.635` is near optimum.
