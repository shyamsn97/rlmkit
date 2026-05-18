# Autoresearch Run Failure Analysis

Internal notes from inspecting `examples/autoresearch/runs/autoresearch`
after the recursive circle-packing runs.

## TL;DR

The current failures are not random. They come from a small set of
prompt/setup problems:

1. Children over-generalize the task and build generic optimization
   frameworks instead of the one concrete circle-packing contract.
2. Children guess or infer `N` even though the task says `26`.
3. Children write complex retry harnesses that do not inspect the
   correct failing row.
4. Children retry low-value or doomed attempts because the prompt says
   "up to 3 fixes" too broadly.
5. Children optimize proxy objectives that are not the actual score.
6. Some fixes are generated from generic coding-agent priors
   (`__all__`, `if __name__ == "__main__"`, CLI behavior, `program`
   imports), which are actively wrong for this target.

The good news: valid improvements happened. The run found
`ilp_relax_round` with `score=2.240226` and `genetic_ea` with
`score=2.116232`, both above the baseline `1.525315`. So recursion
and the evaluator are working. The prompt needs to make children
less generic, less clever, and more directly bound to the evaluator.

## Ledger Summary

Baseline:

- `baseline`: `1.525315`

Successful useful trials:

- `ilp_relax_round`: `2.240226`
- `genetic_ea`: `2.116232`

Successful but useless trials:

- `simulated_annealing`, `_fix1`, `_fix2`, `_fix3`: all `0.0234`
- `l_bfgs_polish_fix1`: `0.0`

Crashes:

- `random_restart_local*`: repeated `ValueError` unpacking `best_move`
- `greedy_constructive`: first an `IndexError`, later fixed attempts
  returned 30 centers instead of 26
- `l_bfgs_polish`: `NameError: re is not defined`
- `beam_search_k5*`: returned 20 or 12 centers instead of 26
- `tabu_search*`: generated centers outside `[0, 1]^2`

## Failure Categories

### 1. Children over-generalize instead of solving circle packing

Examples:

- `random_restart_local` reads `program.md`, then writes comments like:
  "We don't know the exact spec yet; infer typical structure".
- `simulated_annealing` builds adapters for `centers_radii`,
  `permutation`, `bitstring`, and `generic_continuous`.
- `greedy_constructive` implements both circle-packing and TSP
  builders.
- `tabu_search` tries to infer entrypoints, return contracts, and
  objective names from baseline text.

This is wasted context and a source of bugs. The target is not a
generic optimization harness. It is exactly:

```python
def solve():
    import numpy as np
    return centers, radii  # shapes (26, 2), (26,)
```

Prompt fix:

- In child instructions, explicitly forbid generic adapters.
- Say: "Do not infer the problem type. It is circle packing. Hard-code
  `N = 26`."
- Say: "Do not import `program`, inspect env vars, parse baseline text,
  support TSP/permutation/bitstring/generic continuous cases, or write
  CLI/main fallbacks."

Suggested wording:

```text
This is NOT a generic optimizer benchmark. Do not write adapters for
other task types. Do not infer N. Hard-code N = 26. Your source should
only solve this circle-packing contract.
```

### 2. Wrong `N`

Failures:

- `greedy_constructive` defaulted to `N = 30`.
- `beam_search_k5` returned `20`, then `12`, then `20` centers.
- `tabu_search` defaulted to `n = 20`.

Root cause:

The prompt lets children "infer" or "detect" too much. The model
therefore writes fallback heuristics and defaults (`20`, `25`, `30`)
instead of obeying `26`.

Prompt fix:

- Repeat `N = 26` in the child prompt, not just `program.md`.
- In examples, use `N = 26` as a literal assignment.
- Add a self-check before `run_experiment`.

Suggested child checklist:

```python
assert "def solve():" in source
assert "N = 26" in source or "range(26)" in source
```

Better: ask the child to run an in-REPL static sanity check:

```python
import ast
ast.parse(source)
assert "def solve():" in source
```

We should avoid making the check too semantic, but the prompt can
strongly suggest it.

### 3. Retry logic often uses the wrong ledger row

Several child sessions use:

```python
runs = list_runs()
last = runs[0]
```

But `list_runs()` is sorted best-first, not latest-first. `runs[0]`
is often the best scored row (`ilp_relax_round`), not the trial that
just crashed. This caused `tabu_search` to report the `ilp_relax_round`
result as its own final answer.

Prompt fix:

- Never tell children to use `list_runs()[0]` or `list_runs()[-1]` for
  latest crash.
- Tell children: if `run_experiment` raises, use `e.row` directly.
- Do not catch broad `Exception` unless binding it as `e`.

Preferred retry pattern:

```python
try:
    r = run_experiment(source, description=slug, budget_s=BUDGET)
except ExperimentCrashed as e:
    row = e.row
    print(row["stderr_tail"][-1000:])
    # fix source based on row, then retry
```

If we do not want children catching exceptions, then we should not ask
them to retry in the same REPL block. Let the exception land, and on
the resumed turn they can inspect their own crash. But for that to
work reliably, child iteration cap must be high enough for one retry
turn.

### 4. Broad "up to 3 fixes" causes bad retries

Examples:

- `random_restart_local` retried three times but did not fix the actual
  unpacking bug:

```python
best_move = ('center', i, dx, dy) if move_type < 0.7 else ('radius', i, delta)
kind, i, a, b = best_move
```

The radius move has 3 tuple items, but the unpack expects 4.

- `simulated_annealing` produced a valid score `0.0234`, then still ran
  `_fix1`, `_fix2`, `_fix3`. A valid low score is not a crash and
  should not trigger fix attempts.

Prompt fix:

- "Fixes are only for `returncode != 0`; never `_fixN` a valid scored
  trial."
- "A `_fixN` must address the exact `stderr_tail`; do not apply generic
  compatibility fixes."
- "After one repeated identical error, stop and report failure."

Suggested wording:

```text
Only retry when `run_experiment` raises `ExperimentCrashed`. A numeric
score, even a terrible one, is a completed trial: call `done(...)` and
let the parent decide.

Each fix must name the exact error it addresses. Do not add generic
fallbacks (`__all__`, `if __name__ == "__main__"`, alternate task
adapters, CLI code) unless the traceback explicitly asks for it.
```

### 5. Proxy objectives diverge from actual evaluator

Children frequently optimize proxies:

- area (`sum(r*r)`) instead of sum of radii
- overlap-penalized objectives that allow centers outside the unit
  square
- "min clearance" rather than final sum of radii

This produced valid but tiny scores (`0.0234`) and invalid geometry
(`tabu_search` centers outside the square).

Prompt fix:

- State the exact objective in child query: **maximize `sum(radii)`,
  not area, not minimum clearance, not a soft penalty score**.
- Require a final projection/repair before returning:

```python
centers = np.clip(centers, radii[:, None], 1.0 - radii[:, None])
radii = np.maximum(radii, 0.0)
```

Projection alone is not enough for overlap, but it prevents the
obvious outside-square failures.

Also ask children to include a simple local validation helper inside
`solve()` or before return:

```python
assert centers.shape == (26, 2)
assert radii.shape == (26,)
```

Assertions inside `solve()` become evaluator crashes with useful
messages; this is better than returning nonsense.

### 6. The prompt invites generic "robustness" that is harmful here

Current parent-generated child context said things like:

- "implement one complete solution string accordingly"
- "quick fixes as `_fix1`..."
- "read program.md if helpful"

The model interpreted this as permission to make a general-purpose
adapter layer. It wrote:

- `try: from program import N`
- environment-variable `N`
- TSP/permutation fallback
- bitstring fallback
- `if __name__ == "__main__"`
- `__all__`

Prompt fix:

Use narrower child query templates. For example:

```text
slug='greedy_constructive'. Circle packing only. Write a complete
source string defining exactly `def solve():` for N=26. Do not support
other task types. Do not import `program`. Do not use env vars. Do not
write `__main__` or CLI code. Maximize sum(radii). Run exactly one
experiment. If it crashes, fix only the exact stderr_tail.
```

## Proposed Prompt Changes

### System prompt: child section

Replace broad child guidance with this:

```text
Child trial agents are not general coding agents. They are one-trial
research workers for the exact target in program.md.

For circle packing:
- hard-code N = 26
- define exactly `def solve():`
- return `(centers, radii)` with shapes `(26, 2)` and `(26,)`
- maximize `sum(radii)`
- use only numpy + stdlib
- do not import `program`
- do not read env vars
- do not add CLI/main/stdout behavior
- do not support other problem families

Build one complete source string and call:
`run_experiment(source, description=slug, budget_s=BUDGET)`.

If it returns a numeric score, stop and `done(...)` immediately.
If it raises `ExperimentCrashed as e`, use `e.row["stderr_tail"]` to
make at most one targeted fix in the next turn. Do not perform generic
compatibility fixes.
```

### Parent child-query template

The parent should send children a terse template:

```python
q = f"""
slug={slug!r}. Circle packing only.
Idea: {hyp}
Hard contract:
- exactly def solve():
- N = 26
- return centers, radii with shapes (26,2), (26,)
- centers/radii must be valid; maximize sum(radii)
- no program imports, env vars, CLI/main/stdout, or alternate task adapters
Run one experiment with run_experiment(source, description=slug,
budget_s=BUDGET). If it scores, done. If it crashes, use only the
specific stderr_tail to fix.
"""
```

### Program.md

Move "Strategy hints" lower and put a "Do not generalize" section near
the top:

```text
This target is not generic. Do not infer the problem from text. The
problem is exactly circle packing with N=26. Do not write support for
TSP, bitstrings, generic vectors, CLI scripts, stdin/stdout, env vars,
or program imports.
```

### Retry examples

The examples should show `except ExperimentCrashed as e`, not
`except Exception`, and should never use `list_runs()[0]` as latest.

Good:

```python
try:
    r = run_experiment(source, description=slug, budget_s=BUDGET)
    done(json.dumps({"slug": slug, "score": r["score"]}))
except ExperimentCrashed as e:
    print(e.row["stderr_tail"][-1000:])
    # next turn: patch exact issue
```

Bad:

```python
except Exception:
    last = list_runs()[0]  # wrong: best-first, not latest
```

## Possible Tooling Changes (Not Just Prompt)

Prompt fixes help, but two small tool changes would prevent entire
classes of failure:

1. Add `latest_run()` tool that returns max-`n` row. Then children stop
   misusing `list_runs()[0]`.
2. In `run_experiment`, include a clear `ok` boolean:

```python
{"ok": rc == 0 and score is not None, ...}
```

3. Add optional `validate_source(source)` or make `run_experiment`
   preflight more domain-aware for circle packing:

- top-level `def solve():`
- no `def solve(...)` args
- flag obvious `N = 20`, `N = 30`
- flag `from program import N`
- flag `if __name__ == "__main__"`
- flag `sys.stdout.write`

These should probably be warnings surfaced in `stderr_tail`, not hard
global rules, unless this remains a circle-packing-only example.

## Recommended Next Patch

1. Add a `latest_run()` tool.
2. Tighten `AUTORESEARCH_RECURSION_TEXT` child section:
   - no generic adapters
   - no `program` import/env var/CLI
   - numeric score means done, no `_fixN`
   - use `ExperimentCrashed as e`, not broad `Exception`
3. Move "Do not generalize" near the top of
   `examples/autoresearch/circle_packing/program.md`.
4. Reduce or remove prompt examples that contain broad fallback logic.

The main theme: children should be researchers for one exact target,
not generic solver authors.
