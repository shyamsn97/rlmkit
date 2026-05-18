# Autoresearch — what's wrong, ideas for redesign

Internal brainstorm. Not user-facing.

## The complaint

Running `examples/autoresearch/autoresearch.py --target circle_packing
--model gpt-5-mini --rounds 4 --branches-per-turn 4` reliably produces:

- 1 baseline row in the ledger.
- 3 children that crash on first attempt.
- Parent's generator returns `done("best=baseline")`.
- No retries, no recovery, no learning.

User sentiment: "this shouldnt be this hard to get working fucking
code." It's been multiple iterations of prompt and harness tweaks
and the baseline rate of "trial that produces a numeric score" is
still close to 0.

## Diagnosis: three independent failures, all colliding

### 1. The parent-agent-as-loop is a fiction

The prompt describes a multi-turn loop:

> Turn 0 — orient and run baseline … Turn k — fan out … Final
> turn — summarize from the ledger.

The actual control flow:

```
flow.start(query)                  # creates parent agent, prompts it
while not graph.finished:
    graph = flow.step(graph)       # advances one or more agents
```

`max_iterations` in `RLMConfig` caps how many engine steps a single
agent can take, not how many fan-out rounds the parent runs. The
parent generator is one-shot:

```python
# what the parent's generator actually does:
read_brief(); run_baseline(); pick_hypotheses()
handles = [delegate(...) for hyp in hyps]
results = yield wait(*handles)
done(summary)
```

That's one round, period. We were writing the system prompt as if
the parent looped, then the LLM (correctly) did one round and
called done(). The brief mentioning `iterations=20` and the prompt
mentioning "Turn k" both describe behavior the architecture
doesn't run.

### 2. Children give up after one crash

Children get a fresh context. They get the slug + one-line
hypothesis as their query, plus the system prompt (`AUTORESEARCH_*_TEXT`).
Prompt tells them: "retry up to 3 times on failure." Observed:

```python
# Child sees ExperimentCrashed in its REPL output.
# Next turn, model emits:
done(json.dumps({"ok": False, "attempts": 1, "error": "..."}))
```

gpt-5-mini consistently picks the easy exit. The retry prompt
section gets ~10% compliance in practice. We can't change this by
adding more text.

### 3. The function-shape contract is brittle

Common LLM mistakes that all crash the trial:

- `def solve_instance(inst: dict) -> dict:` — fabricates a
  different problem entirely. Returns `{"x": ..., "y": ..., "r": ..., "feasible": bool}`.
- `def solve(N=26):` — adds parameters.
- `return [centers, radii]` — list instead of tuple.
- `return {"centers": centers, "radii": radii}` — dict.
- `return centers, radii.tolist()` — radii as plain list (works,
  actually, after our coercion).
- Random JS / Markdown artifacts in the source (`{` after `def
  solve():`, leftover code fences).

The preflight catches some of these (wrong name, wrong sig, syntax),
fast and clean. But the catch-rate isn't the bottleneck — the
bottleneck is **the child can't recover** (failure 2) and **the
parent doesn't fan out again with the lesson learned** (failure 1).

### 4. circle_packing is a bad first example

Geometry validation has many independent failure modes (count,
range, overlap, return shape). Each is binary: pass or hard fail.
There's no "almost-correct" output that gets a partial score; you
either land a valid packing or you crash. That's the worst possible
shape for an LLM hill-climbing target — every gradient signal we
could give the model is destroyed by the validation gate.

## Ideas for redesign

### A. Single-agent loop. No parent/child.

```python
agent = LLM(query="""
    You are an autoresearch agent for {target}.

    {program_md}

    Tools: run_experiment(source, description), get_runs(),
    read_file, write_file. Loop until budget exhausted or you
    hit the known optimum.
""")
# Engine runs the same agent for many turns. Each turn = one
# experiment + decide-next-strategy.
```

- One context. Every crash stays in it; the agent self-corrects.
- No fan-out, no fresh-context children, no parent/child sync.
- Sequential, not parallel.
- Demonstrates: tool use, multi-turn REPL, persistent ledger.
- Loses: delegation, parallelism, the parent-as-coordinator story.
- **The boring, most likely to actually produce numbers.**

### B. Python-driver fan-out. Parent is a Python loop, not an LLM.

```python
# autoresearch.py becomes:
for round_i in range(args.rounds):
    ledger = read_ledger()
    slugs = pick_fresh_slugs(ledger, n=args.branches_per_turn)
    flows = [
        spawn_child_flow(slug, hyp, llm, runtime)
        for slug, hyp in slugs
    ]
    parallel_run(flows)
print_summary()
```

- "Parent" is deterministic Python. No "parent prompt" to drift.
- Each child is option A (single-agent loop) on its own slug.
- Children share a ledger via filesystem locking.
- Parallel exploration without coordinator drift.
- Loses: the "RLMFlow shows off delegation" demo angle.
- **Preserves parallelism; drops the parent-LLM fiction.**

### C. Keep architecture; default to gpt-5 (non-mini).

- One-line change in autoresearch.py argparse default.
- Mini is genuinely the source of `solve_instance(inst: dict)`.
- gpt-5 produces correct contracts ~95%+ of the time on this kind
  of task in informal testing.
- Works with the existing prompt apparatus.
- **No architectural insight, but probably fixes 80% of the pain
  for one line of code.**

### D. Swap circle_packing for a soft-scoring task.

Examples:

- **Regression**: dataset of `(x, y)` pairs; agent writes
  `predict(x) -> y`; score = -MSE on held-out set. Any callable
  that takes `x` and returns `y` works; bad predictions get bad
  scores, not crashes.
- **Sequence prediction / sorting / arithmetic**: same shape; soft
  metrics.
- **Text classification on a tiny held-out set**: agent writes
  `classify(text) -> label`; score = accuracy.

Why it helps: validation no longer gates progress. A confused
`solve_instance(inst)` that returns garbage scores -100; the agent
sees -100 in the ledger and tries again. There's a continuous
gradient from "totally wrong" → "bad" → "good" instead of
crash-vs-pass.

Loses: visual / geometric appeal of circle-packing. The README
demo "watch it climb from 1.5 to 2.6" is gone unless we find an
equally satisfying new task.

### E. Less-strict evaluator (function name + return type fallbacks).

Cheaper than redesign:

```python
# evaluate.py:
fn = (
    getattr(mod, "solve", None)
    or _find_single_zero_arg_top_level_fn(mod)
)
result = fn()
# Accept tuple or list:
if isinstance(result, (tuple, list)) and len(result) == 2:
    centers, radii = result
elif isinstance(result, dict) and {"centers", "radii"} <= result.keys():
    centers, radii = result["centers"], result["radii"]
elif isinstance(result, dict) and {"x", "y", "r"} <= result.keys():
    centers = np.column_stack([result["x"], result["y"]])
    radii = np.asarray(result["r"])
else:
    return _fail(...)
```

- Absorbs the four most common LLM mistakes (`solve_instance`,
  list, dict-with-centers/radii, dict-with-x/y/r).
- Doesn't fix the "agent gives up on retry" issue.
- Doesn't fix the "parent doesn't loop" issue.
- **A patch, not a fix. But a useful patch on top of any of A/B/C.**

### F. Force retry by making `run_experiment` return-on-failure instead of raise.

```python
# Instead of raising ExperimentCrashed:
return {"score": None, "stderr_tail": "...", "ok": False, ...}
```

- Children can't bail with `done(ok=False, error=...)` after the
  first crash because the API doesn't surface a crash; it just
  returns a bad row.
- They have to look at the row and decide: either declare the slug
  dead, or write a fixed source and call `run_experiment` again.
- Removes the "exception → just call done()" easy-out path.
- Makes the retry pattern the natural expression in code.

### G. Drop branching entirely; "agent" is just the autoresearch loop.

The autoresearch.py script itself is the agent. It uses the LLM as
a code-generation tool but doesn't expose it through RLMFlow's
agent abstraction at all:

```python
for trial in range(args.budget):
    ledger = read_ledger()
    prompt = build_prompt(brief, ledger, best_source)
    new_source = llm.complete(prompt)        # one call
    score = run_evaluator(new_source)         # subprocess
    append_ledger(slug, score, new_source)
print_summary()
```

- Lose: agentic tool use, multi-turn REPL, the whole point of
  RLMFlow.
- Gain: dead simple, predictable, hard to break.
- Probably not what we want for an *RLMFlow example*. Mentioned
  for completeness — if autoresearch isn't really an "agent" task
  in any interesting way, maybe it shouldn't be an RLMFlow demo at
  all.

### H. Retire circle_packing, replace with something flashier.

Other candidates that demonstrate hill-climbing under autoresearch:

- **MNIST-toy**: train a tiny numpy MLP on a 100-sample subset,
  agent edits training-loop hyperparams + arch.
- **Symbolic regression on a small dataset**.
- **Boids parameter tuning** (we already have a boids notebook,
  could share assets): agent picks parameters, score = visual
  measure of "flockingness".
- **CSV transformation puzzles**: input CSV in, target CSV out,
  score = row-similarity.

## My read

The user's problem isn't any single one of these — it's the
compound effect. Best ROI in order:

1. **C (gpt-5 default)** — biggest single behavior delta, zero
   architectural risk, one-line change. Probably moves us from
   "0 valid trials" to "majority valid trials" overnight.
2. **A or B** — pick whichever serves the demo story better. A is
   simpler; B keeps the parallelism + delegation story. For an
   RLMFlow example I lean B because it's still "showing off"
   parallel agents, just without the brittle LLM coordinator.
3. **D or H** — swap the example task. circle_packing is a poor
   first impression; the validation cliff means even a strong
   model has high crash rate, which makes the demo unreliable.
4. **E + F** — patches that compose with any redesign. Keep the
   design simple but absorb common drift modes.

## Concrete minimal next step (if forced to pick one)

Apply C + E + F together. Keep current architecture, but:

- `--model gpt-5` default.
- Evaluator accepts list/tuple/dict and falls back to "the only
  zero-arg top-level function."
- `run_experiment` returns rows on failure (no `ExperimentCrashed`),
  so children can't bail with one-line excuses.

If after that, baseline-rate of valid trials is < 50%, escalate to
A or B + D.
