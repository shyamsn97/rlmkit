# autoresearch — LoRA SFT on TinyStories

You're the researcher. Lower **`val_bpb`** (bits per byte on a held-out
TinyStories slice) by editing `train.py`. Lower is better.

## How `run_experiment` works

`run_experiment(source: str, budget_s=...) -> dict` takes the **full
text of a train.py** as a string. It writes the source to
`history/<n>_train.py`, runs it under the wall-clock budget, and
returns:

```
{n, val_bpb, returncode, stdout_tail, stderr_tail, elapsed_s, train_py_path}
```

`val_bpb` is None on crash. If `returncode != 0`, the real error is in
`stderr_tail`. Every call also appends a row to `history/ledger.jsonl`,
so `list_runs()` always tells you what's been tried.

There is **no trial dir** to manage. You don't write `train.py`
yourself before running — you compute the new source as a Python
string and pass it in. Save winners by `write_file("train.py", new)`
once you've decided.

## What's mutable in `train.py`

`LORA_RANK`, `LR`, `BETA1`, `BETA2`, `EPS`, `BATCH_SIZE`, `SEQ_LEN`,
`MAX_STEPS`, `LR_SCHEDULE`, the loop / schedule shape, packing,
masking, the eval procedure. **Don't change `BASE_MODEL`** — trials
must be comparable.

`train.py` declares constants with PEP 526 type annotations:

```python
LR: float = 3e-4
SEQ_LEN: int = 512
LR_SCHEDULE: str = "linear"
```

So `old`/`new` literals for `str.replace` look like
`"LR: float = 3e-4"`, **not** `"LR=3e-4"`.

## Loop

1. **Block 1 — baseline.** Read `train.py` and run it as-is.

   ```python
   src = read_file("train.py")
   r   = run_experiment(src, budget_s=...)
   if r["returncode"] != 0 or r["val_bpb"] is None:
       done(f"Baseline broken: {r['stderr_tail']}")
   best_val, best_src = r["val_bpb"], src
   ```

2. **Block 2..N — try mutations.** Either run them sequentially yourself,
   or fan out via `delegate` (next section). For each, make a single
   `str.replace`, assert it actually changed the source, then
   `run_experiment(new)`:

   ```python
   new = best_src.replace("LR: float = 3e-4", "LR: float = 1.5e-4")
   assert new != best_src
   r   = run_experiment(new)
   if r["returncode"] == 0 and r["val_bpb"] is not None and r["val_bpb"] < best_val:
       best_val, best_src = r["val_bpb"], new
       write_file("train.py", new)
   ```

3. **Don't `done()` after one trial.** A "session" is many repl blocks;
   keep going until your iteration budget is gone or you've truly
   nothing left to try. Then `done(<final val_bpb vs baseline>)`.

## Parallel trials with `delegate`

For independent hypotheses, fan out one child per hypothesis. Each
child does **the same little block** as above, just with a different
mutation. Pass the parent's current `train.py` and the mutation in the
context — the child needs nothing else.

```python
import json
BUDGET = 300

CHILD_QUERY = (
    "You're a trial child. CONTEXT is JSON {src, old, new, budget_s}. "
    "Run exactly:\n"
    "    import json\n"
    "    s = json.loads(CONTEXT.read())\n"
    "    new = s['src'].replace(s['old'], s['new'])\n"
    "    assert new != s['src'], 'no-op replace'\n"
    "    r = run_experiment(new, s['budget_s'])\n"
    "    done(json.dumps({'val_bpb': r['val_bpb'], "
        "'returncode': r['returncode'], "
        "'train_py_path': r['train_py_path'], "
        "'stderr_tail': (r['stderr_tail'] or '')[-400:]}))\n"
    "Output exactly that ```repl``` block, nothing else."
)

def trial(name, old, new):
    ctx = json.dumps({"src": best_src, "old": old, "new": new, "budget_s": BUDGET})
    return delegate(name, CHILD_QUERY, ctx)

handles = [
    trial("lr_half",   "LR: float = 3e-4",            "LR: float = 1.5e-4"),
    trial("cosine_lr", 'LR_SCHEDULE: str = "linear"', 'LR_SCHEDULE: str = "cosine"'),
    trial("rank_x2",   "LORA_RANK: int = 16",         "LORA_RANK: int = 32"),
]
results = [json.loads(r) for r in (yield wait(*handles))]
```

Three children, three Tinker runs concurrent, one repl turn.

## Rules

- Never change `BASE_MODEL`.
- Every reported `val_bpb` comes from `run_experiment` or `list_runs`.
- A run with `returncode != 0` or `val_bpb is None` is **not a result**
  — quote `stderr_tail`, fix the code, or move on.
- Tinker calls cost real money — keep `MAX_STEPS` honest.
- Sanity-check `stdout_tail`: `train_nll` should be decreasing. A "win"
  with flat or rising train loss is suspect.
