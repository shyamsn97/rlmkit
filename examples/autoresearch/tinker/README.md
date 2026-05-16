# Tinker autoresearch on TinyStories

A Karpathy-style autoresearch target where the agent hill-climbs `val_bpb`
on TinyStories by editing a [Tinker](https://thinkingmachines.ai/tinker/)
LoRA SFT script. The harness (`autoresearch.py`) doesn't need to know
anything about Tinker — it just runs `python train.py` under a wall-clock
budget and parses `val_bpb: <float>` from stdout.

```
examples/autoresearch/tinker/
├── prepare.py    # one-time data prep (downloads TinyStories) — DO NOT EDIT
├── train.py      # Tinker LoRA SFT loop — the agent edits this
├── program.md    # the agent's operating manual
└── data/         # populated by prepare.py
```

## What's different from Karpathy's autoresearch

| | Karpathy autoresearch | Tinker autoresearch |
|---|---|---|
| Training | nanochat from scratch (~50-200M params) | LoRA fine-tune ~600M-1B base model |
| Hardware | single local H100 | Tinker hosted (paid) |
| Cost | free | per-experiment, real money |
| Surface area | full `train.py` (model, optimizer, loss) | hyperparams + data prep + LR schedule |
| `val_bpb` | bits-per-byte on FineWeb val | bits-per-byte on TinyStories val |

The autoresearch loop itself is identical: edit → run → check → commit/reset → repeat.

## One-time setup

```bash
# 1. Tinker SDK + dataset loader
pip install tinker datasets

# 2. API key (https://tinker-console.thinkingmachines.ai/)
export TINKER_API_KEY="..."

# 3. Download + cache TinyStories text
cd examples/autoresearch/tinker
python prepare.py
```

Optional sanity run before letting the agent loose:

```bash
python train.py
# ...
# step=  50  train_nll=2.0987  lr=1.67e-05  elapsed=287s
# final  val_bpb=1.2345  elapsed=305s
# val_bpb: 1.234500
```

Confirm the last line matches `val_bpb: <float>` — that's what the agent
reads.

## Running the agent

The driver lives one level up. Point it at this directory:

```bash
cd /Users/shyam/Code/rlmkit
python examples/autoresearch/autoresearch.py \
    --target examples/autoresearch/tinker \
    --budget-s 360 \
    --rounds 20
```

No git setup is required — the agent edits `train.py` in place inside a
disposable workspace under `runs/autoresearch/`, and every version that
ran is archived to `history/<n>_train.py` for rollback.

## What the agent will mutate

`train.py` exposes a block of hyperparams at the top:

- `BASE_MODEL`, `LORA_RANK`
- `LR`, `BETA1`, `BETA2`, `EPS`, `LR_SCHEDULE`
- `BATCH_SIZE`, `SEQ_LEN`, `MAX_STEPS`
- `EVAL_SEQS`, `LOG_EVERY`, `SEED`

It can also rewrite `_pack_sequences`, `_make_datum`, `_lr_at`, or
`_eval_bpb` if it has a hypothesis about packing, masking, schedule
shape, or eval. It cannot touch `prepare.py` or the `val_bpb: <float>`
contract — those are fixed.

## Cost & speed notes

- Tinker is paid per experiment; 20 rounds × 5min runs adds up fast.
  Start with `--rounds 2 --budget-s 60` to dry-run the loop before
  unleashing it.
- The wall-clock timeout passed via `--budget-s` is enforced by the
  driver; if Tinker queues are slow, your experiment may fail to print
  `val_bpb` before the kill — the driver handles this gracefully (returns
  `val_bpb: null`, agent treats as a failed mutation and resets).

## References

- Karpathy: [autoresearch](https://github.com/karpathy/autoresearch)
- Tinker docs: [quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/),
  [supervised learning recipe](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_loop.py)
- Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
