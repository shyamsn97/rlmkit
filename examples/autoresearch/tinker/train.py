"""Tinker LoRA SFT on TinyStories — the file the agent edits.

Runs a short fine-tune of a small base model on TinyStories and reports
validation **bits-per-byte** on a held-out slice. The autoresearch driver
parses the last line of stdout looking for ``val_bpb: <float>``.

Contract (do NOT change):
- Read ``data/train.txt`` and ``data/val.txt`` (built by ``prepare.py``).
- Print one final line ``val_bpb: <float>`` on success. Lower is better.
- Use ``TINKER_API_KEY`` from the environment.
- bpb is bits-per-utf8-byte: ``total_nll_nats / (total_bytes * ln 2)``.
  This makes it vocab-independent so model swaps are still comparable.

Everything else is yours to mutate: model, hyperparams, schedule, packing,
eval batch, masking strategy.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import tinker
from tinker import types


# ── Mutable hyperparams ──────────────────────────────────────────────
#
# Pick a *small base model* so there's real headroom — a heavily
# pretrained instruct model is already near-optimal on TinyStories and
# LoRA can barely move it. Tinker's available base (non-instruct) models
# in ascending size:
#   meta-llama/Llama-3.2-1B   ← weakest, default
#   meta-llama/Llama-3.2-3B
#   Qwen/Qwen3-4B-Base
#   meta-llama/Llama-3.1-8B
# Note: `Qwen/Qwen3-0.6B` exists but is instruct-tuned, not a base.

BASE_MODEL: str = "meta-llama/Llama-3.2-1B"
LORA_RANK: int = 16

LR: float = 3e-4
BETA1: float = 0.9
BETA2: float = 0.95
EPS: float = 1e-8

BATCH_SIZE: int = 8
SEQ_LEN: int = 512
MAX_STEPS: int = 80          # wall-clock budget caps this anyway
LR_SCHEDULE: str = "linear"  # "linear" | "constant" | "cosine"

EVAL_SEQS: int = 64          # number of val sequences for the bpb estimate
LOG_EVERY: int = 10
SEED: int = 0


# ── Fixed plumbing (don't change unless you know why) ────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_TXT = DATA_DIR / "train.txt"
VAL_TXT = DATA_DIR / "val.txt"


def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Run `export TINKER_API_KEY=...`.")
    if not TRAIN_TXT.exists() or not VAL_TXT.exists():
        raise SystemExit(
            f"Missing data: {TRAIN_TXT} / {VAL_TXT}. Run `python prepare.py` first."
        )

    random.seed(SEED)
    t0 = time.time()

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    tokenizer = client.get_tokenizer()
    print(f"base_model={BASE_MODEL} lora_rank={LORA_RANK}")

    train_seqs = _pack_sequences(TRAIN_TXT.read_text(encoding="utf-8"), tokenizer, SEQ_LEN)
    val_text = VAL_TXT.read_text(encoding="utf-8")
    val_seqs = _pack_sequences(val_text, tokenizer, SEQ_LEN)
    print(f"train_seqs={len(train_seqs):,}  val_seqs={len(val_seqs):,}  seq_len={SEQ_LEN}")
    if not train_seqs or not val_seqs:
        raise SystemExit("Not enough data to form a single sequence — make SEQ_LEN smaller.")

    rng = random.Random(SEED)
    for step in range(MAX_STEPS):
        batch = [
            _make_datum(train_seqs[rng.randrange(len(train_seqs))])
            for _ in range(BATCH_SIZE)
        ]
        lr = _lr_at(step, MAX_STEPS, LR, LR_SCHEDULE)
        adam = types.AdamParams(learning_rate=lr, beta1=BETA1, beta2=BETA2, eps=EPS)

        fwd = client.forward_backward(batch, loss_fn="cross_entropy")
        opt = client.optim_step(adam)
        fwd_result = fwd.result()
        opt.result()

        if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            # Every datum has exactly SEQ_LEN weighted target tokens by
            # construction (see `_pack_sequences` + `_make_datum`).
            mean_nll = float(fwd_result.metrics["loss:sum"]) / (BATCH_SIZE * SEQ_LEN)
            print(
                f"step={step:4d}  train_nll={mean_nll:.4f}  "
                f"lr={lr:.2e}  elapsed={time.time() - t0:.0f}s"
            )

    val_bpb = _eval_bpb(client, val_seqs[:EVAL_SEQS], val_text, tokenizer)
    print(f"final  val_bpb={val_bpb:.4f}  elapsed={time.time() - t0:.0f}s")
    # Last line is the contract.
    print(f"val_bpb: {val_bpb:.6f}")


# ── helpers ──────────────────────────────────────────────────────────

def _pack_sequences(text: str, tokenizer, seq_len: int) -> list[list[int]]:
    """Tokenize the corpus once and split into fixed-length chunks of seq_len + 1.

    Each chunk yields one Datum with input = tokens[:-1] and target = tokens[1:].
    HF tokenizers warn when a single ``encode()`` call exceeds the model's
    ``model_max_length``; we *want* a long sequence here (we're going to chunk
    it ourselves), so silence that warning to keep stderr usable for real errors.
    """
    import logging

    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunk = seq_len + 1
    return [ids[i : i + chunk] for i in range(0, len(ids) - chunk, chunk)]


def _make_datum(token_ids: list[int]) -> types.Datum:
    inputs = token_ids[:-1]
    targets = token_ids[1:]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=inputs),
        loss_fn_inputs=dict(
            weights=[1.0] * len(inputs),
            target_tokens=targets,
        ),
    )


def _lr_at(step: int, total: int, base: float, schedule: str) -> float:
    if total <= 1:
        return base
    if schedule == "constant":
        return base
    if schedule == "cosine":
        progress = step / (total - 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * progress))
    # linear (default): decay to zero by the last step
    return base * max(0.0, 1.0 - step / total)


def _eval_bpb(client, val_seqs: list[list[int]], val_text: str, tokenizer) -> float:
    """Validation bits-per-byte over ``val_seqs`` using the trained LoRA.

    Computes total NLL across the eval sequences (in nats) and divides by the
    UTF-8 byte length of the same tokens to get a vocab-independent metric.
    Token text is reconstructed via the tokenizer so the byte count matches the
    tokens we actually scored.
    """
    if not val_seqs:
        return float("inf")

    batch = [_make_datum(seq) for seq in val_seqs]

    # forward_backward without a follow-up optim_step leaves gradients sitting
    # in the optimizer; that's fine — this is the last thing we do before exit.
    fwd_result = client.forward_backward(batch, loss_fn="cross_entropy").result()

    # cross_entropy reports `loss:sum` — scalar sum of -log p(x_t) * w_t over
    # the whole batch, in nats. Our weights are all 1.0, so that's the total
    # NLL across all scored target positions.
    total_nll_nats = float(fwd_result.metrics["loss:sum"])

    # convert total nats → bits/byte using the actual byte length of this slice.
    target_text = tokenizer.decode(
        [tok for seq in val_seqs for tok in seq[1:]],
        skip_special_tokens=False,
    )
    bytes_count = max(1, len(target_text.encode("utf-8")))
    return total_nll_nats / (bytes_count * math.log(2))


if __name__ == "__main__":
    main()
