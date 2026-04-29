"""OOLONG scorer — standalone, zero rlmkit dependency.

Implements the scoring methodology from the OOLONG paper
(arXiv:2511.02817, §2.3 and §3.2). See
``docs/internal/oolong_scoring.md`` for the full derivation and links.

TL;DR:

  1. Parse ``answer`` with ``ast.literal_eval`` (the HF dataset stores
     it as a Python-repr of a list).
  2. Extract the slot from the prediction by matching the template
     embedded in the question (``Give your final answer in the form
     'TEMPLATE'``). Fall back to the last substring of the expected
     answer type (number / comparison / label / date) if the template
     does not match.
  3. Dispatch on ``answer_type``:
       - NUMERIC        → ``score = 0.75 ** abs(gold - pred)``
       - LABEL / COMPARISON / USER / DATE / MONTH_YEAR
                        → normalized exact match (0/1)
       - list (real)    → set-overlap F1

This module has no runtime dependencies beyond the Python standard
library. It is intended to be reused from OOLONG harnesses (e.g. a
future local runner or lighteval task).

When the official harness at https://github.com/abertsch72/oolong
ships eval code, re-pin on a small sample and resolve any drift.
"""

from __future__ import annotations

import ast
import re
from typing import Any

__all__ = [
    "normalize",
    "parse_gold",
    "normalize_answer_type",
    "template_from_question",
    "extract_from_template",
    "instantiate_template",
    "render_gold_templated",
    "unwrap_done",
    "last_substring_of_type",
    "extract_answer",
    "score_numeric",
    "score_exact",
    "score_set_overlap",
    "score_one",
    "COMPARISON_PHRASES",
    "TEMPLATE_PLACEHOLDERS",
]


# ── Regexes & constants ───────────────────────────────────────────────

WHITESPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

COMPARISON_PHRASES = (
    "more common than",
    "less common than",
    "same frequency as",
    "greater than",
    "less than",
    "equal to",
)

MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
MONTH_YEAR_RE = re.compile(
    r"\b(?:" + "|".join(MONTHS) + r")\s+\d{4}\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:" + "|".join(m[:3] for m in MONTHS) + r")\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

# Placeholder tokens that mark the slot in an OOLONG answer template —
# e.g. "Label: answer", "Answer: number", "Count: N".
TEMPLATE_PLACEHOLDERS = (
    "answer", "answers", "n", "number", "count",
    "value", "label", "word", "name", "date",
)

SLOT_SENTINEL = "\x00SLOT\x00"

# Match ``done("…")`` / ``done('…')`` — used to recover the final answer
# from a raw LLM reply when the engine hit ``max_iterations`` on the very
# iteration that proposed it (so the code block was never executed and
# ``state.result`` fell back to ``state.last_reply``). Single-line argument
# only; multi-line ``done`` calls are vanishingly rare in practice.
DONE_CALL_RE = re.compile(r"""done\s*\(\s*(['"])(.*?)\1\s*\)""", re.DOTALL)


# ── Normalization ─────────────────────────────────────────────────────


def normalize(text: Any) -> str:
    """Light normalization for string-type comparisons: lowercase + ws collapse.

    Deliberately *does not* strip punctuation or articles — OOLONG
    answers like ``"more common than"`` or ``"Jan 21, 2025"`` rely on
    those. Case-folding + whitespace collapse is all the paper calls
    for.
    """
    if not isinstance(text, str):
        text = str(text)
    return WHITESPACE_RE.sub(" ", text.strip().lower())


def parse_gold(raw: Any) -> list[Any]:
    """Parse OOLONG's ``answer`` field into a list.

    The HF dataset stores ``answer`` as a string containing a Python
    repr of a list — ``"['incorrect']"``, ``"[7]"``, ``"['a', 'b']"``.
    Returns the parsed list elements (ints stay ints, strings stay
    strings). Passthrough for actual lists.
    """
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith(("[", "(")) and s.endswith(("]", ")")):
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [raw]
    return [raw]


def normalize_answer_type(raw: str | None) -> str:
    """``"ANSWER_TYPE.LABEL"`` → ``"label"``; unknown → ``"string"``."""
    if not raw:
        return "string"
    tail = str(raw).split(".")[-1].strip().lower()
    return tail or "string"


# ── Extraction ────────────────────────────────────────────────────────


def template_from_question(question: str | None) -> str | None:
    """Pull the quoted answer template out of the question.

    Looks for ``Give your final answer in the form '<template>'`` (or
    double-quoted). Returns the raw template string or ``None``.
    """
    if not question:
        return None
    m = re.search(
        r"in the form\s*[:\-]?\s*['\"]([^'\"]+)['\"]",
        question,
        re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def extract_from_template(prediction: str, template: str) -> str | None:
    """Match the prediction against an OOLONG answer template and return the slot.

    Placeholders recognized:
      - bracketed: ``[X]``, ``[Y]``, ``[ANSWER]``
      - trailing word after ``: ``: ``answer``, ``number``, ``N``, ...

    Substitutions happen on the *raw* template (before ``re.escape``)
    via a sentinel, so whitespace and other regex metacharacters in the
    template are correctly escaped around the capture group. The capture
    stops at characters that mark the end of a string literal in source
    code (quotes, closing parens/brackets/braces, backticks); without
    that fence we would happily slurp ``False")`` out of
    ``done("Label: False")`` and miscompare to the gold ``False``. We
    take the *last* match in the prediction (models often restate the
    template multiple times before settling).
    """
    alt = "|".join(TEMPLATE_PLACEHOLDERS)
    t = re.sub(r"\[[^\]]+\]", SLOT_SENTINEL, template)
    t = re.sub(
        rf":\s+(?:{alt})\s*$",
        ": " + SLOT_SENTINEL,
        t,
        flags=re.IGNORECASE,
    )

    pattern = re.escape(t).replace(
        re.escape(SLOT_SENTINEL), r"([^\n\"'`)\]}]+)"
    )
    try:
        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error:
        return None
    matches = compiled.findall(prediction)
    if not matches:
        return None
    slot = matches[-1]
    if isinstance(slot, tuple):
        slot = next((part for part in reversed(slot) if part), "")
    return slot.strip().strip("'\"").strip()


def instantiate_template(template: str, slot: str) -> str | None:
    """Substitute the slot placeholder in *template* with *slot*.

    Inverse of :func:`extract_from_template` — useful for rendering the
    gold answer back into the question's "in the form '...'" template
    so a JSONL row's gold can be eyeballed against the prediction in a
    compatible shape.

    Returns ``None`` if no recognized placeholder is found.
    """
    bracketed = re.search(r"\[[^\]]+\]", template)
    if bracketed:
        return re.sub(r"\[[^\]]+\]", slot, template, count=1)
    alt = "|".join(TEMPLATE_PLACEHOLDERS)
    m = re.search(rf":\s+(?:{alt})\s*$", template, re.IGNORECASE)
    if m:
        return template[: m.start()] + ": " + slot
    return None


def render_gold_templated(question: str | None, gold: list[Any]) -> str | None:
    """Render *gold* into the question's answer template, if any.

    Single-slot golds get instantiated directly. Multi-element golds
    (OOLONG-real list answers) are joined with ``", "`` and substituted
    as one slot — good enough for human-readable JSONL eyeballing, not
    used for scoring.
    """
    if not question or not gold:
        return None
    template = template_from_question(question)
    if not template:
        return None
    slot = str(gold[0]) if len(gold) == 1 else ", ".join(str(g) for g in gold)
    return instantiate_template(template, slot)


def unwrap_done(text: str) -> str | None:
    """Pull the string argument out of a ``done("…")`` call in *text*.

    Returns the *last* such argument (agents sometimes stage several
    candidate answers before settling). ``None`` if no ``done(...)``
    is present.

    Used as a pre-pass in :func:`extract_answer` so that when an agent
    runs out of iterations on the iteration that proposed the final
    answer — e.g. ``state.last_reply == '```repl\\ndone("Label: False")\\n```'`` —
    we still recover the intended slot instead of returning the raw
    code fence.
    """
    if not text:
        return None
    matches = DONE_CALL_RE.findall(text)
    if not matches:
        return None
    return matches[-1][1]


def last_substring_of_type(text: str, answer_type: str) -> str | None:
    """Paper §2.3 fallback: return the last substring matching the type.

    Used when the question template doesn't match the prediction
    (commonly when the model ran out of output budget).
    """
    t = answer_type
    if t in ("numeric", "number"):
        matches = NUMBER_RE.findall(text)
        return matches[-1] if matches else None
    if t == "comparison":
        low = text.lower()
        hits = [(low.rfind(p), p) for p in COMPARISON_PHRASES]
        hits = [h for h in hits if h[0] >= 0]
        if not hits:
            return None
        return max(hits, key=lambda x: x[0])[1]
    if t == "month_year":
        matches = MONTH_YEAR_RE.findall(text)
        return matches[-1] if matches else None
    if t == "date":
        matches = DATE_RE.findall(text)
        return matches[-1] if matches else None
    if t == "user":
        matches = re.findall(r"\b\d{3,}\b", text)
        return matches[-1] if matches else None
    return None


def extract_answer(
    prediction: str,
    question: str | None,
    answer_type: str = "string",
) -> str:
    """Best-effort slot extraction per OOLONG §2.3.

    Order:
      0. If the prediction is raw source containing ``done("X")``, treat
         X as the prediction and continue. This catches the "agent ran
         out of iterations on the iteration that wrote the final answer"
         case — the engine fell back to ``state.last_reply`` (raw LLM
         text), but the model's intent is still recoverable.
      1. Template match against the question's ``in the form '…'`` spec.
      2. Last substring of the expected type (regex per type).
      3. Last non-empty, non-fence line of the prediction, truncated.
         We stop returning the entire prediction here because mid-
         thought blobs can be kilobytes of code and pollute
         ``results.jsonl``.
    """
    if not prediction:
        return ""
    text = prediction.strip()

    unwrapped = unwrap_done(text)
    if unwrapped is not None:
        text = unwrapped.strip()

    template = template_from_question(question)
    if template:
        slot = extract_from_template(text, template)
        if slot:
            return slot

    fallback = last_substring_of_type(text, answer_type)
    if fallback:
        return fallback

    last_line = ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("```"):
            last_line = stripped
            break
    return (last_line or text)[:200]


# ── Per-type scorers ──────────────────────────────────────────────────


def score_numeric(pred: str, gold_value: Any) -> dict[str, float]:
    """Partial-credit numeric score: ``0.75 ** |gold − pred|``."""
    try:
        g = float(gold_value)
    except (TypeError, ValueError):
        return {"score": 0.0, "parsed_pred": float("nan"), "parsed_gold": float("nan")}
    m = NUMBER_RE.search(pred or "")
    if not m:
        return {"score": 0.0, "parsed_pred": float("nan"), "parsed_gold": g}
    try:
        p = float(m.group(0))
    except ValueError:
        return {"score": 0.0, "parsed_pred": float("nan"), "parsed_gold": g}
    return {"score": 0.75 ** abs(g - p), "parsed_pred": p, "parsed_gold": g}


def score_exact(pred: str, gold: Any) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def score_set_overlap(pred: str, gold: list[Any]) -> float:
    """Set-overlap F1 used by OOLONG-real's list-valued answers."""
    gold_set = {normalize(g) for g in gold if normalize(g)}
    pred_items = [normalize(p) for p in re.split(r"[,\n;]", pred or "")]
    pred_set = {p for p in pred_items if p}
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


# ── Top-level score ───────────────────────────────────────────────────


def score_one(
    prediction: str,
    gold: Any,
    *,
    question: str | None = None,
    answer_type: str | None = None,
) -> dict[str, Any]:
    """Score a single prediction per the OOLONG paper methodology.

    Returns a dict with:
      - ``score``: primary metric in ``[0, 1]`` (the number the paper
        reports; equals exact match for strings, ``0.75 ** |diff|`` for
        numerics, set-F1 for lists).
      - ``exact_match``: 1.0 iff extracted == gold after normalize
        (informational; identical to ``score`` for string types).
      - ``extracted``: the slot our extractor pulled from the
        prediction, so you can eyeball what was actually scored.
      - ``answer_type``: the dispatch key used.
    """
    if prediction is None:
        prediction = ""
    at = normalize_answer_type(answer_type)
    golds = parse_gold(gold)
    extracted = extract_answer(prediction, question, at)

    if at in ("numeric", "number"):
        gold_val = golds[0] if golds else None
        num = score_numeric(extracted, gold_val)
        return {
            "answer_type": at,
            "extracted": extracted,
            "score": num["score"],
            "exact_match": 1.0 if num["parsed_pred"] == num["parsed_gold"] else 0.0,
            "parsed_pred": num["parsed_pred"],
            "parsed_gold": num["parsed_gold"],
        }

    if at == "list":
        s = score_set_overlap(extracted, golds)
        return {
            "answer_type": at,
            "extracted": extracted,
            "score": s,
            "exact_match": 1.0 if s == 1.0 else 0.0,
        }

    em = max(score_exact(extracted, g) for g in golds) if golds else 0.0
    return {
        "answer_type": at,
        "extracted": extracted,
        "score": em,
        "exact_match": em,
    }
