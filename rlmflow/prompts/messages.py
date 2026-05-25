"""Message templates used by the RLMFlow engine.

All user-facing text lives here so rlm.py stays logic-only.

Mirrors the prompts in alexzhang13/rlm (`rlm/utils/prompts.py`):
https://github.com/alexzhang13/rlm/blob/main/rlm/utils/prompts.py
"""

from __future__ import annotations

from typing import Any

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

REPL_BLOCK_RULE = """Use exactly one fenced REPL code block per assistant message. Your entire reply must have this shape:
```repl
# Python code here
```
Do not write bare `repl` without the opening and closing triple backticks."""

# USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the prompt.

# Continue using the REPL environment, which has the `CONTEXT` variable, and querying sub-LLMs / sub-agents by writing to ```repl``` tags, and determine your answer. Your next action:"""

USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: "{query}".

Continue using the REPL environment, which has the `CONTEXT` variable, and querying sub-LLMs / sub-agents by writing to ```repl``` tags, and determine your answer. Your next action:"""


def build_context_metadata(
    info: dict[str, Any] | None,
    context_keys: list[str] | None = None,
) -> str:
    """`Your CONTEXT contains ...` size signal + optional extra-keys note.

    Mirrors the size hint in alexzhang13/rlm's `build_rlm_system_prompt`
    (https://github.com/alexzhang13/rlm/blob/main/rlm/utils/prompts.py).
    Returns ``""`` when there is nothing to say (empty/missing CONTEXT and no
    extra keys). Called once per agent, prepended to the iteration-0 user
    message so the model knows the size of what it's been handed before
    deciding on a chunking strategy.
    """
    lines: list[str] = []
    chars = int(info.get("chars", 0)) if info else 0
    if chars > 0:
        approx_tokens = int(info.get("approx_tokens", chars // 4))
        n_lines = int(info.get("lines", 0))
        lines.append(
            f"Your `CONTEXT` contains {chars} characters across {n_lines} "
            f"lines (~{approx_tokens} tokens)."
        )

    extra_keys = sorted(k for k in (context_keys or []) if k != "context")
    if extra_keys:
        shown = ", ".join(extra_keys[:8])
        suffix = f", ... +{len(extra_keys) - 8} more" if len(extra_keys) > 8 else ""
        lines.append(f"Additional context keys: {shown}{suffix}.")

    return "\n\n".join(lines)


CONTINUE_ACTION = "Continue. Your next action:"

FIRST_TURN_DECOMPOSITION_NUDGE = (
    "If the task decomposes into independent units, your first action should "
    "usually spawn the child batch with `rlm_delegate(...)` or issue a "
    "`llm_query_batched(...)` fanout rather than solving every unit yourself. "
    "When using `rlm_delegate`, put the shared requirements/data in "
    "`context=...` so the child can inspect them through `CONTEXT.read()`; "
    'avoid `context=""` for nontrivial delegated work.'
)


def build_user_prompt(
    *,
    query: str | None = None,
    iteration: int = 0,
    depth: int = 0,
    max_depth: int = 0,
    context_keys: list[str] | None = None,
    context_info: dict[str, Any] | None = None,
) -> str:
    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment or seen your "
            "prompt / context yet. Your next action should be to look through "
            "and figure out how to answer the prompt, so don't just provide a "
            "final answer yet."
        )
        body = USER_PROMPT_WITH_ROOT.format(query=query)
        metadata = build_context_metadata(context_info, context_keys)
        parts = [safeguard]
        if metadata:
            parts.append(metadata)
        parts.append(body)
        parts.append(FIRST_TURN_DECOMPOSITION_NUDGE)
        prompt = "\n\n".join(parts)
    else:
        # Continue turns: the REPL output above already carries all the
        # substantive context. The model knows it's in a loop and what
        # the original query was (from history). A minimal nudge is enough.
        prompt = CONTINUE_ACTION

    if max_depth > 0 and depth >= max_depth:
        prompt += (
            "\n\nNote: You are at the recursion limit; you cannot spawn sub-agents."
        )

    return prompt


FINAL_ANSWER_ACTION = """You have used the full iteration budget without calling done().

Based on the work above, provide the final answer now. The block must call done(answer). The done() argument must be only the final answer string in the exact form the query requested. Do not do more investigation."""

NO_CODE_BLOCK = f"ERROR: Your previous reply did not contain a ```repl``` code block. {REPL_BLOCK_RULE} Try again."

EXECUTION_OUTPUT = "REPL output for previous block:\n{output}"

ORPHANED_DELEGATES = "You delegated [{names}] but never called `await rlm_wait(...)`. You must use `await rlm_wait(*handles)` to collect results."

STATUS_DEPTH_ROOT = " You have the full recursion budget available."
STATUS_DEPTH_MID = " Some recursion budget remains available."
STATUS_DEPTH_NEAR_MAX = " You are near the recursion limit."

TRUNCATION_SUMMARY = """## Query
{query}

## History
{total} messages so far, showing the last {cap}.{session_hint}"""

TRUNCATION_SESSION_HINT = ""
