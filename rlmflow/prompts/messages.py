"""Message templates used by the RLMFlow engine.

All user-facing text lives here so rlm.py stays logic-only.
"""

DEFAULT_QUERY = (
    "Please read through the context and answer any queries "
    "or respond to any instructions contained within it."
)

FIRST_ACTION = (
    "Query: {query}\n\n"
    "Your response MUST contain exactly one ```repl``` code block. "
    "Write Python that makes progress on the query. "
    "If the task has separable parts, delegate them in this first program; "
    "if it is trivial, do it directly."
)

CONTINUE_ACTION = (
    "Continue working on: {query}\n\n"
    "Respond with a ```repl``` code block. "
    "Use delegation for separable remaining work; do direct code for small, coupled work."
)

FINAL_ANSWER_ACTION = (
    "You have used the full iteration budget without calling done().\n\n"
    "Based on the work above, provide the final answer now. "
    "Respond with exactly one ```repl``` code block that calls done(answer). "
    "The done() argument must be only the final answer string in the exact form "
    "the query requested. Do not do more investigation."
)

NO_CODE_BLOCK = (
    "ERROR: Your previous reply did not contain a ```repl``` code block. "
    "You MUST reply with exactly one ```repl``` block every time. "
    "Try again."
)

EXECUTION_OUTPUT = "Code executed:\n```python\n{code}\n```\n\nREPL output:\n{output}"

STUCK_WARNING = (
    "\n\nWARNING: Your code produced no output and did not call done(). "
    "This is the {n}th consecutive empty execution. Your current approach is "
    "NOT working. Do NOT repeat the same code. Either:\n"
    "  1. Fix the root cause (e.g. the variable/data you need doesn't exist)\n"
    "  2. Try a completely different approach\n"
    "  3. Call done() with an error message explaining what went wrong"
)

ORPHANED_DELEGATES = (
    "You delegated [{names}] but never called `yield wait(...)`. "
    "You must use `yield wait(*handles)` to collect results."
)

RESUME_MESSAGE = (
    "## Resuming Previous Session\n\n"
    "**Query:** {query}\n\n"
    "**Agent tree at save time:**\n```\n{tree}\n```\n\n"
    "**Recent history:**\n{recent}\n\n"
    "Pick up where you left off."
)

STATUS_DEPTH_ROOT = (
    " You have the full recursion budget. Delegate based on task structure: "
    "independent subtasks, multiple files, large context analysis, and review/test work are good candidates."
)

STATUS_DEPTH_MID = " You still may delegate when work is naturally separable; avoid unnecessary delegation chains."

STATUS_DEPTH_NEAR_MAX = " You are near the recursion limit. Prefer direct work unless one final child would clearly isolate an independent subtask."

TRUNCATION_SUMMARY = (
    "## Query\n{query}\n\n"
    "## History\n{total} messages so far, showing the last {cap}.{session_hint}"
)

TRUNCATION_SESSION_HINT = ""
