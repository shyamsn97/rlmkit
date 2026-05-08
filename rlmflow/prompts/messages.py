"""Message templates used by the RLMFlow engine.

All user-facing text lives here so rlm.py stays logic-only.
"""

DEFAULT_QUERY = (
    "Please read through the context and answer any queries "
    "or respond to any instructions contained within it."
)

FIRST_ACTION = (
    "Query: {query}\n\n"
    "{context_hint}"
    "Respond with exactly one ```repl``` code block."
)

CONTINUE_ACTION = (
    "Continue working on: {query}\n\n"
    "{context_hint}"
    "Respond with one ```repl``` code block."
)

CONTEXT_HINT_PRESENT = (
    "Relevant data is available as the `CONTEXT` REPL variable — "
    "use `CONTEXT.read/lines/grep` to navigate it.\n\n"
)
CONTEXT_HINT_ABSENT = ""

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

EXECUTION_OUTPUT = "REPL output:\n{output}"

ORPHANED_DELEGATES = (
    "You delegated [{names}] but never called `yield wait(...)`. "
    "You must use `yield wait(*handles)` to collect results."
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
