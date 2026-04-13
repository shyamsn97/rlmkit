"""Message templates used by the RLM engine.

All user-facing text lives here so rlm.py stays logic-only.
"""

FIRST_ACTION = (
    "Task: {task}\n\n"
    "Your response MUST contain exactly one ```repl``` code block. "
    "Write Python that makes progress on the task."
)

CONTINUE_ACTION = (
    "Continue working on: {task}\n\n" "Respond with a ```repl``` code block."
)

NO_CODE_BLOCK = (
    "ERROR: Your previous reply did not contain a ```repl``` code block. "
    "You MUST reply with exactly one ```repl``` block every time. "
    "Try again."
)

EXECUTION_OUTPUT = "Code executed:\n```python\n{code}\n```\n\nREPL output:\n{output}"

ORPHANED_DELEGATES = (
    "You delegated [{names}] but never called `yield wait(...)`. "
    "You must use `yield wait(*handles)` to collect results."
)

RESUME_MESSAGE = (
    "## Resuming Previous Session\n\n"
    "**Task:** {task}\n\n"
    "**Agent tree at save time:**\n```\n{tree}\n```\n\n"
    "**Recent history:**\n{recent}\n\n"
    "Pick up where you left off."
)

STATUS_DEPTH_ROOT = " You are the root agent — delegate freely to sub-agents."

STATUS_DEPTH_MID = " Be more conservative with delegation the deeper you are."

STATUS_DEPTH_NEAR_MAX = (
    " You are near the depth limit — work directly, do not delegate."
)

TRUNCATION_SUMMARY = (
    "## Task\n{task}\n\n"
    "## History\n{total} messages so far, showing the last {cap}.{session_hint}"
)

TRUNCATION_SESSION_HINT = (
    " Earlier messages were trimmed — call `read_history()` to recover them."
)
