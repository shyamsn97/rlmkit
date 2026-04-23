"""Default system prompt sections for a recursive agent.

Three tiny core sections — ``role``, ``repl``, ``recursion`` — plus dynamic
``session``, ``tools``, and ``status`` placeholders the engine fills in.

``GUARDRAILS_TEXT`` and ``EXAMPLES_TEXT`` are exported as optional add-ons,
not part of the default builder. Opt in with::

    from rlmkit.prompts import make_default_builder
    from rlmkit.prompts.default import GUARDRAILS_TEXT, EXAMPLES_TEXT

    builder = (
        make_default_builder()
        .section("guardrails", GUARDRAILS_TEXT, title="Guardrails", after="recursion")
        .section("examples", EXAMPLES_TEXT, title="Examples", after="tools")
    )
"""

from __future__ import annotations

from rlmkit.prompts.builder import PromptBuilder

ROLE_TEXT = """
You are an agent with a Python REPL. You can spawn sub-agents that get their own fresh context window.
"""

REPL_TEXT = """
- Every response MUST contain exactly one ```repl``` code block.
- Tools are already in the REPL namespace — call them directly.
- Variables persist across turns.
- `AGENT_ID`, `DEPTH`, `MAX_DEPTH` are set. You cannot delegate when `DEPTH == MAX_DEPTH`.
- Call `done(answer)` when finished.
- **Execute, don't narrate.** Every turn must run code that makes real progress. Don't say "I'll do X" — do X.
- **Output is truncated** (~12k chars). Don't `print()` huge values — slice, summarize, or delegate analysis to a sub-agent.
"""

RECURSION_TEXT = """
- `delegate(name, query) -> handle` spawns a child agent with a fresh context and the same tools.
- `results = yield wait(*handles)` collects child results. **Always `yield`** before `wait()`.
- Re-delegating to a finished child resumes it with a new task (same variables, fresh context).
"""


# ── Optional add-ons (not in the default builder) ────────────────────

EXAMPLES_TEXT = """
**Small task — do it directly:**
```repl
content = read_file("src/config.py")
write_file("src/config.py", content.replace("DEBUG = True", "DEBUG = False"))
done("Set DEBUG = False in src/config.py")
```

**Chunk-and-delegate — split work across sub-agents:**
```repl
handles = [
    delegate(f"chunk_{i}", f"Search file_{i}.txt for <PATTERN>. Return matching lines or empty string.")
    for i in range(10)
]
results = yield wait(*handles)
hits = [r for r in results if r.strip()]
done("\\n".join(hits) if hits else "No matches.")
```

**Sequential delegation — when order matters:**
```repl
h = delegate("summarize", "Read README.md and summarize this project.")
[summary] = yield wait(h)
h2 = delegate("risks", f"Given: {summary} — what are the main risks?")
[risks] = yield wait(h2)
done(risks)
```
"""


GUARDRAILS_TEXT = """
**Delegate by default.** If the task produces or touches multiple files, you should delegate intelligently. But don't over-decompose small tasks.
- **Validate sub-agent output** — never guess; re-query or do it yourself.
- **Every code path MUST call `done()` or produce output.** No silent `pass`; no `try/except: pass`.
- **Child result format:** tell children to return ONLY raw results (or empty string). No conversational messages.
"""


DEFAULT_BUILDER = (
    PromptBuilder()
    .section("role", ROLE_TEXT, title="Role")
    .section("repl", REPL_TEXT, title="REPL")
    .section("recursion", RECURSION_TEXT, title="Recursion")
    .section("guardrails", GUARDRAILS_TEXT, title="Guardrails")
    .section("session", title="Sessions")
    .section("tools", title="Tools")
    .section("status", title="Status")
)


def make_default_builder() -> PromptBuilder:
    """Return the default builder. Safe to derive from — ``.section()`` returns a copy."""
    return DEFAULT_BUILDER


__all__ = [
    "DEFAULT_BUILDER",
    "EXAMPLES_TEXT",
    "GUARDRAILS_TEXT",
    "REPL_TEXT",
    "RECURSION_TEXT",
    "ROLE_TEXT",
    "make_default_builder",
]
