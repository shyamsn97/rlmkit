"""Baseline (no-delegation) prompt — useful as a control in benchmarks.

Drops every delegation rule, the recursion section, and the multi-agent
examples so the agent doesn't waste turns proposing `rlm_delegate(...)` calls
that the runtime would refuse anyway (the engine filters `rlm_delegate` /
`rlm_wait` out of the tools section when ``max_depth == 0``).

Useful as a control when comparing against the recursive version: same
model, same tools, same task — minus delegation. Lives under
``benchmarks/`` rather than ``rlmflow/prompts/`` because it isn't part
of the core library surface; benchmarks that want it pass it in
explicitly::

    from benchmarks.baseline_prompt import BASELINE_BUILDER

    engine = RLMFlow(
        llm_client=...,
        config=RLMConfig(max_depth=0, ...),
        prompt_builder=BASELINE_BUILDER,
    )
"""

from __future__ import annotations

from rlmflow.prompts.builder import PromptBuilder
from rlmflow.prompts.default import CONTEXT_TEXT

ROLE_BASELINE_TEXT = """
You are an agent with a Python REPL. You solve tasks by writing and executing Python programs.
"""

REPL_BASELINE_TEXT = """
- Every response is exactly one ```repl``` code block. Tools are pre-imported; variables persist across turns.
- Final answer: `done(answer)` exactly once when complete. That string is what the user sees.
- Iterate — run code, observe, decide. Don't one-shot unfamiliar data.
- Verify before `done()` — an empty / zero / surprising result deserves one sanity check first.
- Output is truncated (~12k chars). Slice or summarize — don't `print` huge values.
"""

CORE_EXAMPLES_BASELINE_TEXT = """
**Small task — do it directly:**
```repl
content = read_file("src/config.py")
write_file("src/config.py", content.replace("DEBUG = True", "DEBUG = False"))
done("Set DEBUG = False in src/config.py")
```

**Multi-file output — write, then verify the entry point:**
```repl
write_file("output/app/config.py", "CFG = {'n': 10}\\n")
write_file("output/app/core.py",   "def compute(x: int) -> int:\\n    return x * x\\n")
write_file("output/app/main.py",
    "from core import compute\\n"
    "from config import CFG\\n"
    "if __name__ == '__main__':\\n"
    "    print(compute(CFG['n']))\\n")
main = read_file("output/app/main.py")
assert "from core" in main and "from config" in main, "missing imports"
done("Wrote output/app/{config,core,main}.py")
```

**Long context — chunk and aggregate inline:**
```repl
n = CONTEXT.line_count()
hits = []
for start in range(0, n, 200):
    chunk = CONTEXT.lines(start, min(start + 200, n))
    if "TODO" in chunk:
        hits.append(chunk)
done("\\n---\\n".join(hits) if hits else "No TODOs found.")
```
"""

BASELINE_BUILDER = (
    PromptBuilder()
    .section("role", ROLE_BASELINE_TEXT, title="Role")
    .section("repl", REPL_BASELINE_TEXT, title="REPL")
    .section("tools", title="Tools")
    .section("context", CONTEXT_TEXT, title="Context")
    .section("core_examples", CORE_EXAMPLES_BASELINE_TEXT, title="Examples")
    .section("status", title="Status")
)


__all__ = [
    "BASELINE_BUILDER",
    "CORE_EXAMPLES_BASELINE_TEXT",
    "REPL_BASELINE_TEXT",
    "ROLE_BASELINE_TEXT",
]
