"""Custom agent — subclass RLM to add tools, change the prompt, and log runs.

This example creates a "code reviewer" agent that analyzes a codebase
by delegating file-by-file review to sub-agents, then combines results.
Shows how to override the prompt, add custom tools, and use lifecycle hooks.
"""

import time
from pathlib import Path

from rlmkit.llm import LLMClient
from rlmkit.logging.rich import RichLogger
from rlmkit.prompts.default import (
    IDENTITY_TEXT,
    RECURSION_TEXT,
    GUARDRAILS_TEXT,
)
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.utils import tool


# ── 1. Custom tools ─────────────────────────────────────────────────

@tool("Count lines, words, and characters in a file. Returns a dict.")
def file_stats(path: str) -> dict:
    text = Path(path).read_text()
    return {
        "lines": len(text.splitlines()),
        "words": len(text.split()),
        "chars": len(text),
    }


@tool("Return the current UTC timestamp as ISO-8601.")
def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── 2. Subclass RLM ─────────────────────────────────────────────────

class CodeReviewer(RLM):
    """An agent that reviews code — custom prompt, extra tools, logging."""

    def __init__(self, review_focus: str = "bugs, style, and performance", **kwargs):
        super().__init__(**kwargs)
        self.review_focus = review_focus
        self.runtime.register_tool(file_stats)
        self.runtime.register_tool(now)

    # -- Custom prompt -----------------------------------------------------

    def build_system_prompt(self) -> str:
        tools = self._tool_summary()
        depth = f"Depth: {self.depth}/{self.config.max_depth}."
        if self.depth >= self.config.max_depth - 1:
            depth += " Near limit — do not delegate."

        return f"""\
## Identity

You are a recursive code reviewer. You analyze codebases by breaking
them into pieces, reviewing each piece, and combining findings.

Focus on: {self.review_focus}

{IDENTITY_TEXT}

## Recursion

{RECURSION_TEXT}

## Review Process

For small files (< 100 lines), review directly and report findings.
For large files or directories with many files, delegate per-file
reviews to sub-agents and combine their results into a summary.

Output format for each file:
- **File**: path
- **Issues**: numbered list of findings
- **Severity**: info / warning / error for each

## Guardrails

{GUARDRAILS_TEXT}

## Current State

{depth}

## Tools

{tools}
"""

    # -- Lifecycle hooks for timing ----------------------------------------

    def on_run_start(self, task: str) -> None:
        self._t0 = time.monotonic()
        print(f"[{self.agent_id}] start: {task[:80]}")

    def on_run_end(self, task: str, result: str) -> None:
        dt = time.monotonic() - self._t0
        print(f"[{self.agent_id}] done in {dt:.1f}s")

    def on_iteration_start(self, task: str, iteration: int) -> None:
        print(f"[{self.agent_id}]   iter {iteration}")

    # -- Cap child iterations ----------------------------------------------

    def create_child(self, agent_id, task, *, max_iterations=None):
        child = super().create_child(agent_id, task, max_iterations=max_iterations)
        child.config.max_iterations = min(child.config.max_iterations, 8)
        return child


# ── 3. Run it ───────────────────────────────────────────────────────

def main():
    try:
        import anthropic

        class Client(LLMClient):
            def __init__(self):
                self.client = anthropic.Anthropic()
            def chat(self, messages):
                system = ""
                chat_msgs = []
                for m in messages:
                    if m["role"] == "system":
                        system = m["content"]
                    else:
                        chat_msgs.append(m)
                return self.client.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=4096,
                    system=system, messages=chat_msgs,
                ).content[0].text

    except ImportError:
        raise RuntimeError("pip install anthropic")

    workspace = Path(".")
    runtime = LocalRuntime(workspace=workspace)

    agent = CodeReviewer(
        llm_client=Client(),
        runtime=runtime,
        config=RLMConfig(max_depth=3, max_iterations=15),
        logger=RichLogger(),
        review_focus="bugs, missing error handling, and type safety",
    )

    result = agent.run(
        "Review all Python files under rlmkit/. For each file, identify "
        "bugs, missing error handling, and type safety issues. Delegate "
        "one sub-agent per file for files over 50 lines. Combine all "
        "findings into a final summary."
    )

    print(f"\n{'='*60}")
    print(result)


if __name__ == "__main__":
    main()
