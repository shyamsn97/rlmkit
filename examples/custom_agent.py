"""Custom agent — subclass RLM and RLMState to add tools and state.

This example creates a "code reviewer" agent that tracks review findings
as custom state. Shows how to:
- Subclass RLMState to add fields
- Subclass RLM to set state_cls, override prompt and step hooks
- Use the step-based API with custom state
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rlmkit.llm import OpenAIClient
from rlmkit.prompts import make_default_builder
from rlmkit.prompts.default import ROLE_TEXT
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import CodeExec, RLMState
from rlmkit.utils import tool

from utils import StepLogger, DIM, RESET


# ── 1. Custom state ─────────────────────────────────────────────────

class ReviewState(RLMState):
    """Extends RLMState with a list of findings accumulated across steps."""
    findings: list[str] = []


# ── 2. Custom tools ─────────────────────────────────────────────────

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


# ── 3. Subclass RLM ─────────────────────────────────────────────────

class CodeReviewer(RLM):
    """An agent that reviews code — custom prompt, extra tools, custom state."""

    state_cls = ReviewState

    def __init__(self, review_focus: str = "bugs, style, and performance", **kwargs):
        super().__init__(**kwargs)
        self.review_focus = review_focus
        self.runtime.register_tool(file_stats)
        self.runtime.register_tool(now)

        REVIEW_PROCESS = """\
For small files (< 100 lines), review directly and report findings.
For large files or directories with many files, delegate per-file
reviews to sub-agents and combine their results into a summary.

Output format for each file:
- **File**: path
- **Issues**: numbered list of findings
- **Severity**: info / warning / error for each"""

        self.prompt_builder = (
            make_default_builder()
            .section(
                "role",
                f"You are a recursive code reviewer. You analyze codebases by "
                f"breaking them into pieces, reviewing each piece, and combining "
                f"findings.\n\nFocus on: {self.review_focus}\n\n{ROLE_TEXT}",
                title="Role",
            )
            .section("review_process", REVIEW_PROCESS, title="Review Process", after="recursion")
            .remove("examples")
        )

    def make_state(self, **fields) -> ReviewState:
        return ReviewState(**fields, findings=[])

    def step_exec(self, state: ReviewState) -> ReviewState:
        """After each exec, extract any findings from the output."""
        new_state = super().step_exec(state)
        if isinstance(new_state.event, CodeExec) and "issue" in new_state.event.output.lower():
            return new_state.update(
                findings=state.findings + [new_state.event.output],
            )
        return new_state

    def create_child(self, agent_id, *, max_iterations=None, llm_client=None):
        child = super().create_child(agent_id, max_iterations=max_iterations, llm_client=llm_client)
        child.config.max_iterations = min(child.config.max_iterations, 8)
        return child


# ── 4. Run it ───────────────────────────────────────────────────────

def main():
    llm = OpenAIClient()

    logger = StepLogger(Path(__file__).parent / "custom_agent_log.md")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        runtime = LocalRuntime(workspace=workspace)

        agent = CodeReviewer(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15, session="context"),
            review_focus="bugs, missing error handling, and type safety",
        )

        def _progress(aid, child_state, done, total):
            status = child_state.status.value
            print(f"  {DIM}[{done}/{total}]{RESET} {aid} → {status}")
            with open(logger.path, "a") as f:
                f.write(f"- [{done}/{total}] `{aid}` → {status}\n")
            sys.stdout.flush()

        agent.on_child_stepped = _progress

        state = agent.start(
            "Review all Python files under rlmkit/. For each file, identify "
            "bugs, missing error handling, and type safety issues. Delegate "
            "one sub-agent per file for files over 50 lines. Combine all "
            "findings into a final summary."
        )

        step = 0
        while not state.finished:
            state = agent.step(state)
            step += 1
            extra = ""
            if isinstance(state, ReviewState) and state.findings:
                extra = f"{DIM}({len(state.findings)} findings){RESET}"
            logger.log(step, state, extra_term=extra)

        print(f"\n{'='*60}")
        print(state.result)
        print(f"Trace saved to {logger.path}")


if __name__ == "__main__":
    main()
