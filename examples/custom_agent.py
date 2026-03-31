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

from rlmkit.llm import LLMClient
from rlmkit.prompts.default import IDENTITY_TEXT, RECURSION_TEXT, GUARDRAILS_TEXT
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import CodeExec, RLMState
from rlmkit.utils import tool


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

    def make_state(self, **fields) -> ReviewState:
        return ReviewState(**fields, findings=[])

    def build_system_prompt(self, state: RLMState) -> str:
        tools = self._tool_summary()
        depth = state.config.get("depth", 0)
        max_depth = state.config.get("max_depth", self.config.max_depth)
        depth_note = f"Depth: {depth}/{max_depth}."
        if depth >= max_depth - 1:
            depth_note += " Near limit — do not delegate."

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

{depth_note}

## Tools

{tools}
"""

    def step_exec(self, state: ReviewState) -> ReviewState:
        """After each exec, extract any findings from the output."""
        new_state = super().step_exec(state)
        if isinstance(new_state.event, CodeExec) and "issue" in new_state.event.output.lower():
            return new_state.update(
                findings=state.findings + [new_state.event.output],
            )
        return new_state

    def create_child(self, agent_id, task, *, max_iterations=None):
        child = super().create_child(agent_id, task, max_iterations=max_iterations)
        child.config.max_iterations = min(child.config.max_iterations, 8)
        return child


# ── 4. Step logger ──────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

LOG_PATH = Path(__file__).parent / "custom_agent_log.md"

def _fmt_event(ev, indent=0):
    """Format a single StepEvent for terminal + markdown. Returns (term_lines, md_lines)."""
    from rlmkit.state import LLMReply, CodeExec, ChildStep, NoCodeBlock
    pad = "  " * indent
    term, md = [], []
    tag = f"{CYAN}{ev.agent_id}{RESET}"
    md_prefix = "  " * indent

    if isinstance(ev, LLMReply):
        term.append(f"{pad}{tag} {BOLD}LLM reply{RESET} (iter {ev.iteration})")
        md.append(f"{md_prefix}**`{ev.agent_id}`** LLM reply (iter {ev.iteration})")
        if ev.code:
            term.append(f"{pad}  {YELLOW}```repl{RESET}")
            for line in ev.code.splitlines():
                term.append(f"{pad}  {line}")
            term.append(f"{pad}  {YELLOW}```{RESET}")
            md.append(f"```repl\n{ev.code}\n```")
        else:
            preview = ev.text[:200].replace("\n", " ")
            term.append(f"{pad}  {DIM}{preview}{'...' if len(ev.text) > 200 else ''}{RESET}")
            md.append(f"{md_prefix}> {preview}{'...' if len(ev.text) > 200 else ''}")

    elif isinstance(ev, CodeExec):
        label = f"{RED}suspended{RESET}" if ev.suspended else f"{GREEN}done{RESET}"
        term.append(f"{pad}{tag} {BOLD}exec{RESET} → {label}")
        md.append(f"{md_prefix}**`{ev.agent_id}`** exec → {'suspended' if ev.suspended else 'done'}")
        out = ev.output.strip()
        if out and out != "(no output)":
            for line in out.splitlines()[:10]:
                term.append(f"{pad}  {DIM}│{RESET} {line}")
            if len(out.splitlines()) > 10:
                term.append(f"{pad}  {DIM}│ ...({len(out.splitlines())} lines){RESET}")
            md.append(f"```\n{out[:2000]}\n```")

    elif isinstance(ev, NoCodeBlock):
        term.append(f"{pad}{tag} {YELLOW}no code block{RESET}")
        md.append(f"{md_prefix}**`{ev.agent_id}`** no code block")

    elif isinstance(ev, ChildStep):
        status = f"{GREEN}all done{RESET}" if ev.all_done else "in progress"
        term.append(f"{pad}{tag} {BOLD}children{RESET} {status}")
        md.append(f"{md_prefix}**`{ev.agent_id}`** children — {'all done' if ev.all_done else 'in progress'}")
        for ce in ev.child_events:
            t, m = _fmt_event(ce, indent + 1)
            term.extend(t)
            md.extend(m)

    return term, md

def log_step(step: int, state: RLMState):
    import sys
    from rlmkit.state import ChildStep
    ev = state.event
    if not ev:
        return

    term_lines, md_lines = _fmt_event(ev)

    findings = ""
    findings_md = ""
    if isinstance(state, ReviewState) and state.findings:
        findings = f" {DIM}({len(state.findings)} findings){RESET}"
        findings_md = f" ({len(state.findings)} findings)"
        if term_lines:
            term_lines[0] += findings
        md_lines.insert(0, f"_{len(state.findings)} findings so far_")

    header = f"{DIM}── step {step} ──{RESET}"
    print(f"\n{header}")
    for line in term_lines:
        print(line)

    if isinstance(ev, ChildStep) and state.children:
        n_done = sum(1 for c in state.children if c.finished)
        total = len(state.children)
        print(f"  {DIM}[{n_done}/{total} done]{RESET}")
        for cs in state.children:
            if cs.finished and cs.result:
                preview = cs.result[:120].replace("\n", " ")
                print(f"  {GREEN}✓{RESET} {CYAN}{cs.agent_id}{RESET} → {preview}")
                md_lines.append(f"- ✓ `{cs.agent_id}` → {cs.result[:200]}")
            elif cs.finished:
                print(f"  {GREEN}✓{RESET} {DIM}{cs.agent_id}{RESET}")
            else:
                print(f"  {YELLOW}·{RESET} {DIM}{cs.agent_id} [{cs.status.value}]{RESET}")

    md = [f"### Step {step}"] + md_lines

    if state.finished and state.result:
        print(f"\n{GREEN}{BOLD}result:{RESET} {state.result[:200]}")
        md.append(f"\n---\n**Result:** {state.result}")

    with open(LOG_PATH, "a") as f:
        f.write("\n".join(md) + "\n\n")
    sys.stdout.flush()


# ── 5. Run it ───────────────────────────────────────────────────────

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

    LOG_PATH.write_text("# custom_agent.py trace\n\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        runtime = LocalRuntime(workspace=workspace)

        agent = CodeReviewer(
            llm_client=Client(),
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15, context_path="context.md"),
            review_focus="bugs, missing error handling, and type safety",
        )

        def _progress(aid, child_state, done, total):
            status = child_state.status.value
            msg = f"  {DIM}[{done}/{total}]{RESET} {aid} → {status}"
            print(msg)
            with open(LOG_PATH, "a") as f:
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
            log_step(step, state)

        print(f"\n{'='*60}")
        print(state.result)

        print(f"Trace saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
