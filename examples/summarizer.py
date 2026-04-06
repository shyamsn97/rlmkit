"""Recursive summarization of a long document.

Generates a large document (~10k lines of synthetic meeting notes),
then uses an RLM to summarize it recursively: chunk the document,
delegate each chunk's summary to a sub-agent, then combine the
partial summaries into a final one.

Shows:
- Custom prompt builder (swap role, add a summarization section)
- Custom tools registered on the runtime
- The step-based API with full logging
- `yield wait()` for parallel delegation

Usage:
    python summarizer.py
    python summarizer.py --lines 5000   # smaller doc
    python summarizer.py --viz           # live terminal UI
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

from rlmkit.llm import OpenAIClient
from rlmkit.prompts import make_default_builder
from rlmkit.prompts.default import ROLE_TEXT
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import RLMState
from rlmkit.utils import tool


# ── Generate a long document ────────────────────────────────────────

TOPICS = [
    "Q3 revenue targets", "hiring pipeline", "infrastructure costs",
    "customer churn", "product roadmap", "security audit findings",
    "onboarding flow redesign", "API rate limiting", "database migration",
    "mobile app performance", "competitor analysis", "support ticket backlog",
    "compliance requirements", "cloud spend optimization", "team retro action items",
    "launch timeline", "partnership discussions", "user research findings",
    "A/B test results", "documentation debt",
]

PEOPLE = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Irene", "Jack", "Karen", "Leo", "Mia", "Nate", "Olivia", "Pat",
]

ACTIONS = [
    "proposed", "suggested", "raised a concern about", "committed to",
    "pushed back on", "agreed to investigate", "presented data on",
    "volunteered to lead", "flagged a blocker for", "requested more info on",
    "shared an update on", "escalated", "approved the plan for",
    "asked for a timeline on", "offered to help with", "reported progress on",
]

DETAILS = [
    "Need to finalize by end of week.",
    "Will follow up with the team.",
    "Blocked until the vendor responds.",
    "Requires sign-off from legal.",
    "Budget was already approved.",
    "We need to hire two more engineers for this.",
    "The prototype is ready for review.",
    "Latency numbers are worse than expected.",
    "Customer feedback has been overwhelmingly positive.",
    "This is lower priority than originally thought.",
    "Deadline moved to next quarter.",
    "We should revisit this after the reorg.",
    "Metrics look promising but sample size is small.",
    "The current approach won't scale past 10k users.",
    "Already handled in the last sprint.",
    "There's a dependency on the platform team.",
]


def generate_document(num_lines: int = 10_000) -> str:
    """Generate synthetic meeting notes with many topics and action items."""
    lines = []
    meeting_num = 1
    i = 0
    while i < num_lines:
        topic = random.choice(TOPICS)
        lines.append(f"=== Meeting #{meeting_num}: {topic} ===")
        lines.append(f"Date: 2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
        lines.append(f"Attendees: {', '.join(random.sample(PEOPLE, random.randint(3, 7)))}")
        lines.append("")
        i += 4

        num_items = random.randint(8, 25)
        for _ in range(num_items):
            if i >= num_lines:
                break
            person = random.choice(PEOPLE)
            action = random.choice(ACTIONS)
            detail = random.choice(DETAILS)
            lines.append(f"- {person} {action} {topic.lower()}. {detail}")
            i += 1
            if random.random() < 0.3:
                lines.append(f"  Follow-up: {random.choice(PEOPLE)} to coordinate with {random.choice(PEOPLE)}.")
                i += 1

        lines.append("")
        lines.append(f"Action items from meeting #{meeting_num}:")
        n_actions = random.randint(2, 6)
        for j in range(n_actions):
            if i >= num_lines:
                break
            lines.append(f"  {j+1}. [{random.choice(PEOPLE)}] {random.choice(ACTIONS)} {topic.lower()}")
            i += 1
        lines.append("")
        lines.append("---")
        lines.append("")
        i += 3
        meeting_num += 1

    return "\n".join(lines[:num_lines])


# ── Step logger ─────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

LOG_PATH = Path(__file__).parent / "summarizer_log.md"


def _fmt_event(ev, indent=0):
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
        if ev.agent_finished:
            status = f"{GREEN}finished{RESET}"
            md_status = "agent finished"
        elif ev.all_done:
            status = f"{GREEN}all done{RESET}"
            md_status = "all done"
        else:
            status = "in progress"
            md_status = "in progress"
        term.append(f"{pad}{tag} {BOLD}children{RESET} {status}")
        md.append(f"{md_prefix}**`{ev.agent_id}`** children — {md_status}")
        if ev.exec_output and ev.exec_output.strip():
            for line in ev.exec_output.strip().splitlines()[:10]:
                term.append(f"{pad}  {DIM}│{RESET} {line}")
            md.append(f"```\n{ev.exec_output.strip()[:2000]}\n```")
        for ce in ev.child_events:
            t, m = _fmt_event(ce, indent + 1)
            term.extend(t)
            md.extend(m)

    return term, md


def log_step(step: int, state: RLMState):
    ev = state.event
    if not ev:
        return

    term_lines, md_lines = _fmt_event(ev)
    header = f"{DIM}── step {step} ──{RESET}"
    print(f"\n{header}")
    for line in term_lines:
        print(line)

    from rlmkit.state import ChildStep
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
        print(f"\n{GREEN}{BOLD}result:{RESET} {state.result[:500]}")
        md.append(f"\n---\n**Result:** {state.result}")

    with open(LOG_PATH, "a") as f:
        f.write("\n".join(md) + "\n\n")
    sys.stdout.flush()


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recursive document summarizer")
    parser.add_argument("--lines", type=int, default=10_000, help="Number of lines to generate")
    parser.add_argument("--viz", action="store_true", help="Live terminal visualization")
    args = parser.parse_args()

    llm = OpenAIClient()

    LOG_PATH.write_text("# summarizer.py trace\n\n")
    doc = generate_document(num_lines=args.lines)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        doc_path = workspace / "meeting_notes.txt"
        doc_path.write_text(doc)
        actual_lines = len(doc.splitlines())
        print(f"Generated {actual_lines:,} lines of meeting notes ({len(doc):,} chars)")

        runtime = LocalRuntime(workspace=workspace)

        @tool("Read lines start:end (0-indexed, exclusive) from a file. Returns the text.")
        def read_lines(path: str, start: int, end: int) -> str:
            p = workspace / path
            lines = p.read_text().splitlines()
            return "\n".join(lines[start:end])

        @tool("Count the number of lines in a file.")
        def line_count(path: str) -> int:
            p = workspace / path
            return len(p.read_text().splitlines())

        runtime.register_tool(read_lines)
        runtime.register_tool(line_count)

        SUMMARIZATION_SECTION = """\
When summarizing, follow these rules:
- **Preserve key decisions and action items.** Names, dates, and commitments matter.
- **Drop filler and repetition.** Meeting notes are verbose — compress aggressively.
- **Structure the output.** Use bullet points grouped by topic.
- When combining sub-summaries, merge related topics and remove duplicates.
- The final summary should be readable by someone who missed all the meetings."""

        builder = (
            make_default_builder()
            .section(
                "role",
                "You are a recursive document summarizer. You break large documents "
                "into chunks, summarize each chunk via sub-agents, and combine the "
                f"results into a coherent final summary.\n\n{ROLE_TEXT}",
                title="Role",
            )
            .section("summarization", SUMMARIZATION_SECTION, title="Summarization Rules", after="recursion")
        )

        agent = RLM(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15, session="context"),
            prompt_builder=builder,
        )

        state = agent.start(
            f"Summarize meeting_notes.txt ({actual_lines:,} lines). It contains detailed meeting "
            f"notes from many meetings. Extract key decisions, action items, and themes. "
            f"The file is too long to read at once — chunk it and delegate."
        )

        if args.viz:
            from viz import live
            states = live(agent, state)
        else:
            step = 0
            states = []
            while not state.finished:
                state = agent.step(state)
                step += 1
                states.append(state)
                log_step(step, state)

        final = states[-1] if states else state
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len((final.result or '').splitlines())} lines):\n")
        print(final.result or "(no result)")
        print(f"\nTrace saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
