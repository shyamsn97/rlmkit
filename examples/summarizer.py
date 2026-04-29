"""Recursive summarization of a long document.

Generates a large document (~10k lines of synthetic meeting notes),
then uses an RLMFlow to summarize it recursively: chunk the document,
delegate each chunk's summary to a sub-agent, then combine the
partial summaries into a final one.

Shows:
- Custom prompt builder (swap role, add a summarization section)
- Custom tools registered on the runtime
- ``yield wait()`` for parallel delegation

Usage:
    python examples/summarizer.py
    python examples/summarizer.py --lines 5000
    python examples/summarizer.py --no-viz
    python examples/summarizer.py --docker-image rlmkit:local
"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

from rlmkit.llm import AnthropicClient, OpenAIClient
from rlmkit.prompts import DEFAULT_BUILDER
from rlmkit.prompts.default import ROLE_TEXT
from rlmkit.rlm import RLMConfig, RLMFlow
from rlmkit.runtime.docker import DockerRuntime
from rlmkit.runtime.local import LocalRuntime
from rlmkit.tools import FILE_TOOLS


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
    lines: list[str] = []
    meeting_num = 1
    i = 0
    while i < num_lines:
        topic = random.choice(TOPICS)
        lines.append(f"=== Meeting #{meeting_num}: {topic} ===")
        lines.append(f"Date: 2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
        lines.append(
            f"Attendees: {', '.join(random.sample(PEOPLE, random.randint(3, 7)))}"
        )
        lines.append("")
        i += 4

        for _ in range(random.randint(8, 25)):
            if i >= num_lines:
                break
            lines.append(
                f"- {random.choice(PEOPLE)} {random.choice(ACTIONS)} "
                f"{topic.lower()}. {random.choice(DETAILS)}"
            )
            i += 1
            if random.random() < 0.3:
                lines.append(
                    f"  Follow-up: {random.choice(PEOPLE)} to coordinate with "
                    f"{random.choice(PEOPLE)}."
                )
                i += 1

        lines.append("")
        lines.append(f"Action items from meeting #{meeting_num}:")
        for j in range(random.randint(2, 6)):
            if i >= num_lines:
                break
            lines.append(
                f"  {j + 1}. [{random.choice(PEOPLE)}] "
                f"{random.choice(ACTIONS)} {topic.lower()}"
            )
            i += 1
        lines.extend(["", "---", ""])
        i += 3
        meeting_num += 1

    return "\n".join(lines[:num_lines])


# ── Prompt ──────────────────────────────────────────────────────────

SUMMARIZATION_SECTION = """\
When summarizing, follow these rules:
- **Preserve key decisions and action items.** Names, dates, and commitments matter.
- **Drop filler and repetition.** Meeting notes are verbose — compress aggressively.
- **Structure the output.** Use bullet points grouped by topic.
- When combining sub-summaries, merge related topics and remove duplicates.
- The final summary should be readable by someone who missed all the meetings."""


def build_prompt_builder():
    return (
        DEFAULT_BUILDER.section(
            "role",
            "You are a recursive document summarizer. You break large documents "
            "into chunks, summarize each chunk via sub-agents, and combine the "
            f"results into a coherent final summary.\n\n{ROLE_TEXT}",
            title="Role",
        )
        .section(
            "summarization",
            SUMMARIZATION_SECTION,
            title="Summarization Rules",
            after="recursion",
        )
    )


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recursive document summarizer")
    parser.add_argument("--lines", type=int, default=10_000)
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--fast-model", default=None)
    parser.add_argument("--docker-image", default=None,
                        help="If set, run agent code inside this Docker image (e.g. rlmkit:local).")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    else:
        print(">>> LOCAL RUNTIME")

    doc = generate_document(num_lines=args.lines)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir).resolve()
        (workspace / "meeting_notes.txt").write_text(doc)
        actual_lines = len(doc.splitlines())
        print(f"Generated {actual_lines:,} lines of meeting notes ({len(doc):,} chars)")

        if args.docker_image:
            runtime = DockerRuntime(
                args.docker_image,
                workspace=workspace,
                mounts={str(workspace): "/workspace"},
                workdir="/workspace",
            )
        else:
            runtime = LocalRuntime(workspace=workspace)
        runtime.register_tools(FILE_TOOLS)

        llm = (
            AnthropicClient(args.model)
            if args.model.startswith("claude")
            else OpenAIClient(args.model)
        )
        llm_clients = None
        if args.fast_model:
            fast = (
                AnthropicClient(args.fast_model)
                if args.fast_model.startswith("claude")
                else OpenAIClient(args.fast_model)
            )
            llm_clients = {
                "fast": {"model": fast, "description": "Cheaper model for small sub-tasks."},
            }

        agent = RLMFlow(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=args.max_depth, max_iterations=args.max_iterations),
            llm_clients=llm_clients,
            prompt_builder=build_prompt_builder(),
        )

        state = agent.start(
            f"Summarize meeting_notes.txt ({actual_lines:,} lines). It contains "
            f"detailed meeting notes from many meetings. Extract key decisions, "
            f"action items, and themes. The file is too long to read at once — "
            f"chunk it and delegate."
        )
        trace = [state]

        if args.no_viz:
            while not state.finished:
                state = agent.step(state)
                trace.append(state)
                print(state.tree())
        else:
            from rlmkit.utils.viz import live
            for s in live(agent, state):
                state = s
                trace.append(state)

        print(f"\n{'=' * 60}")
        print(f"SUMMARY ({len((state.result or '').splitlines())} lines):\n")
        print(state.result or "(no result)")

        from rlmkit.utils.trace import save_trace
        trace_dir = Path("traces/summarizer")
        save_trace(trace, trace_dir, metadata={"lines": actual_lines})
        print(f"\nTrace saved to {trace_dir}/")


if __name__ == "__main__":
    main()
