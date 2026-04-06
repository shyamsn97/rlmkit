"""Needle-in-a-haystack with recursive chunking.

Generates 1M lines of noise with a single hidden "magic number."
The agent can't fit it all in one context window, so it chunks
the data and delegates each chunk to a sub-agent in parallel.

Tools are added dynamically — the runtime starts bare, and we
register only what the agent needs for this specific task.

Uses the step-based API to show full state after each step.

Inspired by https://github.com/alexzhang13/rlm-minimal/blob/main/main.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

from rlmkit.llm import OpenAIClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime

from utils import StepLogger


# ── Generate the haystack ───────────────────────────────────────────

def generate_haystack(num_lines: int = 1_000_000) -> tuple[str, str]:
    """Return (haystack_text, answer). The answer is hidden on one line."""
    words = ["blah", "random", "text", "data", "content", "noise", "sample", "filler"]
    answer = str(random.randint(1_000_000, 9_999_999))

    lines = []
    for _ in range(num_lines):
        n = random.randint(3, 8)
        lines.append(" ".join(random.choice(words) for _ in range(n)))

    position = random.randint(num_lines // 3, 2 * num_lines // 3)
    lines[position] = f"The magic number is {answer}"
    print(f"Needle at line {position:,} of {num_lines:,}")

    return "\n".join(lines), answer


# ── Main ─────────────────────────────────────────────────────────────

def main():
    strong = OpenAIClient("gpt-5")
    fast = OpenAIClient("gpt-5-mini")

    logger = StepLogger(Path(__file__).parent / "basic_log.md")
    haystack, answer = generate_haystack(num_lines=1_000_000)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        haystack_path = workspace / "haystack.txt"
        haystack_path.write_text(haystack)
        num_lines = len(haystack.splitlines())
        print(f"Wrote {num_lines:,} lines to haystack.txt")

        runtime = LocalRuntime(workspace=workspace)
        runtime.tools.pop("grep", None)

        agent = RLM(
            llm_client=strong,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15, session="context"),
            llm_clients={
                "fast": {"model": fast, "description": "Cheap model for simple search tasks"},
            },
        )

        state = agent.start(
            f"haystack.txt has {num_lines:,} lines of noise with one line that says "
            f"'The magic number is <N>'. Find the number. The file is too large to "
            f"read at once — chunk it and delegate search to sub-agents. Return just the number."
        )

        if "--viz" in sys.argv:
            from viz import live
            states = live(agent, state)
        else:
            step = 0
            states = []
            while not state.finished:
                state = agent.step(state)
                step += 1
                states.append(state)
                logger.log(step, state)

        final = states[-1] if states else state
        print(f"\n{'='*40}")
        print(f"Agent answer:   {final.result}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in (final.result or '')}")
        print(f"Trace saved to {logger.path}")


if __name__ == "__main__":
    main()
