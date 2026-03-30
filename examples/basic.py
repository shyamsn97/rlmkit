"""Needle-in-a-haystack with recursive chunking.

Generates 1M lines of noise with a single hidden "magic number."
The agent can't fit it all in one context window, so it chunks
the data and delegates each chunk to a sub-agent in parallel.

Tools are added dynamically — the runtime starts bare, and we
register only what the agent needs for this specific task.

Inspired by https://github.com/alexzhang13/rlm-minimal/blob/main/main.py
"""

import random
import tempfile
from pathlib import Path

from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.utils import tool


# ── LLM client (pick one) ───────────────────────────────────────────

try:
    from openai import OpenAI

    class OpenAIClient:
        def __init__(self, model: str = "gpt-4o"):
            self.client = OpenAI()
            self.model = model

        def chat(self, messages: list[dict[str, str]]) -> str:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
            )
            return resp.choices[0].message.content or ""

except ImportError:
    OpenAIClient = None  # type: ignore[misc,assignment]


try:
    import anthropic

    class AnthropicClient:
        def __init__(self, model: str = "claude-sonnet-4-20250514"):
            self.client = anthropic.Anthropic()
            self.model = model

        def chat(self, messages: list[dict[str, str]]) -> str:
            system = ""
            chat_msgs = []
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                else:
                    chat_msgs.append(m)
            resp = self.client.messages.create(
                model=self.model, max_tokens=4096,
                system=system, messages=chat_msgs,
            )
            return resp.content[0].text

except ImportError:
    AnthropicClient = None  # type: ignore[misc,assignment]


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
    if AnthropicClient is not None:
        llm = AnthropicClient()
    elif OpenAIClient is not None:
        llm = OpenAIClient()
    else:
        raise RuntimeError("pip install openai anthropic")

    haystack, answer = generate_haystack(num_lines=1_000_000)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        haystack_path = workspace / "haystack.txt"
        haystack_path.write_text(haystack)
        print(f"Wrote {len(haystack):,} chars to haystack.txt")

        # Bare runtime — just execute + inject, nothing else
        runtime = LocalRuntime(workspace=workspace)

        # Dynamically add only the tools the agent needs for this task
        @tool("Read lines start:end (0-indexed) from a file. Returns the text.")
        def read_lines(path: str, start: int, end: int) -> str:
            p = workspace / path
            lines = p.read_text().splitlines()
            return "\n".join(lines[start:end])

        @tool("Count the number of lines in a file.")
        def line_count(path: str) -> int:
            p = workspace / path
            return len(p.read_text().splitlines())

        @tool("Search for a regex pattern in a line range. Returns matching lines.")
        def search_lines(path: str, pattern: str, start: int = 0, end: int = -1) -> str:
            import re
            p = workspace / path
            lines = p.read_text().splitlines()
            chunk = lines[start:end] if end > 0 else lines[start:]
            regex = re.compile(pattern)
            matches = []
            for i, line in enumerate(chunk, start=start):
                if regex.search(line):
                    matches.append(f"line {i}: {line}")
            return "\n".join(matches) if matches else "(no matches)"

        runtime.register_tool(read_lines)
        runtime.register_tool(line_count)
        runtime.register_tool(search_lines)

        agent = RLM(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15),
        )

        result = agent.run(
            "There is a file called haystack.txt in the workspace. "
            "It has about 1 million lines of random noise, but exactly "
            "one line says 'The magic number is XXXXXXX'. "
            "Find the magic number. The file is too large to read at once — "
            "chunk it and delegate chunks to sub-agents to search in parallel."
        )

        print(f"\n{'='*40}")
        print(f"Agent returned: {result}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in result}")


if __name__ == "__main__":
    main()
