"""Needle in a haystack across many files.

Generates 500 files of random noise in a temp directory. One file
contains a magic string. The agent uses custom tools registered
with @runtime.tool to find it, delegating the search in parallel.
"""

import random
import string
import tempfile
from pathlib import Path

from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime


# ── LLM client ──────────────────────────────────────────────────────

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

def generate_haystack(directory: Path, num_files: int = 500, lines_per_file: int = 200):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    answer = "".join(random.choices(string.digits, k=7))
    needle_file = random.randint(0, num_files - 1)
    needle_line = random.randint(0, lines_per_file - 1)

    for i in range(num_files):
        lines = []
        for j in range(lines_per_file):
            if i == needle_file and j == needle_line:
                lines.append(f"The magic number is {answer}")
            else:
                n = random.randint(3, 8)
                lines.append(" ".join(random.choice(words) for _ in range(n)))
        (directory / f"file_{i:04d}.txt").write_text("\n".join(lines))

    print(f"Needle in file_{needle_file:04d}.txt line {needle_line}")
    return answer


# ── Runtime with custom tools ───────────────────────────────────────

def setup_runtime(workspace: Path) -> LocalRuntime:
    rt = LocalRuntime(workspace=workspace)

    @rt.tool("List files matching a glob pattern.")
    def list_files(pattern: str = "*.txt") -> list[str]:
        return sorted(str(p.relative_to(rt.workspace)) for p in rt.workspace.glob(pattern))

    @rt.tool("Count files matching a glob pattern.")
    def count_files(pattern: str = "*.txt") -> int:
        return len(list(rt.workspace.glob(pattern)))

    @rt.tool("Grep for a regex across files. Pass a list of filenames to scope the search.")
    def grep(pattern: str, files: list[str] | None = None, max_results: int = 20) -> str:
        import re
        regex = re.compile(pattern)
        if files is None:
            targets = sorted(str(p.relative_to(rt.workspace)) for p in rt.workspace.rglob("*.txt"))
        else:
            targets = files
        matches = []
        for f in targets:
            try:
                for i, line in enumerate((rt.workspace / f).read_text().splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{f}:{i}: {line}")
                        if len(matches) >= max_results:
                            return "\n".join(matches)
            except (OSError, UnicodeDecodeError):
                continue
        return "\n".join(matches) if matches else "(no matches)"

    @rt.tool("Read a file's contents.")
    def read_file(path: str) -> str:
        return (rt.workspace / path).read_text()

    return rt


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if AnthropicClient is not None:
        llm = AnthropicClient()
    elif OpenAIClient is not None:
        llm = OpenAIClient()
    else:
        raise RuntimeError("pip install openai anthropic")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        answer = generate_haystack(workspace, num_files=500)
        print(f"Generated 500 files in {workspace}")

        runtime = setup_runtime(workspace)
        agent = RLM(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15),
            runtime_factory=lambda: setup_runtime(workspace),
        )

        result = agent.run(
            "There are 500 text files in the workspace (file_0000.txt to file_0499.txt). "
            "Exactly one line across all files says 'The magic number is XXXXXXX'. "
            "Find it. There are too many files to grep all at once — "
            "split them into batches and delegate each batch to a sub-agent."
        )

        print(f"\n{'='*40}")
        print(f"Agent returned: {result}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in result}")


if __name__ == "__main__":
    main()
