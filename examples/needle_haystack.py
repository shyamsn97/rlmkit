"""Needle in a haystack across many files.

Generates 500 files of random noise in a temp directory. One file
contains a magic string. The agent uses custom tools registered
with @runtime.tool to find it, delegating the search in parallel.

Uses the step-based API for full observability.
"""

from __future__ import annotations

import random
import string
import sys
import tempfile
from pathlib import Path

from rlmkit.llm import AnthropicClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime

from utils import StepLogger, DIM, RESET


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
        return "\n".join(matches)

    @rt.tool("Read a file's contents.")
    def read_file(path: str) -> str:
        return (rt.workspace / path).read_text()

    return rt


# ── Main ─────────────────────────────────────────────────────────────

def main():
    llm = AnthropicClient("claude-opus-4-6")

    logger = StepLogger(Path(__file__).parent / "needle_haystack_log.md")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        answer = generate_haystack(workspace, num_files=500)
        print(f"Generated 500 files in {workspace}")

        runtime = setup_runtime(workspace)
        agent = RLM(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15, session="context"),
            runtime_factory=lambda: setup_runtime(workspace),
        )

        def _progress(aid, child_state, done, total):
            status = child_state.status.value
            print(f"  {DIM}[{done}/{total}]{RESET} {aid} → {status}")
            with open(logger.path, "a") as f:
                f.write(f"- [{done}/{total}] `{aid}` → {status}\n")
            sys.stdout.flush()

        agent.on_child_stepped = _progress

        state = agent.start(
            "There are 500 text files in the workspace (file_0000.txt to file_0499.txt). "
            "Exactly one line across all files says 'The magic number is XXXXXXX'. "
            "Find it. There are too many files to grep all at once — "
            "split them into batches and delegate each batch to a sub-agent."
        )

        if "--viz" in sys.argv:
            from rlmkit.utils.viz import live
            states = live(agent, state)
            state = states[-1]
        else:
            step = 0
            while not state.finished:
                state = agent.step(state)
                step += 1
                logger.log(step, state)

        print(f"\n{'='*40}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in (state.result or '')}")
        print(f"Trace saved to {logger.path}")


if __name__ == "__main__":
    main()
