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
from rlmkit.state import RLMState
from rlmkit.tools import FILE_TOOLS


class LoggingRLM(RLM):
    def extract_code(self, text: str, state: RLMState | None = None) -> str | None:
        code = self.parse_code(text)
        if code is None or state is None:
            return code
        header = f'print("[{state.agent_id} iter {state.iteration}] executing...")'
        return header + "\n" + code


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


# ── Runtime setup ────────────────────────────────────────────────────

def setup_runtime(workspace: Path) -> LocalRuntime:
    rt = LocalRuntime(workspace=workspace)
    rt.register_tools(FILE_TOOLS)
    return rt


# ── Main ─────────────────────────────────────────────────────────────

def main():
    llm = AnthropicClient("claude-opus-4-6")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        answer = generate_haystack(workspace, num_files=500)
        print(f"Generated 500 files in {workspace}")

        runtime = setup_runtime(workspace)
        agent = LoggingRLM(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15, session="context"),
            runtime_factory=lambda: setup_runtime(workspace),
        )

        state = agent.start(
            "There are 500 text files in the workspace (file_0000.txt to file_0499.txt). "
            "Exactly one line across all files says 'The magic number is XXXXXXX'. "
            "Find it. There are too many files to grep all at once — "
            "split them into batches and delegate each batch to a sub-agent."
        )

        if "--no-viz" not in sys.argv:
            from rlmkit.utils.viz import live
            states = live(agent, state)
            state = states[-1]
        else:
            step = 0
            while not state.finished:
                state = agent.step(state)
                step += 1
                print(state.tree())

        print(f"\n{'='*40}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in (state.result or '')}")

        from rlmkit.utils.viewer import save_trace, open_viewer
        trace_dir = Path("traces/needle_haystack")
        save_trace(states, trace_dir, query=state.query, metadata={"answer": answer})
        print(f"Trace saved to {trace_dir}/")

        if "--viewer" in sys.argv:
            open_viewer(states, query=state.query)


if __name__ == "__main__":
    main()
