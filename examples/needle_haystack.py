"""Needle in a haystack across many files.

Generates 500 files of random noise in a temp directory. One file
contains a magic string. The agent uses the standard file tools to
find it, delegating the search in parallel across batches.

Usage:
    python examples/needle_haystack.py
    python examples/needle_haystack.py --no-viz
    python examples/needle_haystack.py --viewer
    python examples/needle_haystack.py --docker-image rlmkit:local
"""

from __future__ import annotations

import argparse
import random
import string
import tempfile
from pathlib import Path

from rlmkit.llm import AnthropicClient, OpenAIClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.docker import DockerRuntime
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import RLMState
from rlmkit.tools import FILE_TOOLS


class LoggingRLM(RLM):
    def extract_code(self, text: str, state: RLMState | None = None) -> str | None:
        code = super().extract_code(text, state)
        if code is None or state is None:
            return code
        header = f'print("[{state.agent_id} iter {state.iteration}] executing...")'
        return header + "\n" + code


# ── Generate the haystack ───────────────────────────────────────────

def generate_haystack(
    directory: Path, num_files: int = 500, lines_per_file: int = 200
) -> str:
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Needle in a haystack across many files")
    parser.add_argument("--num-files", type=int, default=500)
    parser.add_argument("--viewer", action="store_true",
                        help="Open the state viewer after finishing")
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

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir).resolve()
        answer = generate_haystack(workspace, num_files=args.num_files)
        print(f"Generated {args.num_files} files in {workspace}")

        def make_runtime():
            if args.docker_image:
                rt = DockerRuntime(
                    args.docker_image,
                    workspace=workspace,
                    mounts={str(workspace): "/workspace"},
                    workdir="/workspace",
                )
            else:
                rt = LocalRuntime(workspace=workspace)
            rt.register_tools(FILE_TOOLS)
            return rt

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

        agent = LoggingRLM(
            llm_client=llm,
            runtime=make_runtime(),
            config=RLMConfig(
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
                session="needle-haystack/context",
            ),
            llm_clients=llm_clients,
            runtime_factory=make_runtime,
        )

        state = agent.start(
            f"There are {args.num_files} text files in the workspace "
            f"(file_0000.txt to file_{args.num_files - 1:04d}.txt). "
            f"Exactly one line across all files says 'The magic number is XXXXXXX'. "
            f"Find it. There are too many files to grep all at once — "
            f"split them into batches and delegate each batch to a sub-agent."
        )

        if args.no_viz:
            states: list[RLMState] = [state]
            while not state.finished:
                state = agent.step(state)
                states.append(state)
                print(state.tree())
        else:
            from rlmkit.utils.viz import live
            states = live(agent, state)
            state = states[-1]

        print(f"\n{'=' * 40}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in (state.result or '')}")

        from rlmkit.utils.viewer import open_viewer, save_trace
        trace_dir = Path("traces/needle_haystack")
        save_trace(states, trace_dir, query=state.query, metadata={"answer": answer})
        print(f"Trace saved to {trace_dir}/")

        if args.viewer:
            open_viewer(states, query=state.query)


if __name__ == "__main__":
    main()
