"""Showcase: checkpointing, forking, and session persistence.

Demonstrates features that come free from immutable state:

  1. Step-by-step execution with tree inspection
  2. Checkpointing state to JSON mid-run
  3. Forking from a checkpoint to try a different approach
  4. Session persistence: write_tree / from_session round-trip

Usage:
    python showcase.py
    python showcase.py --no-viz
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from rlmkit.llm import AnthropicClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.session import FileSession
from rlmkit.state import RLMState

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def banner(msg: str):
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}\n")


def make_agent(workspace: Path, session_dir: str = "context") -> RLM:
    return RLM(
        llm_client=AnthropicClient("claude-opus-4-6"),
        runtime=LocalRuntime(workspace=workspace),
        config=RLMConfig(max_depth=3, max_iterations=15, session=session_dir),
        llm_clients={
            "fast": {
                "model": AnthropicClient("claude-haiku-4-5"),
                "description": "Fast model for simple sub-tasks",
            },
        },
    )


def main():
    viz = "--no-viz" not in sys.argv

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # ── 1. Step-by-step with tree inspection ────────────────────
        banner("1. Step-by-step execution")

        agent = make_agent(workspace)
        state = agent.start(
            "Create a simple Python calculator module (calc.py) with add, subtract, "
            "multiply, divide functions, plus a test file (test_calc.py) that tests them. "
            "Delegate the test file to a sub-agent."
        )

        history: list[RLMState] = [state]
        checkpoint_step = 3
        checkpoint: str | None = None

        if viz:
            from rlmkit.utils.viz import live
            states = live(agent, state)
            state = states[-1]
            history = states
            if len(states) > checkpoint_step:
                checkpoint = states[checkpoint_step].model_dump_json()
                print(f"\n{DIM}(Checkpoint auto-saved at step {checkpoint_step}){RESET}")
        else:
            step = 0
            while not state.finished:
                state = agent.step(state)
                step += 1
                history.append(state)
                print(f"{DIM}── step {step} ──{RESET}")
                print(state.tree())

                if step == checkpoint_step:
                    checkpoint = state.model_dump_json()
                    print(f"\n{YELLOW}>>> Checkpoint saved at step {step}{RESET}")

        print(f"\n{GREEN}Result:{RESET} {state.result}")

        # ── 2. Checkpoint: save and load ────────────────────────────
        banner("2. Checkpoint round-trip")

        if checkpoint:
            loaded = RLMState.model_validate_json(checkpoint)
            print(f"Loaded checkpoint — status: {loaded.status.value}, "
                  f"iteration: {loaded.iteration}, children: {len(loaded.children)}")
            print(f"\n{loaded.tree()}")

            ckpt_path = workspace / "checkpoint.json"
            ckpt_path.write_text(checkpoint)
            size_kb = ckpt_path.stat().st_size / 1024
            print(f"\n{DIM}Saved to {ckpt_path} ({size_kb:.1f} KB){RESET}")
        else:
            print(f"{DIM}(Run finished in < {checkpoint_step} steps, no checkpoint){RESET}")

        # ── 3. Fork: branch from checkpoint ─────────────────────────
        banner("3. Fork from checkpoint")

        if checkpoint:
            fork_state = RLMState.model_validate_json(checkpoint)
            fork_state = fork_state.update(
                task=(
                    "Create a simple Python calculator module (calc.py) with add, subtract, "
                    "multiply, divide functions, plus a test file (test_calc.py). "
                    "Also add a power() function. Delegate the test file to a sub-agent."
                ),
                status=fork_state.status,
            )

            fork_workspace = workspace / "fork"
            fork_workspace.mkdir(exist_ok=True)

            for f in workspace.glob("*.py"):
                (fork_workspace / f.name).write_text(f.read_text())

            fork_agent = make_agent(fork_workspace, session_dir="fork_context")
            fork_state = fork_agent.start(fork_state.task)

            if viz:
                fork_states = live(fork_agent, fork_state)
                fork_state = fork_states[-1]
            else:
                step = 0
                while not fork_state.finished:
                    fork_state = fork_agent.step(fork_state)
                    step += 1
                    print(f"{DIM}── fork step {step} ──{RESET}")
                    print(fork_state.tree())

            print(f"\n{GREEN}Fork result:{RESET} {fork_state.result}")
        else:
            print(f"{DIM}(Skipped — no checkpoint to fork from){RESET}")

        # ── 4. Session persistence ──────────────────────────────────
        banner("4. Session persistence")

        session_dir = workspace / "session_demo"
        session = FileSession(session_dir)

        session.write_tree(state)
        agents = session.list_agents()
        print(f"Persisted {len(agents)} agent(s): {agents}")

        rebuilt = RLMState.from_session(session, agent_id="root", recursive=True)
        print(f"\nRebuilt tree from session:")
        print(rebuilt.tree())

        # ── 5. Time travel ──────────────────────────────────────────
        banner("5. Time travel (state history)")

        total = len(history)
        show = min(total, 6)
        indices = [0] + list(range(max(1, total - show + 1), total))

        for i in indices:
            s = history[i]
            n_children = len(s.children)
            label = f"step {i}" if i > 0 else "initial"
            print(f"{CYAN}{label}{RESET}: status={s.status.value}, "
                  f"iter={s.iteration}, children={n_children}")

        print(f"\n{DIM}Full history: {total} states. "
              f"Rewind to any point by indexing into the list.{RESET}")

        # ── Done ────────────────────────────────────────────────────
        banner("Done")
        print("Features demonstrated:")
        print("  - step-by-step execution with state.tree()")
        print("  - checkpoint: model_dump_json / model_validate_json")
        print("  - fork: load checkpoint, modify task, run divergent branch")
        print("  - session: write_tree / from_session round-trip")
        print("  - time travel: full history of immutable states")


if __name__ == "__main__":
    main()
