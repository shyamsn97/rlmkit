"""Showcase: seven things you get for free from rlmkit's immutable state.

Every RLMState is a frozen Pydantic model.  Because states are never mutated
in-place, you can checkpoint them, fork from them, rewind to them, or
surgically edit children mid-run — all with plain JSON serialization.

This script walks through each capability end-to-end:

  1. Step-by-step execution    — call agent.step() in a loop, inspect the
                                  tree after every step.
  2. Checkpointing             — serialize a mid-run state to JSON and
                                  reload it losslessly.
  3. Forking                   — load a checkpoint, tweak the query, and
                                  run a divergent branch in a fresh workspace.
  4. Session persistence       — write the full agent tree to disk with
                                  FileSession, then rebuild it from files.
  5. Time travel               — keep a list[RLMState] history; rewind to
                                  any earlier step by indexing into it.
  6. Intervention              — while the root agent is SUPERVISING, peek
                                  at its children, kill one, and let the
                                  rest finish normally.
  7. Gym-style loop            — treat step() like env.step(): observe the
                                  new state, assign a scalar reward, collect
                                  a trajectory.

Usage:
    python examples/showcase.py
    python examples/showcase.py --no-viz
    python examples/showcase.py --docker-image rlmkit:local
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rlmkit.llm import AnthropicClient, OpenAIClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.docker import DockerRuntime
from rlmkit.runtime.local import LocalRuntime
from rlmkit.session import FileSession
from rlmkit.state import RLMState
from rlmkit.tools import FILE_TOOLS

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


def make_agent(args, workspace: Path, session_dir: str = "context") -> RLM:
    workspace = Path(workspace).resolve()
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

    return RLM(
        llm_client=llm,
        runtime=runtime,
        config=RLMConfig(
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
            session=session_dir,
        ),
        llm_clients=llm_clients,
    )


def run(agent: RLM, state: RLMState, no_viz: bool) -> list[RLMState]:
    if no_viz:
        history = [state]
        step = 0
        while not state.finished:
            state = agent.step(state)
            step += 1
            history.append(state)
            print(f"── step {step} ──")
            print(state.tree())
        return history
    from rlmkit.utils.viz import live
    return live(agent, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        workspace = Path(tmpdir)

        # ── 1. Step-by-step with tree inspection ────────────────────
        banner("1. Step-by-step execution")
        agent = make_agent(args, workspace)
        state = agent.start(
            "Create a simple Python calculator module (calc.py) with add, "
            "subtract, multiply, divide functions, plus a test file "
            "(test_calc.py) that tests them. Delegate the test file to a "
            "sub-agent."
        )
        history = run(agent, state, args.no_viz)
        state = history[-1]

        from rlmkit.utils.trace import save_trace
        trace_dir = workspace / "trace"
        save_trace(history, trace_dir)
        print(f"\n{DIM}Trace saved to {trace_dir}/{RESET}")

        checkpoint_step = 3
        ckpt_path: Path | None = None
        if len(history) > checkpoint_step:
            ckpt_path = workspace / "checkpoint.json"
            history[checkpoint_step].save(ckpt_path)
            print(f"{DIM}(Checkpoint auto-saved at step {checkpoint_step}){RESET}")
        print(f"\n{GREEN}Result:{RESET} {state.result}")

        # ── 2. Checkpoint: save and load ────────────────────────────
        banner("2. Checkpoint round-trip")
        if ckpt_path:
            loaded = RLMState.load(ckpt_path)
            print(
                f"Loaded checkpoint — status: {loaded.status.value}, "
                f"iteration: {loaded.iteration}, children: {len(loaded.children)}"
            )
            print(f"\n{loaded.tree()}")
            size_kb = ckpt_path.stat().st_size / 1024
            print(f"\n{DIM}Saved to {ckpt_path} ({size_kb:.1f} KB){RESET}")
        else:
            print(
                f"{DIM}(Run finished in < {checkpoint_step} steps, "
                f"no checkpoint){RESET}"
            )

        # ── 3. Fork: branch from checkpoint ─────────────────────────
        banner("3. Fork from checkpoint")
        if ckpt_path:
            fork_state = RLMState.load(ckpt_path)
            fork_state = fork_state.update(
                query=(
                    "Create a simple Python calculator module (calc.py) with "
                    "add, subtract, multiply, divide functions, plus a test "
                    "file (test_calc.py). Also add a power() function. "
                    "Delegate the test file to a sub-agent."
                ),
                status=fork_state.status,
            )
            fork_workspace = workspace / "fork"
            fork_workspace.mkdir(exist_ok=True)
            for f in workspace.glob("*.py"):
                (fork_workspace / f.name).write_text(f.read_text())
            fork_agent = make_agent(args, fork_workspace, session_dir="fork_context")
            fork_state = fork_agent.start(fork_state.query)
            fork_state = run(fork_agent, fork_state, args.no_viz)[-1]
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
        print("\nRebuilt tree from session:")
        print(rebuilt.tree())

        # ── 5. Time travel ──────────────────────────────────────────
        banner("5. Time travel (state history)")
        total = len(history)
        show = min(total, 6)
        indices = [0] + list(range(max(1, total - show + 1), total))
        for i in indices:
            s = history[i]
            label = f"step {i}" if i > 0 else "initial"
            print(
                f"{CYAN}{label}{RESET}: status={s.status.value}, "
                f"iter={s.iteration}, children={len(s.children)}"
            )
        print(
            f"\n{DIM}Full history: {total} states. "
            f"Rewind to any point by indexing into the list.{RESET}"
        )

        # ── 6. Intervention: inspect, kill, inject ────────────────
        banner("6. Intervention")
        agent2 = make_agent(args, workspace)
        state2 = agent2.start(
            "Create three files: hello.py, goodbye.py, utils.py. "
            "Delegate each file to a sub-agent."
        )
        intervened = False
        while not state2.finished:
            state2 = agent2.step(state2)
            if state2.status.value == "supervising" and not intervened:
                print(f"{YELLOW}Intervening mid-run!{RESET}")
                print(f"Children: {[c.agent_id for c in state2.children]}")
                new_children = []
                for child in state2.children:
                    if "goodbye" in child.agent_id:
                        print(f"  {YELLOW}Killing {child.agent_id}{RESET}")
                        new_children.append(
                            child.update(
                                status=state2.children[0].status.__class__("finished"),
                                result="(killed by intervention)",
                            )
                        )
                    else:
                        new_children.append(child)
                state2 = state2.update(children=new_children)
                intervened = True
        print(f"\n{GREEN}Result:{RESET} {state2.result}")
        for c in state2.children:
            tag = f"{YELLOW}(killed){RESET}" if "killed" in (c.result or "") else ""
            print(f"  {c.agent_id}: {(c.result or '')[:60]} {tag}")

        # ── 7. Gym-style loop ────────────────────────────────────────
        banner("7. Gym-style loop")
        agent3 = make_agent(args, workspace)
        state3 = agent3.start("Write a haiku about recursion to haiku.txt")
        rewards: list[float] = []
        step = 0
        while not state3.finished:
            state3 = agent3.step(state3)
            step += 1
            r = 0.0
            if state3.finished:
                r = 1.0
            elif state3.status.value == "supervising":
                r = 0.5 * len(state3.children)
            rewards.append(r)
            print(f"  step {step}: status={state3.status.value}, reward={r:.1f}")
        print(f"\n{GREEN}Result:{RESET} {state3.result}")
        print(f"Rewards: {rewards}")
        print(f"Total: {sum(rewards):.1f}")

        banner("Done")


if __name__ == "__main__":
    main()
