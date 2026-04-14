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
    python showcase.py            # with live terminal visualization
    python showcase.py --no-viz   # headless, prints tree each step
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rlmkit.llm import AnthropicClient
from rlmkit.rlm import RLM, RLMConfig
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


def make_agent(workspace: Path, session_dir: str = "context") -> RLM:
    """Create an RLM agent wired to a local workspace with filesystem tools.

    Args:
        workspace: Directory the agent reads/writes files in.
        session_dir: Sub-directory name (relative to workspace) where
            FileSession stores per-agent message logs.
    """
    runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools(FILE_TOOLS)
    return RLM(
        llm_client=AnthropicClient("claude-opus-4-6"),
        runtime=runtime,
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
        # The most basic loop: call step() repeatedly until the agent
        # finishes.  Each step advances exactly one phase (LLM call,
        # code exec, or one batch of child work), so you see every
        # intermediate state.  We also stash every state in `history`
        # for the time-travel demo later.
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
        # RLMState is a Pydantic model, so model_dump_json() gives you a
        # complete, self-contained snapshot — messages, children, status,
        # everything.  model_validate_json() restores it byte-for-byte.
        # This means you can persist mid-run states to disk and pick up
        # later without any custom serialization.
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
        # Load the same checkpoint, change the query (add a power()
        # function), copy the workspace files into a new directory, and
        # run from scratch with the modified goal.  The original run is
        # untouched — immutable state means forking is just "deserialize
        # + tweak + re-run".
        banner("3. Fork from checkpoint")

        if checkpoint:
            fork_state = RLMState.model_validate_json(checkpoint)
            fork_state = fork_state.update(
                query=(
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
            fork_state = fork_agent.start(fork_state.query)

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
        # FileSession writes each agent's messages as a separate JSON
        # file on disk.  write_tree() walks the whole tree (root +
        # children recursively) and persists everything.
        # from_session() does the inverse — reads messages back and
        # reconstructs the full RLMState tree, so you can visualize or
        # continue a run that was saved earlier.
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
        # Because step() returns a *new* state each time (never mutates),
        # you can just keep them all in a list.  Want to see what the
        # tree looked like three steps ago?  history[n].  Want to re-run
        # from that point with a different strategy?  Pass history[n]
        # back to step().
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

        # ── 6. Intervention: inspect, kill, inject ────────────────
        # When the root agent delegates and enters SUPERVISING, you can
        # inspect state2.children, decide one branch is unwanted, and
        # surgically mark it finished with a synthetic result.  The root
        # agent never knows — it just sees that child as "done" and
        # continues with the remaining children.  This is possible
        # because state.update() returns a new state with your edits
        # while leaving the original untouched.
        banner("6. Intervention")

        agent2 = make_agent(workspace)
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
                        new_children.append(child.update(
                            status=state2.children[0].status.__class__("finished"),
                            result="(killed by intervention)",
                        ))
                    else:
                        new_children.append(child)
                state2 = state2.update(children=new_children)
                intervened = True

        print(f"\n{GREEN}Result:{RESET} {state2.result}")
        for c in state2.children:
            tag = f"{YELLOW}(killed){RESET}" if "killed" in (c.result or "") else ""
            print(f"  {c.agent_id}: {(c.result or '')[:60]} {tag}")

        # ── 7. Gym-style loop ────────────────────────────────────────
        # step() has the same shape as env.step() in OpenAI Gym:
        #   new_state = agent.step(old_state)
        # This makes it natural to assign rewards after each transition
        # and collect (state, action, reward) trajectories.  Here we use
        # a toy reward: +1 for finishing, +0.5 per child spawned during
        # supervision, 0 otherwise.  In practice you'd score based on
        # code correctness, test pass rate, etc.
        banner("7. Gym-style loop")

        agent3 = make_agent(workspace)
        state3 = agent3.start("Write a haiku about recursion to haiku.txt")

        rewards: list[float] = []
        step = 0
        while not state3.finished:
            prev = state3
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

        # ── Done ────────────────────────────────────────────────────
        banner("Done")
        print("Features demonstrated:")
        print("  1. step-by-step execution with state.tree()")
        print("  2. checkpoint: model_dump_json / model_validate_json")
        print("  3. fork: load checkpoint, modify query, run divergent branch")
        print("  4. session: write_tree / from_session round-trip")
        print("  5. time travel: full history of immutable states")
        print("  6. intervention: inspect children, kill branches mid-run")
        print("  7. gym-style loop: step() with reward signals")


if __name__ == "__main__":
    main()
