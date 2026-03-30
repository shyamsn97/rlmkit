"""Lightweight, overridable logger for the RLM agent loop."""

from __future__ import annotations


class Logger:
    """Base logger — override any method to customize output.

    Every event in the agent loop calls one of these methods.
    The default implementation is silent (no-op). Use ``PrintLogger``
    for readable stdout streaming, or ``RichLogger`` for rich output.
    """

    def on_run_start(self, agent_id: str, task: str, max_iterations: int) -> None:
        pass

    def on_run_end(self, agent_id: str, result: str, iterations_used: int) -> None:
        pass

    def on_iter_start(self, agent_id: str, iteration: int) -> None:
        pass

    def on_llm_start(self, agent_id: str, iteration: int, num_messages: int) -> None:
        pass

    def on_llm_token(self, agent_id: str, token: str) -> None:
        pass

    def on_llm_end(self, agent_id: str, text: str) -> None:
        pass

    def on_exec_start(self, agent_id: str, code: str) -> None:
        pass

    def on_exec_end(self, agent_id: str, output: str) -> None:
        pass

    def on_no_code_block(self, agent_id: str, iteration: int) -> None:
        pass

    def on_stall(self, agent_id: str, iteration: int) -> None:
        pass

    def on_done(self, agent_id: str, message: str) -> None:
        pass

    def on_delegate(self, agent_id: str, child_id: str, task: str, wait: bool) -> None:
        pass

    def on_delegate_refused(self, agent_id: str, max_depth: int) -> None:
        pass


class PrintLogger(Logger):
    """Streams tokens to stdout with clean execution output."""

    def on_run_start(self, agent_id: str, task: str, max_iterations: int) -> None:
        print("\n" + "━" * 60)
        print(f"  {agent_id}  |  {max_iterations} iters max")
        print("━" * 60)

    def on_run_end(self, agent_id: str, result: str, iterations_used: int) -> None:
        print("\n" + "━" * 60)
        print(f"  {agent_id}  done  ({iterations_used} iters)")
        print("━" * 60 + "\n")

    def on_iter_start(self, agent_id: str, iteration: int) -> None:
        print("\n" + "─" * 60)
        print(f"  [{agent_id}] iter {iteration}")
        print("─" * 60)

    def on_llm_token(self, agent_id: str, token: str) -> None:
        print(token, end="", flush=True)

    def on_llm_end(self, agent_id: str, text: str) -> None:
        print()

    def on_exec_end(self, agent_id: str, output: str) -> None:
        print("\n┌── output ──")
        for line in output.splitlines():
            print(f"│ {line}")
        print("└────────────")

    def on_no_code_block(self, agent_id: str, iteration: int) -> None:
        print(f"  ⚠ no code block (iter {iteration})")

    def on_stall(self, agent_id: str, iteration: int) -> None:
        print(f"  ⚠ stall detected (iter {iteration})")

    def on_done(self, agent_id: str, message: str) -> None:
        print(f"  ✓ done: {message[:120]}" if message else "  ✓ done")

    def on_delegate(self, agent_id: str, child_id: str, task: str, wait: bool) -> None:
        mode = "sync" if wait else "async"
        print(f"  → delegate [{mode}] {child_id}: {task[:80]}")

    def on_delegate_refused(self, agent_id: str, max_depth: int) -> None:
        print(f"  ✗ delegation refused (max depth {max_depth})")
