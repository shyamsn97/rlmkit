"""Core RLM agent loop.

Subclass `RLM` and override any of these to customize behavior:
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import count
from typing import Any

from .llm import LLMClient
from .logging import Logger
from .prompts.default import TOOLS_TEXT, make_default_builder
from .runtime import Runtime
from .utils import find_code_blocks, tool


class DelegateHandle:
    def __init__(self, agent_id: str, task: str, future: Future[str]) -> None:
        self.agent_id = agent_id
        self.task = task
        self._future = future

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> str:
        return self._future.result(timeout=timeout)


class ThreadPool:
    """Shared thread pool for all agents in a recursive tree.

    Every RLM in the tree submits to the same pool, so `max_workers`
    is the global concurrency limit across all depths. Tasks queue up
    when all workers are busy.
    """

    def __init__(self, max_workers: int = 8) -> None:
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, agent_id: str, task: str, fn: Callable[[], str]) -> DelegateHandle:
        return DelegateHandle(agent_id, task, self._executor.submit(fn))

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


@dataclass
class RLMConfig:
    """All tuning knobs for an RLM agent."""

    max_depth: int = 5
    max_iterations: int = 30
    max_output_length: int = 12_000
    replay_last_n_turns: int = 0
    single_block: bool = True
    context_path: str | None = None
    system_prompt: str | None = None


class RLM:
    """Recursive Language Model agent.

    Override `build_system_prompt()` to customize the prompt.
    Override any other method to change behavior.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        runtime: Runtime,
        config: RLMConfig | None = None,
        logger: Logger | None = None,
        agent_id: str = "root",
        depth: int = 0,
        pool: ThreadPool | None = None,
        runtime_factory: Callable[[], Runtime] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.runtime = runtime
        self.config = config or RLMConfig()
        self.logger = logger or Logger()
        self.agent_id = agent_id
        self.depth = depth
        self.pool = pool or ThreadPool()
        self.runtime_factory = runtime_factory or self._default_runtime_factory
        self._delegate_ids = count(1)
        self._done = False
        self._final_result = ""
        self.last_messages: list[dict[str, str]] = []

        if self.config.context_path is not None:
            self.runtime.inject("CONTEXT_PATH", self.config.context_path)
        self.runtime.inject("AGENT_ID", self.agent_id)
        self.runtime.inject("DEPTH", str(self.depth))
        self.runtime.inject("MAX_DEPTH", str(self.config.max_depth))
        self.runtime.register_tool(self.done, core=True)
        self.runtime.register_tool(self.delegate, core=True)
        self.runtime.register_tool(self.wait_all, core=True)

    # ── prompt (override this) ───────────────────────────────────────
    def build_system_prompt(self) -> str:
        if self.config.system_prompt:
            return self.config.system_prompt
        tools_body = TOOLS_TEXT.format(tool_summary=self._tool_summary())
        depth_note = f"You are at recursion depth **{self.depth}** of max **{self.config.max_depth}**."
        if self.depth >= self.config.max_depth - 1:
            depth_note += (
                " You are near the depth limit — work directly, do not delegate."
            )
        elif self.depth > 0:
            depth_note += " Be more conservative with delegation the deeper you are."
        default_builder = make_default_builder()
        default_builder.section("tools", tools_body, title="Tools")
        default_builder.section("status", depth_note, title="Status")
        return default_builder.build()

    def _tool_summary(self) -> str:
        lines = []
        for td in self.runtime.get_tool_defs():
            lines.append(f"- `{td.name}{td.signature}`: {td.description}")
        return "\n".join(lines)

    # ── message formatting (override these) ──────────────────────────

    def next_action_message(self, task: str, *, iteration: int) -> dict[str, str]:
        if iteration == 0:
            content = (
                f"Task: {task}\n\n"
                "Start by inspecting the available context and tools. "
                "Reply with exactly one ```repl``` block that takes the next concrete step."
            )
        else:
            content = (
                f"Continue working on the task: {task}\n\n"
                "Reply with exactly one ```repl``` block for the next concrete step. "
                "If you are done, call `done(...)` inside the REPL block."
            )
        return {"role": "user", "content": content}

    def execution_output_message(self, output: str) -> dict[str, str]:
        return {
            "role": "user",
            "content": f"REPL output:\n```\n{output}\n```",
        }

    def no_code_block_message(self) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Your previous reply did not include a ```repl``` block. "
                "Reply with exactly one ```repl``` block and take a concrete action."
            ),
        }

    def stall_recovery_message(self) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "You repeated the same REPL action multiple times. "
                "Do not repeat it again. Choose a different concrete step."
            ),
        }

    # ── LLM interaction (override this) ─────────────────────────────

    def call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM client, emitting tokens through the logger."""
        self.logger.on_llm_start(self.agent_id, 0, len(messages))
        chunks: list[str] = []
        for chunk in self.llm_client.stream(messages):
            chunks.append(chunk)
            self.logger.on_llm_token(self.agent_id, chunk)
        text = "".join(chunks)
        self.logger.on_llm_end(self.agent_id, text)
        return text

    # ── execution (override these) ───────────────────────────────────

    def execute_code(self, code: str) -> str:
        output = self.runtime.execute(code)
        if len(output) > self.config.max_output_length:
            output = output[: self.config.max_output_length] + "\n...<truncated>"
        return output

    def extract_code_blocks(self, text: str) -> list[str]:
        return find_code_blocks(text)

    def truncate_after_first_block(self, text: str) -> str:
        """Cut the response after the first ```repl``` block's closing fence."""
        import re

        m = re.search(r"```repl\s*\n.*?\n```", text, re.DOTALL)
        if m:
            return text[: m.end()]
        return text

    # ── child agents (override these) ────────────────────────────────

    def child_kwargs(self) -> dict[str, Any]:
        return {
            "llm_client": self.llm_client,
            "config": replace(self.config),
            "logger": self.logger,
            "depth": self.depth + 1,
            "pool": self.pool,
            "runtime_factory": self.runtime_factory,
        }

    def create_runtime(self) -> Runtime:
        return self.runtime_factory()

    def create_child(
        self,
        agent_id: str,
        task: str,
        *,
        max_iterations: int | None = None,
    ) -> "RLM":
        child_runtime = self.create_runtime()
        if self.config.context_path:
            child_runtime.write_file(
                self.config.context_path,
                self.runtime.read_file(self.config.context_path),
            )

        kwargs = self.child_kwargs()
        if max_iterations is not None:
            kwargs["config"] = replace(kwargs["config"], max_iterations=max_iterations)

        return self.__class__(
            runtime=child_runtime,
            agent_id=agent_id,
            **kwargs,
        )

    def _default_runtime_factory(self) -> Runtime:
        return type(self.runtime)(workspace=self.runtime.workspace)

    # ── context (override these) ─────────────────────────────────────

    def build_initial_context(self, task: str) -> str:
        return task

    def initialize_context(self, task: str, *, reset: bool) -> None:
        self.runtime.inject("TASK", task)
        if self.config.context_path is None:
            return
        current = self.runtime.read_file(self.config.context_path)
        if reset or not current.strip():
            self.runtime.write_file(
                self.config.context_path, self.build_initial_context(task)
            )

    # ── lifecycle hooks (override these) ─────────────────────────────

    def on_run_start(self, task: str) -> None:
        pass

    def on_run_end(self, task: str, result: str) -> None:
        pass

    def on_iteration_start(self, task: str, iteration: int) -> None:
        pass

    def on_iteration_end(self, task: str, iteration: int) -> None:
        pass

    # ── tools ────────────────────────────────────────────────────────

    @tool(
        """Mark the current agent as finished.
Args:
- message (str): Optional final result text.
Returns:
- str: Confirmation string."""
    )
    def done(self, message: str = "") -> str:
        cleaned = message.strip()
        self._done = True
        self._final_result = cleaned or "Done."
        self.logger.on_done(self.agent_id, cleaned)
        return f"Done.{(' ' + cleaned) if cleaned else ''}"

    @tool(
        """Delegate a subtask to a child agent.
Args:
- task (str): The child task to run.
- wait (bool): If True, block until completion and return the result.
- max_iterations (int | None): Optional iteration limit override.
Returns:
- DelegateHandle | str: A handle when wait=False, otherwise the child result string."""
    )
    def delegate(
        self,
        task: str,
        *,
        wait: bool = False,
        max_iterations: int | None = None,
    ) -> str | DelegateHandle:
        if self.depth >= self.config.max_depth:
            self.logger.on_delegate_refused(self.agent_id, self.config.max_depth)
            return f"[delegation refused: at max depth {self.config.max_depth}] Do this directly."
        agent_id = f"{self.agent_id}.{next(self._delegate_ids)}"
        self.logger.on_delegate(self.agent_id, agent_id, task, wait)
        child = self.create_child(agent_id, task, max_iterations=max_iterations)

        def _run() -> str:
            return child.run(task, reset_context=False)

        handle = self.pool.submit(agent_id, task, _run)
        if wait:
            return handle.result()
        return handle

    @tool(
        """Wait for delegated child agents.
Args:
- handles (list[DelegateHandle]): Handles from delegate(..., wait=False).
- timeout (float | None): Optional timeout per handle.
Returns:
- list[str]: Child results in input order."""
    )
    def wait_all(
        self,
        handles: list[DelegateHandle],
        *,
        timeout: float | None = None,
    ) -> list[str]:
        return [h.result(timeout=timeout) for h in handles]

    # ── run loop ─────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        *,
        max_iterations: int | None = None,
        reset_context: bool = True,
        replay_last_n_turns: int | None = None,
    ) -> str:
        iters = max_iterations or self.config.max_iterations
        replay_turns = (
            self.config.replay_last_n_turns
            if replay_last_n_turns is None
            else max(0, replay_last_n_turns)
        )

        self._done = False
        self._final_result = ""
        self.initialize_context(task, reset=reset_context)
        self.on_run_start(task)

        recent_turns: deque[list[dict[str, str]]] = deque(maxlen=replay_turns)
        pending: list[dict[str, str]] = []
        prev_code = ""
        repeat_count = 0
        invalid_count = 0
        recovered = False
        last_reply = ""
        iteration = -1

        system_msg = {"role": "system", "content": self.build_system_prompt()}
        self.logger.on_run_start(self.agent_id, task, iters)

        for iteration in range(iters):
            self.on_iteration_start(task, iteration)
            self.logger.on_iter_start(self.agent_id, iteration)

            messages = [system_msg]
            for turn in recent_turns:
                messages.extend(turn)
            if pending:
                messages.extend(pending)
                pending = []
            messages.append(self.next_action_message(task, iteration=iteration))

            text = self.call_llm(messages)
            last_reply = text
            assistant_msg = {"role": "assistant", "content": text}

            blocks = self.extract_code_blocks(text)
            if not blocks:
                invalid_count += 1
                self.logger.on_no_code_block(self.agent_id, iteration)
                self.last_messages = messages + [assistant_msg]
                if invalid_count <= 1:
                    pending = [assistant_msg, self.no_code_block_message()]
                    self.on_iteration_end(task, iteration)
                    continue
                self.on_iteration_end(task, iteration)
                break

            code = blocks[0]
            if self.config.single_block:
                text = self.truncate_after_first_block(text)
                assistant_msg = {"role": "assistant", "content": text}
            invalid_count = 0
            self.logger.on_exec_start(self.agent_id, code)
            output = self.execute_code(code)
            self.logger.on_exec_end(self.agent_id, output)

            exec_msg = self.execution_output_message(output)

            norm = code.strip()
            if norm == prev_code:
                repeat_count += 1
            else:
                prev_code = norm
                repeat_count = 1

            if repeat_count >= 3:
                self.logger.on_stall(self.agent_id, iteration)
                self.last_messages = messages + [assistant_msg, exec_msg]
                if not recovered:
                    recovered = True
                    prev_code = ""
                    repeat_count = 0
                    pending = [assistant_msg, exec_msg, self.stall_recovery_message()]
                    self.on_iteration_end(task, iteration)
                    continue
                self.on_iteration_end(task, iteration)
                break

            recovered = False
            if replay_turns > 0:
                recent_turns.append([assistant_msg, exec_msg])
            self.last_messages = messages + [assistant_msg, exec_msg]

            self.on_iteration_end(task, iteration)
            if self._done:
                break

        result = self._final_result or last_reply.strip()
        self.logger.on_run_end(self.agent_id, result, iteration + 1)
        self.on_run_end(task, result)
        return result
