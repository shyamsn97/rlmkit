"""Core RLM engine — step-based execution with thread suspension."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import count
from pathlib import PurePosixPath
from threading import Event, Thread

from .llm import LLMClient
from .prompts.default import TOOLS_TEXT, make_default_builder
from .runtime import Runtime
from .state import (
    ChildHandle,
    ChildStep,
    CodeExec,
    LLMReply,
    NoCodeBlock,
    RLMState,
    Status,
)
from .utils import find_code_blocks, tool

# ── exec thread helper ───────────────────────────────────────────────


class ExecThread:
    """Runs code in a background thread with suspend/resume for delegation."""

    def __init__(self):
        self._phase = Event()
        self._resume = Event()
        self.suspended = False
        self.output = ""

    def run(self, fn: Callable[[str], str], code: str) -> tuple[bool, str]:
        """Execute code via fn. Returns (suspended, output)."""
        self._phase = Event()
        self._resume = Event()
        self.suspended = False
        self.output = ""
        Thread(target=self._target, args=(fn, code), daemon=True).start()
        self._phase.wait()
        return self.suspended, self.output

    def resume(self) -> str | None:
        """Resume after children finish. Returns output or None if re-suspended."""
        self._phase.clear()
        self._resume.set()
        self._phase.wait()
        return None if self.suspended else self.output

    def suspend(self):
        """Called from the exec thread to yield control back to the stepper."""
        self.suspended = True
        self._phase.set()
        self._resume.wait()
        self._resume.clear()
        self.suspended = False

    def _target(self, fn, code):
        self.output = fn(code)
        self._phase.set()


# ── config ───────────────────────────────────────────────────────────


@dataclass
class RLMConfig:
    max_depth: int = 5
    max_iterations: int = 30
    max_output_length: int = 12_000
    max_messages: int | None = None
    max_concurrent_children: int = 8
    child_max_iterations: int | None = None
    single_block: bool = True
    context_path: str | None = None
    system_prompt: str | None = None


# ── engine ───────────────────────────────────────────────────────────


class RLM:
    """Recursive Language Model engine.

    Owns mutable resources (LLM client, runtime, threads). All observable
    state lives in RLMState — frozen, immutable, recursive.
    """

    state_cls: type[RLMState] = RLMState

    def __init__(
        self,
        llm_client: LLMClient,
        runtime: Runtime,
        config: RLMConfig | None = None,
        agent_id: str = "root",
        depth: int = 0,
        runtime_factory: Callable[[], Runtime] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.runtime = runtime
        self.config = config or RLMConfig()
        self.agent_id = agent_id
        self.depth = depth
        self.runtime_factory = runtime_factory

        self._delegate_ids = count(1)
        self.children: dict[str, RLM] = {}
        self.waiting_on: list[str] = []
        self.is_done = False
        self.result: str | None = None
        self._thread = ExecThread()
        self.last_state: RLMState | None = None

        self.runtime.inject("CONTEXT_PATH", self.config.context_path)
        self.runtime.inject("AGENT_ID", self.agent_id)
        self.runtime.inject("DEPTH", str(self.depth))
        self.runtime.inject("MAX_DEPTH", str(self.config.max_depth))
        self.runtime.register_tool(self.done, core=True)
        self.runtime.register_tool(self.delegate, core=True)
        self.runtime.register_tool(self.wait_all, core=True)

    # ── lifecycle ────────────────────────────────────────────────────

    def make_state(self, **fields) -> RLMState:
        """Build initial state. Override to inject custom fields."""
        return self.state_cls(**fields)

    def start(self, task: str, *, reset_context: bool = True) -> RLMState:
        """Reset engine, initialize context, return first state."""
        self.children.clear()
        self.waiting_on.clear()
        self.is_done = False
        self.result = None
        self.initialize_context(task, reset=reset_context)
        config = asdict(self.config)
        config.update(depth=self.depth, task=task)
        state = self.make_state(
            agent_id=self.agent_id,
            status=Status.WAITING,
            config=config,
            context=self.read_context(),
        )
        self.last_state = state
        return state

    def run(self, task: str, *, reset_context: bool = True) -> str:
        """Run to completion. Returns the result string."""
        state = self.start(task, reset_context=reset_context)
        while not state.finished:
            state = self.step(state)
        return state.result or ""

    # ── step ─────────────────────────────────────────────────────────

    def step(self, state: RLMState) -> RLMState:
        """Advance one step. The state machine: WAITING→HAS_REPLY→EXEC→(SUPERVISING→)FINISHED."""
        match state.status:
            case Status.WAITING:
                return self.step_llm(state)
            case Status.HAS_REPLY:
                return self.step_exec(state)
            case Status.SUPERVISING:
                return self.step_children(state)
            case _:
                return state

    def step_llm(self, state: RLMState) -> RLMState:
        """Call the LLM → HAS_REPLY."""
        # Bail if we've hit the iteration cap
        max_iters = state.config.get("max_iterations", self.config.max_iterations)
        if state.iteration >= max_iters:
            return state.update(status=Status.FINISHED, result=state.last_reply or "")

        # Build prompt, call LLM, store reply
        system = self.build_system_prompt(state)
        messages = self.build_messages(state, system_prompt=system)
        text = self.call_llm(messages)
        assistant_message = {"role": "assistant", "content": text}

        return state.update(
            status=Status.HAS_REPLY,
            iteration=state.iteration + 1,
            event=LLMReply(
                agent_id=state.agent_id,
                iteration=state.iteration + 1,
                text=text,
                code=self.extract_code(text),
            ),
            messages=messages + [assistant_message],
            last_reply=text,
        )

    def step_exec(self, state: RLMState) -> RLMState:
        """Execute the code block → WAITING / SUPERVISING / FINISHED."""
        # Extract the ```repl``` block from the LLM's reply
        code = self.extract_code(state.last_reply or "")
        if not code:
            return state.update(
                status=Status.WAITING,
                event=NoCodeBlock(
                    agent_id=state.agent_id,
                    iteration=state.iteration,
                    text=state.last_reply or "",
                ),
                messages=state.messages + [self.no_code_block_message()],
            )

        # Run the code — may suspend if it calls delegate()/wait_all()
        suspended, output = self._thread.run(self.execute_code, code)

        # Determine next status based on what happened
        if self.is_done:
            new_status = Status.FINISHED
        elif suspended:
            new_status = Status.SUPERVISING
        else:
            new_status = Status.WAITING

        # If suspended, register any newly created children
        children = list(state.children)
        if suspended:
            existing = {cs.agent_id for cs in children}
            for cid, engine in self.children.items():
                if cid not in existing:
                    children.append(engine.last_state)

        # Feed the execution output back into message history
        exec_message = self.execution_output_message(code, output)
        new_messages = state.messages + [exec_message]

        return state.update(
            status=new_status,
            event=CodeExec(
                agent_id=state.agent_id,
                iteration=state.iteration,
                code=code,
                output=output,
                suspended=suspended,
            ),
            messages=new_messages,
            result=self.result if self.is_done else None,
            context=self.read_context(),
            children=children,
        )

    def execute_child_steps(
        self, active: list[tuple[RLMState, RLM]]
    ) -> dict[str, RLMState]:
        """Step active children and return {agent_id: new_state}.

        Override to swap in a custom executor (e.g. process pool,
        remote dispatch, rate-limited queue).
        """
        cap = self.config.max_concurrent_children
        results: dict[str, RLMState] = {}
        with ThreadPoolExecutor(max_workers=min(len(active), cap)) as pool:
            futures = {pool.submit(e.step, cs): cs.agent_id for cs, e in active}
            for future in as_completed(futures):
                aid = futures[future]
                results[aid] = future.result()
                self.on_child_stepped(aid, results[aid], len(results), len(active))
        return results

    def on_child_stepped(
        self, agent_id: str, state: RLMState, completed: int, total: int
    ) -> None:
        """Called after each child completes a step. Override for progress."""

    def step_children(self, state: RLMState) -> RLMState:
        """Step all active children in parallel → check if wait is satisfied."""

        # Run one step for every child that hasn't finished yet
        active = [
            (cs, self.children[cs.agent_id]) for cs in state.children if not cs.finished
        ]
        stepped = self.execute_child_steps(active) if active else {}

        # Merge stepped children back into the full children list
        new_children, events = [], []
        for cs in state.children:
            if cs.finished:
                new_children.append(cs)
            else:
                new_cs = stepped[cs.agent_id]
                new_children.append(new_cs)
                if new_cs.event:
                    events.append(new_cs.event)

        # Check if all children the parent is waiting on are done
        all_done = all(
            cs.agent_id not in self.waiting_on or cs.finished for cs in new_children
        )

        if all_done:
            # Resume the parent's exec thread (it was suspended at wait_all)
            output = self._thread.resume()

            if self.is_done:
                new_status = Status.FINISHED
            elif output is None:
                new_status = Status.SUPERVISING
            else:
                new_status = Status.WAITING

            # If done() wasn't reached, feed the output back so the LLM can retry
            new_messages = list(state.messages)
            if output is not None and not self.is_done:
                new_messages.append(
                    {
                        "role": "user",
                        "content": f"REPL output:\n{output}",
                    }
                )

            return state.update(
                status=new_status,
                event=ChildStep(
                    agent_id=state.agent_id,
                    iteration=state.iteration,
                    child_events=events,
                    all_done=True,
                    exec_output=output,
                ),
                messages=new_messages,
                result=self.result if self.is_done else None,
                children=new_children,
            )

        # Not all waited-on children are done yet — stay in SUPERVISING
        return state.update(
            event=ChildStep(
                agent_id=state.agent_id,
                iteration=state.iteration,
                child_events=events,
                all_done=False,
            ),
            children=new_children,
        )

    # ── prompt & messages ────────────────────────────────────────────

    def build_system_prompt(self, state: RLMState) -> str:
        """Build the system prompt. Override to customize."""
        if self.config.system_prompt:
            return self.config.system_prompt
        depth = state.config.get("depth", 0)
        max_depth = state.config.get("max_depth", self.config.max_depth)
        tool_lines = "\n".join(
            f"- `{td.name}{td.signature}`: {td.description}"
            for td in self.runtime.get_tool_defs()
        )
        preimported = self.runtime.available_modules()
        if preimported:
            tool_lines += (
                f"\n\nPre-imported modules (already available, no import needed): "
                f"`{'`, `'.join(preimported)}`"
            )
        depth_note = f"You are at recursion depth **{depth}** of max **{max_depth}**."
        if depth >= max_depth - 1:
            depth_note += (
                " You are near the depth limit — work directly, do not delegate."
            )
        elif depth > 0:
            depth_note += " Be more conservative with delegation the deeper you are."
        builder = make_default_builder()
        builder.section(
            "tools", TOOLS_TEXT.format(tool_summary=tool_lines), title="Tools"
        )
        builder.section("status", depth_note, title="Status")
        return builder.build()

    def build_messages(self, state: RLMState, *, system_prompt: str) -> list[dict]:
        """Assemble the message list for an LLM call. Override to customize."""
        system = {"role": "system", "content": system_prompt}
        if not state.messages:
            task = state.config.get("task", "")
            return [system, self.next_action_message(task, iteration=0)]

        msgs = list(state.messages)
        cap = self.config.max_messages
        if cap and len(msgs) > cap:
            msgs = self.truncate_messages(state, msgs, cap)
        return [system] + msgs

    def truncate_messages(
        self, state: RLMState, msgs: list[dict], cap: int
    ) -> list[dict]:
        """Build a condensed message list when history exceeds max_messages."""
        task = state.config.get("task", "")
        context = self.read_context()
        recent = msgs[-cap:]

        parts = [f"## Task\n{task}"]
        if context:
            parts.append(f"## Context (from {self.config.context_path})\n{context}")
        parts.append(
            f"## History\n{len(msgs)} messages so far, showing the last {cap}. "
            "Read your context file for full progress."
        )

        summary = {"role": "user", "content": "\n\n".join(parts)}
        return [summary] + recent

    def next_action_message(self, task: str, *, iteration: int) -> dict[str, str]:
        """Format the user message that prompts the LLM to act."""
        ctx_hint = (
            "IMPORTANT: `CONTEXT_PATH` is set — start by reading it with "
            "`read_file(CONTEXT_PATH)` to check prior progress before doing anything else. "
            "Append important progress to it as you go.\n\n"
            if self.config.context_path
            else ""
        )
        if iteration == 0:
            content = (
                f"Task: {task}\n\n"
                f"{ctx_hint}"
                "Your response MUST contain exactly one ```repl``` code block. "
                "Put any reasoning as comments inside the block. "
                "Do NOT reply with only text — every response needs a ```repl``` block."
            )
        else:
            content = (
                f"Continue working on the task: {task}\n\n"
                f"{ctx_hint}"
                "Your response MUST contain exactly one ```repl``` code block "
                "with the next concrete action. "
                "If you are done, call `done(result)` inside the block."
            )
        return {"role": "user", "content": content}

    def execution_output_message(self, code: str, output: str) -> dict[str, str]:
        """Format the executed code and its output as a user message."""
        return {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{output}",
        }

    def no_code_block_message(self) -> dict[str, str]:
        """Nudge the LLM when it forgets to include a ```repl``` block."""
        return {
            "role": "user",
            "content": "ERROR: Your previous reply did not contain a ```repl``` code block. "
            "You MUST reply with exactly one ```repl``` block every time. "
            "Write your next action now inside a ```repl``` block.",
        }

    def call_llm(self, messages: list[dict]) -> str:
        """Call the LLM. Override for custom streaming/logging."""
        return "".join(self.llm_client.stream(messages))

    def execute_code(self, code: str) -> str:
        """Run code via the runtime. Override for sandboxing/limits."""
        output = self.runtime.execute(code)
        if len(output) > self.config.max_output_length:
            return output[: self.config.max_output_length] + "\n...<truncated>"
        return output

    def extract_code(self, text: str) -> str | None:
        """Pull the first ```repl``` block from LLM output."""
        blocks = find_code_blocks(text)
        if not blocks:
            return None
        return blocks[0] if self.config.single_block else "\n\n".join(blocks)

    # ── tools (injected into the REPL namespace) ─────────────────────

    @tool(
        "Mark the current agent as finished.\nArgs:\n- message (str): Required. An informative result message — the actual data, answer, or summary. Never pass empty string unless you truly found nothing.\nReturns:\n- str: The result."
    )
    def done(self, message: str) -> str:
        self.is_done = True
        self.result = message.strip()
        return self.result

    @tool(
        "Delegate a subtask to a child agent.\nArgs:\n- task (str): The subtask.\n- wait (bool): Block until done.\n- max_iterations (int | None): Iteration cap.\nReturns:\n- ChildHandle | str: Handle (async) or result (sync)."
    )
    def delegate(
        self, task: str, *, wait: bool = False, max_iterations: int | None = None
    ) -> str | ChildHandle:
        if self.depth >= self.config.max_depth:
            return f"[refused: max depth {self.config.max_depth}] Do this directly."
        agent_id = f"{self.agent_id}.{next(self._delegate_ids)}"
        child = self.create_child(agent_id, task, max_iterations=max_iterations)
        self.children[agent_id] = child
        child.last_state = child.start(task)
        handle = ChildHandle(agent_id)
        if wait:
            self.waiting_on = [agent_id]
            self._thread.suspend()
            return child.result or ""
        return handle

    @tool(
        "Wait for delegated children.\nArgs:\n- *handles: Handles from delegate().\nReturns:\n- list[str]: Results in order."
    )
    def wait_all(self, *handles: ChildHandle) -> list[str]:
        self.waiting_on = [h.agent_id for h in handles]
        self._thread.suspend()
        return [self.children[h.agent_id].result or "" for h in handles]

    # ── children & context ───────────────────────────────────────────

    def create_context_path(self, agent_id: str) -> str:
        """Derive the context file path for a child. Override to customize layout."""
        return f".rlm/{agent_id}/{PurePosixPath(self.config.context_path).name}"

    def create_child(
        self, agent_id: str, task: str, *, max_iterations: int | None = None
    ) -> RLM:
        """Create a child engine. Override for custom child setup."""
        child_iters = (
            max_iterations
            or self.config.child_max_iterations
            or self.config.max_iterations // 3
        )
        child_config = replace(self.config, max_iterations=child_iters)
        if self.config.context_path:
            child_config = replace(
                child_config,
                context_path=self.create_context_path(agent_id),
            )
        rt = self.runtime_factory() if self.runtime_factory else self.runtime.clone()
        return self.__class__(
            llm_client=self.llm_client,
            runtime=rt,
            config=child_config,
            agent_id=agent_id,
            depth=self.depth + 1,
            runtime_factory=self.runtime_factory,
        )

    def read_context(self) -> str | None:
        """Read the context file, if configured."""
        if not self.config.context_path:
            return None
        try:
            return self.runtime.read_file(self.config.context_path)
        except FileNotFoundError:
            return None

    def initialize_context(self, task: str, *, reset: bool) -> None:
        """Set up the context file for a new run."""
        self.runtime.inject("TASK", task)
        if not self.config.context_path:
            return
        try:
            current = self.runtime.read_file(self.config.context_path)
        except FileNotFoundError:
            current = ""
        if reset or not current.strip():
            content = task if self.depth == 0 else ""
            self.runtime.write_file(self.config.context_path, content)
