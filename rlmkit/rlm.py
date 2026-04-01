"""Core RLM engine — step-based execution with thread suspension."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from threading import Event, Thread

from .context import Context, FileContext
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
from .utils import OrphanedDelegatesError, find_code_blocks, tool

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
    context: Context | str | None = None
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
        llm_clients: dict[str, dict] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.runtime = runtime
        self.config = config or RLMConfig()
        self.agent_id = agent_id
        self.depth = depth
        self.runtime_factory = runtime_factory

        self.llm_clients: dict[str, LLMClient] = {}
        self.model_descriptions: dict[str, str] = {}
        for k, entry in (llm_clients or {}).items():
            self.llm_clients[k] = entry["model"]
            if "description" in entry:
                self.model_descriptions[k] = entry["description"]
        if "default" not in self.llm_clients:
            self.llm_clients["default"] = self.llm_client

        self.children: dict[str, RLM] = {}
        self.waiting_on: list[str] = []
        self.is_done = False
        self.result: str | None = None
        self._thread = ExecThread()
        self.last_state: RLMState | None = None

        # Auto-create FileContext from string path
        if isinstance(self.config.context, str):
            self.config = replace(
                self.config, context=FileContext(self.config.context, self.runtime)
            )
        self.context: Context | None = self.config.context

        self.runtime.inject("OrphanedDelegatesError", OrphanedDelegatesError)
        self.runtime.inject("AGENT_ID", self.agent_id)
        self.runtime.inject("DEPTH", str(self.depth))
        self.runtime.inject("MAX_DEPTH", str(self.config.max_depth))
        self.runtime.register_tool(self.done, core=True)
        self.runtime.register_tool(self.delegate, core=True)
        self.runtime.register_tool(self.wait_all, core=True)
        if self.context:
            self.runtime.register_tool(self.read_context, core=True)
            self.runtime.register_tool(self.append_context, core=True)

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
        self.runtime.inject("TASK", task)
        if self.context and reset_context:
            self.context.write(task if self.depth == 0 else "")
        config_dict = {
            f.name: getattr(self.config, f.name)
            for f in fields(self.config)
            if f.name != "context"
        }
        config_dict.update(depth=self.depth, task=task)
        state = self.make_state(
            agent_id=self.agent_id,
            status=Status.WAITING,
            config=config_dict,
            context=self.context.read() if self.context else None,
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

        # Detect newly created children
        children = list(state.children)
        existing = {cs.agent_id for cs in children}
        new_child_ids = [cid for cid in self.children if cid not in existing]

        # Orphan check: delegate() called but wait_all() never was
        if new_child_ids and not suspended:
            for cid in new_child_ids:
                del self.children[cid]
            names = ", ".join(new_child_ids)
            msg = (
                f"delegate() was called for [{names}] but wait_all() was never "
                f"called. You must call wait_all(*handles) to collect results, "
                f"or use delegate(name, task, wait=True) for synchronous execution."
            )
            err_output = self.execute_code(f"raise OrphanedDelegatesError({msg!r})")
            output += "\n\n" + err_output if output.strip() else err_output
            new_child_ids = []

        if suspended:
            for cid in new_child_ids:
                children.append(self.children[cid].last_state)

        # Determine next status based on what happened
        if self.is_done:
            new_status = Status.FINISHED
        elif suspended:
            new_status = Status.SUPERVISING
        else:
            new_status = Status.WAITING

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
            context=self.context.read() if self.context else None,
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

            # Signal event-stream observers if done() was called in resumed exec
            return state.update(
                status=new_status,
                event=ChildStep(
                    agent_id=state.agent_id,
                    iteration=state.iteration,
                    child_events=events,
                    all_done=True,
                    exec_output=output,
                    agent_finished=self.is_done,
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
        if len(self.llm_clients) > 1:
            model_parts = []
            for k in sorted(self.llm_clients):
                desc = self.model_descriptions.get(k)
                model_parts.append(f"- `{k}`: {desc}" if desc else f"- `{k}`")
            tool_lines += (
                "\n\nAvailable models for `delegate(model=...)`:\n"
                + "\n".join(model_parts)
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
        context = self.context.read() if self.context else ""
        recent = msgs[-cap:]

        parts = [f"## Task\n{task}"]
        if context:
            parts.append(f"## Context\n{context}")
        parts.append(
            f"## History\n{len(msgs)} messages so far, showing the last {cap}. "
            "Call read_context() for full progress."
        )

        summary = {"role": "user", "content": "\n\n".join(parts)}
        return [summary] + recent

    def next_action_message(self, task: str, *, iteration: int) -> dict[str, str]:
        """Format the user message that prompts the LLM to act."""
        ctx_hint = (
            "IMPORTANT: Context is available — start by calling `read_context()` "
            "to check prior progress before doing anything else. "
            "Use `append_context(text)` to record progress as you go.\n\n"
            if self.context
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
        "Delegate a subtask to a named child agent.\nArgs:\n"
        "- name (str): Short identifier for the child (e.g. 'search_batch_0').\n"
        "- task (str): The subtask description.\n"
        "- wait (bool): Block until done.\n"
        "- max_iterations (int | None): Iteration cap.\n"
        "- model (str): Which model to use. Defaults to 'default'.\n"
        "Returns:\n- ChildHandle | str: Handle (async) or result (sync)."
    )
    def delegate(
        self,
        name: str,
        task: str,
        *,
        wait: bool = False,
        max_iterations: int | None = None,
        model: str = "default",
    ) -> str | ChildHandle:
        if self.depth >= self.config.max_depth:
            return f"[refused: max depth {self.config.max_depth}] Do this directly."
        if model not in self.llm_clients:
            keys = ", ".join(sorted(self.llm_clients))
            return f"[error: unknown model {model!r}. available: {keys}]"
        agent_id = self._resolve_child_id(name)
        client = self.llm_clients[model]
        child = self.create_child(
            agent_id, task, max_iterations=max_iterations, llm_client=client
        )
        self.children[agent_id] = child
        child.last_state = child.start(task)
        handle = ChildHandle(agent_id)
        if wait:
            self.waiting_on = [agent_id]
            self._thread.suspend()
            return child.result or ""
        return handle

    def _packed_llm_clients(self) -> dict[str, dict]:
        """Re-pack llm_clients with descriptions for propagation to children."""
        result: dict[str, dict] = {}
        for k, client in self.llm_clients.items():
            entry: dict = {"model": client}
            if k in self.model_descriptions:
                entry["description"] = self.model_descriptions[k]
            result[k] = entry
        return result

    def _resolve_child_id(self, name: str) -> str:
        """Derive a unique agent_id, auto-suffixing on collision."""
        base = f"{self.agent_id}.{name}"
        if base not in self.children:
            return base
        n = 2
        while f"{base}_{n}" in self.children:
            n += 1
        return f"{base}_{n}"

    @tool(
        "Wait for delegated children.\nArgs:\n- *handles: Handles from delegate().\nReturns:\n- list[str]: Results in order."
    )
    def wait_all(self, *handles: ChildHandle) -> list[str]:
        self.waiting_on = [h.agent_id for h in handles]
        self._thread.suspend()
        return [self.children[h.agent_id].result or "" for h in handles]

    # ── context tools ──────────────────────────────────────────────

    @tool("Read the agent's durable context. Returns the full context string.")
    def read_context(self) -> str:
        if not self.context:
            return ""
        return self.context.read()

    @tool(
        "Append text to the agent's durable context.\nArgs:\n- text (str): Text to append.\nReturns:\n- str: Confirmation."
    )
    def append_context(self, text: str) -> str:
        if not self.context:
            return "No context configured."
        self.context.append(text)
        return "ok"

    # ── children ─────────────────────────────────────────────────────

    def create_child(
        self,
        agent_id: str,
        task: str,
        *,
        max_iterations: int | None = None,
        llm_client: LLMClient | None = None,
    ) -> RLM:
        """Create a child engine. Override for custom child setup."""
        child_iters = (
            max_iterations
            or self.config.child_max_iterations
            or self.config.max_iterations // 3
        )
        child_context = self.context.clone(agent_id) if self.context else None
        child_config = replace(
            self.config, max_iterations=child_iters, context=child_context
        )
        rt = self.runtime_factory() if self.runtime_factory else self.runtime.clone()
        return self.__class__(
            llm_client=llm_client or self.llm_client,
            runtime=rt,
            config=child_config,
            agent_id=agent_id,
            depth=self.depth + 1,
            runtime_factory=self.runtime_factory,
            llm_clients=self._packed_llm_clients(),
        )
