"""Core RLM engine — step-based execution with generators."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

from .llm import LLMClient
from .pool import CallablePool, ThreadPool
from .prompts.default import make_default_builder
from .runtime import Runtime
from .session import FileSession, Session
from .state import (
    ChildHandle,
    ChildStep,
    CodeExec,
    LLMReply,
    NoCodeBlock,
    RLMState,
    Status,
    WaitRequest,
)
from .utils import OrphanedDelegatesError, find_code_blocks, tool

# ── AST yield checker ────────────────────────────────────────────────


class YieldChecker(ast.NodeVisitor):
    """Detect bare wait() calls that are missing a yield prefix."""

    def __init__(self):
        self.errors: list[str] = []

    def visit_Yield(self, node):  # noqa: N802
        pass  # yield <expr> — don't descend into the call

    def visit_Call(self, node):  # noqa: N802
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name == "wait":
            self.errors.append(
                f"Line {node.lineno}: `wait(...)` must be prefixed with `yield`"
            )
        self.generic_visit(node)


def check_missing_yields(code: str) -> list[str]:
    """Return a list of error strings for any wait() calls missing yield."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    checker = YieldChecker()
    checker.visit(tree)
    return checker.errors


# ── config ───────────────────────────────────────────────────────────


@dataclass
class RLMConfig:
    max_depth: int = 5
    max_iterations: int = 30
    max_output_length: int = 12_000
    max_messages: int | None = None
    max_concurrency: int = 8
    child_max_iterations: int | None = None
    single_block: bool = True
    session: Session | str | None = None
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
        pool: Any = None,
        prompt_builder: Any = None,
    ) -> None:
        self.llm_client = llm_client
        self.runtime = runtime
        self.config = config or RLMConfig()
        self.agent_id = agent_id
        self.depth = depth
        self.runtime_factory = runtime_factory
        self.prompt_builder = prompt_builder or make_default_builder()

        if pool is None:
            pool = ThreadPool(self.config.max_concurrency)
        elif callable(pool) and not hasattr(pool, "execute"):
            pool = CallablePool(pool)
        self.pool = pool

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
        self.last_state: RLMState | None = None

        if isinstance(self.config.session, str):
            self.config = replace(self.config, session=FileSession(self.config.session))
        self.session: Session | None = self.config.session

        self.runtime.inject("OrphanedDelegatesError", OrphanedDelegatesError)
        self.runtime.inject("AGENT_ID", self.agent_id)
        self.runtime.inject("DEPTH", str(self.depth))
        self.runtime.inject("MAX_DEPTH", str(self.config.max_depth))
        self.runtime.register_tool(self.done, core=True)
        self.runtime.register_tool(self.delegate, core=True)
        self.runtime.register_tool(self.wait, core=True)
        if self.session:
            self.runtime.register_tool(self.list_sessions, core=True)
            self.runtime.register_tool(self.read_history, core=True)

    # ── lifecycle ────────────────────────────────────────────────────

    def make_state(self, **fields) -> RLMState:
        """Build initial state. Override to inject custom fields."""
        return self.state_cls(**fields)

    def initialize_state(self, task: str) -> RLMState:
        """Create initial state for a task, reset engine for a fresh run."""
        self.children.clear()
        self.waiting_on.clear()
        self.is_done = False
        self.result = None
        config_dict = {
            f.name: getattr(self.config, f.name)
            for f in fields(self.config)
            if f.name != "session"
        }
        config_dict.update(depth=self.depth)
        state = self.make_state(
            agent_id=self.agent_id,
            task=task,
            status=Status.WAITING,
            config=config_dict,
        )
        self.last_state = state
        return state

    def start(self, task: str) -> RLMState:
        """Alias for initialize_state (backwards compat)."""
        return self.initialize_state(task)

    def run(self, task: str) -> str:
        """Run to completion. Returns the result string."""
        state = self.initialize_state(task)
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
                return self.step_supervise(state)
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

        yield_errors = check_missing_yields(code)
        if yield_errors:
            err_msg = "ERROR: " + "; ".join(yield_errors)
            return state.update(
                status=Status.WAITING,
                event=CodeExec(
                    agent_id=state.agent_id,
                    iteration=state.iteration,
                    code=code,
                    output=err_msg,
                    suspended=False,
                ),
                messages=state.messages
                + [self.execution_output_message(code, err_msg)],
            )

        suspended, result = self.runtime.start_code(code)

        children = list(state.children)
        existing = {cs.agent_id for cs in children}

        redelegated = [
            cid
            for cid in self.children
            if cid in existing and not self.children[cid].is_done
        ]
        for cid in redelegated:
            children = [
                self.children[cid].last_state if cs.agent_id == cid else cs
                for cs in children
            ]
            existing = {cs.agent_id for cs in children}

        new_child_ids = [cid for cid in self.children if cid not in existing]

        if new_child_ids and not suspended:
            for cid in new_child_ids:
                del self.children[cid]
            names = ", ".join(new_child_ids)
            msg = (
                f"delegate() was called for [{names}] but wait() was never "
                f"called. You must use `yield wait(*handles)` to collect results."
            )
            err_output = self.execute_code(f"raise OrphanedDelegatesError({msg!r})")
            output = result if isinstance(result, str) else ""
            output += "\n\n" + err_output if output.strip() else err_output
            new_child_ids = []
            suspended = False
        else:
            output = "" if suspended else result

        if suspended:
            self.waiting_on = (
                result.agent_ids if isinstance(result, WaitRequest) else []
            )
            for cid in new_child_ids:
                children.append(self.children[cid].last_state)

        if self.is_done:
            new_status = Status.FINISHED
        elif suspended:
            new_status = Status.SUPERVISING
        else:
            new_status = Status.WAITING

        exec_message = self.execution_output_message(code, output)
        new_messages = state.messages + [exec_message]

        new_state = state.update(
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
            children=children,
            waiting_on=self.waiting_on if suspended else [],
        )
        self._write_session(new_state)
        return new_state

    def step_supervise(self, state: RLMState) -> RLMState:
        """Flatten tree, step one round of leaves, cascade parent resumes.

        Each call does a single round: step all steppable leaves in the
        pool, then apply results (which may resume parents).  The outer
        ``step()`` loop calls this repeatedly until no longer SUPERVISING.
        This keeps each step granular so live UIs can show progress.
        """
        nodes = self.flatten_all(state)
        nodes.sort(key=lambda x: -x[0])
        steppable = [
            (cs, engine) for _, cs, engine in nodes if cs.status != Status.SUPERVISING
        ]
        if not steppable:
            pending = [cs.agent_id for cs in state.children if not cs.finished]
            raise RuntimeError(
                f"Supervise stall: no steppable leaves but {len(pending)} "
                f"pending children: {pending[:5]}"
            )

        stepped = self.pool.execute(steppable)
        child_events = [new_cs.event for new_cs in stepped.values() if new_cs.event]

        state, resume_output = self.apply_results(state, stepped)

        return state.update(
            event=ChildStep(
                agent_id=state.agent_id,
                iteration=state.iteration,
                child_events=child_events,
                all_done=state.status != Status.SUPERVISING,
                exec_output=resume_output,
                agent_finished=self.is_done,
            ),
        )

    def flatten_all(self, state: RLMState, depth: int = 0) -> list:
        """Collect all non-finished nodes in the subtree as (depth, state, engine)."""
        nodes = []
        for cs in state.children:
            if cs.finished:
                continue
            engine = self.children[cs.agent_id]
            nodes.append((depth, cs, engine))
            if cs.status == Status.SUPERVISING:
                nodes.extend(engine.flatten_all(cs, depth + 1))
        return nodes

    def apply_results(
        self, state: RLMState, stepped: dict[str, RLMState]
    ) -> tuple[RLMState, str | None]:
        """Rebuild the state tree bottom-up, resuming parents whose children are done.

        Returns (new_state, resume_output). resume_output is the stdout
        captured when the parent's generator was resumed, or None.
        """
        new_children = []
        for cs in state.children:
            if cs.finished:
                new_children.append(cs)
                continue
            engine = self.children[cs.agent_id]
            if cs.agent_id in stepped:
                new_children.append(stepped[cs.agent_id])
            elif cs.status == Status.SUPERVISING:
                updated, _ = engine.apply_results(cs, stepped)
                new_children.append(updated)
            else:
                new_children.append(cs)

        state = state.update(children=new_children)

        if self.waiting_on:
            all_done = all(
                cs.agent_id not in self.waiting_on or cs.finished for cs in new_children
            )
            if all_done:
                return self.resume_after_children(state)
        return state, None

    def resume_after_children(self, state: RLMState) -> tuple[RLMState, str | None]:
        """Resume this agent's generator after waited-on children finished.

        Returns (new_state, exec_output).
        """
        existing = {cs.agent_id for cs in state.children}

        results = [self.children[aid].result or "" for aid in self.waiting_on]
        suspended, resume_result = self.runtime.resume_code(results)

        children = list(state.children)

        redelegated = [
            cid
            for cid in self.children
            if cid in existing and not self.children[cid].is_done
        ]
        for cid in redelegated:
            children = [
                self.children[cid].last_state if cs.agent_id == cid else cs
                for cs in children
            ]
            existing = {cs.agent_id for cs in children}

        new_child_ids = [cid for cid in self.children if cid not in existing]

        if new_child_ids and not suspended and not self.is_done:
            for cid in new_child_ids:
                del self.children[cid]
            names = ", ".join(new_child_ids)
            msg = (
                f"delegate() was called for [{names}] but wait() was never "
                f"called. You must use `yield wait(*handles)` to collect results."
            )
            err_output = self.execute_code(f"raise OrphanedDelegatesError({msg!r})")
            output = resume_result if isinstance(resume_result, str) else ""
            resume_result = (
                output + "\n\n" + err_output if output.strip() else err_output
            )
            new_child_ids = []

        if self.is_done:
            new_status = Status.FINISHED
        elif suspended:
            self.waiting_on = (
                resume_result.agent_ids
                if isinstance(resume_result, WaitRequest)
                else []
            )
            for cid in new_child_ids:
                children.append(self.children[cid].last_state)
            new_status = Status.SUPERVISING
        else:
            new_status = Status.WAITING

        output = "" if suspended else resume_result

        new_messages = list(state.messages)
        if not suspended and not self.is_done and output:
            new_messages.append({"role": "user", "content": f"REPL output:\n{output}"})

        new_state = state.update(
            status=new_status,
            messages=new_messages,
            result=self.result if self.is_done else None,
            children=children,
            waiting_on=self.waiting_on if suspended else [],
        )
        return new_state, output if not suspended else None

    def _write_session(self, state: RLMState) -> None:
        """Persist the agent's message history to the session store."""
        if self.session and state.messages:
            self.session.write(state.agent_id, state.messages)

    # ── prompt & messages ────────────────────────────────────────────

    def build_system_prompt(self, state: RLMState) -> str:
        """Build the system prompt. Override to customize.

        The builder holds static sections (role, repl, recursion, etc.).
        Dynamic sections — ``tools`` and ``status`` — are passed as
        keyword overrides to ``build()``, keeping the builder immutable.
        """
        if self.config.system_prompt:
            return self.config.system_prompt
        return self.prompt_builder.build(
            tools=self.build_tools_section(state),
            status=self.build_status_section(state),
        )

    def build_tools_section(self, state: RLMState) -> str:
        """Build the tools content for the system prompt. Override to customize."""
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
        return tool_lines

    def build_status_section(self, state: RLMState) -> str:
        """Build the status note for the system prompt. Override to customize."""
        depth = state.config.get("depth", 0)
        max_depth = state.config.get("max_depth", self.config.max_depth)
        depth_note = f"You are at recursion depth **{depth}** of max **{max_depth}**."
        if depth >= max_depth - 1:
            depth_note += (
                " You are near the depth limit — work directly, do not delegate."
            )
        elif depth > 0:
            depth_note += " Be more conservative with delegation the deeper you are."
        return depth_note

    def build_messages(self, state: RLMState, *, system_prompt: str) -> list[dict]:
        """Assemble the message list for an LLM call. Override to customize."""
        system = {"role": "system", "content": system_prompt}
        if not state.messages:
            return [system, self.next_action_message(state.task, iteration=0)]

        msgs = list(state.messages)
        cap = self.config.max_messages
        if cap and len(msgs) > cap:
            msgs = self.truncate_messages(state, msgs, cap)
        return [system] + msgs

    def truncate_messages(
        self, state: RLMState, msgs: list[dict], cap: int
    ) -> list[dict]:
        """Build a condensed message list when history exceeds max_messages."""
        recent = msgs[-cap:]

        parts = [f"## Task\n{state.task}"]
        hint = (
            "Use `read_history()` to review earlier messages, "
            "or `list_sessions()` to see other agents' progress."
            if self.session
            else ""
        )
        parts.append(
            f"## History\n{len(msgs)} messages so far, showing the last {cap}. "
            f"{hint}"
        )

        summary = {"role": "user", "content": "\n\n".join(parts)}
        return [summary] + recent

    def next_action_message(self, task: str, *, iteration: int) -> dict[str, str]:
        """Format the user message that prompts the LLM to act."""
        session_hint = (
            "You can call `read_history()` to review your earlier messages, "
            "or `list_sessions()` to see all agents in the tree.\n\n"
            if self.session
            else ""
        )
        if iteration == 0:
            content = (
                f"Task: {task}\n\n"
                f"{session_hint}"
                "Your response MUST contain exactly one ```repl``` code block. "
                "Put any reasoning as comments inside the block. "
                "Do NOT reply with only text — every response needs a ```repl``` block."
            )
        else:
            content = (
                f"Continue working on the task: {task}\n\n"
                f"{session_hint}"
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
        if self.is_done:
            return self.result
        self.is_done = True
        self.result = message.strip()
        return self.result

    @tool(
        "Delegate a subtask to a named child agent.\nArgs:\n"
        "- name (str): Short identifier for the child (e.g. 'search_batch_0').\n"
        "- task (str): The subtask description.\n"
        "- max_iterations (int | None): Iteration cap.\n"
        "- model (str): Which model to use. Defaults to 'default'.\n"
        "Returns:\n- ChildHandle: Use `yield wait(handle)` to collect the result."
    )
    def delegate(
        self,
        name: str,
        task: str,
        *,
        max_iterations: int | None = None,
        model: str = "default",
    ) -> ChildHandle:
        if self.depth >= self.config.max_depth:
            return f"[refused: max depth {self.config.max_depth}] Do this directly."
        if model not in self.llm_clients:
            keys = ", ".join(sorted(self.llm_clients))
            return f"[error: unknown model {model!r}. available: {keys}]"

        agent_id = f"{self.agent_id}.{name}"

        if agent_id in self.children and self.children[agent_id].is_done:
            child = self.children[agent_id]
            child.last_state = child.start(task)
            return ChildHandle(agent_id)

        if agent_id in self.children:
            agent_id = self._resolve_child_id(name)

        client = self.llm_clients[model]
        child = self.create_child(
            agent_id, max_iterations=max_iterations, llm_client=client
        )
        self.children[agent_id] = child
        child.last_state = child.start(task)
        return ChildHandle(agent_id)

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
        "Wait for delegated children. Must be called with `yield`.\n"
        "Args:\n- *handles: Handles from delegate().\n"
        "Returns:\n- list[str]: Results in order (via yield)."
    )
    def wait(self, *handles: ChildHandle) -> WaitRequest:
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    # ── session tools ─────────────────────────────────────────────

    @tool(
        "List all agent sessions in the tree.\n"
        "Returns a summary of each agent's ID and task."
    )
    def list_sessions(self) -> str:
        if not self.session:
            return ""
        agents = self.session.list_agents()
        if not agents:
            return "No sessions found."
        lines = []
        for aid in agents:
            msgs = self.session.read(aid)
            task = ""
            for m in msgs:
                if m.get("role") == "user":
                    task = m.get("content", "")[:120]
                    break
            lines.append(f"{aid}: {task}")
        return "\n".join(lines)

    @tool(
        "Read an agent's message history.\nArgs:\n"
        "- agent_id (str | None): Agent to read. Defaults to self.\n"
        "- last_n (int): Number of recent messages. Default 20.\n"
        "Returns:\n- str: Formatted message history."
    )
    def read_history(self, agent_id: str | None = None, last_n: int = 20) -> str:
        if not self.session:
            return ""
        aid = agent_id or self.agent_id
        msgs = self.session.read(aid)
        if not msgs:
            return f"No session found for {aid}."
        recent = msgs[-last_n:]
        lines = []
        for m in recent:
            role = m.get("role", "?")
            content = m.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    # ── children ─────────────────────────────────────────────────────

    def create_child(
        self,
        agent_id: str,
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
        child_config = replace(
            self.config, max_iterations=child_iters, session=self.session
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
            pool=self.pool,
            prompt_builder=self.prompt_builder,
        )
