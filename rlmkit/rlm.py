"""Core RLM engine — step-based execution with delegation via generators.

State machine: READY → EXECUTING → (SUPERVISING →) FINISHED
See docs/internal/engine_flow.md for a full walkthrough.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

from .llm import LLMClient
from .pool import CallablePool, ThreadPool
from .prompts.default import DEFAULT_BUILDER
from .prompts.messages import (
    CONTINUE_ACTION,
    EXECUTION_OUTPUT,
    FIRST_ACTION,
    NO_CODE_BLOCK,
    ORPHANED_DELEGATES,
    RESUME_MESSAGE,
    STATUS_DEPTH_MID,
    STATUS_DEPTH_NEAR_MAX,
    STATUS_DEPTH_ROOT,
    TRUNCATION_SESSION_HINT,
    TRUNCATION_SUMMARY,
)
from .runtime import Runtime
from .session import FileSession, Session
from .state import (
    ChildHandle,
    CodeExec,
    LLMReply,
    NoCodeBlock,
    ResumeExec,
    RLMState,
    Status,
    WaitRequest,
)
from .utils import OrphanedDelegatesError, check_yield_errors, find_code_blocks, tool


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


class RLM:
    """Recursive Language Model engine.

    Owns mutable resources (LLM client, runtime, thread pool).
    All observable state lives in ``RLMState`` — frozen, immutable, recursive.
    The engine advances state via ``step(state) → state``.
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
        self.prompt_builder = prompt_builder or DEFAULT_BUILDER

        # Pool
        if pool is None:
            pool = ThreadPool(self.config.max_concurrency)
        elif callable(pool) and not hasattr(pool, "execute"):
            pool = CallablePool(pool)
        self.pool = pool

        # Multi-model routing
        self.llm_clients: dict[str, LLMClient] = {}
        self.model_descriptions: dict[str, str] = {}
        for k, entry in (llm_clients or {}).items():
            self.llm_clients[k] = entry["model"]
            if "description" in entry:
                self.model_descriptions[k] = entry["description"]
        if "default" not in self.llm_clients:
            self.llm_clients["default"] = self.llm_client

        # Mutable engine state (side effects from code execution)
        self.children: dict[str, RLM] = {}
        self.is_done = False
        self.result: str | None = None

        # Session
        if isinstance(self.config.session, str):
            self.config = replace(self.config, session=FileSession(self.config.session))
        self.session: Session | None = self.config.session

        self.register_tools()

    def register_tools(self) -> None:
        self.runtime.inject("OrphanedDelegatesError", OrphanedDelegatesError)
        self.runtime.inject("AGENT_ID", self.agent_id)
        self.runtime.inject("DEPTH", str(self.depth))
        self.runtime.inject("MAX_DEPTH", str(self.config.max_depth))
        self.runtime.register_tool(self.done, core=True)
        self.runtime.register_tool(self.delegate, core=True)
        self.runtime.register_tool(self.wait, core=True)
        if self.session:
            self.runtime.inject("SESSION", self.session)
            self.runtime.register_tool(self.list_sessions, core=True)
            self.runtime.register_tool(self.read_history, core=True)
            self.prompt_builder = self.prompt_builder.section(
                "session",
                self.session.prompt_hint(),
                title="Sessions",
            )

    # ── public API ────────────────────────────────────────────────────

    def make_state(self, **kw) -> RLMState:
        """Build initial state. Override to inject custom fields."""
        return self.state_cls(**kw)

    def start(self, task: str) -> RLMState:
        """Create initial state for a task and reset engine."""
        self.children.clear()
        self.is_done = False
        self.result = None
        cfg = {
            f.name: getattr(self.config, f.name)
            for f in fields(self.config)
            if f.name != "session"
        }
        cfg.update(depth=self.depth, model=getattr(self.llm_client, "model", None))
        state = self.make_state(
            agent_id=self.agent_id,
            task=task,
            status=Status.READY,
            config=cfg,
            messages=[self.first_action_message(task)],
        )
        return state

    def restore(self, saved_state: RLMState) -> RLMState:
        """Resume from a saved state with a fresh engine."""
        state = self.start(saved_state.task)
        msg = self.build_resume_msg(saved_state)
        return state.update(messages=state.messages + [msg])

    def run(self, task: str) -> str:
        """Run to completion. Returns the result string."""
        state = self.start(task)
        while not state.finished:
            state = self.step(state)
        return state.result or ""

    # ── step ─────────────────────────────────────────────────────────

    def step(self, state: RLMState) -> RLMState:
        """One full cycle: LLM → exec → drive children bottom-up → resume."""
        if state.status == Status.READY:
            state = self.step_llm(state)
        if state.status == Status.EXECUTING:
            state = self.step_exec(state)
        if state.status == Status.SUPERVISING:
            state = self.step_children(state)
            if self.can_resume(state):
                state = self.resume_exec(state)
        return state

    # ── state machine: READY ──────────────────────────────────────────

    def step_llm(self, state: RLMState) -> RLMState:
        """Call the LLM → EXECUTING."""
        max_iter = state.config.get("max_iterations", self.config.max_iterations)
        if state.iteration >= max_iter:
            return state.update(status=Status.FINISHED, result=state.last_reply or "")

        messages = self.build_messages(state)
        text = self.call_llm(messages)
        persisted = state.messages + [{"role": "assistant", "content": text}]

        return state.update(
            status=Status.EXECUTING,
            iteration=state.iteration + 1,
            event=LLMReply(
                agent_id=state.agent_id,
                iteration=state.iteration + 1,
                text=text,
                code=self.extract_code(text),
            ),
            messages=persisted,
            last_reply=text,
        )

    # ── state machine: EXECUTING ──────────────────────────────────────

    def step_exec(self, state: RLMState) -> RLMState:
        """Run code from the LLM reply."""
        code = self.extract_code(state.last_reply or "")

        if not code:
            return state.update(
                status=Status.READY,
                event=NoCodeBlock(
                    agent_id=state.agent_id,
                    iteration=state.iteration,
                    text=state.last_reply or "",
                ),
                messages=state.messages + [{"role": "user", "content": NO_CODE_BLOCK}],
            )

        err = check_yield_errors(code)
        if err:
            return state.update(
                status=Status.READY,
                event=CodeExec(
                    agent_id=state.agent_id,
                    iteration=state.iteration,
                    code=code,
                    output=err,
                ),
                messages=state.messages + [self.execution_output_message(code, err)],
            )

        suspended, raw = self.runtime.start_code(code)
        state, output = self.after_exec(state, suspended, raw)
        new_state = state.update(
            event=CodeExec(
                agent_id=state.agent_id,
                iteration=state.iteration,
                code=code,
                output=output,
                suspended=state.status == Status.SUPERVISING,
            ),
            messages=state.messages + [self.execution_output_message(code, output)],
        )
        self.write_session(new_state)
        return new_state

    # ── state machine: SUPERVISING ───────────────────────────────────

    def step_node(self, state: RLMState) -> RLMState:
        """Advance one node: LLM + exec. No child recursion."""
        if state.status == Status.READY:
            state = self.step_llm(state)
        if state.status == Status.EXECUTING:
            state = self.step_exec(state)
        return state

    def step_children(self, state: RLMState) -> RLMState:
        """Step all descendants once, then cascade resumes upward.

        Flatten the tree into {id: state} and {id: engine} maps.
        Build a queue of everything that can run right now, execute
        in parallel, then check if any parents became resumable.
        Each node runs at most once per step (tracked by ``done``).
        Newly spawned children are NOT in the queue — they wait
        for the next outer step() call.
        """
        nodes: dict[str, RLMState] = {}
        engines: dict[str, RLM] = {}
        self.flatten(state, nodes, engines)

        queue = set(nodes.keys())  # only nodes that exist right now
        done: set[str] = set()  # processed this step — skip next round

        while True:
            # Build a batch of everything runnable in the queue
            batch: list[tuple[str, Any]] = []
            for aid in queue - done:
                s = nodes[aid]
                if s.finished:
                    continue
                e = engines[aid]

                # READY / EXECUTING → run LLM + exec (no child recursion)
                if s.status in (Status.READY, Status.EXECUTING):
                    batch.append((aid, lambda e=e, s=s: e.step_node(s)))

                # SUPERVISING → resume generator if all deps finished
                elif s.status == Status.SUPERVISING:
                    if all(nodes[d].finished for d in s.waiting_on):
                        patched = s.update(
                            children=[nodes.get(c.agent_id, c) for c in s.children]
                        )
                        batch.append((aid, lambda e=e, s=patched: e.resume_exec(s)))

            if not batch:
                break

            # Execute batch in parallel, merge results
            for aid, new_s in self.pool.execute(batch).items():
                done.add(aid)
                nodes[aid] = new_s
                engines[aid].flatten(new_s, nodes, engines)

        return self.rebuild(state, nodes)

    def resume_exec(self, state: RLMState) -> RLMState:
        """Resume the suspended generator after children finish."""
        by_id = {c.agent_id: c for c in state.children}
        results = [by_id[aid].result or "" for aid in state.waiting_on]
        suspended, raw = self.runtime.resume_code(results)
        state, output = self.after_exec(state, suspended, raw)
        return state.update(
            event=ResumeExec(
                agent_id=state.agent_id,
                iteration=state.iteration,
                output=output,
            )
        )

    def flatten(
        self, state: RLMState, nodes: dict[str, RLMState], engines: dict[str, RLM]
    ) -> None:
        """Recursively populate flat maps of all descendants."""
        for child in state.children:
            cid = child.agent_id
            nodes[cid] = child
            engines[cid] = self.children[cid]
            self.children[cid].flatten(child, nodes, engines)

    def rebuild(self, state: RLMState, nodes: dict[str, RLMState]) -> RLMState:
        """Recursively reconstruct the immutable tree from the flat map."""
        new_children = []
        for child in state.children:
            cid = child.agent_id
            updated = nodes.get(cid, child)
            if cid in self.children:
                updated = self.children[cid].rebuild(updated, nodes)
            new_children.append(updated)
        return state.update(children=new_children)

    def can_resume(self, state: RLMState) -> bool:
        """True if all waited-on children are finished."""
        if not state.waiting_on:
            return False
        by_id = {c.agent_id: c for c in state.children}
        return all(by_id.get(aid, state).finished for aid in state.waiting_on)

    def after_exec(
        self, state: RLMState, suspended: bool, raw: Any
    ) -> tuple[RLMState, str]:
        """Process execution result. Returns (new_state, output)."""
        old_ids = {c.agent_id for c in state.children}
        spawned = [cid for cid in self.children if cid not in old_ids]

        if spawned and not suspended and not self.is_done:
            for cid in spawned:
                del self.children[cid]
            msg = ORPHANED_DELEGATES.format(names=", ".join(spawned))
            output = self.execute_code(f"raise OrphanedDelegatesError({msg!r})")
            base = raw if isinstance(raw, str) else ""
            if base.strip():
                output = base + "\n\n" + output
            return state.update(status=Status.READY), output

        if suspended:
            waiting_on = raw.agent_ids if isinstance(raw, WaitRequest) else []
            children = state.children + [
                self.children[cid].last_state for cid in spawned
            ]
            return (
                state.update(
                    status=Status.SUPERVISING,
                    children=children,
                    waiting_on=waiting_on,
                ),
                "",
            )

        output = raw if isinstance(raw, str) else ""
        if self.is_done:
            return state.update(status=Status.FINISHED, result=self.result), output
        return state.update(status=Status.READY), output

    # ── prompt building ───────────────────────────────────────────────

    def build_system_prompt(self, state: RLMState) -> str:
        """Build the full system prompt. Override to customize."""
        if self.config.system_prompt:
            return self.config.system_prompt
        return self.prompt_builder.build(
            tools=self.build_tools_section(state),
            status=self.build_status_section(state),
        )

    def build_tools_section(self, state: RLMState) -> str:
        """Build the tools listing for the system prompt. Override to customize."""
        lines = [
            f"- `{t.name}{t.signature}`: {t.description}"
            for t in self.runtime.get_tool_defs()
        ]
        if len(self.llm_clients) > 1:
            lines.append("\nAvailable models for `delegate(model=...)`:")
            for k in sorted(self.llm_clients):
                desc = self.model_descriptions.get(k)
                lines.append(f"- `{k}`: {desc}" if desc else f"- `{k}`")
        mods = self.runtime.available_modules()
        if mods:
            lines.append(f"\nPre-imported: `{'`, `'.join(mods)}`")
        return "\n".join(lines)

    def build_status_section(self, state: RLMState) -> str:
        """Build the depth/status note for the system prompt. Override to customize."""
        depth = state.config.get("depth", 0)
        max_depth = state.config.get("max_depth", self.config.max_depth)
        note = f"You are at recursion depth **{depth}** of max **{max_depth}**."
        if depth == 0 and max_depth > 0:
            note += STATUS_DEPTH_ROOT
        elif depth >= max_depth - 1:
            note += STATUS_DEPTH_NEAR_MAX
        elif depth > 0:
            note += STATUS_DEPTH_MID
        return note

    # ── message construction ──────────────────────────────────────────

    def build_messages(
        self, state: RLMState, *, system_prompt: str | None = None
    ) -> list[dict]:
        """Assemble the message list for an LLM call. Override to customize.

        The action prompt is transient — re-injected each turn, never stored.
        """
        system = {
            "role": "system",
            "content": system_prompt or self.build_system_prompt(state),
        }
        msgs = list(state.messages)
        cap = self.config.max_messages
        if cap and len(msgs) > cap:
            hint = TRUNCATION_SESSION_HINT if self.session else ""
            summary = {
                "role": "user",
                "content": TRUNCATION_SUMMARY.format(
                    task=state.task,
                    total=len(msgs),
                    cap=cap,
                    session_hint=hint,
                ),
            }
            msgs = [summary] + msgs[-cap:]

        if state.iteration > 0:
            msgs.append(self.next_action_message(state.task))
        return [system] + msgs

    def first_action_message(self, task: str) -> dict[str, str]:
        """First user message (stored in state.messages). Override to customize."""
        return {"role": "user", "content": FIRST_ACTION.format(task=task)}

    def next_action_message(self, task: str) -> dict[str, str]:
        """Transient continuation prompt appended every turn (not stored)."""
        return {"role": "user", "content": CONTINUE_ACTION.format(task=task)}

    def execution_output_message(self, code: str, output: str) -> dict[str, str]:
        """Format code + output as a user message. Override to customize."""
        return {
            "role": "user",
            "content": EXECUTION_OUTPUT.format(code=code, output=output),
        }

    # ── LLM / code helpers ────────────────────────────────────────────

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

    # ── REPL tools (injected into agent namespace) ────────────────────

    @tool(
        "Mark the current agent as finished.\nArgs:\n"
        "- message (str): The result — actual data, answer, or summary.\n"
        "Returns: str"
    )
    def done(self, message: str) -> str:
        if self.is_done:
            return self.result
        self.is_done = True
        self.result = message.strip()
        return self.result

    @tool(
        "Delegate a subtask to a named child agent.\nArgs:\n"
        "- name (str): Short identifier (e.g. 'search_batch_0').\n"
        "- task (str): The subtask description.\n"
        "- max_iterations (int | None): Iteration cap.\n"
        "- model (str): Which model to use. Defaults to 'default'.\n"
        "Returns: ChildHandle — use `yield wait(handle)` to collect result."
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
        if agent_id in self.children:
            agent_id = self.unique_child_id(name)

        child = self.create_child(
            agent_id,
            max_iterations=max_iterations,
            llm_client=self.llm_clients[model],
        )
        self.children[agent_id] = child
        child.last_state = child.start(task)
        return ChildHandle(agent_id)

    @tool(
        "Wait for delegated children. Must be called with `yield`.\n"
        "Args:\n- *handles: Handles from delegate().\n"
        "Returns: list[str] — results in order (via yield)."
    )
    def wait(self, *handles: ChildHandle) -> WaitRequest:
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    # ── session tools ─────────────────────────────────────────────────

    @tool("List all agents with IDs and tasks.\nReturns: str")
    def list_sessions(self) -> str:
        agents = self.session.list_agents() if self.session else []
        if not agents:
            return "No sessions found."
        lines = []
        for aid in agents:
            msgs = self.session.read(aid)
            task = next(
                (m["content"][:120] for m in msgs if m.get("role") == "user"),
                "",
            )
            lines.append(f"{aid}: {task}")
        return "\n".join(lines)

    @tool(
        "Read conversation transcript for any agent.\nArgs:\n"
        "- agent_id (str | None): Agent to read. Defaults to self.\n"
        "- last_n (int): Number of recent messages. Default 20.\n"
        "Returns: str"
    )
    def read_history(self, agent_id: str | None = None, last_n: int = 20) -> str:
        if not self.session:
            return ""
        aid = agent_id or self.agent_id
        msgs = self.session.read(aid)
        if not msgs:
            return f"No session found for {aid}."
        return "\n\n".join(
            f"[{m.get('role', '?')}] {m.get('content', '')[:500]}"
            for m in msgs[-last_n:]
        )

    # ── child factory ─────────────────────────────────────────────────

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

        packed = {
            k: {
                "model": c,
                **(
                    {"description": self.model_descriptions[k]}
                    if k in self.model_descriptions
                    else {}
                ),
            }
            for k, c in self.llm_clients.items()
        }
        return self.__class__(
            llm_client=llm_client or self.llm_client,
            runtime=rt,
            config=child_config,
            agent_id=agent_id,
            depth=self.depth + 1,
            runtime_factory=self.runtime_factory,
            llm_clients=packed,
            pool=self.pool,
            prompt_builder=self.prompt_builder,
        )

    def unique_child_id(self, name: str) -> str:
        base = f"{self.agent_id}.{name}"
        n = 2
        while f"{base}_{n}" in self.children:
            n += 1
        return f"{base}_{n}"

    def build_resume_msg(self, saved_state: RLMState) -> dict:
        tree = saved_state.tree(color=False)
        recent = ""
        if self.session:
            msgs = self.session.read(saved_state.agent_id or "root")
            recent = "\n".join(
                f"[{m.get('role', '?')}] {m.get('content', '')[:300]}"
                for m in msgs[-5:]
            )
        return {
            "role": "user",
            "content": RESUME_MESSAGE.format(
                task=saved_state.task,
                tree=tree,
                recent=recent or "(no session history available)",
            ),
        }

    def write_session(self, state: RLMState) -> None:
        if self.session and state.messages:
            msgs = [m for m in state.messages if m.get("role") != "system"]
            self.session.write(state.agent_id, msgs)
