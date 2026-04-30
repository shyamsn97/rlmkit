"""RLMFlow — one recursive interpreter over typed nodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.node import (
    ActionNode,
    ChildHandle,
    ErrorNode,
    Node,
    ObservationNode,
    QueryNode,
    ResultNode,
    ResumeNode,
    RuntimeRef,
    SupervisingNode,
    WaitRequest,
)
from rlmflow.pool import CallablePool, Pool, SequentialPool, ThreadPool
from rlmflow.prompts.default import DEFAULT_BUILDER
from rlmflow.prompts.messages import (
    CONTINUE_ACTION,
    DEFAULT_QUERY,
    EXECUTION_OUTPUT,
    FINAL_ANSWER_ACTION,
    FIRST_ACTION,
    NO_CODE_BLOCK,
    ORPHANED_DELEGATES,
    STATUS_DEPTH_MID,
    STATUS_DEPTH_NEAR_MAX,
    STATUS_DEPTH_ROOT,
    TRUNCATION_SESSION_HINT,
    TRUNCATION_SUMMARY,
)
from rlmflow.runtime import Runtime
from rlmflow.tools import tool
from rlmflow.utils import OrphanedDelegatesError, check_yield_errors, find_code_blocks
from rlmflow.workspace import (
    Context,
    ContextVariable,
    InMemoryContext,
    InMemorySession,
    Session,
    Workspace,
)

ROOT_RUNTIME_ID = "root"


@dataclass
class RLMConfig:
    """Engine-level knobs."""

    max_depth: int = 5
    max_iterations: int = 30
    max_output_length: int = 12000
    max_messages: int | None = None
    max_concurrency: int | None = None
    child_max_iterations: int | None = None
    single_block: bool = True
    system_prompt: str | None = None
    max_budget: int | None = None


@dataclass
class ActiveStep:
    """Action-local state captured by done/delegate/wait tool calls."""

    node: ActionNode
    done_result: str | None = None
    delegated: dict[str, QueryNode] = field(default_factory=dict)


def create_pool(config: RLMConfig, pool: Pool | Callable | None = None) -> Pool:
    if pool is not None:
        return pool if hasattr(pool, "execute") else CallablePool(pool)
    if config.max_concurrency is not None:
        return ThreadPool(config.max_concurrency)
    return SequentialPool()


class NodeScheduler:
    """Step runnable supervised descendants as one pool batch."""

    def __init__(self, pool: Pool | None = None) -> None:
        self.pool = pool

    def runnable_nodes(self, root: SupervisingNode) -> list[Node]:
        runnable: list[Node] = []

        def visit(node: Node) -> None:
            if node.terminal:
                return
            if isinstance(node, SupervisingNode):
                if all(child.terminal for child in node.children):
                    runnable.append(node)
                    return
                for child in node.children:
                    visit(child)
                return
            runnable.append(node)

        for child in root.children:
            visit(child)
        return runnable

    def rebuild(
        self, root: SupervisingNode, updates: dict[str, Node]
    ) -> SupervisingNode:
        def replace_node(node: Node) -> Node:
            updated = updates.get(node.id) or updates.get(node.agent_id)
            if updated is not None:
                return updated
            if isinstance(node, SupervisingNode):
                return node.update(
                    children=[replace_node(child) for child in node.children]
                )
            return node

        return root.update(children=[replace_node(child) for child in root.children])

    def step_runnable_nodes(
        self,
        root: SupervisingNode,
        flow: RLMFlow,
    ) -> SupervisingNode:
        runnable = self.runnable_nodes(root)
        tasks = [
            (node.id, lambda node=node: flow.step_local(node)) for node in runnable
        ]
        pool = self.pool or flow.pool
        updates = pool.execute(tasks) if tasks else {}
        return self.rebuild(root, updates)


class RLMFlow(LLMClient):
    """Recursive language-model flow engine.

    One RLMFlow interprets the recursive node graph. Agent-specific scope lives
    on each node, and REPL continuity lives in runtime sessions.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        runtime: Runtime | None = None,
        config: RLMConfig | None = None,
        runtime_factory: Callable[[], Runtime] | None = None,
        llm_clients: dict[str, dict] | None = None,
        pool: Any = None,
        prompt_builder: Any = None,
        *,
        workspace: Workspace | None = None,
        node_scheduler: NodeScheduler | None = None,
    ) -> None:
        if workspace is not None and runtime is None:
            runtime = workspace.materialize_runtime()
        if runtime is None:
            raise ValueError("RLMFlow requires either runtime= or workspace=.")

        self.llm_client = llm_client
        self.runtime = runtime
        self.workspace = workspace
        self.session: Session = workspace.session if workspace else InMemorySession()
        self.context: Context = workspace.context if workspace else InMemoryContext()
        self.config = config or RLMConfig()
        self.runtime_factory = runtime_factory
        self.prompt_builder = prompt_builder or DEFAULT_BUILDER
        self.pool = create_pool(self.config, pool)
        self.node_scheduler = node_scheduler or NodeScheduler()

        self.llm_clients: dict[str, LLMClient] = {}
        self.model_descriptions: dict[str, str] = {}
        for key, entry in (llm_clients or {}).items():
            self.llm_clients[key] = entry["model"]
            if "description" in entry:
                self.model_descriptions[key] = entry["description"]
        if "default" not in self.llm_clients:
            self.llm_clients["default"] = self.llm_client

        self._runtime_sessions: dict[str, Runtime] = {ROOT_RUNTIME_ID: runtime}
        self.register_tools(runtime)

    # ── lifecycle ────────────────────────────────────────────────────

    def start(
        self,
        query: str | None = None,
        *,
        context: str | None = None,
        contexts: dict[str, str] | None = None,
        context_metadata: dict[str, Any] | None = None,
        agent_id: str = "root",
    ) -> QueryNode:
        query = query or DEFAULT_QUERY

        if context is not None:
            self.context.write(
                "context",
                context,
                agent_id=agent_id,
                metadata=context_metadata,
            )
        for key, value in (contexts or {}).items():
            self.context.write(key, value, agent_id=agent_id)

        root = QueryNode(
            agent_id=agent_id,
            depth=0,
            query=query,
            runtime=RuntimeRef(id=ROOT_RUNTIME_ID),
            system_prompt=self.build_system_prompt_for(
                query=query,
                agent_id=agent_id,
                depth=0,
            ),
            config=self.node_config(),
            workspace=self.workspace.ref() if self.workspace else None,
            branch_id=self.workspace.branch_id if self.workspace else "main",
            content=FIRST_ACTION.format(query=query),
        )
        return self.record(root)

    def run(self, query: str | None = None, **kwargs) -> str:
        node = self.start(query, **kwargs)
        while not node.terminal:
            node = self.step(node)
        return node.result if isinstance(node, ResultNode) else ""

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        query = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        return self.run(query)

    def step(self, node: Node, *, use_cache: bool = True) -> Node:
        if node.terminal:
            return node
        if isinstance(node, SupervisingNode):
            return self.step_supervising(node)
        if isinstance(node, ActionNode):
            return self.step_action(node)
        if isinstance(node, ObservationNode):
            return self.step_observation(node, use_cache=use_cache)
        raise TypeError(f"Unknown node type: {type(node).__name__}")

    def step_local(self, node: Node) -> Node:
        return self.step(node)

    def terminate(self, node: Node) -> Node:
        if node.terminal:
            return node
        max_iter = node.config.get("max_iterations", self.config.max_iterations)
        iteration = self.iteration_of(node)
        config = node.config
        if iteration >= max_iter:
            config = {**node.config, "max_iterations": iteration + 1}
        if isinstance(node, SupervisingNode):
            return node.update(
                terminate_requested=True,
                config=config,
                children=[self.terminate(child) for child in node.children],
            )
        return node.update(terminate_requested=True, config=config)

    # ── stepping ──────────────────────────────────────────────────────

    def step_observation(
        self, node: ObservationNode, *, use_cache: bool = True
    ) -> Node:
        if self.exceeds_budget(node):
            return self.record_successor(
                node,
                node.successor(
                    ResultNode, result=f"[budget exceeded: {node.tree_tokens} tokens]"
                ),
            )

        max_iter = node.config.get("max_iterations", self.config.max_iterations)
        if self.iteration_of(node) >= max_iter and not node.terminate_requested:
            node = node.update(terminate_requested=True)

        action = self.reply_to(node, use_cache=use_cache)
        self.record_successor(node, action)
        if isinstance(action, ErrorNode):
            return action
        return self.step_action(action)

    def reply_to(
        self,
        node: ObservationNode,
        *,
        use_cache: bool = True,
    ) -> ActionNode | ErrorNode:
        del use_cache
        messages = self.build_messages(node)
        client = self.llm_client_for(node)
        raw = self.call_llm(messages, node=node, client=client)
        usage = client.last_usage or LLMUsage()
        code = self.extract_code(raw)
        if not code:
            return node.successor(
                ErrorNode, content=NO_CODE_BLOCK, error="no_code_block"
            )
        return node.successor(
            ActionNode,
            reply=raw,
            code=code,
            model=getattr(client, "model", None),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_input_tokens=node.total_input_tokens + usage.input_tokens,
            total_output_tokens=node.total_output_tokens + usage.output_tokens,
        )

    def step_action(self, node: ActionNode) -> Node:
        err = check_yield_errors(node.code)
        if err:
            return self.record_successor(
                node,
                node.successor(
                    ErrorNode, code=node.code, content=err, error="invalid_yield"
                ),
            )

        with self.active_step(node) as step:
            suspended, raw = self.run_code(node, node.code)

        spawned = list(step.delegated)
        if spawned and not suspended and step.done_result is None:
            msg = ORPHANED_DELEGATES.format(names=", ".join(spawned))
            base = raw if isinstance(raw, str) else ""
            output = self.execute_code(node, f"raise OrphanedDelegatesError({msg!r})")
            content = (base + "\n\n" + output).strip()
            return self.record_successor(
                node,
                node.successor(
                    ErrorNode,
                    code=node.code,
                    content=self.format_exec_output(node.code, content),
                    error="orphaned_delegates",
                ),
            )

        if step.done_result is not None:
            return self.record_successor(
                node,
                node.successor(
                    ResultNode,
                    result=step.done_result.strip(),
                    children=list(node.children),
                ),
            )

        if suspended:
            request, pre_output = raw
            children = [
                step.delegated[aid]
                for aid in request.agent_ids
                if aid in step.delegated
            ]
            return self.record_successor(
                node,
                node.successor(
                    SupervisingNode,
                    code=node.code,
                    output=pre_output,
                    waiting_on=request.agent_ids,
                    children=children,
                ),
            )

        output = raw if isinstance(raw, str) else ""
        if not output.strip():
            output = "(no output)"
        return self.record_successor(
            node,
            node.successor(
                ObservationNode,
                code=node.code,
                output=output,
                content=self.format_exec_output(node.code, output),
            ),
        )

    def step_supervising(self, node: SupervisingNode) -> Node:
        if self.can_resume(node):
            return self.resume_supervisor(node)
        return self.node_scheduler.step_runnable_nodes(node, self)

    def resume_supervisor(self, node: SupervisingNode) -> Node:
        by_id = {child.agent_id: child for child in node.children}
        results = [
            (
                by_id[agent_id].result
                if isinstance(by_id.get(agent_id), ResultNode)
                else ""
            )
            for agent_id in node.waiting_on
        ]

        with self.active_step(node) as step:
            suspended, raw = self.resume_code(node, results)

        if step.done_result is not None:
            return self.record_successor(
                node,
                node.successor(
                    ResultNode,
                    result=step.done_result.strip(),
                    children=list(node.children),
                ),
            )

        if suspended:
            request, pre_output = raw
            children = [
                step.delegated[aid]
                for aid in request.agent_ids
                if aid in step.delegated
            ]
            return self.record_successor(
                node,
                node.successor(
                    SupervisingNode,
                    output=pre_output,
                    waiting_on=request.agent_ids,
                    children=children,
                ),
            )

        output = raw if isinstance(raw, str) else ""
        child_summary = "\n".join(
            f"  {agent_id}: {getattr(by_id.get(agent_id), 'result', '') or '(no result)'}"
            for agent_id in node.waiting_on
        )
        content = (
            f"Children finished:\n{child_summary}\n\n"
            f"Generator resumed. Output:\n{output or '(no output)'}"
        )
        return self.record_successor(
            node,
            node.successor(
                ResumeNode,
                output=output,
                content=content,
                resumed_from=list(node.waiting_on),
                children=[child.id for child in node.children],
            ),
        )

    # ── graph/context bookkeeping ─────────────────────────────────────

    def record(self, node: Node) -> Node:
        self.session.write(node)
        return node

    def record_successor(self, prev: Node, node: Node) -> Node:
        self.record(prev.update(children=[*prev.children, node.id]))
        return self.record(node)

    def chain_to_root(self, node: Node) -> list[Node]:
        return self.session.chain_to(node)

    def iteration_of(self, node: Node) -> int:
        return sum(isinstance(item, ActionNode) for item in self.chain_to_root(node))

    def exceeds_budget(self, node: Node) -> bool:
        return (
            self.config.max_budget is not None
            and node.tree_tokens >= self.config.max_budget
        )

    def can_resume(self, node: SupervisingNode) -> bool:
        if not node.waiting_on:
            return False
        by_id = {child.agent_id: child for child in node.children}
        return all(
            isinstance(by_id.get(agent_id), ResultNode) for agent_id in node.waiting_on
        )

    # ── runtime sessions ──────────────────────────────────────────────

    def runtime_for(self, node_or_ref: Node | RuntimeRef | None) -> Runtime:
        ref = node_or_ref.runtime if isinstance(node_or_ref, Node) else node_or_ref
        session_id = ref.id if ref is not None else ROOT_RUNTIME_ID
        return self._runtime_sessions[session_id]

    def create_runtime_session(self, parent: Node, *, agent_id: str) -> RuntimeRef:
        parent_runtime = self.runtime_for(parent)
        session_id = f"{agent_id}:{uuid4().hex[:8]}"
        runtime = (
            self.runtime_factory() if self.runtime_factory else parent_runtime.clone()
        )
        self._runtime_sessions[session_id] = runtime
        self.register_tools(runtime)
        return RuntimeRef(id=session_id)

    def prepare_runtime(self, node: Node) -> Runtime:
        runtime = self.runtime_for(node)
        runtime.inject("OrphanedDelegatesError", OrphanedDelegatesError)
        runtime.inject("AGENT_ID", node.agent_id)
        runtime.inject("DEPTH", str(node.depth))
        runtime.inject("MAX_DEPTH", str(self.config.max_depth))
        runtime.inject(
            "RLM_SESSION",
            {
                "agent_id": node.agent_id,
                "node_id": node.id,
                "branch_id": node.branch_id,
            },
        )
        if self.context.list_contexts(agent_id=node.agent_id):
            runtime.inject(
                "CONTEXT", ContextVariable(self.context, agent_id=node.agent_id)
            )
        return runtime

    def run_code(self, node: ActionNode, code: str) -> tuple[bool, object]:
        runtime = self.prepare_runtime(node)
        suspended, raw = runtime.start_code(code)
        if isinstance(raw, str) and len(raw) > self.config.max_output_length:
            raw = raw[: self.config.max_output_length] + "\n...<truncated>"
        return suspended, raw

    def resume_code(
        self, node: SupervisingNode, results: list[str]
    ) -> tuple[bool, object]:
        runtime = self.prepare_runtime(node)
        suspended, raw = runtime.resume_code(results)
        if isinstance(raw, str) and len(raw) > self.config.max_output_length:
            raw = raw[: self.config.max_output_length] + "\n...<truncated>"
        return suspended, raw

    def execute_code(self, node: Node, code: str) -> str:
        output = self.prepare_runtime(node).execute(code)
        if len(output) > self.config.max_output_length:
            return output[: self.config.max_output_length] + "\n...<truncated>"
        return output

    # ── LLM / messages ────────────────────────────────────────────────

    def build_messages(self, leaf: Node) -> list[dict[str, str]]:
        system = {
            "role": "system",
            "content": leaf.system_prompt or self.build_system_prompt(leaf),
        }
        msgs: list[dict[str, str]] = []
        for node in self.chain_to_root(leaf):
            if isinstance(node, ResultNode):
                continue
            if isinstance(node, ObservationNode):
                msgs.append({"role": "user", "content": node.content})
            elif isinstance(node, ActionNode):
                msgs.append({"role": "assistant", "content": node.reply})

        cap = self.config.max_messages
        if cap and len(msgs) > cap:
            hint = TRUNCATION_SESSION_HINT if self.workspace else ""
            msgs = [
                {
                    "role": "user",
                    "content": TRUNCATION_SUMMARY.format(
                        query=leaf.query,
                        total=len(msgs),
                        cap=cap,
                        session_hint=hint,
                    ),
                }
            ] + msgs[-cap:]

        if leaf.terminate_requested:
            msgs.append({"role": "user", "content": FINAL_ANSWER_ACTION})
        elif self.iteration_of(leaf) > 0:
            msgs.append(
                {"role": "user", "content": CONTINUE_ACTION.format(query=leaf.query)}
            )
        return [system] + msgs

    def llm_client_for(self, node: Node) -> LLMClient:
        model = node.config.get("model", "default")
        return self.llm_clients.get(model, self.llm_client)

    def call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        node: Node | None = None,
        client: LLMClient | None = None,
    ) -> str:
        del node
        active_client = client or self.llm_client
        result = "".join(active_client.stream(messages))
        self.last_usage = active_client.last_usage
        return result

    def extract_code(self, text: str) -> str | None:
        blocks = find_code_blocks(text)
        if not blocks:
            return None
        return blocks[0] if self.config.single_block else "\n\n".join(blocks)

    def format_exec_output(self, code: str, output: str) -> str:
        return EXECUTION_OUTPUT.format(code=code, output=output or "(no output)")

    def build_system_prompt_for(self, *, query: str, agent_id: str, depth: int) -> str:
        node = QueryNode(
            agent_id=agent_id,
            depth=depth,
            query=query,
            config=self.node_config(),
        )
        return self.build_system_prompt(node)

    def build_system_prompt(self, node: Node) -> str:
        if self.config.system_prompt:
            return self.config.system_prompt
        builder = self.prompt_builder
        hint = self.context.context_prompt_hint(agent_id=node.agent_id)
        if hint:
            builder = builder.section("context", hint, title="Context")
        return builder.build(
            tools=self.build_tools_section(),
            status=self.build_status_section(node),
        )

    def build_tools_section(self) -> str:
        lines = [
            f"- `{tool_def.name}{tool_def.signature}`: {tool_def.description}"
            for tool_def in self.runtime.get_tool_defs()
        ]
        if len(self.llm_clients) > 1:
            lines.append("\nAvailable models for `delegate(model=...)`:")
            for key in sorted(self.llm_clients):
                desc = self.model_descriptions.get(key)
                lines.append(f"- `{key}`: {desc}" if desc else f"- `{key}`")
        modules = self.runtime.available_modules()
        if modules:
            lines.append(f"\nPre-imported: `{'`, `'.join(modules)}`")
        return "\n".join(lines)

    def build_status_section(self, node: Node) -> str:
        max_depth = node.config.get("max_depth", self.config.max_depth)
        note = f"You are at recursion depth **{node.depth}** of max **{max_depth}**."
        if node.depth == 0 and max_depth > 0:
            note += STATUS_DEPTH_ROOT
        elif node.depth >= max_depth - 1:
            note += STATUS_DEPTH_NEAR_MAX
        elif node.depth > 0:
            note += STATUS_DEPTH_MID
        return note

    def node_config(self) -> dict[str, Any]:
        return {
            "model": "default",
            "max_depth": self.config.max_depth,
            "max_iterations": self.config.max_iterations,
            "max_output_length": self.config.max_output_length,
            "max_messages": self.config.max_messages,
            "child_max_iterations": self.config.child_max_iterations,
            "single_block": self.config.single_block,
            "max_budget": self.config.max_budget,
        }

    def child_config(
        self, parent: ActionNode, max_iterations: int | None
    ) -> dict[str, Any]:
        child_iters = (
            max_iterations
            or self.config.child_max_iterations
            or max(
                1, parent.config.get("max_iterations", self.config.max_iterations) // 3
            )
        )
        return {**parent.config, "max_iterations": child_iters}

    # ── tools ─────────────────────────────────────────────────────────

    def register_tools(self, runtime: Runtime | None = None) -> None:
        runtime = runtime or self.runtime
        runtime.inject("OrphanedDelegatesError", OrphanedDelegatesError)
        runtime.register_tool(self.done, core=True)
        runtime.register_tool(self.delegate, core=True)
        runtime.register_tool(self.wait, core=True)

    def active_step(self, node: ActionNode):
        flow = self

        class ActiveStepScope:
            def __enter__(self) -> ActiveStep:
                step = ActiveStep(node=node)
                runtime = flow.prepare_runtime(node)
                flow.bind_step_tools(runtime, step)
                return step

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        return ActiveStepScope()

    def bind_step_tools(self, runtime: Runtime, step: ActiveStep) -> None:
        def done(message: str) -> str:
            return self.done_for_step(step, message)

        def delegate(
            name: str,
            query: str,
            *,
            context: str | None = None,
            max_iterations: int | None = None,
            model: str = "default",
        ) -> ChildHandle | str:
            return self.delegate_for_step(
                step,
                name,
                query,
                context=context,
                max_iterations=max_iterations,
                model=model,
            )

        def wait(*handles: ChildHandle) -> WaitRequest:
            return self.wait_for_step(step, *handles)

        runtime.inject("done", done)
        runtime.inject("delegate", delegate)
        runtime.inject("wait", wait)

    @tool("Mark the current agent as finished.")
    def done(self, message: str) -> str:
        raise RuntimeError("done() is bound to the active runtime step")

    def done_for_step(self, step: ActiveStep, message: str) -> str:
        if step.done_result is not None:
            return step.done_result
        step.done_result = message.strip()
        print(f"[done] {step.done_result}")
        return step.done_result

    @tool("Delegate a subtask to a named child agent.")
    def delegate(
        self,
        name: str,
        query: str,
        *,
        context: str | None = None,
        max_iterations: int | None = None,
        model: str = "default",
    ) -> ChildHandle | str:
        raise RuntimeError("delegate() is bound to the active runtime step")

    def delegate_for_step(
        self,
        step: ActiveStep,
        name: str,
        query: str,
        *,
        context: str | None = None,
        max_iterations: int | None = None,
        model: str = "default",
    ) -> ChildHandle | str:
        parent = step.node
        if parent.depth >= self.config.max_depth:
            return f"[refused: max depth {self.config.max_depth}] Do this directly."
        if model not in self.llm_clients:
            keys = ", ".join(sorted(self.llm_clients))
            return f"[error: unknown model {model!r}. available: {keys}]"

        agent_id = self.unique_child_id(parent, name, step.delegated)
        if context is not None:
            self.context.write("context", context, agent_id=agent_id)
        runtime_ref = self.create_runtime_session(parent, agent_id=agent_id)
        child = QueryNode(
            agent_id=agent_id,
            depth=parent.depth + 1,
            query=query,
            runtime=runtime_ref,
            workspace=parent.workspace,
            branch_id=parent.branch_id,
            config={**self.child_config(parent, max_iterations), "model": model},
            system_prompt=self.build_system_prompt_for(
                query=query,
                agent_id=agent_id,
                depth=parent.depth + 1,
            ),
            content=FIRST_ACTION.format(query=query),
        )
        step.delegated[agent_id] = self.record(child)
        return ChildHandle(agent_id)

    @tool("Wait for delegated children. Must be called with `yield`.")
    def wait(self, *handles: ChildHandle) -> WaitRequest:
        raise RuntimeError("wait() is bound to the active runtime step")

    def wait_for_step(self, step: ActiveStep, *handles: ChildHandle) -> WaitRequest:
        del step
        return WaitRequest(agent_ids=[handle.agent_id for handle in handles])

    def unique_child_id(
        self,
        parent: ActionNode,
        name: str,
        delegated: dict[str, QueryNode],
    ) -> str:
        base = f"{parent.agent_id}.{name}"
        if base not in delegated:
            return base
        i = 1
        while f"{base}_{i}" in delegated:
            i += 1
        return f"{base}_{i}"

    def _pack_llm_clients(self, default: LLMClient | None = None) -> dict[str, dict]:
        out = {
            key: {"model": model, "description": self.model_descriptions.get(key, "")}
            for key, model in self.llm_clients.items()
        }
        if default is not None:
            out["default"] = {
                "model": default,
                "description": self.model_descriptions.get("default", ""),
            }
        return out


__all__ = ["NodeScheduler", "RLMConfig", "RLMFlow"]
