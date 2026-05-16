"""RLMFlow — the recursive language-model orchestrator.

This module holds the public :class:`RLMFlow` class. Mechanics live
under :mod:`rlmflow.engine`:

- per-agent step dispatch and transitions  (``engine.transitions``)
- LLM call / code extraction              (``engine.code``)
- replay-of-one for cold-start resume     (``engine.replay``)
- runtime sessions / env injection / tool registration
                                          (``engine.sessions``)
- ``build_messages`` / system prompt      (``engine.messages``)
- shared ``append_node`` / ``unique_child_id`` (``engine.seq``)

The :class:`RLMFlow` class itself does the orchestration plus a small
amount of core logic (``spawn_child``, the runtime-session binding
methods) that is engine-shaped enough that splitting it into a free
function would just mean passing ``self`` to it as the first argument.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rlmflow.config import RLMConfig
from rlmflow.engine.code import call_llm, extract_code, llm_client_for
from rlmflow.engine.messages import (
    build_messages,
    build_status_section,
    build_system_prompt,
    build_system_prompt_for,
    build_tools_section,
    node_config,
)
from rlmflow.engine.seq import (
    ROOT_RUNTIME_ID,
    append_node,
    create_pool,
    format_exec_output,
    unique_child_id,
)
from rlmflow.engine.sessions import (
    create_runtime_session,
    inject_env,
    register_tools,
    runtime_for,
)
from rlmflow.engine.transitions import step_agent
from rlmflow.graph import ChildHandle, Graph, Node, RuntimeRef, UserQuery
from rlmflow.llm import LLMClient
from rlmflow.prompts.default import BASELINE_BUILDER, DEFAULT_BUILDER
from rlmflow.prompts.messages import (
    CONTEXT_HINT_ABSENT,
    CONTEXT_HINT_PRESENT,
    DEFAULT_QUERY,
    FIRST_ACTION,
)
from rlmflow.runtime import Runtime
from rlmflow.scheduler import NodeScheduler
from rlmflow.workspace import (
    Context,
    InMemoryContext,
    InMemorySession,
    Session,
    Workspace,
)


def _child_config(
    parent: Graph,
    *,
    max_iterations: int | None,
    default_max_iterations: int,
    child_max_iterations: int | None,
) -> dict[str, Any]:
    """Derive the per-child config dict from ``parent.config``.

    ``max_iterations`` (caller override) wins if set. Otherwise
    ``child_max_iterations`` (engine default for children). Otherwise
    a third of the parent's max iterations, floored at 1.
    """
    child_iters = (
        max_iterations
        or child_max_iterations
        or max(
            1,
            parent.config.get("max_iterations", default_max_iterations) // 3,
        )
    )
    return {**parent.config, "max_iterations": child_iters}


class RLMFlow(LLMClient):
    """Recursive language-model flow engine.

    Holds the prompt builder, runtime sessions, pool, and persistence
    handles. The execution graph itself lives in the session — every
    step reloads it through
    :meth:`~rlmflow.workspace.session.Session.load_graph`.
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
        if workspace is None:
            runtime_workspace = getattr(runtime, "workspace", None)
            if runtime_workspace is not None:
                runtime_root = Path(runtime_workspace).resolve()
                if runtime_root != Path.cwd().resolve():
                    workspace = Workspace.create(runtime_root)

        self.llm_client = llm_client
        self.runtime = runtime
        self.workspace = workspace
        self.session: Session = workspace.session if workspace else InMemorySession()
        self.context: Context = workspace.context if workspace else InMemoryContext()
        self.config = config or RLMConfig()
        self.runtime_factory = runtime_factory
        default_builder = (
            BASELINE_BUILDER if self.config.max_depth == 0 else DEFAULT_BUILDER
        )
        self.prompt_builder = prompt_builder or default_builder
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

        self.runtime_sessions: dict[str, Runtime] = {ROOT_RUNTIME_ID: runtime}
        self.terminate_requested: set[str] = set()
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
    ) -> Graph:
        query = query or DEFAULT_QUERY

        self.context.write(
            "context",
            context if context is not None else "",
            agent_id=agent_id,
            metadata=context_metadata,
        )
        for key, value in (contexts or {}).items():
            self.context.write(key, value, agent_id=agent_id)

        context_hint = CONTEXT_HINT_PRESENT if context else CONTEXT_HINT_ABSENT
        root = Graph(
            agent_id=agent_id,
            branch_id=self.workspace.branch_id if self.workspace else "main",
            depth=0,
            query=query,
            system_prompt=build_system_prompt_for(
                self, query=query, agent_id=agent_id, depth=0
            ),
            config=node_config(self),
            workspace=self.workspace.ref() if self.workspace else None,
            runtime=RuntimeRef(id=ROOT_RUNTIME_ID),
        )
        self.session.write_agent(root)
        append_node(
            self.session,
            root,
            UserQuery(
                content=FIRST_ACTION.format(query=query, context_hint=context_hint)
            ),
        )
        return self.session.load_graph()

    def run(self, query: str | None = None, **kwargs) -> str:
        graph = self.start(query, **kwargs)
        while not graph.finished:
            graph = self.step(graph)
        return graph.result()

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

    def step(self, graph: Graph) -> Graph:
        """Advance the run by one synchronized batch.

        All currently-runnable agents are stepped (in parallel if a
        :class:`Pool` is configured). Returns a freshly-loaded
        :class:`Graph` snapshot.
        """
        runnable = self.node_scheduler.runnable_agents(graph)
        if not runnable:
            return graph
        tasks = [(aid, (lambda aid=aid: step_agent(self, aid))) for aid in runnable]
        self.pool.execute(tasks)
        return self.session.load_graph()

    def terminate(self, graph: Graph) -> Graph:
        """Mark every still-running agent for a final-answer turn.

        Equivalent to giving every agent one last chance to emit ``done()``.
        The engine then drives those agents to terminal states as normal.
        """
        for aid in graph.agents:
            if not graph.agents[aid].finished:
                self.terminate_requested.add(aid)
        return self.session.load_graph()

    # ── runtime-session binding methods ──────────────────────────────
    #
    # These pull the relevant state off ``self`` and forward to the
    # flat helpers in :mod:`rlmflow.engine.sessions`. Call sites use
    # ``engine.runtime_for(ref)`` etc. — the binding work happens here,
    # the actual logic lives in the flat helpers.

    def runtime_for(self, ref: RuntimeRef | None) -> Runtime:
        return runtime_for(
            self.runtime_sessions,
            ref,
            root=self.runtime,
            factory=self.runtime_factory,
            on_create=self.register_tools,
        )

    def create_runtime_session(
        self, parent_runtime: Runtime, *, agent_id: str
    ) -> RuntimeRef:
        return create_runtime_session(
            self.runtime_sessions,
            parent_runtime,
            agent_id=agent_id,
            factory=self.runtime_factory,
            on_create=self.register_tools,
        )

    def inject_env(self, graph: Graph, node: Node) -> Runtime:
        return inject_env(
            self.runtime_for(graph.runtime),
            graph,
            node,
            max_depth=self.config.max_depth,
            session=self.session,
            context=self.context,
        )

    def register_tools(self, runtime: Runtime | None = None) -> None:
        register_tools(runtime or self.runtime, spawn_child=self.spawn_child)

    # ── child spawning ───────────────────────────────────────────────

    def spawn_child(
        self,
        parent_agent_id: str,
        parent_node_id: str,
        name: str,
        query: str,
        context: str,
        *,
        max_iterations: int | None = None,
        model: str = "default",
    ) -> ChildHandle | str:
        """Spawn a child agent under ``parent_agent_id`` and return its handle.

        Public seam invoked by the ``delegate(...)`` REPL closure.
        Creates a child :class:`~rlmflow.graph.Graph`, allocates a new
        runtime session, writes the initial seed action, and returns
        a :class:`~rlmflow.graph.ChildHandle`. Returns a refusal
        string instead of a handle if the child cannot be created
        (max depth reached, unknown model, …).
        """
        parent = self.session.load_graph().agents[parent_agent_id]
        if parent.depth >= self.config.max_depth:
            return f"[refused: max depth {self.config.max_depth}] Do this directly."
        if model not in self.llm_clients:
            keys = ", ".join(sorted(self.llm_clients))
            return f"[error: unknown model {model!r}. available: {keys}]"

        child_aid = unique_child_id(parent_agent_id, name, set(parent.children))
        self.context.write("context", context, agent_id=child_aid)

        parent_runtime = self.runtime_for(parent.runtime)
        runtime_ref = self.create_runtime_session(parent_runtime, agent_id=child_aid)

        cfg = {
            **_child_config(
                parent,
                max_iterations=max_iterations,
                default_max_iterations=self.config.max_iterations,
                child_max_iterations=self.config.child_max_iterations,
            ),
            "model": model,
        }
        context_hint = CONTEXT_HINT_PRESENT if context else CONTEXT_HINT_ABSENT
        child_graph = Graph(
            agent_id=child_aid,
            branch_id=parent.branch_id,
            depth=parent.depth + 1,
            query=query,
            system_prompt=self.build_system_prompt_for(
                query=query,
                agent_id=child_aid,
                depth=parent.depth + 1,
                config=cfg,
            ),
            config=cfg,
            workspace=parent.workspace,
            runtime=runtime_ref,
            model=None,
            parent_agent_id=parent.agent_id,
            parent_node_id=parent_node_id,
        )
        self.session.write_agent(child_graph)
        append_node(
            self.session,
            child_graph,
            UserQuery(
                content=FIRST_ACTION.format(query=query, context_hint=context_hint)
            ),
        )
        return ChildHandle(child_aid)

    # ── prompts / messages / LLM ─────────────────────────────────────
    #
    # These functions still take ``engine`` as their first arg because
    # each one touches several engine attributes (config, runtime,
    # llm_clients, model_descriptions, prompt_builder, …) — flattening
    # them would balloon their signatures past ten parameters. The
    # methods below pull the engine forward.

    def build_messages(
        self, graph: Graph, *, force_final: bool = False
    ) -> list[dict[str, str]]:
        return build_messages(self, graph, force_final=force_final)

    def build_system_prompt(self, graph: Graph) -> str:
        return build_system_prompt(self, graph)

    def build_system_prompt_for(
        self,
        *,
        query: str,
        agent_id: str,
        depth: int,
        config: dict[str, Any] | None = None,
    ) -> str:
        return build_system_prompt_for(
            self, query=query, agent_id=agent_id, depth=depth, config=config
        )

    def build_tools_section(self) -> str:
        return build_tools_section(self)

    def build_status_section(self, graph: Graph) -> str:
        return build_status_section(self, graph)

    def node_config(self) -> dict[str, Any]:
        return node_config(self)

    def call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        client: LLMClient | None = None,
    ) -> str:
        return call_llm(self, messages, client=client)

    def llm_client_for(self, graph: Graph) -> LLMClient:
        return llm_client_for(self, graph)

    def extract_code(self, text: str) -> str | None:
        return extract_code(self, text)

    def format_exec_output(self, output: str) -> str:
        return format_exec_output(output)


__all__ = ["NodeScheduler", "RLMConfig", "RLMFlow", "create_pool"]
