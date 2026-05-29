"""RLMFlow — the recursive language-model orchestrator.

This module holds :class:`RLMFlow`, the engine. Every piece of
behavior a user might want to customize is a method on this class —
override what you want, call ``super()`` for default behavior.

Pure helpers live under :mod:`rlmflow.engine`:

- :mod:`rlmflow.engine.actions` — :class:`Action` types and the pure
  projection ``Graph -> ActionPlan``.
- :mod:`rlmflow.engine.replay` — cold-start replay-of-one for
  rebuilding a suspended coroutine after a fork or process restart.
- :mod:`rlmflow.engine.scheduling` — implementation of the outer
  ``step`` loop and async-child refill policy.
- :mod:`rlmflow.engine.transitions` — implementation of action-to-state
  transition handlers.
- :mod:`rlmflow.engine.seq` — tiny pure helpers (sequence numbers,
  output truncation, the pool factory).
- :mod:`rlmflow.engine.config` — :class:`RLMConfig` (pure data).

Nothing under ``engine/`` holds engine state; ``RLMFlow`` does.

The class is grouped:

1. Construction
2. Lifecycle           — ``start`` / ``run`` / ``chat`` / ``step`` / ``terminate``
3. Per-step transitions — ``apply_one`` and the three half-step
                          handlers (LLM / exec / resume-after-supervising)
4. LLM half-step       — ``reply_to`` / ``call_llm`` / ``llm_client_for`` /
                          ``extract_code`` (+ private transcript writer)
5. Messages / prompt   — ``build_messages`` / ``build_system_prompt`` /
                          ``build_tools_section`` / ``build_status_section``
6. Runtime / env       — ``runtime_for`` / ``create_runtime_session`` /
                          ``inject_env`` / ``register_tools`` /
                          ``format_exec_output``
7. Child spawning      — ``spawn_child``
8. Bookkeeping         — ``record_usage`` / ``node_config``
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import uuid4

from rlmflow.engine import scheduling, transitions
from rlmflow.engine.actions import Action
from rlmflow.engine.config import RLMConfig
from rlmflow.engine.scheduler import NodeScheduler
from rlmflow.engine.seq import (
    ROOT_RUNTIME_ID,
    append_node,
    create_pool,
    format_exec_output,
    unique_child_id,
)
from rlmflow.engine.transcript import TranscriptRecorder
from rlmflow.graph import (
    ChildHandle,
    ExecAction,
    Graph,
    LLMOutput,
    Node,
    RuntimeRef,
    SupervisingOutput,
    UserQuery,
    is_errored,
    is_exec_output,
    is_llm_output,
    is_user_query,
)
from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.prompts.default import DEFAULT_BUILDER
from rlmflow.prompts.messages import (
    DEFAULT_QUERY,
    FINAL_ANSWER_ACTION,
    STATUS_DEPTH_MID,
    STATUS_DEPTH_NEAR_MAX,
    STATUS_DEPTH_ROOT,
    TRUNCATION_SESSION_HINT,
    TRUNCATION_SUMMARY,
    build_user_prompt,
)
from rlmflow.runtime import LocalRuntime, Runtime
from rlmflow.tools import tool
from rlmflow.tools.builtins import (
    SHOW_VARS,
    make_delegate,
    make_done,
    make_launch_subagent,
    make_launch_subagents,
    make_wait,
)
from rlmflow.utils import find_code_blocks
from rlmflow.workspace import (
    Context,
    ContextVariable,
    InMemoryContext,
    InMemorySession,
    Session,
    SessionVariable,
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
    :meth:`~rlmflow.workspace.Session.load_graph`.

    Every method below is an extension seam. Subclass and override
    what you want; the default implementations call ``super()`` paths
    or pure helpers from :mod:`rlmflow.engine`.

    Overridable surface:

    - **Lifecycle:** :meth:`start` / :meth:`run` / :meth:`chat` /
      :meth:`step` / :meth:`terminate`
    - **Per-step transitions:** :meth:`apply_one` / :meth:`step_llm` /
      :meth:`step_exec` / :meth:`step_after_supervising`
    - **LLM half-step:** :meth:`reply_to` / :meth:`call_llm` /
      :meth:`llm_client_for` / :meth:`extract_code`
    - **Messages / prompt:** :meth:`build_messages` /
      :meth:`build_system_prompt` / :meth:`build_system_prompt_for` /
      :meth:`build_tools_section` / :meth:`build_status_section`
    - **Runtime / env:** :meth:`runtime_for` /
      :meth:`create_runtime_session` / :meth:`inject_env` /
      :meth:`register_tools` / :meth:`format_exec_output`
    - **Child spawning:** :meth:`spawn_child`
    - **Bookkeeping:** :meth:`record_usage` / :meth:`node_config`
    """

    # ── construction ─────────────────────────────────────────────────

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
        if workspace is None and runtime is None:
            raise ValueError("RLMFlow requires either runtime= or workspace=.")
        if workspace is not None and runtime is None:
            runtime = LocalRuntime(workspace=workspace)
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
        self.transcript_recorder = TranscriptRecorder(self.session)
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

        self.runtime_sessions: dict[str, Runtime] = {ROOT_RUNTIME_ID: runtime}
        self.terminate_requested: set[str] = set()
        self.last_usage: LLMUsage | None = None
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

        context_hint = self.context.list_contexts(agent_id=agent_id)
        context_info = self._context_info(agent_id)
        root = Graph(
            agent_id=agent_id,
            branch_id=self.workspace.branch_id if self.workspace else "main",
            depth=0,
            query=query,
            system_prompt=self.build_system_prompt_for(
                query=query,
                agent_id=agent_id,
                depth=0,
            ),
            config=self.node_config(),
            workspace=self.workspace.ref() if self.workspace else None,
            runtime=RuntimeRef(id=ROOT_RUNTIME_ID),
        )
        self.session.write_agent(root)
        append_node(
            self.session,
            root,
            UserQuery(
                content=build_user_prompt(
                    query=query,
                    iteration=0,
                    depth=0,
                    max_depth=self.config.max_depth,
                    context_keys=context_hint,
                    context_info=context_info,
                )
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

        Two phases:

        1. **Plan** — :func:`rlmflow.engine.actions.act` projects
           every runnable agent's current observation into an
           :class:`~rlmflow.engine.actions.Action` (pure, no I/O).
        2. **Apply** — every action is materialized in parallel via
           :meth:`apply_one`, which writes the resulting
           ``(ActionNode, ObservationNode)`` pair through the session.

        Returns a freshly-loaded :class:`Graph` snapshot.
        """
        return scheduling.step(self, graph)

    def _refill_eager_children(
        self,
        _done_id: str,
        _result: object,
        active_ids: set[str],
    ) -> list[tuple[str, Callable[[], None]]]:
        """Return newly runnable eager-child tasks after one task completes."""
        return scheduling.refill_eager_children(self, _done_id, _result, active_ids)

    def terminate(self, graph: Graph) -> Graph:
        """Mark every still-running agent for a final-answer turn.

        Equivalent to giving every agent one last chance to emit ``done()``.
        The engine then drives those agents to terminal states as normal.
        """
        for aid in graph.agents:
            if not graph.agents[aid].finished:
                self.terminate_requested.add(aid)
        return self.session.load_graph()

    # ── per-step transitions ─────────────────────────────────────────

    def apply_one(self, action: Action) -> None:
        """Materialize one :class:`Action` against the persisted graph.

        Reloads the graph from ``self.session``, enforces the global
        token budget, and dispatches to the half-step handler keyed
        by action type. The dispatch logic itself lives in
        :func:`rlmflow.engine.actions.act_one`; this method does no
        re-decisioning.
        """
        return transitions.apply_one(self, action)

    def step_llm(
        self,
        graph: Graph,
        last: Node,
        *,
        force_final: bool,
        model: str | None = None,
    ) -> None:
        """LLM half of one turn: write ``LLMAction → LLMOutput``.

        ``last`` is the observation the LLM is replying to (a
        :class:`UserQuery`, :class:`ExecOutput`, or :class:`ErrorOutput`).
        ``force_final`` is the policy decision (computed by
        :func:`~rlmflow.engine.actions.act_one`) to force a terminal
        answer this turn. ``model`` optionally overrides
        ``graph.config['model']`` for this single call.

        The next :meth:`apply_one` round will see :class:`LLMOutput`
        as the current state and run :meth:`step_exec` against it.
        """
        return transitions.step_llm(
            self,
            graph,
            last,
            force_final=force_final,
            model=model,
        )

    def step_exec(self, graph: Graph, llm_output: LLMOutput) -> None:
        """Exec half of one turn: write ``ExecAction → CodeObservation``.

        Reads the code from ``llm_output`` (the assistant's reply
        rendered as a code block), runs it through the runtime, and
        persists the resulting :class:`CodeObservation` (one of
        :class:`ExecOutput` / :class:`SupervisingOutput` /
        :class:`ErrorOutput` / :class:`DoneOutput`).
        """
        return transitions.step_exec(self, graph, llm_output)

    def _run_exec(
        self,
        graph: Graph,
        exec_action: ExecAction,
        code: str,
    ) -> None:
        return transitions.run_exec(self, graph, exec_action, code)

    def step_after_supervising(
        self,
        graph: Graph,
        last: SupervisingOutput,
    ) -> None:
        """Resume half: write ``ResumeAction → CodeObservation``.

        Drives the supervising agent forward after its waited-on
        children have settled. On a cold start (process restart or
        fork), the live coroutine is gone — we replay the action code
        with ``rlm_delegate`` in replay mode so it pauses at the same
        await before the regular resume path takes over.
        """
        return transitions.step_after_supervising(self, graph, last)

    # ── LLM half-step ────────────────────────────────────────────────

    def reply_to(
        self,
        graph: Graph,
        last: Node,
        *,
        force_final: bool,
    ) -> tuple[LLMOutput, LLMUsage]:
        """Ask the LLM for the next turn; return ``(LLMOutput, LLMUsage)``.

        Always returns an :class:`LLMOutput`, even when the reply has
        no parseable code block (in which case ``LLMOutput.code`` is
        ``""``). The caller is responsible for handling the empty-code
        case by appending a follow-up :class:`ErrorOutput` (with
        ``error="no_code_block"``).

        The returned ``LLMUsage`` is the per-call usage; the caller
        (typically :meth:`step_llm`) decides whether to cache it as
        ``self.last_usage`` via :meth:`record_usage`.
        """
        messages = self.build_messages(graph, force_final=force_final)
        client = self.llm_client_for(graph)
        t0 = time.time()
        raw, usage = self.call_llm(messages, client=client)
        elapsed_s = round(time.time() - t0, 3)
        code = self.extract_code(raw)
        self.transcript_recorder.record_turn(
            graph=graph,
            last=last,
            messages=messages,
            client=client,
            force_final=force_final,
            raw=raw,
            usage=usage,
            elapsed_s=elapsed_s,
        )
        output = LLMOutput(
            agent_id=graph.agent_id,
            seq=last.seq + 1,
            reply=raw,
            code=code or "",
            model=getattr(client, "model", None),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return output, usage

    def call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        client: LLMClient | None = None,
    ) -> tuple[str, LLMUsage]:
        """Stream a chat completion and return ``(text, usage)``.

        Override to add retries, caching, mocking, etc. Defaults to
        streaming from ``client`` (or ``self.llm_client`` if omitted)
        and returning the post-call ``LLMUsage``.
        """
        active = client or self.llm_client
        text = "".join(active.stream(messages))
        usage = active.last_usage or LLMUsage()
        return text, usage

    @tool(
        "Run multiple independent one-shot LLM prompts concurrently. "
        "Returns strings in the same order as the prompts."
    )
    def llm_query_batched(
        self,
        prompts: list[str],
        *,
        model: str = "default",
    ) -> list[str]:
        """Run one-shot LLM prompts concurrently without spawning child agents."""

        if isinstance(prompts, str) or not isinstance(prompts, list):
            raise TypeError("llm_query_batched() requires a list[str] of prompts")
        if not all(isinstance(prompt, str) for prompt in prompts):
            raise TypeError("llm_query_batched() requires a list[str] of prompts")
        if model not in self.llm_clients:
            keys = ", ".join(sorted(self.llm_clients))
            raise ValueError(f"unknown model {model!r}. available: {keys}")
        if not prompts:
            return []

        client = self.llm_clients[model]

        def run_one(prompt: str) -> tuple[str, LLMUsage]:
            return self.call_llm([{"role": "user", "content": prompt}], client=client)

        cap = self.config.max_concurrency or len(prompts)
        max_workers = max(1, min(len(prompts), cap))
        if max_workers == 1:
            pairs = [run_one(prompt) for prompt in prompts]
        else:
            pairs_by_index: dict[int, tuple[str, LLMUsage]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(run_one, prompt): i for i, prompt in enumerate(prompts)
                }
                for future in as_completed(futures):
                    pairs_by_index[futures[future]] = future.result()
            pairs = [pairs_by_index[i] for i in range(len(prompts))]

        total_usage = LLMUsage(
            input_tokens=sum(usage.input_tokens for _, usage in pairs),
            output_tokens=sum(usage.output_tokens for _, usage in pairs),
        )
        self.record_usage(total_usage)
        return [text for text, _ in pairs]

    def llm_client_for(self, graph: Graph) -> LLMClient:
        """Pick the per-agent LLM client.

        The agent's ``config["model"]`` is the lookup key into
        ``self.llm_clients``; when missing, fall back to
        ``self.llm_client``. Override to add per-graph routing.
        """
        model = graph.config.get("model", "default")
        return self.llm_clients.get(model, self.llm_client)

    def extract_code(self, text: str) -> str | None:
        """Pull the first (or merged) ```repl block from an LLM reply.

        Override to recognize different fence syntax, inject a
        preamble, or filter blocks before they reach the runtime.
        """
        blocks = find_code_blocks(text)
        if not blocks:
            return None
        return blocks[0] if self.config.single_block else "\n\n".join(blocks)

    # ── messages / system prompt ─────────────────────────────────────

    def build_messages(
        self,
        graph: Graph,
        *,
        force_final: bool = False,
    ) -> list[dict[str, str]]:
        """Render ``graph``'s trajectory as a chat-message list."""
        system_content = graph.system_prompt or self.build_system_prompt_for(
            query=graph.query,
            agent_id=graph.agent_id,
            depth=graph.depth,
        )
        system = {"role": "system", "content": system_content}

        context_keys = self.context.list_contexts(agent_id=graph.agent_id)
        max_depth = (graph.config or {}).get("max_depth", self.config.max_depth)

        msgs: list[dict[str, str]] = []
        for state in graph.states:
            if is_user_query(state):
                _append_message(msgs, "user", state.content)
            elif is_llm_output(state):
                _append_message(msgs, "assistant", state.reply)
            elif is_exec_output(state):
                _append_message(msgs, "user", state.content or state.output)
            elif is_errored(state):
                _append_message(msgs, "user", state.content)
            # SupervisingOutput, DoneOutput, and every ActionNode are
            # engine bookkeeping — not part of the LLM projection.

        cap = self.config.max_messages
        if cap and len(msgs) > cap:
            msgs = [
                {
                    "role": "user",
                    "content": TRUNCATION_SUMMARY.format(
                        query=graph.query,
                        total=len(msgs),
                        cap=cap,
                        session_hint=TRUNCATION_SESSION_HINT,
                    ),
                }
            ] + msgs[-cap:]
            msgs = _coalesce_messages(msgs)

        # Gate on LLMOutput count — not LLMAction count — so we don't
        # double up the user prompt on the very first turn. The
        # transition writes the paired ``LLMAction`` *before* calling
        # ``build_messages``, so the action for the in-progress turn is
        # already in ``graph.states`` here. ``LLMOutput``s only exist
        # for *completed* prior turns, which is what should gate the
        # continuation nudge.
        prior_turns = sum(1 for s in graph.states if is_llm_output(s))
        has_prior_turn = prior_turns > 0
        nudge: str | None = None
        if force_final:
            nudge = FINAL_ANSWER_ACTION
        elif has_prior_turn:
            nudge = build_user_prompt(
                query=graph.query,
                iteration=prior_turns,
                depth=graph.depth,
                max_depth=max_depth,
                context_keys=context_keys,
            )

        if nudge is not None:
            # Merge the continue-turn / final-answer nudge into the last
            # message if it's already user-role — otherwise providers
            # like Anthropic reject (or silently merge) consecutive
            # `{role: "user"}` blocks. Fall back to appending a new
            # message only if the trailing message is assistant-role,
            # which shouldn't happen in practice but keeps us safe.
            if msgs and msgs[-1]["role"] == "user":
                _append_message(msgs, "user", nudge)
            else:
                _append_message(msgs, "user", nudge)
        return [system] + msgs

    def build_system_prompt(self, graph: Graph) -> str:
        """Render the system prompt for the agent rooted in ``graph``."""
        if self.config.system_prompt:
            return self.config.system_prompt
        return self.prompt_builder.build(
            tools=self.build_tools_section(),
            status=self.build_status_section(graph),
        )

    def build_system_prompt_for(
        self,
        *,
        query: str,
        agent_id: str,
        depth: int,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Render the system prompt for a (possibly not-yet-instantiated) agent."""
        stub = Graph(
            agent_id=agent_id,
            depth=depth,
            query=query,
            config=config or self.node_config(),
        )
        return self.build_system_prompt(stub)

    def build_tools_section(self) -> str:
        """Render the tools section that lands inside the system prompt."""
        baseline = self.config.max_depth == 0
        tool_defs = [t for t in self.runtime.get_tool_defs() if not t.core]
        lines = [
            "Tool functions are already available in the REPL namespace; "
            "do not import them from a `tools` module. Call them directly by name.",
            "",
        ]
        lines += [
            f"- `{tool_def.name}{tool_def.signature}`: {tool_def.description}"
            for tool_def in tool_defs
        ]
        if len(self.llm_clients) > 1 and not baseline:
            lines.append(
                "\nAvailable models for `launch_subagent(model=...)` "
                "and `llm_query_batched(model=...)`:"
            )
            for key in sorted(self.llm_clients):
                desc = self.model_descriptions.get(key)
                lines.append(f"- `{key}`: {desc}" if desc else f"- `{key}`")
        modules = self.runtime.available_modules()
        if modules:
            lines.append(f"\nPre-imported: `{'`, `'.join(modules)}`")
        return "\n".join(lines)

    def build_status_section(self, graph: Graph) -> str:
        """Render the depth/status note that lands inside the system prompt."""
        effective_max = graph.config.get("max_depth", self.config.max_depth)
        if effective_max == 0:
            return (
                "Baseline mode: no sub-agents available. Do all work directly "
                "in this REPL."
            )
        note = (
            f"You are at recursion depth **{graph.depth}** of max "
            f"**{effective_max}**."
        )
        if graph.depth == 0:
            note += STATUS_DEPTH_ROOT
        elif graph.depth >= effective_max - 1:
            note += STATUS_DEPTH_NEAR_MAX
        elif graph.depth > 0:
            note += STATUS_DEPTH_MID
        return note

    # ── runtime / env ────────────────────────────────────────────────

    def runtime_for(self, ref: RuntimeRef | None) -> Runtime:
        """Return the runtime session bound to ``ref``, restoring lazily.

        On a fresh engine attached to a forked or reloaded workspace,
        ``self.runtime_sessions`` only holds the ``ROOT_RUNTIME_ID``
        runtime. Any other agent ``RuntimeRef`` would otherwise
        ``KeyError``. Instead, we materialize a fresh runtime via
        ``runtime_factory`` (or by cloning the root) and call
        :meth:`register_tools` against it. The REPL namespace and any
        suspended generator are *not* restored — callers that need a
        paused generator (the supervising transition) ask for
        replay-of-one separately.
        """
        session_id = ref.id if ref is not None else ROOT_RUNTIME_ID
        runtime = self.runtime_sessions.get(session_id)
        if runtime is None:
            runtime = (
                self.runtime_factory() if self.runtime_factory else self.runtime.clone()
            )
            self.runtime_sessions[session_id] = runtime
        return runtime

    def create_runtime_session(
        self, parent_runtime: Runtime, *, agent_id: str
    ) -> RuntimeRef:
        """Allocate a fresh runtime session for a child agent."""
        session_id = f"{agent_id}:{uuid4().hex[:8]}"
        runtime = (
            self.runtime_factory() if self.runtime_factory else parent_runtime.clone()
        )
        self.runtime_sessions[session_id] = runtime
        self.register_tools(runtime)
        return RuntimeRef(id=session_id)

    def inject_env(self, graph: Graph, node: Node) -> Runtime:
        """Reset per-execution state on the runtime and seed env-style vars.

        ``runtime.env`` is the host-side dict shared with ``done`` /
        ``rlm_delegate`` closures (cleared + seeded each call). ``rlm_delegate``
        / ``rlm_wait`` are the internal primitives the ``launch_subagent`` /
        ``launch_subagents`` REPL launchers compose over. The same
        per-agent facts plus ``CONTEXT`` / ``SESSION`` are also pushed
        into the REPL namespace so user code can reference them by
        bare name.
        """
        runtime = self.runtime_for(graph.runtime)
        facts: dict[str, Any] = {
            "AGENT_ID": graph.agent_id,
            "DEPTH": graph.depth,
            "MAX_DEPTH": self.config.max_depth,
            "PARENT_NODE_ID": node.id,
        }
        runtime.env.clear()
        runtime.env.update({**facts, "DONE_RESULT": None})
        preserve_suspension = runtime.suspended
        if preserve_suspension:
            runtime.prepare_for_resume()
            for name, value in facts.items():
                runtime.inject(name, value)
            return runtime

        runtime.prepare_for_execution()

        repl_vars = {
            **facts,
            "SESSION": SessionVariable(
                self.session,
                agent_id=graph.agent_id,
                node_id=node.id,
                branch_id=graph.branch_id,
            ),
            "CONTEXT": ContextVariable(self.context, agent_id=graph.agent_id),
        }
        for name, value in repl_vars.items():
            runtime.inject(name, value)
        return runtime

    def register_tools(self, runtime: Runtime | None = None) -> None:
        """Bind ``done`` / ``rlm_wait`` / ``rlm_delegate`` closures to ``runtime.env``.

        The ``rlm_delegate`` tool needs a way to spawn child agents — we
        pass :meth:`spawn_child` (bound to ``self``) so the tool can
        call back into engine state.

        Closures live in :mod:`rlmflow.tools.builtins` and capture the
        same ``env`` dict the engine reads back after each execution
        (so ``DONE_RESULT`` / ``DELEGATED`` round-trip cleanly).
        """
        runtime = runtime or self.runtime
        runtime.register_tool(SHOW_VARS, core=True)
        runtime.register_tool(make_done(runtime.env), core=True)
        rlm_wait = make_wait()
        runtime.register_tool(rlm_wait, core=True, hidden=True)
        runtime.register_tool(self.llm_query_batched, core=True)
        rlm_delegate = make_delegate(self.spawn_child, runtime.env)
        runtime.register_tool(rlm_delegate, core=True, hidden=True)
        runtime.register_tool(make_launch_subagent(rlm_delegate, rlm_wait), core=True)
        runtime.register_tool(make_launch_subagents(rlm_delegate, rlm_wait), core=True)

    def format_exec_output(self, output: str) -> str:
        """Wrap REPL stdout for inclusion in the next user message."""
        return format_exec_output(output)

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
        """Spawn a child agent under ``parent_agent_id``.

        Public seam invoked by the ``rlm_delegate(...)`` REPL closure.
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
        context_keys = self.context.list_contexts(agent_id=child_aid)
        context_info = self._context_info(child_aid)
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
                content=build_user_prompt(
                    query=query,
                    iteration=0,
                    depth=parent.depth + 1,
                    max_depth=cfg.get("max_depth", self.config.max_depth),
                    context_keys=context_keys,
                    context_info=context_info,
                )
            ),
        )
        return ChildHandle(child_aid)

    # ── bookkeeping ──────────────────────────────────────────────────

    def record_usage(self, usage: LLMUsage) -> None:
        """Cache the most recent ``LLMUsage``. Override for metrics."""
        self.last_usage = usage

    def _context_info(self, agent_id: str) -> dict[str, Any] | None:
        """Best-effort size/shape signal for the agent's primary ``CONTEXT``.

        Returns ``None`` when the agent has no `context` blob written —
        avoids lying about a 0-char chunk for the on-disk haystack case.
        """
        try:
            return self.context.info("context", agent_id=agent_id)
        except KeyError:
            return None

    def node_config(self) -> dict[str, Any]:
        """The default config dict written onto every fresh :class:`Graph`."""
        return {
            "model": "default",
            "max_depth": self.config.max_depth,
            "max_iterations": self.config.max_iterations,
            "max_output_length": self.config.max_output_length,
            "max_messages": self.config.max_messages,
            "child_max_iterations": self.config.child_max_iterations,
            "eager_children": self.config.eager_children,
            "single_block": self.config.single_block,
            "max_budget": self.config.max_budget,
        }


def _append_message(msgs: list[dict[str, str]], role: str, content: str) -> None:
    if msgs and role == "user" and msgs[-1]["role"] == "user":
        msgs[-1] = {
            "role": "user",
            "content": msgs[-1]["content"] + "\n\n" + content,
        }
        return
    msgs.append({"role": role, "content": content})


def _coalesce_messages(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in msgs:
        _append_message(out, msg["role"], msg["content"])
    return out


__all__ = ["NodeScheduler", "RLMConfig", "RLMFlow", "create_pool"]
