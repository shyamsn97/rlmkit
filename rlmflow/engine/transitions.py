"""Engine transitions: project actions, apply them to the graph.

The engine advances by **one observation-to-observation step** per
``apply_one`` call: the agent rests at an observation, the engine
runs ``new_obs = act(obs)``, and persists ``(action, new_obs)``
atomically. Every leaf of the persisted graph is therefore an
observation; intermediate action nodes only exist *inside* a
single ``apply_one`` call's writes.

Two-phase split:

1. :func:`act_one` / :func:`act` — pure projection
   ``Graph -> ActionPlan``: which transition each ready agent
   should take next. No I/O, no writes.
2. :func:`apply_one` / :func:`apply` — the I/O half: dispatch
   one ``Action`` to its handler (:func:`step_llm` /
   :func:`step_exec` / :func:`step_after_supervising`) and write
   the resulting ``(ActionNode, ObservationNode)`` pair through
   the session.

A logical "reasoning turn" is therefore two ``apply_one``
rounds: an LLM half (call the model) and an exec half (run the
model's code). Concurrency is finer-grained — one agent can be
mid-LLM while another is mid-exec.

``Action`` values are intentionally **lite** — just the policy
intent plus inputs that aren't recoverable from the graph
(``force_final``, model override). The persisted ``ActionNode``
written by the handlers carries the full record of what
happened. See :doc:`docs/internal/act_apply.md`.

Every function takes only the concrete dependencies it needs as
keyword arguments. Nothing here imports :class:`~rlmflow.rlm.RLMFlow`.
"""

from __future__ import annotations

from collections.abc import Callable, Set
from dataclasses import dataclass
from typing import Any

from rlmflow.engine.code import reply_to
from rlmflow.engine.config import RLMConfig
from rlmflow.engine.replay import can_resume, replay_to_yield, results_for_supervise
from rlmflow.engine.seq import (
    append_node,
    budget_exceeded,
    format_exec_output,
    iteration_count,
    truncate_output,
)
from rlmflow.graph import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMAction,
    LLMOutput,
    Node,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    is_done,
    is_llm_output,
    is_supervising,
)
from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.prompts.messages import NO_CODE_BLOCK, ORPHANED_DELEGATES
from rlmflow.runtime import Runtime
from rlmflow.utils import check_yield_errors
from rlmflow.workspace import Context, Session

# Type alias for the runtime-injection callable that ``RLMFlow``
# exposes to engine helpers. Hides the runtime-session map behind a
# simple ``(graph, node) -> Runtime`` shape.
InjectEnvFn = Callable[[Graph, Node], Runtime]

# Type alias for an optional last-usage sink. Engine helpers don't
# read engine state; they call this each turn so the engine can
# cache the most recent ``LLMUsage`` if it wants to.
RecordUsageFn = Callable[[LLMUsage], None]


# ── Action union (lite policy values) ───────────────────────────────


@dataclass(frozen=True, slots=True)
class CallLLM:
    """Plan to call the LLM. Materialized as ``LLMAction → LLMOutput``.

    ``force_final`` carries the policy decision to force a terminal
    answer this turn (max iterations exhausted or terminate
    requested). ``model`` is an optional override; ``None`` means
    "use whatever ``graph.config['model']`` says."
    """

    agent_id: str
    force_final: bool = False
    model: str | None = None


@dataclass(frozen=True, slots=True)
class Exec:
    """Plan to run the code from the preceding ``LLMOutput``.

    Materialized as ``ExecAction → CodeObservation``. The code
    itself is recovered from the graph in ``apply_one``; it is
    not duplicated here.
    """

    agent_id: str


@dataclass(frozen=True, slots=True)
class Resume:
    """Plan to resume a paused generator with settled child results.

    Materialized as ``ResumeAction → CodeObservation``. The child
    results are recovered from the graph (the agents named in the
    preceding ``SupervisingOutput.waiting_on``).
    """

    agent_id: str


Action = CallLLM | Exec | Resume
ActionPlan = dict[str, Action]


# ── act: pure projection Graph -> ActionPlan ────────────────────────


def act_one(
    graph: Graph,
    *,
    config: RLMConfig,
    terminate_requested: Set[str] = frozenset(),
) -> Action | None:
    """Project one agent's current observation into the next ``Action``.

    Returns ``None`` if the agent has nothing to do this round
    (terminal, empty trajectory, or stray half-written state).
    Pure: no I/O, no writes.
    """
    if graph.finished:
        return None
    cur = graph.current()
    if cur is None or is_done(cur):
        return None

    if is_supervising(cur):
        # Caller (scheduler) only routes here when children are
        # all settled; ``apply_one`` re-checks via ``can_resume``
        # inside :func:`step_after_supervising`.
        return Resume(agent_id=graph.agent_id)

    if is_llm_output(cur):
        return Exec(agent_id=graph.agent_id)

    if isinstance(cur, (UserQuery, ExecOutput, ErrorOutput)):
        iters = iteration_count(graph)
        max_iter = graph.config.get("max_iterations", config.max_iterations)
        force_final = iters >= max_iter or graph.agent_id in terminate_requested
        return CallLLM(agent_id=graph.agent_id, force_final=force_final)

    # Stray ActionNode without its paired observation — half-written
    # state from a crash mid-step. Skip; the next round sees the
    # rewritten observation.
    return None


def act(
    graph: Graph,
    *,
    config: RLMConfig,
    runnable: list[str] | None = None,
    terminate_requested: Set[str] = frozenset(),
) -> ActionPlan:
    """Project every ready agent's next action into an ``ActionPlan``.

    If ``runnable`` is provided, only those agent ids are
    considered (the typical caller is :class:`NodeScheduler`,
    which already filters supervising-with-pending-children).
    Otherwise every agent in ``graph.agents`` is considered.

    Pure: no I/O, no writes, no engine state.
    """
    aids = runnable if runnable is not None else list(graph.agents)
    plan: ActionPlan = {}
    for aid in aids:
        if aid not in graph.agents:
            continue
        action = act_one(
            graph.agents[aid],
            config=config,
            terminate_requested=terminate_requested,
        )
        if action is not None:
            plan[aid] = action
    return plan


# ── apply: dispatch one Action to its handler ───────────────────────


def apply_one(
    action: Action,
    *,
    session: Session,
    context: Context,
    config: RLMConfig,
    runtime: Runtime,
    llm_client: LLMClient,
    llm_clients: dict[str, Any],
    model_descriptions: dict[str, str],
    prompt_builder: Any,
    inject_env: InjectEnvFn,
    record_usage: RecordUsageFn | None = None,
) -> None:
    """Materialize one :class:`Action` against the persisted graph.

    Reloads the graph from ``session``, enforces the global token
    budget, and dispatches to the handler keyed by action type.
    The dispatch logic itself lives in
    :func:`rlmflow.engine.policy.act_one`; this function does no
    re-decisioning.
    """
    graph = session.load_graph().agents[action.agent_id]

    over = budget_exceeded(graph, config.max_budget)
    if over is not None:
        append_node(
            session,
            graph,
            DoneOutput(result=f"[budget exceeded: {over} tokens]"),
        )
        return

    cur = graph.current()
    if isinstance(action, CallLLM):
        step_llm(
            graph,
            cur,
            force_final=action.force_final,
            model=action.model,
            session=session,
            context=context,
            config=config,
            runtime=runtime,
            llm_client=llm_client,
            llm_clients=llm_clients,
            model_descriptions=model_descriptions,
            prompt_builder=prompt_builder,
            record_usage=record_usage,
        )
    elif isinstance(action, Exec):
        step_exec(
            graph,
            cur,
            session=session,
            config=config,
            inject_env=inject_env,
        )
    elif isinstance(action, Resume):
        step_after_supervising(
            graph,
            cur,
            session=session,
            config=config,
            inject_env=inject_env,
        )


def apply(plan: ActionPlan, **deps: Any) -> None:
    """Apply every action in ``plan`` sequentially.

    Useful for tests and replay. Production engines drive
    parallelism at a higher level (see
    :meth:`rlmflow.rlm.RLMFlow.step`) and call :func:`apply_one`
    per task.
    """
    for action in plan.values():
        apply_one(action, **deps)


# ── obs → LLMAction → LLMOutput  (the "LLM call" half-step) ─────────


def step_llm(
    graph: Graph,
    last: Node,
    *,
    force_final: bool,
    model: str | None = None,
    session: Session,
    context: Context,
    config: RLMConfig,
    runtime: Runtime,
    llm_client: LLMClient,
    llm_clients: dict[str, Any],
    model_descriptions: dict[str, str],
    prompt_builder: Any,
    record_usage: RecordUsageFn | None = None,
) -> None:
    """Perform the LLM half of one turn: write LLMAction → LLMOutput.

    ``last`` is the observation the LLM is replying to (a
    :class:`UserQuery`, :class:`ExecOutput`, or :class:`ErrorOutput`).
    ``force_final`` is the policy decision (computed by
    :func:`~rlmflow.engine.policy.act_one`) to force a terminal
    answer this turn. ``model`` optionally overrides
    ``graph.config['model']`` for this single call.

    The next ``apply_one`` round will see :class:`LLMOutput` as
    the current state and run :func:`step_exec` against it.
    """
    llm_model = model or graph.config.get("model", "default")
    llm_action = LLMAction(
        agent_id=graph.agent_id,
        seq=last.seq + 1,
        model=llm_model,
    )
    append_node(session, graph, llm_action)

    llm_output, usage = reply_to(
        graph,
        llm_action,
        force_final=force_final,
        session=session,
        context=context,
        config=config,
        runtime=runtime,
        llm_client=llm_client,
        llm_clients=llm_clients,
        model_descriptions=model_descriptions,
        prompt_builder=prompt_builder,
    )
    if record_usage is not None:
        record_usage(usage)
    append_node(session, graph, llm_output)


# ── LLMOutput → ExecAction → CodeObservation  (the "exec" half) ──────


def step_exec(
    graph: Graph,
    llm_output: LLMOutput,
    *,
    session: Session,
    config: RLMConfig,
    inject_env: InjectEnvFn,
) -> None:
    """Perform the exec half of one turn: write ExecAction → CodeObs.

    Reads the code from ``llm_output`` (the assistant's reply
    rendered as a code block), runs it through the runtime, and
    persists the resulting :class:`CodeObservation` (one of
    :class:`ExecOutput` / :class:`SupervisingOutput` /
    :class:`ErrorOutput` / :class:`DoneOutput`).
    """
    code = llm_output.code

    exec_action = ExecAction(
        agent_id=graph.agent_id,
        seq=llm_output.seq + 1,
        code=code,
    )
    exec_state = append_node(session, graph, exec_action)

    if not code:
        # LLM produced no parseable code block — surface a retry
        # message; the next apply_one round routes back to step_llm.
        append_node(
            session,
            graph,
            ErrorOutput(content=NO_CODE_BLOCK, error="no_code_block"),
        )
        return

    full = session.load_graph()
    graph = full.agents[graph.agent_id]
    _run_exec(
        graph,
        exec_state,
        code,
        session=session,
        config=config,
        inject_env=inject_env,
    )


def _run_exec(
    graph: Graph,
    exec_action: ExecAction,
    code: str,
    *,
    session: Session,
    config: RLMConfig,
    inject_env: InjectEnvFn,
) -> None:
    err = check_yield_errors(code)
    if err:
        append_node(
            session,
            graph,
            ErrorOutput(content=err, error="invalid_yield", output=""),
        )
        return

    runtime = inject_env(graph, exec_action)
    suspended, raw, errored = runtime.start_code(code)
    raw = truncate_output(raw, config.max_output_length)
    env = runtime.env
    delegated = list(env.get("DELEGATED") or [])
    done_result = env.get("DONE_RESULT")

    if delegated and not suspended and done_result is None:
        msg = ORPHANED_DELEGATES.format(names=", ".join(delegated))
        base = raw if isinstance(raw, str) else ""
        output = truncate_output(
            runtime.execute(f"raise OrphanedDelegatesError({msg!r})"),
            config.max_output_length,
        )
        content = (base + "\n\n" + output).strip()
        append_node(
            session,
            graph,
            ErrorOutput(
                content=format_exec_output(content),
                error="orphaned_delegates",
                output=content,
            ),
        )
        return

    if done_result is not None:
        append_node(
            session,
            graph,
            DoneOutput(result=done_result.strip()),
        )
        return

    if suspended:
        request, pre_output = raw
        append_node(
            session,
            graph,
            SupervisingOutput(
                output=pre_output,
                waiting_on=list(request.agent_ids),
            ),
        )
        return

    output = raw if isinstance(raw, str) else ""
    if not output.strip():
        output = "(no output)"
    if errored:
        append_node(
            session,
            graph,
            ErrorOutput(
                content=format_exec_output(output),
                error="exec_exception",
                output=output,
            ),
        )
        return
    append_node(
        session,
        graph,
        ExecOutput(
            output=output,
            content=format_exec_output(output),
        ),
    )


# ── supervising → ResumeAction → CodeObservation ─────────────────────


def step_after_supervising(
    graph: Graph,
    last: SupervisingOutput,
    *,
    session: Session,
    config: RLMConfig,
    inject_env: InjectEnvFn,
) -> None:
    if not can_resume(graph, last):
        # Children still need to advance. The scheduler picks them up
        # on the next outer step; nothing for this agent to do now.
        return

    results = results_for_supervise(graph, last)

    # Record the system action of resuming.
    resume_action = ResumeAction(
        agent_id=graph.agent_id,
        seq=last.seq + 1,
        resumed_from=list(last.waiting_on),
    )
    resume_state = append_node(session, graph, resume_action)

    runtime = inject_env(graph, resume_state)
    if not runtime.suspended:
        # The live generator is gone — process restart, fork, or any
        # other cold start. Re-execute the action code with delegate
        # in replay mode so the generator is paused at the same yield
        # we recorded, then drop into the regular resume path.
        replay_to_yield(graph, last, runtime)

    suspended, raw, errored = runtime.resume_code(results)
    raw = truncate_output(raw, config.max_output_length)
    env = runtime.env
    done_result = env.get("DONE_RESULT")

    if suspended:
        request, output = raw
    else:
        output = raw if isinstance(raw, str) else ""
    if not output.strip():
        output = "(no output)"

    graph = session.load_graph().agents[graph.agent_id]
    resumed_from = list(last.waiting_on)

    if done_result is not None:
        append_node(
            session,
            graph,
            DoneOutput(
                result=done_result.strip(),
                output=output,
                resumed_from=resumed_from,
            ),
        )
        return

    if suspended:
        append_node(
            session,
            graph,
            SupervisingOutput(
                output=output,
                waiting_on=list(request.agent_ids),
                resumed_from=resumed_from,
            ),
        )
        return

    if errored:
        append_node(
            session,
            graph,
            ErrorOutput(
                content=format_exec_output(output),
                error="exec_exception",
                output=output,
                resumed_from=resumed_from,
            ),
        )
        return
    append_node(
        session,
        graph,
        ExecOutput(
            output=output,
            content=format_exec_output(output),
            resumed_from=resumed_from,
        ),
    )


__all__ = [
    "Action",
    "ActionPlan",
    "CallLLM",
    "Exec",
    "InjectEnvFn",
    "RecordUsageFn",
    "Resume",
    "act",
    "act_one",
    "apply",
    "apply_one",
    "step_after_supervising",
    "step_exec",
    "step_llm",
]
