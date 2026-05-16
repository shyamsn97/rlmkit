"""Per-agent step dispatch and the two transition functions.

The engine's per-tick behaviour is:

1. :func:`step_agent` loads the agent's sub-graph, checks budgets,
   and dispatches on the kind of node the agent is sitting at.
2. :func:`step_after_observation` handles "agent is at an observation,
   call the LLM and run its code". Each step writes the four-node
   sequence ``LLMAction → LLMOutput → ExecAction → <CodeObservation>``
   (where the trailing observation is one of :class:`ExecOutput` /
   :class:`SupervisingOutput` / :class:`ErrorOutput` /
   :class:`DoneOutput`).
3. :func:`step_after_supervising` handles "agent is at a
   :class:`SupervisingOutput`, resume the suspended generator with
   the children's results". Writes
   ``ResumeAction → <CodeObservation>``.

Each function takes ``engine: RLMFlow`` as its first argument so its
dependencies are explicit and the engine class itself stays slim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlmflow.engine.code import reply_to
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
    Node,
    ResumeAction,
    SupervisingOutput,
    is_done,
    is_exec_action,
    is_llm_action,
    is_supervising,
)
from rlmflow.prompts.messages import NO_CODE_BLOCK, ORPHANED_DELEGATES
from rlmflow.utils import check_yield_errors

if TYPE_CHECKING:
    from rlmflow.rlm import RLMFlow


# ── per-agent step dispatch ──────────────────────────────────────────


def step_agent(engine: "RLMFlow", agent_id: str) -> None:
    """Advance the agent rooted at ``agent_id`` by one step."""
    full = engine.session.load_graph()
    if agent_id not in full.agents:
        return
    graph = full.agents[agent_id]
    if graph.finished:
        return

    over = budget_exceeded(full, engine.config.max_budget)
    if over is not None:
        append_node(
            engine.session,
            graph,
            DoneOutput(result=f"[budget exceeded: {over} tokens]"),
        )
        return

    cur = graph.current()
    if cur is None:
        return
    if is_supervising(cur):
        step_after_supervising(engine, graph, cur)
        return
    # Anything except a terminal/yielded observation is a "ready for
    # the next LLM turn" state: UserQuery (start), ExecOutput,
    # ErrorOutput. Done/Supervising are handled above (terminal /
    # paused).
    if is_done(cur):
        return
    # Skip stray ActionNode without its paired observation — should
    # not normally happen, but guard against half-written runs.
    if is_llm_action(cur) or is_exec_action(cur):
        return
    step_after_observation(engine, graph, cur)


# ── observation → LLMAction → LLMOutput → ExecAction → ... ───────────


def step_after_observation(engine: "RLMFlow", graph: Graph, last: Node) -> None:
    iters = iteration_count(graph)
    max_iter = graph.config.get("max_iterations", engine.config.max_iterations)
    terminate = iters >= max_iter or graph.agent_id in engine.terminate_requested

    # 1. Record the system action of calling the LLM.
    llm_action = LLMAction(
        agent_id=graph.agent_id,
        seq=last.seq + 1,
        model=graph.config.get("model", "default"),
    )
    append_node(engine.session, graph, llm_action)

    # 2. Make the call; record what came back.
    llm_output = reply_to(engine, graph, llm_action, force_final=terminate)
    output_state = append_node(engine.session, graph, llm_output)

    # 3. Synthesize ExecAction (the code we're about to run).
    exec_action = ExecAction(
        agent_id=graph.agent_id,
        seq=output_state.seq + 1,
        code=output_state.code,
    )
    exec_state = append_node(engine.session, graph, exec_action)

    # 3a. No code in the reply → ErrorOutput, the LLM retries next turn.
    if not output_state.code:
        append_node(
            engine.session,
            graph,
            ErrorOutput(content=NO_CODE_BLOCK, error="no_code_block"),
        )
        return

    # 4. Run the code; observation depends on what came back.
    full = engine.session.load_graph()
    graph = full.agents[graph.agent_id]
    _run_exec(engine, graph, exec_state, output_state.code)


def _run_exec(
    engine: "RLMFlow",
    graph: Graph,
    exec_action: ExecAction,
    code: str,
) -> None:
    err = check_yield_errors(code)
    if err:
        append_node(
            engine.session,
            graph,
            ErrorOutput(content=err, error="invalid_yield", output=""),
        )
        return

    runtime = engine.inject_env(graph, exec_action)
    suspended, raw, errored = runtime.start_code(code)
    raw = truncate_output(raw, engine.config.max_output_length)
    env = runtime.env
    delegated = list(env.get("DELEGATED") or [])
    done_result = env.get("DONE_RESULT")

    if delegated and not suspended and done_result is None:
        msg = ORPHANED_DELEGATES.format(names=", ".join(delegated))
        base = raw if isinstance(raw, str) else ""
        output = truncate_output(
            runtime.execute(f"raise OrphanedDelegatesError({msg!r})"),
            engine.config.max_output_length,
        )
        content = (base + "\n\n" + output).strip()
        append_node(
            engine.session,
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
            engine.session,
            graph,
            DoneOutput(result=done_result.strip()),
        )
        return

    if suspended:
        request, pre_output = raw
        append_node(
            engine.session,
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
            engine.session,
            graph,
            ErrorOutput(
                content=format_exec_output(output),
                error="exec_exception",
                output=output,
            ),
        )
        return
    append_node(
        engine.session,
        graph,
        ExecOutput(
            output=output,
            content=format_exec_output(output),
        ),
    )


# ── supervising → ResumeAction → CodeObservation ─────────────────────


def step_after_supervising(
    engine: "RLMFlow",
    graph: Graph,
    last: SupervisingOutput,
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
    resume_state = append_node(engine.session, graph, resume_action)

    runtime = engine.inject_env(graph, resume_state)
    if not runtime.suspended:
        # The live generator is gone — process restart, fork, or any
        # other cold start. Re-execute the action code with delegate
        # in replay mode so the generator is paused at the same yield
        # we recorded, then drop into the regular resume path.
        replay_to_yield(engine, graph, last, runtime)

    suspended, raw, errored = runtime.resume_code(results)
    raw = truncate_output(raw, engine.config.max_output_length)
    env = runtime.env
    done_result = env.get("DONE_RESULT")

    if suspended:
        request, output = raw
    else:
        output = raw if isinstance(raw, str) else ""
    if not output.strip():
        output = "(no output)"

    graph = engine.session.load_graph().agents[graph.agent_id]
    resumed_from = list(last.waiting_on)

    if done_result is not None:
        append_node(
            engine.session,
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
            engine.session,
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
            engine.session,
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
        engine.session,
        graph,
        ExecOutput(
            output=output,
            content=format_exec_output(output),
            resumed_from=resumed_from,
        ),
    )


__all__ = [
    "step_after_observation",
    "step_after_supervising",
    "step_agent",
]
