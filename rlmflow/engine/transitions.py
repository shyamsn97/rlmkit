"""Graph transition handlers for :class:`rlmflow.rlm.RLMFlow`.

These helpers materialize one planned action into persisted graph state.
``RLMFlow`` keeps the public methods as override seams; this module keeps the
implementation details out of the façade.
"""

from __future__ import annotations

from rlmflow.engine.actions import CallLLM, Exec, Resume
from rlmflow.engine.replay import (
    can_resume,
    replay_to_suspension,
    results_for_supervise,
)
from rlmflow.engine.seq import append_node, budget_exceeded, truncate_output
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
)
from rlmflow.prompts.messages import NO_CODE_BLOCK
from rlmflow.utils import check_wait_syntax


def apply_one(engine, action) -> None:
    """Materialize one action against the persisted graph."""

    graph = engine.session.load_graph().agents[action.agent_id]

    over = budget_exceeded(graph, engine.config.max_budget)
    if over is not None:
        append_node(
            engine.session,
            graph,
            DoneOutput(result=f"[budget exceeded: {over} tokens]"),
        )
        return

    cur = graph.current()
    if isinstance(action, CallLLM):
        engine.step_llm(
            graph,
            cur,
            force_final=action.force_final,
            model=action.model,
        )
    elif isinstance(action, Exec):
        engine.step_exec(graph, cur)
    elif isinstance(action, Resume):
        engine.step_after_supervising(graph, cur)


def step_llm(
    engine,
    graph: Graph,
    last: Node,
    *,
    force_final: bool,
    model: str | None = None,
) -> None:
    """LLM half of one turn: write ``LLMAction -> LLMOutput``."""

    llm_model = model or graph.config.get("model", "default")
    llm_action = LLMAction(
        agent_id=graph.agent_id,
        seq=last.seq + 1,
        model=llm_model,
    )
    append_node(engine.session, graph, llm_action)

    llm_output, usage = engine.reply_to(graph, llm_action, force_final=force_final)
    engine.record_usage(usage)
    append_node(engine.session, graph, llm_output)


def step_exec(engine, graph: Graph, llm_output: LLMOutput) -> None:
    """Exec half of one turn: write ``ExecAction -> CodeObservation``."""

    code = llm_output.code

    exec_action = ExecAction(
        agent_id=graph.agent_id,
        seq=llm_output.seq + 1,
        code=code,
    )
    exec_state = append_node(engine.session, graph, exec_action)

    if not code:
        append_node(
            engine.session,
            graph,
            ErrorOutput(content=NO_CODE_BLOCK, error="no_code_block"),
        )
        return

    full = engine.session.load_graph()
    graph = full.agents[graph.agent_id]
    engine._run_exec(graph, exec_state, code)


def run_exec(
    engine,
    graph: Graph,
    exec_action: ExecAction,
    code: str,
) -> None:
    """Run an LLM code block and persist the resulting observation."""

    err = check_wait_syntax(code)
    if err:
        append_node(
            engine.session,
            graph,
            ErrorOutput(content=err, error="invalid_wait", output=""),
        )
        return

    try:
        runtime = engine.inject_env(graph, exec_action)
        suspended, raw, errored = runtime.start_code(code)
        runtime.after_execution_transition(engine.runtime_sessions.values())
    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}"
        append_node(
            engine.session,
            graph,
            ErrorOutput(
                content=engine.format_exec_output(output),
                error="runtime_exception",
                output=output,
            ),
        )
        return
    raw = truncate_output(raw, engine.config.max_output_length)
    env = runtime.env
    done_result = env.get("DONE_RESULT")

    if done_result is not None:
        output = raw if isinstance(raw, str) else ""
        if not output.strip():
            output = "(no output)"
        terminal = append_node(
            engine.session,
            graph,
            DoneOutput(
                result=done_result.strip(),
                output=output,
                content=engine.format_exec_output(output),
            ),
        )
        engine.transcript_recorder.record_terminal(graph, terminal)
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
                content=engine.format_exec_output(output),
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
            content=engine.format_exec_output(output),
        ),
    )


def step_after_supervising(
    engine,
    graph: Graph,
    last: SupervisingOutput,
) -> None:
    """Resume half: write ``ResumeAction -> CodeObservation``."""

    if not can_resume(graph, last):
        return

    results = results_for_supervise(graph, last)

    resume_action = ResumeAction(
        agent_id=graph.agent_id,
        seq=last.seq + 1,
        resumed_from=list(last.waiting_on),
    )
    resume_state = append_node(engine.session, graph, resume_action)

    try:
        runtime = engine.inject_env(graph, resume_state)
        if not runtime.suspended:
            replay_to_suspension(graph, last, runtime)

        suspended, raw, errored = runtime.resume_code(results)
        runtime.after_execution_transition(engine.runtime_sessions.values())
    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}"
        append_node(
            engine.session,
            graph,
            ErrorOutput(
                content=engine.format_exec_output(output),
                error="runtime_exception",
                output=output,
                resumed_from=list(last.waiting_on),
            ),
        )
        return
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
        terminal = append_node(
            engine.session,
            graph,
            DoneOutput(
                result=done_result.strip(),
                output=output,
                content=engine.format_exec_output(output),
                resumed_from=resumed_from,
            ),
        )
        engine.transcript_recorder.record_terminal(graph, terminal)
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
                content=engine.format_exec_output(output),
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
            content=engine.format_exec_output(output),
            resumed_from=resumed_from,
        ),
    )


__all__ = [
    "apply_one",
    "run_exec",
    "step_after_supervising",
    "step_exec",
    "step_llm",
]
