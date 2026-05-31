"""Scheduling helpers for :class:`rlmflow.rlm.RLMFlow`.

This module owns the outer ``step`` loop and async-child refill policy. It is
intentionally not a standalone scheduler object: ``RLMFlow`` still owns engine
state, public override methods, sessions, runtimes, and pools.
"""

from __future__ import annotations

from collections.abc import Callable

from rlmflow.engine.actions import act
from rlmflow.engine.replay import can_resume
from rlmflow.graph import (
    ActionNode,
    DoneOutput,
    ExecAction,
    Graph,
    LLMAction,
    ResumeAction,
    SupervisingOutput,
)


def step(engine, graph: Graph) -> Graph:
    """Advance the run by one synchronized or async-child batch."""

    graph, action_materialized = materialize_injected_nodes(engine, graph)
    if action_materialized:
        return graph

    runnable = engine.node_scheduler.runnable_agents(graph)
    if not runnable:
        return graph
    plan = act(
        graph,
        config=engine.config,
        runnable=runnable,
        terminate_requested=engine.terminate_requested,
    )
    if not plan:
        return graph
    tasks = [
        (aid, (lambda action=action: engine.apply_one(action)))
        for aid, action in plan.items()
    ]
    if engine.config.eager_children:
        engine.pool.run_until_idle(tasks, engine._refill_eager_children)
    else:
        engine.pool.execute(tasks)
    return engine.session.load_graph()


def materialize_injected_nodes(engine, graph: Graph) -> tuple[Graph, bool]:
    """Persist nodes appended to the caller's graph before planning.

    ``Graph.inject(...)`` is immutable and does not touch the active session.
    ``agent.step(graph)`` is the commit point: observation nodes are appended
    directly, while an appended ``ExecAction`` is executed before returning.
    """
    persisted = engine.session.load_graph()
    materialized = False
    action_materialized = False

    for aid, candidate in graph.agents.items():
        if aid not in persisted.agents:
            raise ValueError(f"cannot materialize injected unknown agent {aid!r}")
        current = persisted.agents[aid]
        prefix = candidate.states[: len(current.states)]
        if [n.id for n in prefix] != [n.id for n in current.states]:
            if [n.id for n in candidate.states] == [
                n.id for n in current.states[: len(candidate.states)]
            ]:
                continue
            raise ValueError(f"graph for {aid!r} is not based on current session state")

        extra = candidate.states[len(current.states) :]
        for index, node in enumerate(extra):
            if action_materialized:
                raise ValueError("cannot materialize nodes after an appended action")

            engine.session.write_state(node)
            materialized = True

            if isinstance(node, ExecAction):
                if index != len(extra) - 1:
                    raise ValueError(
                        "an appended action must be the final pending node"
                    )
                fresh = engine.session.load_graph().agents[aid]
                engine._run_exec(fresh, node, node.code)
                action_materialized = True
            elif isinstance(node, (LLMAction, ResumeAction, ActionNode)):
                raise NotImplementedError(
                    f"appended {node.type!r} actions are not executable yet"
                )
            elif isinstance(node, DoneOutput):
                fresh = engine.session.load_graph().agents[aid]
                engine.transcript_recorder.record_terminal(fresh, node)

    if materialized:
        return engine.session.load_graph(), action_materialized
    return graph, False


def refill_eager_children(
    engine,
    done_id: str,
    _result: object,
    active_ids: set[str],
) -> list[tuple[str, Callable[[], None]]]:
    """Return newly runnable eager-child tasks after one task completes."""

    graph = engine.session.load_graph()
    tasks: list[tuple[str, Callable[[], None]]] = []
    scheduled: set[str] = set(active_ids)

    for supervisor in graph.walk():
        cur = supervisor.current()
        if not isinstance(cur, SupervisingOutput):
            continue
        if not supervisor.config.get("eager_children", engine.config.eager_children):
            continue

        runnable = (
            [supervisor.agent_id]
            if can_resume(supervisor, cur)
            else engine.node_scheduler.runnable_descendants(supervisor)
        )
        runnable = [aid for aid in runnable if aid not in scheduled]
        if not runnable:
            continue

        plan = act(
            graph,
            config=engine.config,
            runnable=runnable,
            terminate_requested=engine.terminate_requested,
        )
        for aid, action in plan.items():
            if aid in scheduled:
                continue
            scheduled.add(aid)
            tasks.append((aid, lambda action=action: engine.apply_one(action)))

    return tasks


__all__ = ["refill_eager_children", "step"]
