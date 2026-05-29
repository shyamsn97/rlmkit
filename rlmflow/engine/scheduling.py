"""Scheduling helpers for :class:`rlmflow.rlm.RLMFlow`.

This module owns the outer ``step`` loop and async-child refill policy. It is
intentionally not a standalone scheduler object: ``RLMFlow`` still owns engine
state, public override methods, sessions, runtimes, and pools.
"""

from __future__ import annotations

from collections.abc import Callable

from rlmflow.engine.actions import act
from rlmflow.engine.replay import can_resume
from rlmflow.graph import Graph, SupervisingOutput


def step(engine, graph: Graph) -> Graph:
    """Advance the run by one synchronized or async-child batch."""

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
