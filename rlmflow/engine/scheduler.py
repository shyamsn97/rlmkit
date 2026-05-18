"""Pick the agents that can take a step right now.

Pure projection over a :class:`~rlmflow.graph.Graph` — no I/O, no
engine state. Walks the graph top-down from root; a supervising
agent is *runnable* iff all the children it is ``waiting_on`` are
terminal, otherwise the scheduler recurses into those still-running
children.

Lives in :mod:`rlmflow.engine` alongside the other pure helpers
(:mod:`~rlmflow.engine.actions`, :mod:`~rlmflow.engine.replay`,
:mod:`~rlmflow.engine.seq`). The engine owns scheduling; users
override :class:`NodeScheduler` to plug in custom policies.
"""

from __future__ import annotations

from rlmflow.graph import Graph, is_supervising
from rlmflow.utils.pool import Pool


class NodeScheduler:
    """Pick the agents that can take a step right now."""

    def __init__(self, pool: Pool | None = None) -> None:
        self.pool = pool

    def runnable_agents(self, graph: Graph) -> list[str]:
        runnable: list[str] = []

        def visit(aid: str) -> None:
            agent = graph.agents[aid]
            if agent.finished:
                return
            cur = agent.current()
            if cur is None:
                return
            if is_supervising(cur):
                waiting = [
                    graph.agents[child_aid]
                    for child_aid in cur.waiting_on
                    if child_aid in graph.agents
                ]
                if all(child.finished for child in waiting):
                    runnable.append(aid)
                    return
                for child in waiting:
                    if not child.finished:
                        visit(child.agent_id)
                return
            runnable.append(aid)

        visit(graph.agent_id)
        return runnable


__all__ = ["NodeScheduler"]
