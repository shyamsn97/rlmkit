"""Immutable graph injection helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from rlmflow.graph.node import ActionNode, ExecOutput, Node


def inject(
    graph: Any,
    *,
    target: str | re.Pattern[str] | Callable[[Any], Iterable[str | Any]],
    node: Node,
    mode: str = "append",
) -> Any:
    """Return a new graph with ``node`` injected at ``target``."""

    if mode != "append":
        raise NotImplementedError("Graph.inject currently supports append mode only")
    out = graph.copy(deep=True)
    targets = resolve_injection_targets(out, target)
    if not targets:
        raise KeyError(f"no injection targets matched {target!r}")
    for sub in targets:
        fixed = node_for_injection(sub, node)
        cur = sub.current()
        if cur is not None and cur.terminal:
            raise ValueError(f"cannot inject into finished agent {sub.agent_id!r}")
        if cur is not None and is_action_like(cur) and is_action_like(fixed):
            raise ValueError(
                f"cannot queue multiple pending actions for {sub.agent_id!r}"
            )
        sub.states.append(fixed)
    return out


def inject_output(
    graph: Any,
    *,
    target: str | re.Pattern[str] | Callable[[Any], Iterable[str | Any]],
    output: str,
    content: str | None = None,
) -> Any:
    return inject(
        graph,
        target=target,
        node=ExecOutput(output=output, content=content or output),
    )


def resolve_injection_targets(
    graph: Any,
    target: str | re.Pattern[str] | Callable[[Any], Iterable[str | Any]],
) -> list[Any]:
    if callable(target):
        from rlmflow.graph.graph import Graph

        raw = list(target(graph))
        out = []
        for item in raw:
            out.append(item if isinstance(item, Graph) else graph[item])
        return out
    if isinstance(target, str) and target in graph.agents:
        return [graph[target]]
    compiled = re.compile(target) if isinstance(target, str) else target
    return [g for g in graph.walk() if compiled.search(g.agent_id)]


def is_action_like(node: Node) -> bool:
    return isinstance(node, ActionNode)


def node_for_injection(sub: Any, node: Node) -> Node:
    fields = node.model_dump(
        exclude={"id", "agent_id", "seq"},
        mode="python",
    )
    next_seq = (sub.states[-1].seq + 1) if sub.states else 0
    return node.__class__(
        agent_id=sub.agent_id,
        seq=next_seq,
        **fields,
    )


__all__ = [
    "inject",
    "inject_output",
    "is_action_like",
    "node_for_injection",
    "resolve_injection_targets",
]
