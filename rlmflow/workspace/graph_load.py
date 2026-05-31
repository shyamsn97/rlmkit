"""Helpers for reconstructing workspace session graphs."""

from __future__ import annotations

from typing import Any

from rlmflow.graph import Graph, Node


def build_graph(
    *,
    root_agent_id: str,
    agent_dicts: dict[str, dict[str, Any]],
    agent_states: dict[str, tuple[Node, ...]],
) -> Graph:
    """Recover the recursive :class:`Graph` from flat per-agent dicts."""

    children_by_parent: dict[str, list[str]] = {}
    for aid, data in agent_dicts.items():
        if aid == root_agent_id:
            continue
        parent = data.get("parent_agent_id") or root_agent_id
        children_by_parent.setdefault(parent, []).append(aid)

    def build(aid: str) -> Graph:
        data = agent_dicts.get(aid, {"agent_id": aid})
        states = agent_states.get(aid, ())
        children = {
            child_aid: build(child_aid) for child_aid in children_by_parent.get(aid, [])
        }
        return Graph.from_meta_dict(data, states=states, children=children)

    if root_agent_id not in agent_dicts:
        return Graph(agent_id=root_agent_id)
    return build(root_agent_id)


__all__ = ["build_graph"]
