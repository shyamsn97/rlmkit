"""Shared context lookup behavior for workspace backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def context_agent_ids(agent_id: str | None) -> list[str]:
    """Return lookup order for agent-specific context with root fallback."""

    if not agent_id:
        return ["root"]
    if agent_id == "root":
        return ["root"]
    return [agent_id, "root"]


def first_context_value(
    key: str,
    *,
    agent_id: str,
    exists: Callable[[str, str], bool],
    read: Callable[[str, str], str],
) -> str:
    for aid in context_agent_ids(agent_id):
        if exists(aid, key):
            return read(aid, key)
    raise KeyError(f"context {key!r} not found for {agent_id!r}")


def context_keys_for_agents(
    agent_id: str | None,
    keys_for_agent: Callable[[str], Iterable[str]],
) -> list[str]:
    keys: set[str] = set()
    for aid in context_agent_ids(agent_id):
        keys.update(keys_for_agent(aid))
    return sorted(keys)


__all__ = ["context_agent_ids", "context_keys_for_agents", "first_context_value"]
