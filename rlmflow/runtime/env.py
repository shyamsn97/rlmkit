"""Constants and helpers for per-runtime execution state."""

from __future__ import annotations

from typing import Any

AGENT_ID = "AGENT_ID"
DEPTH = "DEPTH"
MAX_DEPTH = "MAX_DEPTH"
PARENT_NODE_ID = "PARENT_NODE_ID"
DONE_RESULT = "DONE_RESULT"
REPLAY_QUEUE = "_REPLAY_QUEUE"


def execution_facts(
    *,
    agent_id: str,
    depth: int,
    max_depth: int,
    parent_node_id: str,
) -> dict[str, Any]:
    return {
        AGENT_ID: agent_id,
        DEPTH: depth,
        MAX_DEPTH: max_depth,
        PARENT_NODE_ID: parent_node_id,
    }


def seed_execution_env(env: dict[str, object], facts: dict[str, Any]) -> None:
    env.clear()
    env.update({**facts, DONE_RESULT: None})


def done_result(env: dict[str, object]) -> object:
    return env.get(DONE_RESULT)


def replay_queue(env: dict[str, object]) -> list[str] | None:
    queue = env.get(REPLAY_QUEUE)
    return queue if isinstance(queue, list) else None


def set_replay_queue(env: dict[str, object], agent_ids: list[str]) -> None:
    env[REPLAY_QUEUE] = list(agent_ids)


def clear_replay_queue(env: dict[str, object]) -> None:
    env.pop(REPLAY_QUEUE, None)


__all__ = [
    "AGENT_ID",
    "DEPTH",
    "DONE_RESULT",
    "MAX_DEPTH",
    "PARENT_NODE_ID",
    "REPLAY_QUEUE",
    "clear_replay_queue",
    "done_result",
    "execution_facts",
    "replay_queue",
    "seed_execution_env",
    "set_replay_queue",
]
