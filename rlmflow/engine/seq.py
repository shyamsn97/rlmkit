"""Small shared helpers used across the engine.

These are pure (or close to it) utilities that don't fit on a specific
transition module. Keeping them in one place avoids cross-imports
between ``transitions``, ``code``, ``replay``, etc.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from rlmflow.graph import Graph, Node, is_llm_action
from rlmflow.pool import CallablePool, Pool, SequentialPool, ThreadPool
from rlmflow.prompts.messages import EXECUTION_OUTPUT
from rlmflow.workspace import Session

if TYPE_CHECKING:
    from rlmflow.engine.config import RLMConfig

ROOT_RUNTIME_ID = "root"


def append_node(session: Session, graph: Graph, node: Node) -> Node:
    """Append ``node`` to ``graph``'s trajectory with the next ``seq``.

    Single source of truth for sequence numbering. Call sites never set
    ``agent_id`` or ``seq`` themselves — pass a node with the payload
    fields populated and this helper assigns identity.

    Also mirrors the new node into the local ``graph.states`` list so
    consecutive appends within one step compute ``seq`` correctly without
    needing to reload from the session between calls.
    """
    next_seq = (graph.states[-1].seq + 1) if graph.states else 0
    fields_dict = node.model_dump(exclude={"id", "agent_id", "seq"}, mode="python")
    fixed = node.__class__(agent_id=graph.agent_id, seq=next_seq, **fields_dict)
    session.write_state(fixed)
    graph.states.append(fixed)
    return fixed


def unique_child_id(parent_aid: str, name: str, existing: set[str]) -> str:
    base = f"{parent_aid}.{name}"
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def create_pool(config: "RLMConfig", pool: Pool | Callable | None = None) -> Pool:
    if pool is not None:
        return pool if hasattr(pool, "execute") else CallablePool(pool)
    if config.max_concurrency is None or config.max_concurrency <= 1:
        return SequentialPool()
    return ThreadPool(config.max_concurrency)


def iteration_count(graph: Graph) -> int:
    """How many :class:`LLMAction` nodes the agent has emitted so far."""
    return sum(is_llm_action(s) for s in graph.states)


def budget_exceeded(graph: Graph, max_budget: int | None) -> int | None:
    """Return total tokens if the run is over budget, else ``None``."""
    if max_budget is None:
        return None
    total = graph.total_tokens()
    return total if total >= max_budget else None


def truncate_output(raw: object, max_length: int) -> object:
    """Cap REPL output at ``max_length`` chars; passthrough non-strings."""
    if isinstance(raw, str) and len(raw) > max_length:
        return raw[:max_length] + "\n...<truncated>"
    return raw


def format_exec_output(output: str) -> str:
    return EXECUTION_OUTPUT.format(output=output or "(no output)")


__all__ = [
    "ROOT_RUNTIME_ID",
    "append_node",
    "budget_exceeded",
    "create_pool",
    "format_exec_output",
    "iteration_count",
    "truncate_output",
    "unique_child_id",
]
