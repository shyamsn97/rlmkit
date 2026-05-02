"""Tracing integrations: emit rlmflow runs into external observability stacks.

Today: JSON Lines event logs. Future: OpenTelemetry, LangSmith, W&B.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from rlmflow.node import Node


def json_logs(states: Node | Iterable[Node], path: str | Path) -> Path:
    """Write a newline-delimited JSON event log: one node per line.

    Useful for piping into Loki / Datadog / Splunk / DuckDB. Pass either a
    final state (its subtree is walked) or a step history.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(states, Node):
        nodes: Iterable[Node] = states.walk()
    else:
        seen: set[str] = set()
        ordered: list[Node] = []
        for state in states:
            for node in state.walk():
                if node.id not in seen:
                    seen.add(node.id)
                    ordered.append(node)
        nodes = ordered
    with p.open("w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps(node.to_dict(), default=str) + "\n")
    return p


__all__ = ["json_logs"]
