"""Tracing integrations: emit rlmflow runs into external observability stacks.

Today: JSON Lines state logs. Future: OpenTelemetry, LangSmith, W&B.
"""

from __future__ import annotations

import json
from pathlib import Path

from rlmflow.graph import Graph


def json_logs(graph: Graph, path: str | Path) -> Path:
    """Write a newline-delimited JSON state log — one node per line.

    Useful for piping into Loki / Datadog / Splunk / DuckDB.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for node in graph.nodes:
            f.write(json.dumps(node.to_dict(), default=str) + "\n")
    return p


__all__ = ["json_logs"]
