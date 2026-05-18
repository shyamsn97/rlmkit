"""Save and load :class:`Graph` traces.

A trace is a list of graph snapshots — typically one per
:meth:`RLMFlow.step` call. The file format is JSON: a top-level dict with
``"steps"`` (list of ``Graph.to_dict()`` payloads) and optional
``"metadata"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

from rlmflow.graph import Graph


class Trace(NamedTuple):
    graphs: list[Graph]
    metadata: dict[str, Any]


def _resolve(path: str | Path, *, writing: bool) -> Path:
    p = Path(path)
    if p.is_dir() or not p.suffix:
        if writing:
            p.mkdir(parents=True, exist_ok=True)
        p = p / "trace.json"
    elif writing:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_trace(
    graphs: list[Graph],
    path: str | Path = "trace.json",
    metadata: dict | None = None,
) -> Path:
    """Persist a list of :class:`Graph` snapshots."""
    p = _resolve(path, writing=True)
    data: dict[str, Any] = {"steps": [graph.to_dict() for graph in graphs]}
    if metadata:
        data["metadata"] = metadata
    p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    return p


def load_trace(path: str | Path) -> Trace:
    p = _resolve(path, writing=False)
    data = json.loads(p.read_text(encoding="utf-8"))
    graphs = [Graph.from_dict(step) for step in data.get("steps", [])]
    return Trace(graphs=graphs, metadata=data.get("metadata") or {})


__all__ = ["Trace", "load_trace", "save_trace"]
