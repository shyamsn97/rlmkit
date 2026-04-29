"""Save and load typed RLMFlow node traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

from rlmkit.node import Node, parse_node_obj


class Trace(NamedTuple):
    states: list[Node]
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
    states: list[Node],
    path: str | Path = "trace.json",
    metadata: dict | None = None,
) -> Path:
    p = _resolve(path, writing=True)
    data: dict[str, Any] = {"steps": [state.to_dict() for state in states]}
    if metadata:
        data["metadata"] = metadata
    p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    return p


def load_trace(path: str | Path) -> Trace:
    p = _resolve(path, writing=False)
    data = json.loads(p.read_text(encoding="utf-8"))
    return Trace(
        states=[parse_node_obj(state) for state in data["steps"]],
        metadata=data.get("metadata") or {},
    )
