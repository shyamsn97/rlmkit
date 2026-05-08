"""Save and load typed RLMFlow node traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

from rlmflow.node import Node, parse_node_obj
from rlmflow.utils.viz import session_events


class Trace(NamedTuple):
    states: list[Node]
    metadata: dict[str, Any]
    events: list[Node] = []


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
    *,
    include_events: bool = False,
    events: list[Node] | None = None,
) -> Path:
    p = _resolve(path, writing=True)
    data: dict[str, Any] = {"steps": [state.to_dict() for state in states]}
    trace_events = list(events or [])
    if include_events and not trace_events:
        trace_events = _events_from_states(states)
    if trace_events:
        data["events"] = [event.to_dict() for event in trace_events]
    if metadata:
        data["metadata"] = metadata
    p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    return p


def _events_from_states(states: list[Node]) -> list[Node]:
    for state in states:
        ws = getattr(state, "workspace", None)
        root = getattr(ws, "root", None) if ws else None
        if not root:
            continue
        events = session_events(Path(root) / "session")
        if events:
            return events
    return []


def load_trace(path: str | Path) -> Trace:
    p = _resolve(path, writing=False)
    data = json.loads(p.read_text(encoding="utf-8"))
    states = [parse_node_obj(state) for state in data["steps"]]
    events = [parse_node_obj(event) for event in data.get("events", [])]
    return Trace(
        states=states,
        metadata=data.get("metadata") or {},
        events=events,
    )
