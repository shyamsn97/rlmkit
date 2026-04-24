"""Save and load traces — ordered lists of ``RLMState`` snapshots.

A trace is just a sequence of states. The per-turn query already lives
on each ``state.query``; long-running multi-turn sessions accumulate
into a single trace without losing that information.

Layout on disk::

    {"steps": [dumped_state, ...], "metadata": {...}}

``steps`` preserve ``StepEvent`` subclass info via a ``_type``
discriminator — ``LLMReply`` / ``CodeExec`` / ``ResumeExec`` /
``NoCodeBlock`` / ``ChildStep`` round-trip cleanly.

Usage::

    from rlmkit.utils.trace import save_trace, load_trace

    save_trace(states, "traces/run1")
    save_trace(states, "traces/run1", metadata={"model": "gpt-5"})

    t = load_trace("traces/run1")
    t.states          # list[RLMState] — typed events preserved
    t.metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlmkit.state import RLMState, _dump_state, _hydrate_state


def _resolve(path: str | Path, *, writing: bool) -> Path:
    """Accept a .json file or a directory; normalize to ``<dir>/trace.json``."""
    p = Path(path)
    if p.is_dir() or not p.suffix:
        if writing:
            p.mkdir(parents=True, exist_ok=True)
        p = p / "trace.json"
    elif writing:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Trace:
    """A sequence of ``RLMState`` snapshots plus optional metadata."""

    states: list[RLMState]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Write the trace to *path* (file or directory) as JSON."""
        return save_trace(self.states, path, metadata=self.metadata)

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        """Load a trace previously written by :func:`save_trace`."""
        return load_trace(path)


def save_trace(
    states: list[RLMState],
    path: str | Path = "trace.json",
    metadata: dict | None = None,
) -> Path:
    """Write *states* to *path* as a JSON trace.

    *path* may be a ``.json`` file or a directory; when a directory is
    given the trace is written to ``<path>/trace.json``.
    """
    p = _resolve(path, writing=True)
    data: dict[str, Any] = {"steps": [_dump_state(s) for s in states]}
    if metadata:
        data["metadata"] = metadata
    p.write_text(json.dumps(data, default=str, indent=2))
    return p


def load_trace(path: str | Path) -> Trace:
    """Load a trace written by :func:`save_trace`."""
    p = _resolve(path, writing=False)
    data = json.loads(p.read_text())
    return Trace(
        states=[_hydrate_state(s) for s in data["steps"]],
        metadata=data.get("metadata") or {},
    )
