"""Session — agent conversation persistence.

Subclass ``Session`` for remote or DB-backed storage (Redis, S3, etc.).
The default ``FileSession`` writes JSON files to a directory tree that
mirrors the agent hierarchy::

    context/
    ├── session.json              ← root
    ├── search_0/
    │   ├── session.json          ← root.search_0
    │   └── sub_a/
    │       └── session.json      ← root.search_0.sub_a
    └── search_1/
        └── session.json
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class Session(ABC):
    """Persist and query agent message histories.

    The engine writes after each step.  Agents can read their own
    or other agents' sessions via tools backed by the same store.
    """

    @abstractmethod
    def write(self, agent_id: str, messages: list[dict]) -> None:
        """Write the full message history for an agent."""

    @abstractmethod
    def read(self, agent_id: str) -> list[dict]:
        """Read the message history for an agent.  Empty list if none."""

    @abstractmethod
    def list_agents(self) -> list[str]:
        """List all agent IDs that have stored sessions."""

    @abstractmethod
    def exists(self, agent_id: str) -> bool:
        """Check whether a session exists for the given agent."""


class FileSession(Session):
    """Session backed by JSON files in a local directory."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _agent_path(self, agent_id: str) -> Path:
        parts = agent_id.split(".")
        if len(parts) <= 1:
            return self.base_dir / "session.json"
        return self.base_dir / "/".join(parts[1:]) / "session.json"

    def write(self, agent_id: str, messages: list[dict]) -> None:
        path = self._agent_path(agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(messages, indent=2, default=str))

    def read(self, agent_id: str) -> list[dict]:
        path = self._agent_path(agent_id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def list_agents(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        results = []
        for f in sorted(self.base_dir.rglob("session.json")):
            rel = f.parent.relative_to(self.base_dir)
            parts = [p for p in rel.parts if p != "."]
            agent_id = "root." + ".".join(parts) if parts else "root"
            results.append(agent_id)
        return results

    def exists(self, agent_id: str) -> bool:
        return self._agent_path(agent_id).exists()
