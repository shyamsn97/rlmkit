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

**Contract:** ``state.messages`` is the source of truth during execution.
Session is a write-through mirror — the engine writes ``state.messages``
to the session store after every step.  Session data is used for:

- Cross-agent visibility (``read_history`` / ``list_sessions`` tools)
- Crash recovery (``RLMState.from_session()``)
- Context truncation recovery (agent re-reads its own earlier messages)

To persist a full state tree at once, use ``session.write_tree(state)``.
To reconstruct a state tree from session, use ``RLMState.from_session(session)``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlmkit.state import RLMState

SESSION_TOOLS_HINT = """\

**Tools for reading sessions:**
- `list_sessions()` — lists every agent in the tree with its ID and task. Use this to see who has run and what they worked on.
- `read_history(agent_id, last_n=20)` — reads the conversation transcript for any agent. Defaults to your own. Use this to review what you or any sibling/child agent has done.

**When to use sessions:**
- **Before starting work** — call `list_sessions()` to see if other agents have already done relevant work. Don't redo what a sibling already finished.
- **After children return** — if a child's `done()` result is too terse, use `read_history(child_id)` to read the full transcript of what they did.
- **On resumption** — if you are a re-delegated agent (same name, new task), call `read_history()` to recall your previous work. Your variables are still set, but your context window is fresh.
- **When context is truncated** — if your history was trimmed, use `read_history()` to re-read your earlier messages."""


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

    def write_tree(self, state: RLMState) -> None:
        """Recursively persist messages for every node in the state tree.

        Strips system messages — the prompt is rebuilt dynamically.
        """
        if state.messages:
            msgs = [m for m in state.messages if m.get("role") != "system"]
            self.write(state.agent_id or "root", msgs)
        for child in state.children:
            self.write_tree(child)

    def prompt_hint(self) -> str:
        """Return prompt text explaining how sessions work for this backend."""
        return SESSION_TOOLS_HINT


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

    def prompt_hint(self) -> str:
        return (
            f"""\
`SESSION` points to the session store. Every agent's conversation history is \
persisted as JSON files under `{self.base_dir}/`, mirroring the agent tree:
```
{self.base_dir}/
├── session.json              ← root agent's history
├── scanner_auth/
│   ├── session.json          ← root.scanner_auth
│   └── chunk_0/
│       └── session.json      ← root.scanner_auth.chunk_0
└── scanner_api/
    └── session.json
```

You can also read session files directly via `read_file()` at the paths above."""
            + SESSION_TOOLS_HINT
        )
