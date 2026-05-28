"""In-memory workspace data implementations."""

from __future__ import annotations

from typing import Any

from rlmflow.workspace.base import Context, Session, build_graph


class InMemoryContext(Context):
    """Process-local payload store for runs without a filesystem workspace."""

    def __init__(self) -> None:
        self.blobs: dict[tuple[str, str], str] = {}

    def write(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del metadata
        self.blobs[(agent_id, key)] = value

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        for aid in (agent_id, "root"):
            if (aid, key) in self.blobs:
                return self.blobs[(aid, key)]
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        agent_ids = [agent_id] if agent_id else ["root"]
        if agent_id and agent_id != "root":
            agent_ids.append("root")
        return sorted({key for aid, key in self.blobs if aid in agent_ids})

    def fork(self, new_location: object) -> Context:
        del new_location
        out = InMemoryContext()
        out.blobs = dict(self.blobs)
        return out


class InMemorySession(Session):
    """Process-local session for runs without a filesystem workspace."""

    def __init__(self) -> None:
        self.agent_dicts: dict[str, dict[str, Any]] = {}
        self.agent_states = {}
        self.agent_transcripts: dict[str, dict[str, Any]] = {}
        self.root_agent_id: str = "root"

    def write_agent(self, graph) -> None:
        if not self.agent_dicts:
            self.root_agent_id = graph.agent_id
        self.agent_dicts[graph.agent_id] = graph.meta_dict()
        self.agent_states.setdefault(graph.agent_id, [])
        self.agent_transcripts.setdefault(graph.agent_id, {})

    def write_state(self, state) -> None:
        self.agent_states.setdefault(state.agent_id, []).append(state)

    def read_transcript(self, agent_id: str) -> dict[str, Any] | None:
        existing = self.agent_transcripts.get(agent_id)
        if not existing:
            return None
        return {k: list(v) if isinstance(v, list) else v for k, v in existing.items()}

    def write_transcript(self, agent_id: str, transcript: dict[str, Any]) -> None:
        self.agent_transcripts[agent_id] = {
            k: list(v) if isinstance(v, list) else v for k, v in transcript.items()
        }

    def load_graph(self):
        return build_graph(
            root_agent_id=self.root_agent_id,
            agent_dicts=self.agent_dicts,
            agent_states={aid: tuple(s) for aid, s in self.agent_states.items()},
        )

    def fork(self, new_location: object) -> Session:
        del new_location
        out = InMemorySession()
        out.agent_dicts = {aid: dict(d) for aid, d in self.agent_dicts.items()}
        out.agent_states = {aid: list(s) for aid, s in self.agent_states.items()}
        out.agent_transcripts = {
            aid: {k: list(v) if isinstance(v, list) else v for k, v in t.items()}
            for aid, t in self.agent_transcripts.items()
        }
        out.root_agent_id = self.root_agent_id
        return out


__all__ = ["InMemoryContext", "InMemorySession"]
