"""Session — persistent per-agent invariants + per-turn state log.

Each agent owns one append-only ``session.jsonl`` of state snapshots and
one ``agent.json`` of its run-invariants (query, depth, config, system
prompt, refs to workspace + runtime). Cross-agent topology is recovered
from each agent's ``parent_agent_id`` field; no edges are stored.

Layout::

    <workspace>/
      graph.json                                # workspace manifest (root + agent list)
      session/<aid>/agent.json                  # per-agent invariants (one-shot)
      session/<aid>/session.jsonl               # state log (append-only)
      session/<aid>/latest.json                 # latest-state summary
      session/<aid>/transcript.json             # exact LLM I/O, flat & merged

``transcript.json`` is a *single* document per agent that grows
turn-by-turn. ``messages`` is the flat conversation as the LLM
saw it across all turns combined — every turn appends only the
new entries (the user nudge, if any, plus the assistant reply).
``metadata`` is a parallel list with one entry per message;
all entries are ``{}`` except each assistant message, which
carries the call-specific fields::

    {
      "agent_id": <aid>,
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": [
        {}, {},
        {ts, model, force_final, input_tokens, output_tokens,
         elapsed_s, after_node_id, after_seq},
        {},
        {ts, model, force_final, input_tokens, output_tokens,
         elapsed_s, after_node_id, after_seq}
      ]
    }

The transcript is the ground truth for "what did the LLM
actually see?" — useful for debugging prompt issues, replaying
a turn under a different model, or auditing context bloat.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rlmflow.graph import Graph, Node, is_done, is_observation, parse_node_obj
from rlmflow.workspace.store import Store, copy_workspace_paths, resolve_backend

SESSION_VARIABLE_PROMPT = """
`SESSION` is a read-only view of other agents in this recursive tree.

- `SESSION.agent_id` / `SESSION.node_id`
- `SESSION.list_agents()`
- `SESSION.read(agent_id)`
- `SESSION.grep(pattern, max_results=50)`
- `SESSION.parent(agent_id=None)` / `SESSION.ancestors(agent_id=None)`
- `SESSION.children(agent_id=None)` / `SESSION.subtree(agent_id=None)`
- `SESSION.tree()`

Use it after delegation to inspect child results/transcripts and target repairs.
"""


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "root"


# ── Session ABC ──────────────────────────────────────────────────────


class Session(ABC):
    """Persist per-agent invariants + per-turn state logs.

    Engine surface is narrow: register an agent (its per-run invariants)
    at spawn, append a state per turn, and rebuild the recursive
    :class:`Graph` from the persisted layout on every read.
    """

    @abstractmethod
    def write_agent(self, graph: Graph) -> None: ...

    @abstractmethod
    def write_state(self, state: Node) -> None: ...

    @abstractmethod
    def read_transcript(self, agent_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def write_transcript(self, agent_id: str, transcript: dict[str, Any]) -> None: ...

    @abstractmethod
    def load_graph(self) -> Graph: ...

    @abstractmethod
    def fork(self, new_location: object) -> Session: ...


# ── FileSession ──────────────────────────────────────────────────────


class FileSession(Session):
    """Filesystem-backed :class:`Session`."""

    def __init__(self, root: Store | str | Path) -> None:
        self.store, self.root = resolve_backend(root)

    # ── writes ───────────────────────────────────────────────────────

    def write_agent(self, graph: Graph) -> None:
        self.store.write_json(
            f"session/{_safe_name(graph.agent_id)}/agent.json",
            graph.meta_dict(),
        )
        self._touch_graph_agent(graph.agent_id)

    def write_state(self, state: Node) -> None:
        path = f"session/{_safe_name(state.agent_id)}/session.jsonl"
        self.store.append_jsonl(path, state)
        self._write_latest(state)

    def read_transcript(self, agent_id: str) -> dict[str, Any] | None:
        path = f"session/{_safe_name(agent_id)}/transcript.json"
        if not self.store.exists(path):
            return None
        return self.store.read_json(path)

    def write_transcript(self, agent_id: str, transcript: dict[str, Any]) -> None:
        path = f"session/{_safe_name(agent_id)}/transcript.json"
        self.store.write_json(path, transcript)

    # ── reads ────────────────────────────────────────────────────────

    def load_graph(self) -> Graph:
        manifest = self._load_manifest()
        agent_dicts: dict[str, dict[str, Any]] = {}
        agent_states: dict[str, tuple[Node, ...]] = {}
        for aid in manifest["agents"]:
            meta_path = f"session/{_safe_name(aid)}/agent.json"
            if not self.store.exists(meta_path):
                continue
            agent_dicts[aid] = self.store.read_json(meta_path)
            session_path = f"session/{_safe_name(aid)}/session.jsonl"
            agent_states[aid] = tuple(
                parse_node_obj(line) for line in self.store.read_jsonl(session_path)
            )
        return _build_graph(
            root_agent_id=manifest["root_agent_id"],
            agent_dicts=agent_dicts,
            agent_states=agent_states,
        )

    # ── helpers ──────────────────────────────────────────────────────

    def _load_manifest(self) -> dict[str, Any]:
        if self.store.exists("graph.json"):
            return self.store.read_json("graph.json")
        return {"root_agent_id": "root", "agents": []}

    def _touch_graph_agent(self, agent_id: str) -> None:
        manifest = self._load_manifest()
        dirty = False
        if not manifest["agents"]:
            # First write into a fresh workspace: this agent becomes the root.
            manifest["root_agent_id"] = agent_id
            dirty = True
        if agent_id not in manifest["agents"]:
            manifest["agents"].append(agent_id)
            dirty = True
        if dirty:
            self.store.write_json("graph.json", manifest)

    def _write_latest(self, state: Node) -> None:
        summary = {
            "agent_id": state.agent_id,
            "latest_node_id": state.id,
            "seq": state.seq,
            "type": state.type,
            "terminal": state.terminal,
            "result": getattr(state, "result", None),
        }
        self.store.write_json(
            f"session/{_safe_name(state.agent_id)}/latest.json", summary
        )

    def fork(self, new_location: object) -> Session:
        return FileSession(
            copy_workspace_paths(
                self.store,
                new_location,
                ("graph.json", "session"),
            )
        )


# ── InMemorySession ──────────────────────────────────────────────────


class InMemorySession(Session):
    """Process-local session for runs without a Workspace."""

    def __init__(self) -> None:
        self.agent_dicts: dict[str, dict[str, Any]] = {}
        self.agent_states: dict[str, list[Node]] = {}
        self.agent_transcripts: dict[str, dict[str, Any]] = {}
        self.root_agent_id: str = "root"

    def write_agent(self, graph: Graph) -> None:
        if not self.agent_dicts:
            self.root_agent_id = graph.agent_id
        self.agent_dicts[graph.agent_id] = graph.meta_dict()
        self.agent_states.setdefault(graph.agent_id, [])
        self.agent_transcripts.setdefault(graph.agent_id, {})

    def write_state(self, state: Node) -> None:
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

    def load_graph(self) -> Graph:
        return _build_graph(
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


# ── recursive Graph builder ──────────────────────────────────────────


def _build_graph(
    *,
    root_agent_id: str,
    agent_dicts: dict[str, dict[str, Any]],
    agent_states: dict[str, tuple[Node, ...]],
) -> Graph:
    """Recover the recursive :class:`Graph` from flat per-agent dicts.

    Orphan agents (non-root with no ``parent_agent_id``) are attached
    to the root so nothing is silently dropped.
    """
    children_by_parent: dict[str, list[str]] = {}
    for aid, data in agent_dicts.items():
        if aid == root_agent_id:
            continue
        parent = data.get("parent_agent_id") or root_agent_id
        children_by_parent.setdefault(parent, []).append(aid)

    def build(aid: str) -> Graph:
        data = agent_dicts.get(aid, {"agent_id": aid})
        states = agent_states.get(aid, ())
        children = {
            child_aid: build(child_aid) for child_aid in children_by_parent.get(aid, [])
        }
        return Graph.from_meta_dict(data, states=states, children=children)

    if root_agent_id not in agent_dicts:
        return Graph(agent_id=root_agent_id)
    return build(root_agent_id)


# ── SessionVariable ──────────────────────────────────────────────────


class SessionVariable:
    """REPL handle giving an agent read-only access to every other agent's
    session in the same recursive tree.

    Lazily loads a :class:`Graph` from the underlying session on each call.
    """

    def __init__(
        self,
        session: Session,
        *,
        agent_id: str,
        node_id: str = "",
        branch_id: str = "main",
    ) -> None:
        self.store = session
        self.agent_id = agent_id
        self.node_id = node_id
        self.branch_id = branch_id

    def _graph(self) -> Graph:
        return self.store.load_graph()

    def list_agents(self) -> list[dict[str, Any]]:
        """Summarize every other agent in the session."""
        return _summarize(self._graph(), exclude={self.agent_id})

    def read(self, agent_id: str) -> str:
        """Render the named agent's transcript."""
        graph = self._graph()
        if agent_id not in graph.agents:
            return f"(no nodes for agent {agent_id!r})"
        from rlmflow.utils.viewer import agent_transcript

        return agent_transcript(graph[agent_id], include_system=True)

    _SEARCH_FIELDS: tuple[str, ...] = (
        "content",
        "reply",
        "output",
        "result",
        "error",
    )

    def grep(self, pattern: str, *, max_results: int = 50) -> str:
        """Regex-search across every message field of every other agent."""
        compiled = re.compile(pattern, re.IGNORECASE)
        matches: list[str] = []
        for agent in self._graph().walk():
            if agent.agent_id == self.agent_id:
                continue
            if agent.query:
                for line in agent.query.splitlines():
                    if compiled.search(line):
                        matches.append(f"{agent.agent_id}:query:{line.strip()[:160]}")
                        if len(matches) >= max_results:
                            return "\n".join(matches)
            for state in agent.states:
                for field in self._SEARCH_FIELDS:
                    text = getattr(state, field, "") or ""
                    if not text:
                        continue
                    for line in text.splitlines():
                        if compiled.search(line):
                            matches.append(
                                f"{agent.agent_id}:{state.type}:{line.strip()[:160]}"
                            )
                            if len(matches) >= max_results:
                                return "\n".join(matches)
        return "\n".join(matches)

    def parent(self, agent_id: str | None = None) -> str | None:
        graph = self._graph()
        target = agent_id or self.agent_id
        if target not in graph.agents:
            return None
        return graph[target].parent_id

    def ancestors(self, agent_id: str | None = None) -> list[str]:
        graph = self._graph()
        out: list[str] = []
        target = agent_id or self.agent_id
        while target in graph.agents:
            parent = graph[target].parent_id
            if not parent:
                break
            out.append(parent)
            target = parent
        return list(reversed(out))

    def children(self, agent_id: str | None = None) -> list[str]:
        graph = self._graph()
        target = agent_id or self.agent_id
        if target not in graph.agents:
            return []
        return sorted(graph[target].children.keys())

    def subtree(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        graph = self._graph()
        target = agent_id or self.agent_id
        if target not in graph.agents:
            return []
        ids = {sub.agent_id for sub in graph[target].walk()} - {target}
        return _summarize(graph, include=ids)

    def tree(self) -> str:
        return self._graph().tree()


# ── shared summarization ─────────────────────────────────────────────


def _summarize(
    graph: Graph,
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for agent in graph.walk():
        aid = agent.agent_id
        if include is not None and aid not in include:
            continue
        if exclude is not None and aid in exclude:
            continue
        tip = agent.current()
        if tip is None:
            continue
        preview_src = ""
        if is_done(tip):
            preview_src = tip.result or ""
        elif is_observation(tip):
            preview_src = (
                getattr(tip, "content", "")
                or getattr(tip, "output", "")
                or getattr(tip, "reply", "")
                or ""
            )
        preview = " ".join(preview_src.split())
        out.append(
            {
                "agent_id": aid,
                "type": tip.type,
                "depth": agent.depth,
                "terminal": bool(tip.terminal),
                "result_preview": preview[:120] + ("…" if len(preview) > 120 else ""),
            }
        )
    out.sort(key=lambda r: (r["depth"], r["agent_id"]))
    return out


__all__ = [
    "FileSession",
    "InMemorySession",
    "Session",
    "SessionVariable",
    "SESSION_VARIABLE_PROMPT",
]
