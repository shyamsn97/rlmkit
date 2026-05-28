"""Base workspace, session, and context interfaces."""

from __future__ import annotations

import posixpath
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rlmflow.graph import (
    Graph,
    Node,
    WorkspaceRef,
    is_done,
    is_errored,
    is_exec_output,
    is_llm_output,
    is_observation,
    is_user_query,
    retrace_steps,
)
from rlmflow.workspace.sync import engine_state_path, excluded, sync_lock_for


class ContextVariable:
    """Lazy handle over the current task/data payload."""

    def __init__(
        self,
        context: Context,
        *,
        agent_id: str = "root",
        key: str = "context",
    ) -> None:
        self.store = context
        self.agent_id = agent_id
        self.key = key

    def info(self) -> dict[str, Any]:
        return self.store.info(self.key, agent_id=self.agent_id)

    def read(self, start: int = 0, end: int | None = None) -> str:
        return self.store.read(self.key, agent_id=self.agent_id)[start:end]

    def lines(self, start: int = 0, end: int | None = None) -> list[str]:
        return self.store.read(self.key, agent_id=self.agent_id).splitlines()[start:end]

    def line_count(self) -> int:
        return len(self.store.read(self.key, agent_id=self.agent_id).splitlines())

    def grep(self, pattern: str, *, max_results: int = 50) -> str:
        compiled = re.compile(pattern)
        matches: list[str] = []
        text = self.store.read(self.key, agent_id=self.agent_id)
        for idx, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                matches.append(f"{idx}:{line}")
                if len(matches) >= max_results:
                    break
        return "\n".join(matches)


class Context(ABC):
    """Store task/data payloads exposed to the REPL as ``context``."""

    @abstractmethod
    def write(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("This context backend does not support blobs.")

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        return []

    @abstractmethod
    def fork(self, new_location: object) -> Context:
        """Return a deep copy of this context payload store."""

    def info(
        self,
        key: str = "context",
        *,
        agent_id: str = "root",
    ) -> dict[str, Any]:
        text = self.read(key, agent_id=agent_id)
        return {
            "key": key,
            "agent_id": agent_id,
            "chars": len(text),
            "approx_tokens": len(text) // 4,
            "lines": len(text.splitlines()),
        }


class Session(ABC):
    """Persist per-agent invariants + per-turn state logs."""

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


class BaseWorkspace(ABC):
    """Base interface for durable workspaces."""

    root: Path
    session: Session
    context: Context
    branch_id: str
    uri: str | None = None

    @abstractmethod
    def materialize(self) -> Path:
        """Ensure the workspace is available as a local filesystem root."""

    @abstractmethod
    def commit(self) -> None:
        """Persist materialized changes back to durable storage."""

    @abstractmethod
    def fork(
        self,
        new_location: str | Path | None = None,
        *,
        new_branch_id: str | None = None,
    ) -> BaseWorkspace:
        """Create a durable branch copy."""

    def path(self, *parts: str) -> Path:
        """Return a path inside the materialized workspace root."""

        return self.materialize().joinpath(*parts)

    @property
    def sync_lock(self):
        """Process-local lock for sync operations on this workspace."""

        return sync_lock_for(self.root)

    def push_to(
        self,
        runtime,
        remote_root: str = "/workspace",
        *,
        replace: bool = True,
    ) -> None:
        """Sync this workspace into a runtime execution filesystem."""

        root = self.materialize()
        with self.sync_lock:
            if replace:
                runtime.remove_path(remote_root, recursive=True)
            for path in root.rglob("*"):
                rel = path.relative_to(root).as_posix()
                if path.is_dir() or excluded(rel):
                    continue
                runtime.upload_file(path, posixpath.join(remote_root, rel))

    def pull_from(
        self,
        runtime,
        remote_root: str = "/workspace",
        *,
        merge: bool = False,
        skip_engine_state: bool = False,
    ) -> None:
        """Sync runtime filesystem changes back into this workspace."""

        root = self.materialize()
        incoming = root.parent / f".{root.name}.incoming"
        with self.sync_lock:
            if incoming.exists():
                shutil.rmtree(incoming)
            incoming.mkdir(parents=True, exist_ok=True)

            for rel in runtime.list_files(remote_root):
                if excluded(rel) or (skip_engine_state and engine_state_path(rel)):
                    continue
                runtime.download_file(
                    posixpath.join(remote_root, rel),
                    incoming / rel,
                )

            if not merge:
                for item in list(root.iterdir()):
                    rel = item.relative_to(root).as_posix()
                    if excluded(rel) or (skip_engine_state and engine_state_path(rel)):
                        continue
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

            for path in incoming.rglob("*"):
                rel = path.relative_to(incoming)
                dst = root / rel
                if path.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dst)

            shutil.rmtree(incoming)
            self.commit()

    def ref(self) -> WorkspaceRef:
        return WorkspaceRef(root=str(self.root), branch_id=self.branch_id)

    def load_graph(self) -> Graph:
        """Load the current graph snapshot from this workspace's session."""
        return self.session.load_graph()

    def load_steps(self) -> list[Graph]:
        """Load the run as a list of snapshots, one per state-append."""
        return retrace_steps(self.load_graph())

    def open_viewer(self, **kwargs):
        """Open the interactive viewer for this workspace."""
        from rlmflow.utils.viewer import open_viewer

        return open_viewer(self, **kwargs)


class SessionVariable:
    """REPL handle for reading sibling agent sessions in the same tree."""

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

    def list_agents(self) -> list[str]:
        agents = [
            agent for agent in self._graph().walk() if agent.agent_id != self.agent_id
        ]
        agents.sort(key=lambda agent: (agent.depth, agent.agent_id))
        return [agent.agent_id for agent in agents]

    def summarize_agent(self, agent_id: str) -> dict[str, Any] | None:
        graph = self._graph()
        if agent_id not in graph.agents:
            return None
        return _summarize_agent(graph[agent_id])

    def read(self, agent_id: str) -> str:
        graph = self._graph()
        if agent_id not in graph.agents:
            return f"(no nodes for agent {agent_id!r})"
        from rlmflow.utils.viewer import agent_transcript

        return agent_transcript(graph[agent_id], include_system=True)

    def messages(self, agent_id: str) -> list[dict[str, str]]:
        graph = self._graph()
        if agent_id not in graph.agents:
            return []
        transcript = self.store.read_transcript(agent_id)
        if transcript and transcript.get("messages"):
            return [
                {"role": str(msg["role"]), "content": str(msg["content"])}
                for msg in transcript["messages"]
            ]
        return _messages_from_graph(graph[agent_id])

    def recent(self, agent_id: str, n: int = 5) -> list[dict[str, str]]:
        if n <= 0:
            return []
        return self.messages(agent_id)[-n:]

    _SEARCH_FIELDS: tuple[str, ...] = (
        "content",
        "reply",
        "output",
        "result",
        "error",
    )

    def grep(self, pattern: str, *, max_results: int = 50) -> str:
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


def build_graph(
    *,
    root_agent_id: str,
    agent_dicts: dict[str, dict[str, Any]],
    agent_states: dict[str, tuple[Node, ...]],
) -> Graph:
    """Recover the recursive :class:`Graph` from flat per-agent dicts."""

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
        summary = _summarize_agent(agent)
        if summary is not None:
            out.append(summary)
    out.sort(key=lambda r: (r["depth"], r["agent_id"]))
    return out


def _messages_from_graph(agent: Graph) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if agent.system_prompt:
        msgs.append({"role": "system", "content": agent.system_prompt})
    for state in agent.states:
        if is_user_query(state):
            msgs.append({"role": "user", "content": state.content})
        elif is_llm_output(state):
            msgs.append({"role": "assistant", "content": state.reply})
        elif is_exec_output(state):
            body = state.content or state.output or ""
            if body:
                msgs.append({"role": "user", "content": body})
        elif is_errored(state):
            body = state.content or ""
            if body:
                msgs.append({"role": "user", "content": body})
    return msgs


def _summarize_agent(agent: Graph) -> dict[str, Any] | None:
    tip = agent.current()
    if tip is None:
        return None
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
    return {
        "agent_id": agent.agent_id,
        "type": tip.type,
        "depth": agent.depth,
        "terminal": bool(tip.terminal),
        "result_preview": preview[:120] + ("..." if len(preview) > 120 else ""),
    }


__all__ = [
    "BaseWorkspace",
    "Context",
    "ContextVariable",
    "Session",
    "SessionVariable",
    "build_graph",
]
