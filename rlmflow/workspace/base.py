"""Base workspace, session, and context interfaces."""

from __future__ import annotations

import posixpath
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rlmflow.graph import Graph, Node, WorkspaceRef, retrace_steps
from rlmflow.workspace.graph_load import build_graph
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


__all__ = [
    "BaseWorkspace",
    "Context",
    "ContextVariable",
    "Session",
    "build_graph",
]
