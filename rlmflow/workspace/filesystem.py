"""Filesystem-backed workspace implementations."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from rlmflow.graph import WorkspaceRef, parse_node_obj
from rlmflow.workspace.base import BaseWorkspace, Context, Session, build_graph
from rlmflow.workspace.store import Store, copy_workspace_paths, resolve_backend


def _safe_name(value: str, *, default: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or default


class FileContext(Context):
    """Store-backed context persistence."""

    def __init__(self, root: Store | str | Path) -> None:
        self.store, self.root = resolve_backend(root)

    def _context_paths(self, key: str, *, agent_id: str) -> tuple[Path, Path]:
        safe = _safe_name(key, default="context")
        base = Path("context") / _safe_name(agent_id, default="root")
        if safe == "context":
            return base / "context.txt", base / "context_metadata.json"
        return base / f"{safe}.txt", base / f"{safe}_metadata.json"

    def write(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path, meta_path = self._context_paths(key, agent_id=agent_id)
        self.store.write_text(str(path), value)
        self.store.write_json(
            str(meta_path),
            {
                "key": key,
                "agent_id": agent_id,
                "chars": len(value),
                "metadata": metadata or {},
            },
        )

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        for aid in (agent_id, "root"):
            path, _ = self._context_paths(key, agent_id=aid)
            if self.store.exists(str(path)):
                return self.store.read_text(str(path))
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        agent_ids = [agent_id] if agent_id else ["root"]
        if agent_id and agent_id != "root":
            agent_ids.append("root")
        keys: set[str] = set()
        for aid in agent_ids:
            if aid is None:
                continue
            base = Path("context") / _safe_name(aid, default="root")
            for path in self.store.list(str(base)):
                if path.endswith(".txt"):
                    keys.add(Path(path).stem)
        return sorted(keys)

    def fork(self, new_location: object) -> Context:
        return FileContext(copy_workspace_paths(self.store, new_location, ("context",)))


class FileSession(Session):
    """Filesystem-backed :class:`Session`."""

    def __init__(self, root: Store | str | Path) -> None:
        self.store, self.root = resolve_backend(root)

    def write_agent(self, graph) -> None:
        self.store.write_json(
            f"session/{_safe_name(graph.agent_id, default='root')}/agent.json",
            graph.meta_dict(),
        )
        self._touch_graph_agent(graph.agent_id)

    def write_state(self, state) -> None:
        path = f"session/{_safe_name(state.agent_id, default='root')}/session.jsonl"
        self.store.append_jsonl(path, state)
        self._write_latest(state)

    def read_transcript(self, agent_id: str) -> dict[str, Any] | None:
        path = f"session/{_safe_name(agent_id, default='root')}/transcript.json"
        if not self.store.exists(path):
            return None
        return self.store.read_json(path)

    def write_transcript(self, agent_id: str, transcript: dict[str, Any]) -> None:
        path = f"session/{_safe_name(agent_id, default='root')}/transcript.json"
        self.store.write_json(path, transcript)

    def load_graph(self):
        manifest = self._load_manifest()
        agent_dicts: dict[str, dict[str, Any]] = {}
        agent_states = {}
        for aid in manifest["agents"]:
            safe = _safe_name(aid, default="root")
            meta_path = f"session/{safe}/agent.json"
            if not self.store.exists(meta_path):
                continue
            agent_dicts[aid] = self.store.read_json(meta_path)
            session_path = f"session/{safe}/session.jsonl"
            agent_states[aid] = tuple(
                parse_node_obj(line) for line in self.store.read_jsonl(session_path)
            )
        return build_graph(
            root_agent_id=manifest["root_agent_id"],
            agent_dicts=agent_dicts,
            agent_states=agent_states,
        )

    def _load_manifest(self) -> dict[str, Any]:
        if self.store.exists("graph.json"):
            return self.store.read_json("graph.json")
        return {"root_agent_id": "root", "agents": []}

    def _touch_graph_agent(self, agent_id: str) -> None:
        manifest = self._load_manifest()
        dirty = False
        if not manifest["agents"]:
            manifest["root_agent_id"] = agent_id
            dirty = True
        if agent_id not in manifest["agents"]:
            manifest["agents"].append(agent_id)
            dirty = True
        if dirty:
            self.store.write_json("graph.json", manifest)

    def _write_latest(self, state) -> None:
        self.store.write_json(
            f"session/{_safe_name(state.agent_id, default='root')}/latest.json",
            {
                "agent_id": state.agent_id,
                "latest_node_id": state.id,
                "seq": state.seq,
                "type": state.type,
                "terminal": state.terminal,
                "result": getattr(state, "result", None),
            },
        )

    def fork(self, new_location: object) -> Session:
        return FileSession(
            copy_workspace_paths(
                self.store,
                new_location,
                ("graph.json", "session"),
            )
        )


class Workspace(BaseWorkspace):
    """Local filesystem workspace."""

    def __init__(
        self,
        root: str | Path,
        *,
        session: Session | None = None,
        context: Context | None = None,
        branch_id: str = "main",
        uri: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = session or FileSession(self.root)
        self.context = context or FileContext(self.root)
        self.branch_id = branch_id
        self.uri = uri or str(self.root)

    @classmethod
    def create(
        cls,
        dir: str | Path,
        *,
        branch_id: str = "main",
        session: Session | None = None,
        context: Context | None = None,
    ) -> Workspace:
        return cls(dir, session=session, context=context, branch_id=branch_id)

    @classmethod
    def open(cls, ref: WorkspaceRef) -> Workspace:
        return cls.create(ref.root, branch_id=ref.branch_id)

    @classmethod
    def open_path(
        cls,
        dir: str | Path,
        *,
        branch_id: str = "main",
    ) -> Workspace:
        return cls.create(dir, branch_id=branch_id)

    @staticmethod
    def check_path(path: str | Path) -> bool:
        root = Path(path)
        return (
            root.is_dir()
            and (root / "graph.json").is_file()
            and (root / "session").is_dir()
        )

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def materialize(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def commit(self) -> None:
        """Local filesystem workspaces are already durable."""

    def fork(
        self,
        new_location: str | Path | None = None,
        *,
        new_branch_id: str | None = None,
        new_dir: str | Path | None = None,
    ) -> Workspace:
        location = new_location if new_location is not None else new_dir
        if location is None:
            raise TypeError("fork() requires a new workspace location")

        new_root = Path(location).resolve()
        branch_id = new_branch_id or _branch_id_from_location(location)
        if new_root.exists():
            shutil.rmtree(new_root)
        new_root.mkdir(parents=True, exist_ok=True)

        reserved = {"session", "context", "graph.json", "trace", "checkpoint.json"}
        for item in self.root.iterdir():
            if item.name in reserved:
                continue
            dst = new_root / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        return type(self)(
            new_root,
            session=self.session.fork(new_root),
            context=self.context.fork(new_root),
            branch_id=branch_id,
        )


def _branch_id_from_location(location: str | Path) -> str:
    text = str(location).rstrip("/")
    if "://" in text:
        return text.rsplit("/", 1)[-1] or "branch"
    return Path(text).name or "branch"


__all__ = ["FileContext", "FileSession", "Workspace"]
