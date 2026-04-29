"""Context store — durable graph persistence and agent-facing context."""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rlmkit.node import Node, parse_node_json

CONTEXT_TOOLS_HINT = """
**Context:**
- `CONTEXT.info()` — metadata about the default context.
- `CONTEXT.read(start=0, end=None)` — read a character slice.
- `CONTEXT.lines(start=0, end=None)` — read line range `start:end`.
- `CONTEXT.line_count()` — count lines.
- `CONTEXT.grep(pattern, max_results=50)` — search the context by regex.
- `CONTEXT.append(text)` — append text to the default context.

Use `CONTEXT` for long task input. Do not print the whole context; inspect
samples, split it into chunks, delegate chunk work, and aggregate structured
results.
"""


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "context"


def _agent_path(agent_id: str) -> Path:
    parts = [p for p in agent_id.split(".") if p and p != "root"]
    return Path(*parts) if parts else Path()


class ContextTools:
    """Lazy handle over workspace context variables."""

    def __init__(
        self,
        context: ContextStore,
        *,
        agent_id: str = "root",
        key: str = "context",
    ) -> None:
        self.context = context
        self.agent_id = agent_id
        self.key = key

    def info(self) -> dict[str, Any]:
        return self.context.context_info(self.key, agent_id=self.agent_id)

    def read(self, start: int = 0, end: int | None = None) -> str:
        return self.context.read_context(self.key, agent_id=self.agent_id)[start:end]

    def lines(self, start: int = 0, end: int | None = None) -> str:
        lines = self.context.read_context(self.key, agent_id=self.agent_id).splitlines(
            True
        )
        return "".join(lines[start:end])

    def line_count(self) -> int:
        return len(
            self.context.read_context(self.key, agent_id=self.agent_id).splitlines()
        )

    def grep(self, pattern: str, *, max_results: int = 50) -> str:
        compiled = re.compile(pattern)
        matches: list[str] = []
        text = self.context.read_context(self.key, agent_id=self.agent_id)
        for idx, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                matches.append(f"{idx}:{line}")
                if len(matches) >= max_results:
                    break
        return "\n".join(matches)

    def append(self, text: str) -> None:
        self.context.append_context(self.key, text, agent_id=self.agent_id)


class ContextStore(ABC):
    """Persist typed graph nodes and render agent-facing context."""

    @abstractmethod
    def write(self, node: Node) -> None:
        """Append or store a typed node."""

    @abstractmethod
    def load(self) -> dict[str, Node]:
        """Load nodes keyed by id."""

    @abstractmethod
    def fork(self, new_location: object) -> ContextStore:
        """Return a deep copy of this context store."""

    def chain_to(self, node: Node) -> list[Node]:
        """Return the single-agent chain ending at ``node``.

        The graph topology is node-owned: parents record successor ids in
        ``children``. This helper derives the parent chain from persisted nodes
        so RLMFlow does not need its own graph index.
        """
        nodes = self.load()
        nodes[node.id] = node

        parent_by_child: dict[str, Node] = {}
        for candidate in nodes.values():
            for child in candidate.children:
                child_id = child.id if isinstance(child, Node) else str(child)
                parent_by_child[child_id] = candidate

        chain = [node]
        current = node
        seen = {node.id}
        while current.id in parent_by_child:
            parent = parent_by_child[current.id]
            if parent.id in seen or parent.agent_id != node.agent_id:
                break
            chain.append(parent)
            seen.add(parent.id)
            current = parent
        return list(reversed(chain))

    def write_context(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("This context backend does not support blobs.")

    def append_context(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
    ) -> None:
        self.write_context(
            key,
            self.read_context(key, agent_id=agent_id) + value,
            agent_id=agent_id,
        )

    def read_context(self, key: str = "context", *, agent_id: str = "root") -> str:
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        return []

    def context_info(
        self,
        key: str = "context",
        *,
        agent_id: str = "root",
    ) -> dict[str, Any]:
        text = self.read_context(key, agent_id=agent_id)
        return {
            "key": key,
            "agent_id": agent_id,
            "chars": len(text),
            "approx_tokens": len(text) // 4,
            "lines": len(text.splitlines()),
        }

    def context_prompt_hint(self, *, agent_id: str | None = None) -> str:
        contexts = self.list_contexts(agent_id=agent_id)
        if not contexts:
            return ""
        context_list = "\n".join(f"- `{key}`" for key in contexts)
        return (
            CONTEXT_TOOLS_HINT.strip() + "\n\nAvailable context keys:\n" + context_list
        )


class FileContext(ContextStore):
    """Filesystem context store under ``workspace/context``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "agents").mkdir(parents=True, exist_ok=True)
        self.nodes_path = self.root / "nodes.jsonl"

    def write(self, node: Node) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.nodes_path.open("a", encoding="utf-8") as f:
            f.write(node.model_dump_json() + "\n")
        self.write_agent_view(node)

    def write_agent_view(self, node: Node) -> None:
        view = {
            "agent_id": node.agent_id,
            "latest_leaf_id": node.id,
            "branch_id": node.branch_id,
            "type": node.type,
            "terminal": node.terminal,
            "result": getattr(node, "result", None),
        }
        path = self.root / "agents" / f"{_safe_name(node.agent_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(view, indent=2), encoding="utf-8")

    def load(self) -> dict[str, Node]:
        nodes: dict[str, Node] = {}
        if not self.nodes_path.exists():
            return nodes
        for line in self.nodes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                node = parse_node_json(line)
                nodes[node.id] = node
        return nodes

    def fork(self, new_location: object) -> ContextStore:
        dst = Path(new_location).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        (
            shutil.copytree(self.root, dst)
            if self.root.exists()
            else dst.mkdir(parents=True)
        )
        return FileContext(dst)

    def _context_paths(self, key: str, *, agent_id: str) -> tuple[Path, Path]:
        base = self.root / _agent_path(agent_id)
        safe = _safe_name(key)
        return base / f"{safe}.txt", base / f"{safe}.json"

    def write_context(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path, meta_path = self._context_paths(key, agent_id=agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        meta = {
            "key": key,
            "agent_id": agent_id,
            "chars": len(value),
            "metadata": metadata or {},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def read_context(self, key: str = "context", *, agent_id: str = "root") -> str:
        for aid in (agent_id, "root"):
            path, _ = self._context_paths(key, agent_id=aid)
            if path.exists():
                return path.read_text(encoding="utf-8")
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        agent_ids = [agent_id] if agent_id else ["root"]
        if agent_id and agent_id != "root":
            agent_ids.append("root")
        keys: set[str] = set()
        for aid in agent_ids:
            if aid is None:
                continue
            base = self.root / _agent_path(aid)
            if base.exists():
                keys.update(p.stem for p in base.glob("*.txt"))
        return sorted(keys)


class InMemoryContext(ContextStore):
    """Process-local context store for runs without a Workspace."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.blobs: dict[tuple[str, str], str] = {}

    def write(self, node: Node) -> None:
        self.nodes[node.id] = node

    def load(self) -> dict[str, Node]:
        return dict(self.nodes)

    def fork(self, new_location: object) -> ContextStore:
        del new_location
        out = InMemoryContext()
        out.nodes = dict(self.nodes)
        out.blobs = dict(self.blobs)
        return out

    def write_context(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del metadata
        self.blobs[(agent_id, key)] = value

    def read_context(self, key: str = "context", *, agent_id: str = "root") -> str:
        for aid in (agent_id, "root"):
            if (aid, key) in self.blobs:
                return self.blobs[(aid, key)]
        raise KeyError(f"context {key!r} not found for {agent_id!r}")

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        agent_ids = [agent_id] if agent_id else ["root"]
        if agent_id and agent_id != "root":
            agent_ids.append("root")
        return sorted({key for aid, key in self.blobs if aid in agent_ids})


__all__ = ["ContextStore", "ContextTools", "FileContext", "InMemoryContext"]
