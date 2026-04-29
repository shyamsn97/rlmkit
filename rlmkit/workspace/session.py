"""Session store — durable node/message history for an RLMFlow run."""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from rlmkit.node import Node, parse_node_json


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "root"


class Session(ABC):
    """Persist typed graph nodes and reconstruct per-agent message chains."""

    @abstractmethod
    def write(self, node: Node) -> None:
        """Append or store a typed node."""

    @abstractmethod
    def load(self) -> dict[str, Node]:
        """Load latest nodes keyed by id."""

    @abstractmethod
    def fork(self, new_location: object) -> "Session":
        """Return a deep copy of this session."""

    def chain_to(self, node: Node) -> list[Node]:
        """Return the single-agent chain ending at ``node``."""
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


class FileSession(Session):
    """Filesystem session store under ``workspace/session``."""

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
            "depth": node.depth,
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

    def fork(self, new_location: object) -> Session:
        dst = Path(new_location).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        if self.root.exists():
            shutil.copytree(self.root, dst)
        else:
            dst.mkdir(parents=True)
        return FileSession(dst)


class InMemorySession(Session):
    """Process-local session for runs without a Workspace."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}

    def write(self, node: Node) -> None:
        self.nodes[node.id] = node

    def load(self) -> dict[str, Node]:
        return dict(self.nodes)

    def fork(self, new_location: object) -> Session:
        del new_location
        out = InMemorySession()
        out.nodes = dict(self.nodes)
        return out


__all__ = ["FileSession", "InMemorySession", "Session"]
