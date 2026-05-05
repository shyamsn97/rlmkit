"""Session store — durable node/message history for an RLMFlow run."""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rlmflow.node import Node, parse_node_json

# Terminal-first ordering. Used when an agent has multiple persisted nodes
# and we want the most-evolved one as the chain endpoint.
_NODE_TERMINALITY: dict[str, int] = {
    "result": 0,
    "error": 1,
    "supervising": 2,
    "observation": 3,
    "resume": 4,
    "action": 5,
    "query": 6,
}


SESSION_VARIABLE_PROMPT = """
**Session variable:** read-only view of every *other* agent in this recursive tree.

- `SESSION.agent_id` / `SESSION.node_id` — your own ids.
- `SESSION.list_agents()` → `[{agent_id, type, depth, terminal, result_preview}, ...]`.
- `SESSION.read(agent_id)` — rendered transcript (system + query + actions + observations + result).
- `SESSION.grep(pattern, max_results=50)` — regex across every other agent's messages; `agent_id:type:line` rows.
- `SESSION.parent(agent_id=None)` / `SESSION.ancestors(agent_id=None)` / `SESSION.children(agent_id=None)` / `SESSION.subtree(agent_id=None)` — tree nav.
- `SESSION.tree()` — ASCII tree of the whole run.

Use it to coordinate with siblings/parent: grep before redoing work, read a failed
sibling's transcript before retrying, check `tree()` before delegating.
"""


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

        # Multiple candidates can list the same child id — e.g. a supervising
        # parent agent's node and the sub-agent's own predecessor both claim
        # the result node. Prefer same-agent parents so we walk inside the
        # agent's chain rather than jumping out of it on the first lookup.
        parents_by_child: dict[str, list[Node]] = {}
        for candidate in nodes.values():
            for child in candidate.children:
                child_id = child.id if isinstance(child, Node) else str(child)
                parents_by_child.setdefault(child_id, []).append(candidate)

        chain = [node]
        current = node
        seen = {node.id}
        while True:
            candidates = parents_by_child.get(current.id, [])
            parent = next(
                (c for c in candidates if c.agent_id == node.agent_id),
                None,
            )
            if parent is None or parent.id in seen:
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


class SessionVariable:
    """REPL handle giving an agent read-only access to every other agent's
    session in the same recursive tree.

    The cross-agent analog of :class:`ContextVariable`. Methods mirror ypi's
    ``rlm_sessions`` (list / read / grep) and operate over the same
    :class:`Session` the engine is already writing to.
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

    # ── sibling discovery ─────────────────────────────────────────────

    def _agents(self, *, include_self: bool = False) -> dict[str, list[Node]]:
        groups: dict[str, list[Node]] = {}
        for n in self.store.load().values():
            if not include_self and n.agent_id == self.agent_id:
                continue
            groups.setdefault(n.agent_id, []).append(n)
        return groups

    @staticmethod
    def _terminal_node(nodes: list[Node]) -> Node:
        return min(nodes, key=lambda n: _NODE_TERMINALITY.get(n.type, 99))

    def list_agents(self) -> list[dict[str, Any]]:
        """Summarize every other agent in the session."""
        return self._summarize(self._agents())

    # ── transcript ────────────────────────────────────────────────────

    def read(self, agent_id: str) -> str:
        """Render the named agent's chain as a clean transcript.

        Includes the system prompt, original query, each assistant turn
        (with the ```repl``` block intact), each observation, and the
        terminal result / error.
        """
        nodes = self._agents(include_self=True).get(agent_id, [])
        if not nodes:
            return f"(no nodes for agent {agent_id!r})"
        chain = self.store.chain_to(self._terminal_node(nodes))

        parts: list[str] = []
        if chain and chain[0].system_prompt:
            parts.append(f"--- system ---\n{chain[0].system_prompt.strip()}")
        for node in chain:
            if node.type == "query":
                parts.append(f"--- query ---\n{(node.query or '').strip()}")
            elif node.type == "action":
                parts.append(
                    f"--- assistant ---\n{(getattr(node, 'reply', '') or '').strip()}"
                )
            elif node.type == "observation":
                parts.append(
                    f"--- observation ---\n{(getattr(node, 'content', '') or '').strip()}"
                )
            elif node.type == "supervising":
                wait_on = ", ".join(getattr(node, "waiting_on", []) or [])
                parts.append(f"--- supervising ---\nwaiting on: {wait_on}")
            elif node.type == "resume":
                parts.append(
                    f"--- resume ---\n{(getattr(node, 'content', '') or '').strip()}"
                )
            elif node.type == "error":
                parts.append(
                    f"--- error ({getattr(node, 'error', '')}) ---\n"
                    f"{(getattr(node, 'content', '') or '').strip()}"
                )
            elif node.type == "result":
                parts.append(
                    f"--- result ---\n{(getattr(node, 'result', '') or '').strip()}"
                )
        return "\n\n".join(parts)

    # ── search ────────────────────────────────────────────────────────

    _SEARCH_FIELDS: tuple[str, ...] = (
        "query",
        "reply",
        "content",
        "output",
        "result",
        "error",
    )

    def grep(self, pattern: str, *, max_results: int = 50) -> str:
        """Regex-search across every message field of every other agent."""
        compiled = re.compile(pattern, re.IGNORECASE)
        matches: list[str] = []
        for agent_id, nodes in self._agents().items():
            for n in nodes:
                for field in self._SEARCH_FIELDS:
                    text = getattr(n, field, "") or ""
                    if not text:
                        continue
                    for line in text.splitlines():
                        if compiled.search(line):
                            matches.append(f"{agent_id}:{n.type}:{line.strip()[:160]}")
                            if len(matches) >= max_results:
                                return "\n".join(matches)
        return "\n".join(matches)

    # ── recursive-tree navigation ─────────────────────────────────────
    #
    # Parent-of-agent is derived from the actual cross-agent edges that
    # already live in the session: when a node in agent A has a child
    # node in agent B (e.g. A's ``SupervisingNode`` listing B's
    # ``QueryNode`` as a child), B's parent agent is A. We don't parse
    # dots in agent ids — that breaks the moment an agent name contains
    # a dot (``root.script.js`` is a single child of ``root``, not nested).

    def _agent_parents(self) -> dict[str, str | None]:
        """Map ``agent_id -> parent agent_id`` (``None`` for roots)."""
        all_nodes = list(self.store.load().values())
        by_id = {n.id: n for n in all_nodes}
        parents: dict[str, str | None] = {}
        for n in all_nodes:
            for child in n.children:
                child_id = child.id if isinstance(child, Node) else str(child)
                child_node = by_id.get(child_id)
                if child_node and child_node.agent_id != n.agent_id:
                    parents.setdefault(child_node.agent_id, n.agent_id)
        for n in all_nodes:
            parents.setdefault(n.agent_id, None)
        return parents

    def parent(self, agent_id: str | None = None) -> str | None:
        """Parent ``agent_id`` of ``agent_id`` (defaults to self). ``None`` at root."""
        return self._agent_parents().get(agent_id or self.agent_id)

    def ancestors(self, agent_id: str | None = None) -> list[str]:
        """Agent ids walking root-ward, root first, excluding ``agent_id`` itself."""
        parents = self._agent_parents()
        out: list[str] = []
        seen: set[str] = set()
        cur = parents.get(agent_id or self.agent_id)
        while cur and cur not in seen:
            out.append(cur)
            seen.add(cur)
            cur = parents.get(cur)
        return list(reversed(out))

    def children(self, agent_id: str | None = None) -> list[str]:
        """Direct child agent ids of ``agent_id`` (defaults to self)."""
        aid = agent_id or self.agent_id
        return sorted(
            child
            for child, parent_id in self._agent_parents().items()
            if parent_id == aid
        )

    def subtree(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        """Every descendant agent under ``agent_id``, summarized like ``list_agents``."""
        aid = agent_id or self.agent_id
        parents = self._agent_parents()
        descendants: set[str] = set()
        frontier = [aid]
        while frontier:
            current = frontier.pop()
            for child, parent_id in parents.items():
                if parent_id == current and child not in descendants:
                    descendants.add(child)
                    frontier.append(child)
        groups = self._agents(include_self=True)
        return self._summarize({a: groups[a] for a in descendants if a in groups})

    def tree(self) -> str:
        """ASCII rendering of the full recursive agent tree.

        Each line is ``agent_id [type] result_preview``. Depth-first,
        alphabetical within siblings.
        """
        groups = self._agents(include_self=True)
        if not groups:
            return ""

        parents = self._agent_parents()
        labels: dict[str, str] = {}
        kids: dict[str, list[str]] = {aid: [] for aid in groups}
        roots: list[str] = []

        for aid, nodes in groups.items():
            tip = self._terminal_node(nodes)
            preview = (getattr(tip, "result", None) or "").strip().split("\n", 1)[0]
            label = f"{aid} [{tip.type}]"
            if preview:
                label += f" {preview[:60]}"
            labels[aid] = label
            parent_id = parents.get(aid)
            if parent_id and parent_id in groups:
                kids[parent_id].append(aid)
            else:
                roots.append(aid)

        for siblings in kids.values():
            siblings.sort()
        roots.sort()

        lines: list[str] = []

        def walk(aid: str, indent: str) -> None:
            for i, child in enumerate(kids.get(aid, [])):
                last = i == len(kids[aid]) - 1
                connector = "└── " if last else "├── "
                lines.append(f"{indent}{connector}{labels[child]}")
                walk(child, indent + ("    " if last else "│   "))

        for root in roots:
            lines.append(labels[root])
            walk(root, "")
        return "\n".join(lines)

    # ── shared summarization (used by list_agents and subtree) ────────

    @staticmethod
    def _summarize(groups: dict[str, list[Node]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for agent_id, nodes in groups.items():
            tip = SessionVariable._terminal_node(nodes)
            preview = (getattr(tip, "result", None) or "").strip().replace("\n", " ")
            out.append(
                {
                    "agent_id": agent_id,
                    "type": tip.type,
                    "depth": tip.depth or 0,
                    "terminal": bool(tip.terminal),
                    "result_preview": preview[:120]
                    + ("…" if len(preview) > 120 else ""),
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
