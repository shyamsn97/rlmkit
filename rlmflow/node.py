"""Typed graph-flow nodes and REPL protocol handles."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

_MISSING = object()


class NodeDiff(NamedTuple):
    """Difference between two graph snapshots, indexed by node id."""

    added: list[Node]
    removed: list[Node]


def new_id() -> str:
    return f"n_{uuid4().hex[:12]}"


class WorkspaceRef(BaseModel):
    """Serializable reference to branch-local workspace storage."""

    root: str
    branch_id: str = "main"

    @property
    def context_dir(self) -> Path:
        return Path(self.root) / "context"


class RuntimeRef(BaseModel):
    """Serializable reference to a durable runtime/REPL session."""

    id: str


class ChildHandle:
    """Opaque reference returned by delegate(), passed to wait()."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def __repr__(self) -> str:
        return f"ChildHandle({self.agent_id!r})"

    def to_dict(self) -> dict:
        return {"child_handle": self.agent_id}

    @classmethod
    def from_dict(cls, data: dict) -> ChildHandle:
        return cls(data["child_handle"])


class WaitRequest:
    """Yielded by wait() to request suspension until children finish."""

    def __init__(self, agent_ids: list[str]) -> None:
        self.agent_ids = agent_ids

    def __repr__(self) -> str:
        return f"WaitRequest({self.agent_ids!r})"

    def to_dict(self) -> dict:
        return {"wait_request": self.agent_ids}

    @classmethod
    def from_dict(cls, data: dict) -> WaitRequest:
        return cls(data["wait_request"])


class Node(BaseModel):
    """One immutable event in an RLMFlow graph."""

    model_config = ConfigDict(frozen=True)

    type: str
    id: str = Field(default_factory=new_id)
    branch_id: str = "main"
    children: list[Any] = Field(default_factory=list)

    agent_id: str = "root"
    depth: int = 0
    query: str = ""
    system_prompt: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    workspace: WorkspaceRef | None = None
    runtime: RuntimeRef | None = None
    model: str | None = None
    terminate_requested: bool = False

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def terminal(self) -> bool:
        return False

    @property
    def finished(self) -> bool:
        return self.current().terminal

    def get_result(self) -> str:
        """Result string from this graph's terminal leaf, or ``""`` if not
        finished. On a ``ResultNode`` it returns its own ``result`` field;
        on every other node it walks ``current()`` to the terminal leaf."""
        leaf = self.current()
        return getattr(leaf, "result", "") or ""

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def model_key(self) -> str:
        return str(self.config.get("model") or "default")

    @property
    def model_label(self) -> str:
        actual = getattr(self, "model", None)
        if actual and actual != self.model_key:
            return f"{self.model_key}:{actual}"
        return self.model_key

    def update(self, **changes: Any) -> Node:
        return self.model_copy(update=changes)

    def successor(self, cls: type[Node], **fields: Any) -> Node:
        values = {
            "branch_id": self.branch_id,
            "agent_id": self.agent_id,
            "depth": self.depth,
            "query": self.query,
            "system_prompt": self.system_prompt,
            "config": self.config,
            "workspace": self.workspace,
            "runtime": self.runtime,
            "model": self.model,
            "terminate_requested": self.terminate_requested,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        values.update(fields)
        return cls(**values)

    def child_nodes(self) -> list[Node]:
        return [c for c in self.children if isinstance(c, Node)]

    def current(self) -> Node:
        """Latest same-agent node reachable from this graph root."""
        for child in reversed(self.child_nodes()):
            if child.agent_id == self.agent_id:
                return child.current()
        return self

    def tree_usage(self) -> tuple[int, int]:
        inp = self.total_input_tokens
        out = self.total_output_tokens
        for child in self.child_nodes():
            ci, co = child.tree_usage()
            inp += ci
            out += co
        return inp, out

    @property
    def tree_tokens(self) -> int:
        inp, out = self.tree_usage()
        return inp + out

    def walk(self) -> list[Node]:
        nodes = [self]
        for child in self.child_nodes():
            nodes.extend(child.walk())
        return nodes

    def tree(self, *, color: bool = False, indent: str = "") -> str:
        del color
        return self._snapshot_tree(indent=indent)

    def _snapshot_tree(self, *, indent: str = "") -> str:
        label = f"{self.agent_id or 'root'} [{self.type}] {{{self.model_label}}}"
        result = getattr(self, "result", None)
        if result:
            label += f" -> {str(result)[:80]}"
        lines = [indent + label]
        for child in self.child_nodes():
            lines.append(child._snapshot_tree(indent=indent + "  "))
        return "\n".join(lines)

    def transcript(
        self,
        agent_id: str | None = None,
        *,
        session: Any | None = None,
        include_system: bool = True,
    ) -> str:
        """Render one agent's transcript from this graph snapshot.

        Defaults to the current node's agent. Pass ``agent_id`` to inspect a
        child, or ``session=...`` to reconstruct from a persisted session log.
        """
        from rlmflow.utils.viewer import node_transcript

        return node_transcript(
            self,
            agent_id=agent_id,
            session=session,
            include_system=include_system,
        )

    def session(self, *, include_system: bool = False) -> str:
        """Flat message log of every agent in this subtree, in graph order.

        Walks the subtree depth-first and renders each node as a labeled
        message (``--- [agent] kind ---``). Cross-agent flow shows up
        inline so you can read the run as one chat log. Pass
        ``include_system=True`` to prepend each agent's system prompt the
        first time it appears.
        """
        from rlmflow.utils.viewer import node_session

        return node_session(self, include_system=include_system)

    def plot(
        self,
        kind: str = "graph",
        *,
        states: list[Node] | None = None,
        events: list[Node] | None = None,
        step: int | None = None,
        mode: str = "snapshot",
        height: int = 420,
        title: str | None = None,
        session: Any | None = None,
        include_results: bool = True,
    ) -> Any:
        """Render this node as graph, Mermaid, Gantt, tree, DOT, or D2.

        Examples:
            ``node.plot()`` returns the Plotly graph used by the viewer.
            ``node.plot("mermaid")`` returns a Mermaid state diagram.
            ``node.plot("flowchart")`` returns a Mermaid flowchart.
            ``node.plot("gantt", states=history)`` returns an HTML swimlane.
        """
        from rlmflow.utils.viewer import node_plot

        return node_plot(
            self,
            kind,
            states=states,
            events=events,
            step=step,
            mode=mode,
            height=height,
            title=title,
            session=session,
            include_results=include_results,
        )

    def plot_html(
        self,
        kind: str = "graph",
        *,
        states: list[Node] | None = None,
        events: list[Node] | None = None,
        step: int | None = None,
        mode: str = "snapshot",
        height: int = 420,
        title: str | None = None,
        session: Any | None = None,
        include_results: bool = True,
        include_plotlyjs: str | bool = "cdn",
    ) -> str:
        """Return an HTML fragment for ``self.plot(kind)``."""
        from rlmflow.utils.viewer import node_plot_html

        return node_plot_html(
            self,
            kind,
            states=states,
            events=events,
            step=step,
            mode=mode,
            height=height,
            title=title,
            session=session,
            include_results=include_results,
            include_plotlyjs=include_plotlyjs,
        )

    def find(self, node_id: str) -> Node | None:
        if self.id == node_id or self.agent_id == node_id:
            return self
        for child in self.child_nodes():
            found = child.find(node_id)
            if found is not None:
                return found
        return None

    def leaves(self) -> list[Node]:
        """Every node in the subtree with no materialized children."""
        children = self.child_nodes()
        if not children:
            return [self]
        out: list[Node] = []
        for child in children:
            out.extend(child.leaves())
        return out

    def errors(self) -> list[Node]:
        """Every `ErrorNode` in the subtree."""
        return [n for n in self.walk() if n.type == "error"]

    def results(self) -> list[Node]:
        """Every `ResultNode` in the subtree."""
        return [n for n in self.walk() if n.type == "result"]

    def where(
        self,
        predicate: Callable[[Node], bool] | None = None,
        /,
        **filters: Any,
    ) -> list[Node]:
        """Subtree search by predicate, attribute kwargs, or both.

        ``state.where(type="error")`` returns every error node.
        ``state.where(lambda n: n.depth > 2)`` returns deep nodes.
        Kwargs match attributes by equality; missing attributes never match.
        """

        def matches(node: Node) -> bool:
            if predicate is not None and not predicate(node):
                return False
            for key, expected in filters.items():
                if getattr(node, key, _MISSING) != expected:
                    return False
            return True

        return [n for n in self.walk() if matches(n)]

    def path_to(self, node_id: str) -> list[Node]:
        """Ancestor chain from this node to ``node_id`` (inclusive). Empty if not found."""
        if self.id == node_id or self.agent_id == node_id:
            return [self]
        for child in self.child_nodes():
            path = child.path_to(node_id)
            if path:
                return [self, *path]
        return []

    def diff(self, other: Node) -> NodeDiff:
        """Compare two snapshots by node id. Returns added/removed nodes.

        ``other`` is the *prior* snapshot. ``added`` is what exists in
        ``self`` but not ``other``; ``removed`` is the inverse. Nodes are
        compared by id, so node-content changes are not part of the diff
        (every step produces a new node id, so they show up as additions).
        """
        self_by_id = {n.id: n for n in self.walk()}
        other_by_id = {n.id: n for n in other.walk()}
        added = [n for nid, n in self_by_id.items() if nid not in other_by_id]
        removed = [n for nid, n in other_by_id.items() if nid not in self_by_id]
        return NodeDiff(added=added, removed=removed)

    def replace_many(self, updates: dict[str, Node]) -> Node:
        replacement = updates.get(self.id) or updates.get(self.agent_id)
        if replacement is not None:
            return replacement
        new_children = [
            child.replace_many(updates) if isinstance(child, Node) else child
            for child in self.children
        ]
        return self.update(children=new_children)

    def _repr_html_(self) -> str:
        """Inline HTML render for Jupyter notebooks."""
        from html import escape

        return (
            '<pre style="font-family: ui-monospace, SFMono-Regular, monospace; '
            "background: #0d1117; color: #c9d1d9; padding: 12px; "
            'border-radius: 6px; overflow-x: auto;">'
            f"{escape(self.tree())}"
            "</pre>"
        )

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        """Mime bundle: HTML + Mermaid + plain text for rich Jupyter rendering."""
        from rlmflow.utils.export import to_mermaid

        return {
            "text/plain": self.tree(),
            "text/html": self._repr_html_(),
            "text/x-mermaid": to_mermaid(self),
        }

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> Node:
        return parse_node_obj(data)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Node:
        return parse_node_json(Path(path).read_text(encoding="utf-8"))


class ObservationNode(Node):
    type: Literal["observation"] = "observation"
    content: str = ""
    code: str | None = None
    output: str | None = None


class QueryNode(ObservationNode):
    type: Literal["query"] = "query"


class ErrorNode(ObservationNode):
    type: Literal["error"] = "error"
    error: str = ""


class ResumeNode(ObservationNode):
    type: Literal["resume"] = "resume"
    resumed_from: list[str] = Field(default_factory=list)


class ResultNode(ObservationNode):
    type: Literal["result"] = "result"
    result: str = ""

    @property
    def terminal(self) -> bool:
        return True


class ActionNode(Node):
    type: Literal["action"] = "action"
    reply: str = ""
    code: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class SupervisingNode(ActionNode):
    type: Literal["supervising"] = "supervising"
    output: str = ""
    waiting_on: list[str] = Field(default_factory=list)
    children: list[Any] = Field(default_factory=list)


NodeUnion = Annotated[
    Union[
        QueryNode,
        ErrorNode,
        ResumeNode,
        ResultNode,
        SupervisingNode,
        ActionNode,
        ObservationNode,
    ],
    Field(discriminator="type"),
]


NODE_ADAPTER: TypeAdapter[Node] = TypeAdapter(NodeUnion)


def parse_node_obj(data: dict) -> Node:
    if isinstance(data.get("children"), list):
        data = {
            **data,
            "children": [
                (
                    parse_node_obj(child)
                    if isinstance(child, dict) and "type" in child
                    else child
                )
                for child in data["children"]
            ],
        }
    return NODE_ADAPTER.validate_python(data)


def parse_node_json(data: str) -> Node:
    return parse_node_obj(json.loads(data))


__all__ = [
    "ActionNode",
    "ChildHandle",
    "ErrorNode",
    "NODE_ADAPTER",
    "Node",
    "NodeDiff",
    "ObservationNode",
    "QueryNode",
    "ResultNode",
    "RuntimeRef",
    "ResumeNode",
    "SupervisingNode",
    "WaitRequest",
    "WorkspaceRef",
    "new_id",
    "parse_node_json",
    "parse_node_obj",
]
