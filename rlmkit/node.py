"""Typed graph-flow nodes and REPL protocol handles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id() -> str:
    return f"n_{uuid4().hex[:12]}"


class WorkspaceRef(BaseModel):
    """Serializable reference to branch-local workspace storage."""

    root: str
    branch_id: str = "main"

    @property
    def files(self) -> Path:
        return Path(self.root) / "files"

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
    def from_dict(cls, data: dict) -> "ChildHandle":
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
    def from_dict(cls, data: dict) -> "WaitRequest":
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
    terminate_requested: bool = False

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def terminal(self) -> bool:
        return False

    @property
    def finished(self) -> bool:
        return self.terminal

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def update(self, **changes: Any) -> "Node":
        return self.model_copy(update=changes)

    def successor(self, cls: type["Node"], **fields: Any) -> "Node":
        values = {
            "branch_id": self.branch_id,
            "agent_id": self.agent_id,
            "depth": self.depth,
            "query": self.query,
            "system_prompt": self.system_prompt,
            "config": self.config,
            "workspace": self.workspace,
            "runtime": self.runtime,
            "terminate_requested": self.terminate_requested,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        values.update(fields)
        return cls(**values)

    def child_nodes(self) -> list["Node"]:
        return [c for c in self.children if isinstance(c, Node)]

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

    def walk(self) -> list["Node"]:
        nodes = [self]
        for child in self.child_nodes():
            nodes.extend(child.walk())
        return nodes

    def tree(self, *, color: bool = False, indent: str = "") -> str:
        del color
        label = f"{self.agent_id or 'root'} [{self.type}]"
        result = getattr(self, "result", None)
        if result:
            label += f" -> {str(result)[:80]}"
        lines = [indent + label]
        for child in self.child_nodes():
            lines.append(child.tree(indent=indent + "  "))
        return "\n".join(lines)

    def find(self, node_id: str) -> "Node | None":
        if self.id == node_id or self.agent_id == node_id:
            return self
        for child in self.child_nodes():
            found = child.find(node_id)
            if found is not None:
                return found
        return None

    def replace_many(self, updates: dict[str, "Node"]) -> "Node":
        replacement = updates.get(self.id) or updates.get(self.agent_id)
        if replacement is not None:
            return replacement
        new_children = [
            child.replace_many(updates) if isinstance(child, Node) else child
            for child in self.children
        ]
        return self.update(children=new_children)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return parse_node_obj(data)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Node":
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
    children: list[str] = Field(default_factory=list)


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
    model: str | None = None
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


class _NodeParser(BaseModel):
    node: NodeUnion


def parse_node_obj(data: dict) -> Node:
    return _NodeParser.model_validate({"node": data}).node


def parse_node_json(data: str) -> Node:
    return parse_node_obj(json.loads(data))


__all__ = [
    "ActionNode",
    "ChildHandle",
    "ErrorNode",
    "Node",
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
