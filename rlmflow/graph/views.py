"""Flat query and mutation views over a recursive graph subtree."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

_MISSING = object()


class Edge(NamedTuple):
    """A derived flow- or spawn-edge between two nodes."""

    from_: str
    to: str
    kind: str  # "flows_to" | "spawns"


class NodesView:
    """``graph.nodes`` — flat view over every node in the subtree."""

    __slots__ = ("_g",)

    def __init__(self, g: Any) -> None:
        self._g = g

    def _iter(self) -> Iterator[Any]:
        for g in self._g.walk():
            yield from g.states

    def __iter__(self) -> Iterator[Any]:
        return self._iter()

    def __len__(self) -> int:
        return sum(1 for _ in self._iter())

    def __contains__(self, node_id: object) -> bool:
        return any(n.id == node_id for n in self._iter())

    def __repr__(self) -> str:
        return f"NodesView({len(self)} nodes)"

    def find(self, node_id: str) -> Any | None:
        return self._g.find(node_id)

    def _locate(self, node_id: str) -> tuple[Any, int]:
        for g in self._g.walk():
            for i, s in enumerate(g.states):
                if s.id == node_id:
                    return g, i
        raise KeyError(node_id)

    def replace(self, node_id: str, new_node: Any) -> Any:
        """Find a node anywhere in the subtree and swap it in place."""
        g, i = self._locate(node_id)
        g.states[i] = new_node
        return new_node

    def update(self, node_id: str, **changes: Any) -> Any:
        """Apply ``changes`` to the node with ``node_id`` anywhere in subtree."""
        g, i = self._locate(node_id)
        g.states[i] = g.states[i].update(**changes)
        return g.states[i]

    def remove(self, node_id: str) -> Any:
        """Drop a node from the subtree by id and return it."""
        g, i = self._locate(node_id)
        return g.states.pop(i)

    def where(
        self,
        predicate: Callable[[Any], bool] | None = None,
        /,
        **filters: Any,
    ) -> list[Any]:
        return _filter(self, predicate, filters)

    def queries(self) -> list[Any]:
        """Bootstrap user queries (``type == "user_query"``)."""
        return self.where(type="user_query")

    def llm_actions(self) -> list[Any]:
        """LLM action records (``type == "llm_action"``)."""
        return self.where(type="llm_action")

    def llm_outputs(self) -> list[Any]:
        """LLM replies (``type == "llm_output"``)."""
        return self.where(type="llm_output")

    def exec_actions(self) -> list[Any]:
        """Code-execution actions (``type == "exec_action"``)."""
        return self.where(type="exec_action")

    def resume_actions(self) -> list[Any]:
        """Resume actions (``type == "resume_action"``)."""
        return self.where(type="resume_action")

    def observations(self) -> list[Any]:
        """:class:`ExecOutput` nodes that were not produced by a resume."""
        return [
            n
            for n in self._iter()
            if n.type == "exec_output" and not getattr(n, "resumed_from", None)
        ]

    def supervising(self) -> list[Any]:
        """Yielded code observations (``type == "supervising_output"``)."""
        return self.where(type="supervising_output")

    def resumes(self) -> list[Any]:
        """Code observations produced by a :class:`ResumeAction`."""
        return [
            n
            for n in self._iter()
            if n.type
            in ("exec_output", "supervising_output", "error_output", "done_output")
            and bool(getattr(n, "resumed_from", None))
        ]

    def results(self) -> list[Any]:
        """Terminal results (``type == "done_output"``)."""
        return self.where(type="done_output")

    def errors(self) -> list[Any]:
        """Error observations (``type == "error_output"``)."""
        return self.where(type="error_output")


class AgentsView(Mapping[str, Any]):
    """``graph.agents`` — Mapping[agent_id, sub-Graph] across the subtree."""

    __slots__ = ("_g",)

    def __init__(self, g: Any) -> None:
        self._g = g

    def __iter__(self) -> Iterator[str]:
        for g in self._g.walk():
            yield g.agent_id

    def __len__(self) -> int:
        return sum(1 for _ in self._g.walk())

    def __getitem__(self, aid: str) -> Any:
        return self._g[aid]

    def __repr__(self) -> str:
        return f"AgentsView({list(self)})"


class EdgesView:
    """``graph.edges`` — derived flow + spawn edges across the subtree."""

    __slots__ = ("_g",)

    def __init__(self, g: Any) -> None:
        self._g = g

    def _build(self) -> list[Edge]:
        out: list[Edge] = []
        for g in self._g.walk():
            for prev, curr in zip(g.states, g.states[1:]):
                out.append(Edge(from_=prev.id, to=curr.id, kind="flows_to"))
            for child in g.children.values():
                if child.parent_node_id and child.states:
                    out.append(
                        Edge(
                            from_=child.parent_node_id,
                            to=child.states[0].id,
                            kind="spawns",
                        )
                    )
        return out

    def __iter__(self) -> Iterator[Edge]:
        return iter(self._build())

    def __len__(self) -> int:
        return len(self._build())

    def __repr__(self) -> str:
        return f"EdgesView({len(self)} edges)"

    def where(
        self,
        predicate: Callable[[Edge], bool] | None = None,
        /,
        **filters: Any,
    ) -> list[Edge]:
        return _filter(self._build(), predicate, filters)

    def spawns(self) -> list[Edge]:
        return [e for e in self._build() if e.kind == "spawns"]

    def flows_to(self) -> list[Edge]:
        return [e for e in self._build() if e.kind == "flows_to"]


def _filter(items, predicate, filters):
    def matches(x):
        if predicate is not None and not predicate(x):
            return False
        return all(getattr(x, k, _MISSING) == v for k, v in filters.items())

    return [x for x in items if matches(x)]


__all__ = ["AgentsView", "Edge", "EdgesView", "NodesView"]
