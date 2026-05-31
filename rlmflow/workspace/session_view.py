"""REPL-facing session inspection helpers."""

from __future__ import annotations

import re
from typing import Any

from rlmflow.graph import Graph, is_done, is_observation
from rlmflow.prompts.projection import project_state_messages
from rlmflow.workspace.base import Session


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
    msgs.extend(project_state_messages(agent.states, skip_empty=True))
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


__all__ = ["SessionVariable"]
