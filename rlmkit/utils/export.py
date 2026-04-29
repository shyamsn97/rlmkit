"""Static topology exports for typed RLMFlow node trees."""

from __future__ import annotations

from rlmkit.node import Node


def _sanitize(node_id: str) -> str:
    return node_id.replace(".", "_").replace("-", "_") or "root"


def _truncate(text: str, n: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text[: n - 1] + "..." if len(text) > n else text


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def _escape_dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def to_mermaid(state: Node, *, include_results: bool = True) -> str:
    lines = ["stateDiagram-v2"]

    def walk(node: Node, is_root: bool) -> None:
        nid = _sanitize(node.id)
        label = f"{node.agent_id or 'root'} ({node.type})"
        lines.append(f'    state "{_escape_mermaid(label)}" as {nid}')
        if is_root:
            lines.append(f"    [*] --> {nid}")
        for child in node.child_nodes():
            cid = _sanitize(child.id)
            lines.append(f"    {nid} --> {cid}")
            walk(child, is_root=False)
        result = getattr(node, "result", None)
        if include_results and node.terminal and result:
            lines.append(f'    {nid} --> [*] : "{_escape_mermaid(_truncate(result))}"')

    walk(state, True)
    return "\n".join(lines)


_NODE_COLOR = {
    "query": "#58a6ff",
    "observation": "#58a6ff",
    "action": "#d29922",
    "supervising": "#bc8cff",
    "resume": "#7ee787",
    "error": "#f85149",
    "result": "#3fb950",
}


def to_dot(state: Node, *, include_results: bool = True) -> str:
    lines = [
        "digraph rlmkit {",
        "    rankdir=TB;",
        '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]

    def walk(node: Node) -> None:
        nid = _sanitize(node.id)
        color = _NODE_COLOR.get(node.type, "#8b949e")
        parts = [node.agent_id or "root", node.type]
        result = getattr(node, "result", None)
        if include_results and node.terminal and result:
            parts.append(_truncate(result, 40))
        label = "\\n".join(_escape_dot(part) for part in parts)
        lines.append(
            f'    {nid} [label="{label}", fillcolor="{color}22", color="{color}"];'
        )
        for child in node.child_nodes():
            cid = _sanitize(child.id)
            lines.append(f"    {nid} -> {cid};")
            walk(child)

    walk(state)
    lines.append("}")
    return "\n".join(lines)
