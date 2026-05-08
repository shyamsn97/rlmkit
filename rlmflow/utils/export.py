"""Static topology exports for typed RLMFlow node trees."""

from __future__ import annotations

from rlmflow.node import Node
from rlmflow.utils.viz import VizGraph


def _sanitize(node_id: str) -> str:
    return node_id.replace(".", "_").replace("-", "_") or "root"


def _truncate(text: str, n: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text[: n - 1] + "..." if len(text) > n else text


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def _escape_dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


_MERMAID_FLOW_CLASS = {
    "query": "query",
    "observation": "obs",
    "action": "action",
    "supervising": "sup",
    "resume": "resume",
    "error": "err",
    "result": "result",
}


def to_mermaid(state: Node, *, include_results: bool = True) -> str:
    """Render the node tree as a Mermaid ``stateDiagram-v2``.

    All ``state ... as <id>`` declarations are emitted before any
    transitions, and transition labels are unquoted plain text. Both
    are required to render reliably on GitHub's mermaid version, which
    crashes on forward-referenced states and quoted transition labels
    with a "Cannot read properties of undefined (reading 'shape')"
    error.
    """
    declarations: list[str] = []
    transitions: list[str] = []

    def walk(node: Node, is_root: bool) -> None:
        nid = _sanitize(node.id)
        label = f"{node.agent_id or 'root'} ({node.type})"
        declarations.append(f'    state "{_escape_mermaid(label)}" as {nid}')
        if is_root:
            transitions.append(f"    [*] --> {nid}")
        for child in node.child_nodes():
            cid = _sanitize(child.id)
            walk(child, is_root=False)
            transitions.append(f"    {nid} --> {cid}")
        result = getattr(node, "result", None)
        if include_results and node.terminal and result:
            transitions.append(
                f"    {nid} --> [*] : {_escape_mermaid(_truncate(result))}"
            )

    walk(state, True)
    return "\n".join(["stateDiagram-v2", *declarations, *transitions])


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
        "digraph rlmflow {",
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


def to_mermaid_flowchart(state: Node, *, include_results: bool = True) -> str:
    """Render the node tree as a Mermaid ``flowchart TD`` graph.

    Reads better than ``stateDiagram-v2`` for wide trees with many siblings.
    """
    lines = ["flowchart TD"]

    def walk(node: Node) -> None:
        nid = _sanitize(node.id)
        agent = node.agent_id or "root"
        body = f"{agent}<br/><i>{node.type}</i>"
        result = getattr(node, "result", None)
        if include_results and node.terminal and result:
            body += f"<br/>{_escape_mermaid(_truncate(result, 40))}"
        lines.append(
            f'    {nid}["{body}"]:::{_MERMAID_FLOW_CLASS.get(node.type, "obs")}'
        )
        for child in node.child_nodes():
            cid = _sanitize(child.id)
            lines.append(f"    {nid} --> {cid}")
            walk(child)

    walk(state)

    lines.extend(
        [
            "    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;",
            "    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;",
            "    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;",
            "    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;",
            "    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;",
            "    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;",
            "    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;",
        ]
    )
    return "\n".join(lines)


def to_mermaid_sequence(state: Node) -> str:
    """Render delegate / wait / done flow between agents as a Mermaid ``sequenceDiagram``.

    One participant per agent. Edges encode parent → child delegation, child
    completion (``done()``), and parent resumes (``yield wait``).
    """
    agents: list[str] = []
    seen: set[str] = set()

    def collect_agents(node: Node) -> None:
        aid = node.agent_id or "root"
        if aid not in seen:
            seen.add(aid)
            agents.append(aid)
        for child in node.child_nodes():
            collect_agents(child)

    collect_agents(state)

    lines = ["sequenceDiagram"]
    for aid in agents:
        lines.append(f"    participant {_sanitize(aid)} as {aid}")

    def walk(node: Node) -> None:
        parent_id = _sanitize(node.agent_id or "root")
        for child in node.child_nodes():
            child_id = _sanitize(child.agent_id or "root")
            if child.agent_id != node.agent_id:
                lines.append(f"    {parent_id}->>+{child_id}: delegate")
            walk(child)
            current = child.current()
            if child.agent_id != node.agent_id and current.terminal:
                kind = "done" if current.type == "result" else current.type
                result = getattr(current, "result", None)
                summary = _truncate(result, 30) if result else kind
                lines.append(
                    f"    {child_id}-->>-{parent_id}: {_escape_mermaid(summary)}"
                )

    walk(state)
    return "\n".join(lines)


def to_d2(state: Node, *, include_results: bool = True) -> str:
    """Render the node tree as a `D2 <https://d2lang.com>`_ diagram."""
    _D2_STYLES = {
        "query": '{ style: { fill: "#1f6feb22"; stroke: "#58a6ff" } }',
        "observation": '{ style: { fill: "#1f6feb22"; stroke: "#58a6ff" } }',
        "action": '{ style: { fill: "#d2992222"; stroke: "#d29922" } }',
        "supervising": '{ style: { fill: "#bc8cff22"; stroke: "#bc8cff" } }',
        "resume": '{ style: { fill: "#7ee78722"; stroke: "#7ee787" } }',
        "error": '{ style: { fill: "#f8514922"; stroke: "#f85149" } }',
        "result": '{ style: { fill: "#3fb95022"; stroke: "#3fb950" } }',
    }
    lines: list[str] = []

    def walk(node: Node) -> None:
        nid = _sanitize(node.id)
        agent = node.agent_id or "root"
        label = f"{agent}\\n{node.type}"
        result = getattr(node, "result", None)
        if include_results and node.terminal and result:
            label += f"\\n{_truncate(result, 40)}"
        style = _D2_STYLES.get(node.type, "")
        lines.append(f'{nid}: "{label}" {style}'.rstrip())
        for child in node.child_nodes():
            cid = _sanitize(child.id)
            lines.append(f"{nid} -> {cid}")
            walk(child)

    walk(state)
    return "\n".join(lines)


def viz_graph_to_mermaid_flowchart(
    graph: VizGraph, *, include_results: bool = True
) -> str:
    lines = ["flowchart TD"]
    for item in graph.nodes:
        node = item.payload
        nid = _sanitize(node.id)
        body = f"{item.label}<br/><i>{node.type}</i>"
        result = getattr(node, "result", None)
        if include_results and node.terminal and result:
            body += f"<br/>{_escape_mermaid(_truncate(result, 40))}"
        current = "<br/><b>current</b>" if item.current else ""
        lines.append(
            f'    {nid}["{body}{current}"]:::{_MERMAID_FLOW_CLASS.get(node.type, "obs")}'
        )
    for edge in graph.edges:
        lines.append(
            f"    {_sanitize(edge.source)} -->|{edge.kind}| {_sanitize(edge.target)}"
        )
    lines.extend(
        [
            "    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;",
            "    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;",
            "    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;",
            "    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;",
            "    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;",
            "    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;",
            "    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;",
        ]
    )
    return "\n".join(lines)


def viz_graph_to_dot(graph: VizGraph, *, include_results: bool = True) -> str:
    lines = [
        "digraph rlmflow {",
        "    rankdir=LR;",
        '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]
    for item in graph.nodes:
        node = item.payload
        color = _NODE_COLOR.get(node.type, "#8b949e")
        parts = [item.label, node.agent_id or "root", node.type]
        result = getattr(node, "result", None)
        if include_results and node.terminal and result:
            parts.append(_truncate(result, 40))
        label = "\\n".join(_escape_dot(part) for part in parts)
        penwidth = "2.5" if item.current else "1.0"
        lines.append(
            f'    {_sanitize(node.id)} [label="{label}", fillcolor="{color}22", color="{color}", penwidth={penwidth}];'
        )
    for edge in graph.edges:
        style = "solid" if edge.kind == "next" else "dashed"
        lines.append(
            f'    {_sanitize(edge.source)} -> {_sanitize(edge.target)} [label="{edge.kind}", style={style}];'
        )
    lines.append("}")
    return "\n".join(lines)


def viz_graph_to_d2(graph: VizGraph, *, include_results: bool = True) -> str:
    styles = {
        "query": '{ style: { fill: "#1f6feb22"; stroke: "#58a6ff" } }',
        "observation": '{ style: { fill: "#1f6feb22"; stroke: "#58a6ff" } }',
        "action": '{ style: { fill: "#d2992222"; stroke: "#d29922" } }',
        "supervising": '{ style: { fill: "#bc8cff22"; stroke: "#bc8cff" } }',
        "resume": '{ style: { fill: "#7ee78722"; stroke: "#7ee787" } }',
        "error": '{ style: { fill: "#f8514922"; stroke: "#f85149" } }',
        "result": '{ style: { fill: "#3fb95022"; stroke: "#3fb950" } }',
    }
    lines: list[str] = []
    for item in graph.nodes:
        node = item.payload
        label = f"{item.label}\\n{node.agent_id}\\n{node.type}"
        result = getattr(node, "result", None)
        if include_results and node.terminal and result:
            label += f"\\n{_truncate(result, 40)}"
        lines.append(
            f'{_sanitize(node.id)}: "{label}" {styles.get(node.type, "")}'.rstrip()
        )
    for edge in graph.edges:
        lines.append(
            f"{_sanitize(edge.source)} -> {_sanitize(edge.target)}: {edge.kind}"
        )
    return "\n".join(lines)
