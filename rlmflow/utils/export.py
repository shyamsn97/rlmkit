"""Static topology exports (Mermaid, DOT, D2) for :class:`Graph` snapshots."""

from __future__ import annotations

from rlmflow.graph import Graph, Node, is_done


def _kind(state: Node) -> str:
    """Display label / palette key for ``state``.

    One key per node ``type``, with a single special case: an
    :class:`ExecOutput` produced by a :class:`ResumeAction` (i.e.
    ``resumed_from`` is non-empty) is bucketed as ``"resume"`` so
    the post-resume continuation can be coloured differently from a
    fresh exec.
    """
    t = state.type
    if t == "exec_output" and getattr(state, "resumed_from", None):
        return "resume"
    return _DISPLAY.get(t, t)


_DISPLAY: dict[str, str] = {
    "user_query": "query",
    # Each ActionNode shares its display kind with the observation it
    # pairs with. When both are present (the normal path) the viewer
    # hides the action; when only the action is present (mid-tick,
    # before the observation is written) the action stands in for
    # that step in the figure.
    "llm_action": "llm",
    "llm_output": "llm",
    "exec_action": "exec",
    "exec_output": "exec",
    "supervising_output": "supervising",
    "error_output": "errored",
    "done_output": "done",
    "resume_action": "resume",
}


def _sanitize(node_id: str) -> str:
    return node_id.replace(".", "_").replace("-", "_") or "root"


def _truncate(text: str, n: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text[: n - 1] + "..." if len(text) > n else text


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def _escape_dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# Display kinds → colour. Actions get a desaturated tint of the
# observation they pair with so the obs/action alternation reads
# at a glance.
_NODE_COLOR: dict[str, str] = {
    "query": "#58a6ff",  # blue
    "llm_call": "#a98a2a",  # dim yellow
    "llm": "#d29922",  # yellow
    "exec_call": "#b87650",  # dim orange
    "exec": "#ff9e64",  # orange
    "supervising": "#bc8cff",  # purple
    "resume_call": "#5fa067",  # dim green
    "resume": "#7ee787",  # green
    "errored": "#f85149",  # red
    "done": "#3fb950",  # bright green
}


_MERMAID_FLOW_CLASS: dict[str, str] = {k: k.replace("_", "") for k in _NODE_COLOR}


def _state_label(state: Node) -> str:
    """Short human-readable state label for diagram nodes."""
    return f"{state.agent_id} ({_kind(state)})"


def _state_result_text(state: Node) -> str | None:
    if is_done(state) and state.result:
        return state.result
    return None


# ── Mermaid state diagram ────────────────────────────────────────────


def to_mermaid(graph: Graph, *, include_results: bool = True) -> str:
    """Render ``graph`` as a Mermaid ``stateDiagram-v2``."""
    declarations: list[str] = []
    transitions: list[str] = []
    roots = _root_nodes(graph)

    for state in graph.nodes:
        nid = _sanitize(state.id)
        declarations.append(
            f'    state "{_escape_mermaid(_state_label(state))}" as {nid}'
        )
    for root in roots:
        transitions.append(f"    [*] --> {_sanitize(root.id)}")
    for edge in graph.edges:
        transitions.append(f"    {_sanitize(edge.from_)} --> {_sanitize(edge.to)}")
    if include_results:
        for state in graph.nodes:
            res = _state_result_text(state)
            if res:
                transitions.append(
                    f"    {_sanitize(state.id)} --> [*] : {_escape_mermaid(_truncate(res))}"
                )

    return "\n".join(["stateDiagram-v2", *declarations, *transitions])


# ── DOT ──────────────────────────────────────────────────────────────


def to_dot(graph: Graph, *, include_results: bool = True) -> str:
    lines = [
        "digraph rlmflow {",
        "    rankdir=TB;",
        '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]
    for state in graph.nodes:
        nid = _sanitize(state.id)
        kind = _kind(state)
        color = _NODE_COLOR.get(kind, "#8b949e")
        parts = [state.agent_id or "root", kind]
        if include_results:
            res = _state_result_text(state)
            if res:
                parts.append(_truncate(res, 40))
        label = "\\n".join(_escape_dot(part) for part in parts)
        lines.append(
            f'    {nid} [label="{label}", fillcolor="{color}22", color="{color}"];'
        )
    for edge in graph.edges:
        style = "solid" if edge.kind == "flows_to" else "dashed"
        lines.append(
            f"    {_sanitize(edge.from_)} -> {_sanitize(edge.to)} "
            f'[label="{edge.kind}", style={style}];'
        )
    lines.append("}")
    return "\n".join(lines)


# ── Mermaid flowchart ────────────────────────────────────────────────


def to_mermaid_flowchart(graph: Graph, *, include_results: bool = True) -> str:
    lines = ["flowchart TD"]
    for state in graph.nodes:
        nid = _sanitize(state.id)
        agent = state.agent_id or "root"
        kind = _kind(state)
        body = f"{agent}<br/><i>{kind}</i>"
        if include_results:
            res = _state_result_text(state)
            if res:
                body += f"<br/>{_escape_mermaid(_truncate(res, 40))}"
        lines.append(f'    {nid}["{body}"]:::{_MERMAID_FLOW_CLASS.get(kind, "obs")}')
    for edge in graph.edges:
        lines.append(
            f"    {_sanitize(edge.from_)} -->|{edge.kind}| {_sanitize(edge.to)}"
        )
    for kind, color in _NODE_COLOR.items():
        cls = _MERMAID_FLOW_CLASS[kind]
        lines.append(f"    classDef {cls} fill:{color}22,stroke:{color},color:#c9d1d9;")
    return "\n".join(lines)


# ── Mermaid sequence diagram ─────────────────────────────────────────


def to_mermaid_sequence(graph: Graph) -> str:
    """Delegate / wait / done flow between agents."""
    lines = ["sequenceDiagram"]
    for aid in graph.agents:
        lines.append(f"    participant {_sanitize(aid)} as {aid}")

    spawns = graph.edges.spawns()
    by_id = {e.id: e for e in graph.nodes}
    for edge in spawns:
        parent = by_id.get(edge.from_)
        child = by_id.get(edge.to)
        if parent is None or child is None:
            continue
        parent_id = _sanitize(parent.agent_id)
        child_id = _sanitize(child.agent_id)
        lines.append(f"    {parent_id}->>+{child_id}: delegate")
        child_sub = graph.agents[child.agent_id]
        cur = child_sub.current()
        if cur is not None and cur.terminal:
            kind = "done"
            res = getattr(cur, "result", None)
            summary = _truncate(res, 30) if res else kind
            lines.append(f"    {child_id}-->>-{parent_id}: {_escape_mermaid(summary)}")
    return "\n".join(lines)


# ── D2 ───────────────────────────────────────────────────────────────


_D2_STYLES = {
    kind: f'{{ style: {{ fill: "{color}22"; stroke: "{color}" }} }}'
    for kind, color in _NODE_COLOR.items()
}


def to_d2(graph: Graph, *, include_results: bool = True) -> str:
    lines: list[str] = []
    for state in graph.nodes:
        nid = _sanitize(state.id)
        agent = state.agent_id or "root"
        kind = _kind(state)
        label = f"{agent}\\n{kind}"
        if include_results:
            res = _state_result_text(state)
            if res:
                label += f"\\n{_truncate(res, 40)}"
        style = _D2_STYLES.get(kind, "")
        lines.append(f'{nid}: "{label}" {style}'.rstrip())
    for edge in graph.edges:
        lines.append(f"{_sanitize(edge.from_)} -> {_sanitize(edge.to)}: {edge.kind}")
    return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────


def _root_nodes(graph: Graph) -> list[Node]:
    """Nodes with no incoming edge — used as ``[*] --> X`` Mermaid roots."""
    targets = {edge.to for edge in graph.edges}
    return [n for n in graph.nodes if n.id not in targets]


__all__ = [
    "to_d2",
    "to_dot",
    "to_mermaid",
    "to_mermaid_flowchart",
    "to_mermaid_sequence",
]
