"""Static topology exports of an ``RLMState`` tree.

Turns the delegation tree into formats other tools already render:

* :func:`to_mermaid` — Mermaid ``stateDiagram-v2`` (GitHub, Notion, docs).
* :func:`to_dot`     — Graphviz DOT (``dot -Tsvg run.dot -o run.svg``).

Both are pure strings with no optional dependencies.

Usage::

    from rlmkit.utils.export import to_mermaid, to_dot

    print(to_mermaid(final_state))
    Path("run.dot").write_text(to_dot(final_state))
"""

from __future__ import annotations

from rlmkit.state import RLMState, Status


def _sanitize(agent_id: str) -> str:
    """Mermaid / DOT IDs can't contain dots. Replace with underscores."""
    return agent_id.replace(".", "_") or "root"


def _truncate(s: str, n: int = 60) -> str:
    s = s.replace("\n", " ").strip()
    return s[: n - 1] + "…" if len(s) > n else s


def _escape_mermaid(s: str) -> str:
    return s.replace('"', "'").replace("\n", " ")


def _escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ── Mermaid ──────────────────────────────────────────────────────────


def to_mermaid(state: RLMState, *, include_results: bool = True) -> str:
    """Render a state tree as a Mermaid ``stateDiagram-v2`` string.

    Each agent becomes a state; each parent→child edge is labeled ``delegate``;
    each finished agent also emits a ``--> [*]`` edge carrying the truncated
    result text.
    """
    lines: list[str] = ["stateDiagram-v2"]

    def walk(node: RLMState, is_root: bool) -> None:
        nid = _sanitize(node.agent_id)
        status = node.status.value
        label = f"{node.agent_id or 'root'} ({status}, iter {node.iteration})"
        lines.append(f'    state "{_escape_mermaid(label)}" as {nid}')

        if is_root:
            lines.append(f"    [*] --> {nid}")

        for child in node.children:
            cid = _sanitize(child.agent_id)
            lines.append(f"    {nid} --> {cid} : delegate")
            walk(child, is_root=False)

        if include_results and node.status == Status.FINISHED and node.result:
            result = _escape_mermaid(_truncate(node.result, 60))
            lines.append(f'    {nid} --> [*] : "{result}"')

    walk(state, is_root=True)
    return "\n".join(lines)


# ── Graphviz DOT ─────────────────────────────────────────────────────


_STATUS_COLOR = {
    "ready": "#58a6ff",
    "executing": "#d29922",
    "supervising": "#bc8cff",
    "finished": "#3fb950",
}


def to_dot(state: RLMState, *, include_results: bool = True) -> str:
    """Render a state tree as Graphviz DOT.

    Nodes are colored by ``Status``. Edges are labeled ``delegate``. Finished
    agents get their result text appended to the node label.
    """
    lines: list[str] = [
        "digraph rlmkit {",
        "    rankdir=TB;",
        '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]

    def walk(node: RLMState) -> None:
        nid = _sanitize(node.agent_id)
        status = node.status.value
        color = _STATUS_COLOR.get(status, "#8b949e")
        parts = [node.agent_id or "root", f"{status} · iter {node.iteration}"]
        if include_results and node.status == Status.FINISHED and node.result:
            parts.append(_truncate(node.result, 40))
        label = "\\n".join(_escape_dot(p) for p in parts)
        lines.append(
            f'    {nid} [label="{label}", fillcolor="{color}22", color="{color}"];'
        )
        for child in node.children:
            cid = _sanitize(child.agent_id)
            lines.append(f'    {nid} -> {cid} [label="delegate"];')
            walk(child)

    walk(state)
    lines.append("}")
    return "\n".join(lines)
