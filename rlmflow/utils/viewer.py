"""Small Gradio viewer for typed RLMFlow traces.

Layout (top to bottom):

- a step slider (1 .. N steps). Drag to scrub the execution graph
  through time — nodes appear as the agent runs.
- the interactive Plotly graph for the current step. EVERY node is
  drawn, color-coded by type. Click any node to drill into it.
- a detail card for the clicked node: full code, output, content,
  result, error — whichever fields the node carries — plus the full
  combined conversation for the agent that produced it.

The viewer prefers a `Session` if you pass one (full event log), and
falls back to the snapshots in `states`. With states, each step =
``states[i]`` (a tree snapshot). With only a session, each step adds
or refreshes one node from the jsonl in insertion order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlmflow.node import Node, parse_node_obj
from rlmflow.utils.viz import (
    GraphMode,
    VizEdge,
    build_viz_graph,
    node_tree,
    session_events,
)
from rlmflow.workspace.session import FileSession, Session

# Imported at module top so annotations on nested handlers (e.g. `evt:
# gradio.SelectData`) resolve via `typing.get_type_hints()` even with
# `from __future__ import annotations`. Falls back gracefully if the
# optional `viewer` extra isn't installed — `open_viewer` will then raise
# the standard ImportError when called.
try:  # pragma: no cover - optional dep
    import gradio  # noqa: F401  (re-exported for type-hint resolution)
except ImportError:  # pragma: no cover - optional dep
    gradio = None  # type: ignore[assignment]


def _as_node(value: Node | dict) -> Node:
    return value if isinstance(value, Node) else parse_node_obj(value)


def _resolve_session(
    states: list[Node] | None,
    session: Session | str | Path | None,
) -> Session | None:
    if isinstance(session, Session):
        return session
    if isinstance(session, (str, Path)):
        return FileSession(session)
    if not states:
        return None
    # Best-effort: any node may carry a workspace pointer with a sibling session/.
    for node in states:
        ws = getattr(node, "workspace", None)
        root = getattr(ws, "root", None) if ws else None
        if not root:
            continue
        cand = Path(root) / "session"
        if (cand / "nodes.jsonl").exists():
            return FileSession(cand)
    return None


def _all_nodes(states: list[Node] | None, session: Session | None) -> list[Node]:
    if session is not None:
        nodes = list(session.load().values())
        if nodes:
            return nodes
    if not states:
        return []
    # Use the richest snapshot's full subtree.
    richest = max(states, key=lambda s: len(list(s.walk())))
    return list(richest.walk())


def _agent_chain(agent_id: str, nodes: list[Node]) -> list[Node]:
    """Linear in-agent chain ordered by parent → child within the agent.

    Supervising nodes embed sub-agent result objects in their `children`
    list; we ignore cross-agent edges so we don't accidentally walk into
    a child agent's chain.
    """
    same = [n for n in nodes if n.agent_id == agent_id]
    in_agent = {n.id for n in same}
    parent_of: dict[str, str] = {}
    for n in same:
        for child in n.children:
            cid = child.id if isinstance(child, Node) else str(child)
            if cid in in_agent:
                parent_of[cid] = n.id
    children_of: dict[str, list[str]] = {n.id: [] for n in same}
    for cid, pid in parent_of.items():
        children_of.setdefault(pid, []).append(cid)

    by_id = {n.id: n for n in same}
    roots = [n for n in same if n.id not in parent_of]
    if not roots:
        return same  # malformed; just give back what we have
    # Prefer a query node as the root.
    roots.sort(key=lambda n: (n.type != "query", n.id))
    root = roots[0]

    chain: list[Node] = []
    cur: Node | None = root
    while cur is not None:
        chain.append(cur)
        kids = children_of.get(cur.id, [])
        if not kids:
            break
        # Within an agent the chain is linear; if there happen to be
        # multiple in-agent children pick the latest/terminal-leaning one.
        kids_sorted = sorted(
            (by_id[k] for k in kids),
            key=lambda n: (n.terminal, n.type == "result", n.id),
            reverse=True,
        )
        cur = kids_sorted[0]
    return chain


def _agent_status(chain: list[Node]) -> tuple[str, str]:
    """Return (marker, kind) for an agent given its chain.

    kind is one of: 'result', 'error', 'running'.
    """
    last = chain[-1] if chain else None
    if last is None:
        return ("·", "running")
    if last.type == "result":
        return ("✓", "result")
    if last.type == "error" or any(n.type == "error" for n in chain):
        return ("✗", "error")
    return ("●", "running")


def _agent_result_preview(chain: list[Node], limit: int = 64) -> str:
    last = chain[-1] if chain else None
    if last is None:
        return ""
    if last.type == "result":
        body = (getattr(last, "result", "") or "").strip()
    elif last.type == "error":
        body = f"{getattr(last, 'error', '') or 'error'}: {getattr(last, 'content', '') or ''}".strip()
    else:
        body = f"<in {last.type}>"
    body = body.replace("\n", " ")
    return body[:limit] + ("…" if len(body) > limit else "")


def node_session(node: Node, *, include_system: bool = False) -> str:
    """Render every message in this graph subtree, across all agents, in graph order.

    Each message line is prefixed with the agent it came from, so
    cross-agent flows read top-to-bottom like a chat log. When
    ``include_system`` is true, an agent's system prompt is emitted the
    first time that agent appears.
    """
    parts: list[str] = []
    seen_systems: set[str] = set()
    for item in node.walk():
        agent = item.agent_id or "root"
        if include_system and agent not in seen_systems and item.system_prompt:
            seen_systems.add(agent)
            parts.append(f"--- [{agent}] system ---\n{item.system_prompt.strip()}")

        if item.type == "query":
            parts.append(f"--- [{agent}] query ---\n{(item.query or '').strip()}")
        elif item.type == "action":
            parts.append(
                f"--- [{agent}] assistant ---\n{(getattr(item, 'reply', '') or '').strip()}"
            )
        elif item.type == "observation":
            parts.append(
                f"--- [{agent}] observation ---\n{(getattr(item, 'content', '') or '').strip()}"
            )
        elif item.type == "supervising":
            wait_on = ", ".join(getattr(item, "waiting_on", []) or [])
            parts.append(f"--- [{agent}] supervising ---\nwaiting on: {wait_on}")
        elif item.type == "resume":
            parts.append(
                f"--- [{agent}] resume ---\n{(getattr(item, 'content', '') or '').strip()}"
            )
        elif item.type == "error":
            parts.append(
                f"--- [{agent}] error ({getattr(item, 'error', '')}) ---\n"
                f"{(getattr(item, 'content', '') or '').strip()}"
            )
        elif item.type == "result":
            parts.append(
                f"--- [{agent}] result ---\n{(getattr(item, 'result', '') or '').strip()}"
            )
    return "\n\n".join(parts)


def node_transcript(
    node: Node,
    agent_id: str | None = None,
    *,
    session: Session | str | Path | None = None,
    include_system: bool = True,
) -> str:
    """Render a clean transcript for one agent in a node snapshot.

    By default this only uses the subtree visible from ``node``. Pass a
    session if you want to reconstruct from the persisted event log.
    """
    agent_id = agent_id or node.agent_id
    sess = _resolve_session([node], session)
    nodes = _all_nodes([node], sess) if sess is not None else list(node.walk())
    chain = _agent_chain(agent_id, nodes)
    if not chain:
        return f"(no nodes for agent {agent_id!r})"

    parts: list[str] = []
    if include_system and chain[0].system_prompt:
        parts.append(f"--- system ---\n{chain[0].system_prompt.strip()}")
    for item in chain:
        if item.type == "query":
            parts.append(f"--- query ---\n{(item.query or '').strip()}")
        elif item.type == "action":
            parts.append(
                f"--- assistant ---\n{(getattr(item, 'reply', '') or '').strip()}"
            )
        elif item.type == "observation":
            parts.append(
                f"--- observation ---\n{(getattr(item, 'content', '') or '').strip()}"
            )
        elif item.type == "supervising":
            wait_on = ", ".join(getattr(item, "waiting_on", []) or [])
            parts.append(f"--- supervising ---\nwaiting on: {wait_on}")
        elif item.type == "resume":
            parts.append(
                f"--- resume ---\n{(getattr(item, 'content', '') or '').strip()}"
            )
        elif item.type == "error":
            parts.append(
                f"--- error ({getattr(item, 'error', '')}) ---\n"
                f"{(getattr(item, 'content', '') or '').strip()}"
            )
        elif item.type == "result":
            parts.append(
                f"--- result ---\n{(getattr(item, 'result', '') or '').strip()}"
            )
    return "\n\n".join(parts)


def _normalize_reply(reply: str, code: str) -> str:
    """Re-fence custom code blocks (e.g. ```repl) as ```python so the chat
    panel gets proper syntax highlighting. Falls back to the raw reply if
    we can't find a clean substitution."""
    if not reply:
        return ""
    if not code or code not in reply:
        return reply
    # Find the fence that wraps `code` and rewrite its language tag.
    idx = reply.find(code)
    fence_start = reply.rfind("```", 0, idx)
    if fence_start == -1:
        return reply
    nl = reply.find("\n", fence_start)
    if nl == -1 or nl > idx:
        return reply
    return reply[:fence_start] + "```python" + reply[nl:]


def _render_action_message(
    node: Node, prev: Node | None = None
) -> dict[str, str] | None:
    """Render an action / supervising / error / result node as ONE assistant turn.

    The action node already carries `reply`, `code`, and `output` — fold
    them into a single Markdown bubble with a `<details>` block for the
    REPL output so the conversation reads like a normal chat.
    Supervising nodes typically replay the prior action's code; we strip
    that and keep just the "waiting on" status to avoid duplication.
    """
    parts: list[str] = []
    reply = (getattr(node, "reply", "") or "").strip()
    code = (getattr(node, "code", "") or "").strip()
    output = (getattr(node, "output", "") or "").strip()

    prev_code = (getattr(prev, "code", "") or "").strip() if prev is not None else ""
    suppress_code = node.type == "supervising" and bool(code) and code == prev_code

    if not suppress_code:
        if reply and code and code in reply:
            parts.append(_normalize_reply(reply, code))
        else:
            if reply:
                parts.append(reply)
            if code:
                parts.append(f"```python\n{code}\n```")

    if node.type == "supervising":
        wait_on = getattr(node, "waiting_on", []) or []
        if wait_on:
            badges = ", ".join(f"`{w}`" for w in wait_on)
            parts.append(f"_⏳ delegated to {badges}_")
        else:
            parts.append("_⏳ supervising_")

    if node.type == "error":
        kind = getattr(node, "error", "") or "error"
        body = (getattr(node, "content", "") or "").strip()
        parts.append(f"**❌ {kind}**\n```\n{body[:2000]}\n```")
    elif output and not suppress_code:
        snippet = output if len(output) <= 2000 else output[:2000] + "\n…(truncated)"
        parts.append(
            f"<details><summary>output</summary>\n\n```\n{snippet}\n```\n\n</details>"
        )

    if node.type == "result":
        result = (getattr(node, "result", "") or "").strip()
        parts.append(f"**✓ done()** — {result}" if result else "**✓ done()**")

    body = "\n\n".join(p for p in parts if p)
    if not body:
        return None
    return {"role": "assistant", "content": body}


def _looks_like_continue_ping(content: str) -> bool:
    """Heuristic: skip observations whose content is just the boilerplate
    'Your response MUST contain ...' or trivial REPL echoes."""
    head = content.strip().lower()
    if not head:
        return True
    if "your response must contain" in head:
        return True
    if head.startswith("query:") and len(head) < 400:
        return True
    return False


# ── graph figure ─────────────────────────────────────────────────────

# Plotly hue per node type. Same palette is reused by the message
# bubbles in the detail panel so each conversation row visually matches
# the node it came from.
_GRAPH_NODE_COLORS: dict[str, str] = {
    "query": "#58a6ff",  # sky blue   — input
    "action": "#bc8cff",  # lavender   — compute step
    "observation": "#ff9e64",  # peach      — feedback / tool output
    "supervising": "#ffd33d",  # gold       — orchestrator
    "resume": "#56d4dd",  # cyan       — continuation
    "result": "#56d364",  # emerald    — terminal success
    "error": "#ff7b72",  # coral      — failure
}

# Marker symbols pair with color so types stay distinguishable at small
# sizes / for color-blind viewers / in screenshots.
_GRAPH_NODE_SYMBOLS: dict[str, str] = {
    "query": "circle",
    "action": "diamond",
    "observation": "square",
    "supervising": "star",
    "resume": "triangle-right",
    "result": "hexagon",
    "error": "x",
}


def _node_hover_text(node: Node) -> str:
    """Hover string for a Plotly scatter node — supports <br> + basic HTML.

    Kept deliberately minimal: just identity (agent_id / type / depth /
    model / tokens). Click the node to see code, output, result, or
    error content in the detail panel.
    """
    rows = [
        f"<b>{node.agent_id or 'root'}</b>",
        f"<i>{node.type}</i> · depth {node.depth or 0}",
    ]
    model = getattr(node, "model_label", "") or ""
    if model:
        rows.append(f"model: {model}")
    tin = getattr(node, "total_input_tokens", 0) or 0
    tout = getattr(node, "total_output_tokens", 0) or 0
    if tin or tout:
        rows.append(f"tokens: {tin + tout:,} (in {tin:,} / out {tout:,})")
    return "<br>".join(rows)


def _node_display_label(
    node: Node,
    *,
    repeated_agent: bool = False,
    is_agent_entry: bool = False,
    limit: int = 22,
) -> str:
    label = node.agent_id or node.id[:8]
    if label.startswith("root."):
        label = label[len("root.") :]
    if (node.depth or 0) >= 2 and "." in label:
        label = label.rsplit(".", 1)[-1]
    if repeated_agent and not is_agent_entry:
        label = node.type
    if len(label) > limit:
        label = label[: limit - 1] + "…"
    return label


def _node_children_ids(node: Node) -> list[str]:
    out: list[str] = []
    for child in node.children:
        out.append(child.id if isinstance(child, Node) else str(child))
    return out


def _build_graph_figure(
    nodes: list[Node],
    *,
    height: int = 360,
    title: str = "execution graph",
    id_to_agent: dict[str, str] | None = None,
    edges: list[VizEdge] | None = None,
):
    """Plotly figure showing every node, colored by type, edges = children.

    Works with a flat list (where `children` may be id strings) or with
    a list pulled from a state subtree (where `children` are real Node
    objects). We re-derive the parent → child edges from id references
    so both representations behave the same.

    `id_to_agent` is an optional GLOBAL id → agent_id map (covering nodes
    not yet present in the snapshot). Lets us route cross-agent edges
    even when supervising's `children` references future result ids that
    haven't been created yet at this step.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover - optional dep
        return None

    by_id: dict[str, Node] = {n.id: n for n in nodes}
    children_of: dict[str, list[str]] = {nid: [] for nid in by_id}
    parent_of: dict[str, str] = {}

    # Global id → agent_id (snapshot ⊕ caller-supplied future nodes).
    agent_of_id: dict[str, str] = {n.id: n.agent_id for n in nodes}
    if id_to_agent:
        for k, v in id_to_agent.items():
            agent_of_id.setdefault(k, v)

    if edges is not None:
        for edge in edges:
            if edge.source not in by_id or edge.target not in by_id:
                continue
            if edge.target in parent_of:
                continue
            parent_of[edge.target] = edge.source
            children_of[edge.source].append(edge.target)
    else:
        # Pass 1 — intra-agent edges. The agent's chain (query → action → ...
        # → result) is the *real* parent chain. We do this first so the chain
        # claims its nodes before any cross-agent embed (e.g. supervising
        # listing each child agent's `result` in its `children`) tries to.
        for n in nodes:
            for cid in _node_children_ids(n):
                if cid not in by_id or cid == n.id:
                    continue
                child = by_id[cid]
                if child.agent_id != n.agent_id:
                    continue
                if cid in parent_of:
                    continue
                parent_of[cid] = n.id
                children_of[n.id].append(cid)

        # Pass 2 — cross-agent / delegation edges.
        by_agent: dict[str, list[Node]] = {}
        for n in nodes:
            by_agent.setdefault(n.agent_id, []).append(n)
        for agent_id in by_agent:
            chain = _agent_chain(agent_id, nodes)
            if not chain:
                continue
            first = chain[0]
            if first.id in parent_of:
                continue
            attached = False
            for other in nodes:
                if other.agent_id == agent_id:
                    continue
                for cid in _node_children_ids(other):
                    if agent_of_id.get(cid) == agent_id:
                        parent_of[first.id] = other.id
                        children_of[other.id].append(first.id)
                        attached = True
                        break
                if attached:
                    break

    # Roots = nodes without a known parent.
    roots = [nid for nid in by_id if nid not in parent_of]
    roots.sort(key=lambda nid: (by_id[nid].depth or 0, nid))

    # Tidy-tree layout: each subtree gets a horizontal slot proportional
    # to its leaf count; parent sits centered above its children.
    pos: dict[str, tuple[float, float]] = {}

    def leaf_count(nid: str, seen: set[str]) -> int:
        if nid in seen:
            return 1
        seen.add(nid)
        kids = children_of.get(nid, [])
        if not kids:
            return 1
        return sum(leaf_count(k, seen) for k in kids)

    def place(nid: str, left: float, right: float, depth: int, seen: set[str]) -> None:
        if nid in seen:
            return
        seen.add(nid)
        pos[nid] = ((left + right) / 2, -float(depth))
        kids = children_of.get(nid, [])
        if not kids:
            return
        widths = [leaf_count(k, set()) for k in kids]
        total = sum(widths) or 1
        span = right - left
        cursor = left
        for cid, w in zip(kids, widths):
            child_span = span * (w / total)
            place(cid, cursor, cursor + child_span, depth + 1, seen)
            cursor += child_span

    cursor_left = 0.0
    placed: set[str] = set()
    for root in roots:
        width = leaf_count(root, set())
        place(root, cursor_left, cursor_left + float(max(width, 1)), 0, placed)
        cursor_left += float(max(width, 1)) + 1.0
    # Stragglers (cycles, broken edges) — drop them at depth 0 to the right.
    for nid in by_id:
        if nid not in pos:
            cursor_left += 1.0
            pos[nid] = (cursor_left, 0.0)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for nid, kids in children_of.items():
        if nid not in pos:
            continue
        x0, y0 = pos[nid]
        for cid in kids:
            if cid not in pos:
                continue
            x1, y1 = pos[cid]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"color": "#30363d", "width": 1},
        hoverinfo="skip",
        showlegend=False,
    )

    ordered = [by_id[nid] for nid in pos]
    node_x = [pos[n.id][0] for n in ordered]
    node_y = [pos[n.id][1] for n in ordered]
    agent_counts: dict[str, int] = {}
    for node in ordered:
        agent_counts[node.agent_id] = agent_counts.get(node.agent_id, 0) + 1
    agent_first_id: dict[str, str] = {}
    for node in ordered:
        agent_first_id.setdefault(node.agent_id, node.id)
    labels = [
        _node_display_label(
            n,
            repeated_agent=agent_counts.get(n.agent_id, 0) > 1,
            is_agent_entry=agent_first_id.get(n.agent_id) == n.id,
        )
        for n in ordered
    ]
    colors = [_GRAPH_NODE_COLORS.get(n.type, "#8b949e") for n in ordered]
    symbols = [_GRAPH_NODE_SYMBOLS.get(n.type, "circle") for n in ordered]
    sizes = [
        14
        + min(
            22,
            (((n.total_input_tokens or 0) + (n.total_output_tokens or 0)) ** 0.5 / 3.5),
        )
        for n in ordered
    ]
    hover = [_node_hover_text(n) for n in ordered]

    def text_position(node: Node) -> str:
        parent = parent_of.get(node.id)
        siblings = children_of.get(parent, []) if parent is not None else []
        if len(siblings) > 1 and node.id in siblings and not children_of.get(node.id):
            midpoint = (len(siblings) - 1) / 2
            idx = siblings.index(node.id)
            return "bottom left" if idx < midpoint else "bottom right"
        return "top center" if (node.depth or 0) % 2 else "bottom center"

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition=[text_position(n) for n in ordered],
        textfont={
            "color": "#c9d1d9",
            "family": "ui-monospace, SFMono-Regular, Menlo, monospace",
            "size": 10,
        },
        hovertext=hover,
        hoverinfo="text",
        customdata=[n.id for n in ordered],
        marker={
            "color": colors,
            "symbol": symbols,
            "size": sizes,
            "line": {"color": "#0d1117", "width": 1.5},
        },
        cliponaxis=False,
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    for ntype, color in _GRAPH_NODE_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "color": color,
                    "symbol": _GRAPH_NODE_SYMBOLS.get(ntype, "circle"),
                    "size": 11,
                    "line": {"color": "#0d1117", "width": 1},
                },
                name=ntype,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    n_edges = sum(1 for e in edge_x if e is None)
    if node_x:
        x_span = max(node_x) - min(node_x)
        x_pad = max(1.0, x_span * 0.18)
        x_range = [min(node_x) - x_pad, max(node_x) + x_pad]
    else:
        x_range = [-1.0, 1.0]
    if node_y:
        y_span = max(node_y) - min(node_y)
        y_pad = max(0.75, y_span * 0.16)
        y_range = [min(node_y) - y_pad, max(node_y) + y_pad]
    else:
        y_range = [-1.0, 1.0]
    fig.update_layout(
        title={
            "text": f"<b>{title}</b> · {len(ordered)} nodes · {n_edges} edges",
            "font": {"color": "#e6edf3", "size": 12},
            "x": 0.0,
            "xanchor": "left",
        },
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font={"color": "#e6edf3"},
        height=height,
        hoverlabel={
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "font": {
                "color": "#e6edf3",
                "family": "ui-monospace, SFMono-Regular, Menlo, monospace",
                "size": 11,
            },
            "align": "left",
        },
        xaxis={"visible": False, "range": x_range},
        yaxis={"visible": False, "range": y_range},
        margin={"l": 44, "r": 44, "t": 48, "b": 92},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": "#8b949e", "size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return fig


def node_plot(
    node: Node,
    kind: str = "graph",
    *,
    states: list[Node] | None = None,
    events: list[Node] | None = None,
    step: int | None = None,
    mode: GraphMode = "snapshot",
    height: int = 420,
    title: str | None = None,
    session: Session | str | Path | None = None,
    include_results: bool = True,
):
    """Render ``node`` in one of several formats.

    ``kind`` may be:
    - ``"graph"`` / ``"plotly"``: Plotly figure matching the Gradio viewer.
    - ``"mermaid"`` / ``"state"``: Mermaid state diagram string.
    - ``"flowchart"``: Mermaid flowchart string.
    - ``"sequence"``: Mermaid sequence diagram string.
    - ``"dot"`` / ``"d2"``: static graph language strings.
    - ``"tree"``: plain text tree.
    - ``"gantt"``: HTML swimlane. Pass ``states=[...]`` for a full run.
    """
    fmt = kind.lower().replace("-", "_")
    if fmt in {"mermaid", "state", "state_diagram"}:
        from rlmflow.utils.export import to_mermaid

        return to_mermaid(node, include_results=include_results)
    if fmt in {"flow", "flowchart", "mermaid_flowchart"}:
        from rlmflow.utils.export import (
            to_mermaid_flowchart,
            viz_graph_to_mermaid_flowchart,
        )

        if mode == "events":
            graph = build_viz_graph(
                states=states or [node],
                events=events,
                session=session,
                step=step,
                mode=mode,
            )
            return viz_graph_to_mermaid_flowchart(
                graph, include_results=include_results
            )
        return to_mermaid_flowchart(node, include_results=include_results)
    if fmt in {"sequence", "sequence_diagram"}:
        from rlmflow.utils.export import to_mermaid_sequence

        return to_mermaid_sequence(node)
    if fmt == "dot":
        from rlmflow.utils.export import to_dot, viz_graph_to_dot

        if mode == "events":
            graph = build_viz_graph(
                states=states or [node],
                events=events,
                session=session,
                step=step,
                mode=mode,
            )
            return viz_graph_to_dot(graph, include_results=include_results)
        return to_dot(node, include_results=include_results)
    if fmt == "d2":
        from rlmflow.utils.export import to_d2, viz_graph_to_d2

        if mode == "events":
            graph = build_viz_graph(
                states=states or [node],
                events=events,
                session=session,
                step=step,
                mode=mode,
            )
            return viz_graph_to_d2(graph, include_results=include_results)
        return to_d2(node, include_results=include_results)
    if fmt in {"tree", "ascii"}:
        return node.tree()
    if fmt in {"node_tree", "nodes_tree", "event_tree"}:
        graph = build_viz_graph(
            states=states or [node],
            events=events,
            session=session,
            step=step,
            mode=mode,
        )
        return node_tree(graph)
    if fmt in {"gantt", "swimlane", "swimlanes"}:
        from rlmflow.utils.viz import gantt_html

        return gantt_html(states or [node], title=title or "rlmflow gantt")
    if fmt not in {"graph", "plotly", "viewer"}:
        supported = (
            "graph, mermaid, flowchart, sequence, dot, d2, tree, node_tree, gantt"
        )
        raise ValueError(f"Unknown plot kind {kind!r}. Supported: {supported}.")

    graph = build_viz_graph(
        states=states or [node],
        events=events,
        session=session,
        step=step,
        mode=mode,
    )
    visible_nodes = graph.payloads
    id_to_agent = {item.id: item.agent_id for item in visible_nodes}
    fig = _build_graph_figure(
        visible_nodes,
        height=height,
        title=title or f"{node.agent_id} · {node.type}",
        id_to_agent=id_to_agent,
        edges=graph.edges,
    )
    if fig is None:
        raise ImportError(
            "Node.plot() requires the viewer extra: `pip install rlmflow[viewer]`."
        )
    return fig


def node_plot_html(
    node: Node,
    kind: str = "graph",
    *,
    states: list[Node] | None = None,
    events: list[Node] | None = None,
    step: int | None = None,
    mode: GraphMode = "snapshot",
    height: int = 420,
    title: str | None = None,
    session: Session | str | Path | None = None,
    include_results: bool = True,
    include_plotlyjs: str | bool = "cdn",
) -> str:
    """Return an HTML fragment for ``node.plot(kind)``."""
    rendered = node_plot(
        node,
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
    if hasattr(rendered, "to_html"):
        return rendered.to_html(include_plotlyjs=include_plotlyjs, full_html=False)
    if kind.lower().replace("-", "_") in {"gantt", "swimlane", "swimlanes"}:
        return str(rendered)
    from html import escape

    return (
        '<pre style="background:#0d1117;color:#c9d1d9;padding:12px;'
        'border-radius:6px;overflow:auto;">'
        f"{escape(str(rendered))}</pre>"
    )


def open_viewer(
    states: list[Node] | list[dict] | None = None,
    *,
    session: Session | str | Path | None = None,
    mode: GraphMode | None = None,
    **launch_kwargs: Any,
):
    import gradio as gr

    norm_states: list[Node] | None = None
    if states:
        norm_states = [_as_node(s) for s in states]

    sess = _resolve_session(norm_states, session)
    nodes = _all_nodes(norm_states, sess)
    events = session_events(sess)
    if not nodes:
        raise ValueError(
            "open_viewer needs either `states` (a list of snapshots) "
            "or `session=` (a Session / path to a session/ dir)."
        )

    resolved_mode: GraphMode = mode or ("events" if events else "snapshot")
    if resolved_mode == "events" and not events:
        raise ValueError("open_viewer(mode='events') requires a persisted session log.")
    n_steps = len(norm_states) if norm_states else max(1, len(events))
    step_graphs = [
        build_viz_graph(
            states=norm_states,
            events=events,
            session=sess,
            step=i,
            mode=resolved_mode,
        )
        for i in range(n_steps)
    ]
    nodes_by_id_global = {n.id: n for graph in step_graphs for n in graph.payloads}

    def _coerce_step(value: Any) -> int:
        # Gradio's server-function bridge sometimes hands the JS arg list in
        # as a single positional, so unwrap one-element list/tuple wrappers.
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        return max(0, min(int(value), n_steps - 1))

    def _fig_json_for_step(step: int) -> str:
        step = _coerce_step(step)
        graph = step_graphs[step]
        fig = _build_graph_figure(
            graph.payloads,
            height=420,
            title=f"step {step + 1} / {n_steps}",
            id_to_agent={node.id: node.agent_id for node in graph.payloads},
            edges=graph.edges,
        )
        return fig.to_json() if fig is not None else "{}"

    def get_step_fig(*args: Any) -> str:
        # Accept ``(step,)`` or ``([step, ...],)`` – the JS bridge is loose.
        step_arg = args[0] if len(args) == 1 else args
        if isinstance(step_arg, (list, tuple)) and len(step_arg) >= 1:
            step_arg = step_arg[0]
        return _fig_json_for_step(step_arg)

    def get_node_detail(*args: Any) -> str:
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            payload = args[0]
        else:
            payload = args
        if len(payload) < 2:
            return "<i style='color:#8b949e'>(missing node id)</i>"
        step_arg, node_id = payload[0], payload[1]
        step = _coerce_step(step_arg)
        graph = step_graphs[step]
        nodes_by_id = graph.nodes_by_id
        node = nodes_by_id.get(node_id)
        if node is None:
            return "<i style='color:#8b949e'>(node not visible at this step)</i>"
        return _node_detail_html(node, graph.payloads)

    initial_fig_json = _fig_json_for_step(n_steps - 1)
    initial_detail = (
        "<i style='color:#8b949e'>Click any node above to see its full payload.</i>"
    )

    js_on_load = _GRAPH_JS_ON_LOAD

    with gr.Blocks(title="RLMFlow Viewer", fill_height=True) as demo:
        gr.Markdown(
            f"### RLMFlow Viewer · {n_steps} steps · "
            f"{len(nodes_by_id_global)} unique nodes · {resolved_mode} mode\n"
            "Drag the slider to scrub the graph through time. "
            "Click any node to see its step-local code, output, and conversation."
        )

        slider = gr.Slider(
            minimum=0,
            maximum=max(n_steps - 1, 0),
            step=1,
            value=n_steps - 1,
            label=f"Step (0 .. {n_steps - 1})",
            interactive=n_steps > 1,
        )

        gr.HTML(
            value="",
            html_template=_GRAPH_HTML_TEMPLATE,
            css_template=_GRAPH_CSS,
            js_on_load=js_on_load,
            server_functions=[get_step_fig, get_node_detail],
            initial_fig_json=initial_fig_json,
            initial_detail=initial_detail,
            initial_step=n_steps - 1,
            min_height=720,
        )

        # JS-only handler: relay slider changes to the HTML component
        # via a window CustomEvent. No Python round-trip.
        slider.change(
            fn=None,
            inputs=slider,
            outputs=None,
            js=(
                "(step) => { window.dispatchEvent("
                "new CustomEvent('rlmflow-step-change', {detail: step})); "
                "return step; }"
            ),
        )

    return demo.launch(**launch_kwargs)


# ── HTML template, CSS, and JS for the interactive graph + detail card

_GRAPH_HTML_TEMPLATE = """
<div class="rlmflow-shell" data-initial-step="${initial_step}">
  <div class="rlmflow-plot"></div>
  <div class="rlmflow-detail">${initial_detail}</div>
  <script type="application/json" class="rlmflow-bootstrap">${initial_fig_json}</script>
</div>
""".strip()

_GRAPH_CSS = """
.rlmflow-shell { display: flex; flex-direction: column; gap: 12px; }
.rlmflow-plot {
    width: 100%; height: 420px;
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
}
.rlmflow-detail {
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 14px; color: #e6edf3;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
}
.rlmflow-detail .header {
    display: flex; justify-content: space-between; gap: 8px;
    color: #8b949e; font-size: 11px; margin-bottom: 8px;
}
.rlmflow-detail h4 {
    margin: 0 0 6px 0; color: #e6edf3; font-size: 13px;
    font-family: -apple-system, system-ui, sans-serif;
}
.rlmflow-detail h5 {
    margin: 12px 0 4px 0; color: #8b949e; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em;
    font-family: -apple-system, system-ui, sans-serif;
}
.rlmflow-detail pre {
    background: #161b22; border: 1px solid #30363d; border-radius: 4px;
    padding: 10px; margin: 0; overflow: auto;
    color: #e6edf3; font-size: 11px; line-height: 1.4;
    white-space: pre-wrap; word-break: break-word;
    max-height: 320px;
}
.rlmflow-detail .pill {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    background: #161b22; border: 1px solid #30363d; font-size: 10px;
    color: #8b949e; margin-right: 4px;
}
/* Pills + message accents share one palette — same hues used by the
   plotly graph above so the message list color-matches the node it
   came from. */
.rlmflow-detail .pill.type-query       { color: #58a6ff; border-color: #58a6ff; }
.rlmflow-detail .pill.type-action      { color: #bc8cff; border-color: #bc8cff; }
.rlmflow-detail .pill.type-observation { color: #ff9e64; border-color: #ff9e64; }
.rlmflow-detail .pill.type-supervising { color: #ffd33d; border-color: #ffd33d; }
.rlmflow-detail .pill.type-resume      { color: #56d4dd; border-color: #56d4dd; }
.rlmflow-detail .pill.type-result      { color: #56d364; border-color: #56d364; }
.rlmflow-detail .pill.type-error       { color: #ff7b72; border-color: #ff7b72; }
.rlmflow-detail .messages {
    display: flex; flex-direction: column; gap: 8px;
}
.rlmflow-detail .msg {
    padding: 8px 10px 8px 12px; border-radius: 6px;
    border: 1px solid #30363d;
    border-left: 3px solid #30363d;
    background: #0d1117; color: #e6edf3;
}
.rlmflow-detail .msg .role {
    text-transform: uppercase; font-size: 9px; color: #8b949e;
    letter-spacing: 0.06em; margin-bottom: 4px; display: block;
    font-weight: 600;
}
.rlmflow-detail .msg pre { background: rgba(255,255,255,0.03); }

/* Color-code each bubble by source node type. Left border = the type's
   accent hue, background = a heavily desaturated tint of the same. */
.rlmflow-detail .msg.kind-query {
    border-left-color: #58a6ff; background: #0e1822;
}
.rlmflow-detail .msg.kind-query .role { color: #58a6ff; }

.rlmflow-detail .msg.kind-action {
    border-left-color: #bc8cff; background: #18142a;
}
.rlmflow-detail .msg.kind-action .role { color: #bc8cff; }

.rlmflow-detail .msg.kind-observation {
    border-left-color: #ff9e64; background: #241710;
}
.rlmflow-detail .msg.kind-observation .role { color: #ff9e64; }

.rlmflow-detail .msg.kind-supervising {
    border-left-color: #ffd33d; background: #221d08;
}
.rlmflow-detail .msg.kind-supervising .role { color: #ffd33d; }

.rlmflow-detail .msg.kind-resume {
    border-left-color: #56d4dd; background: #0d2226;
}
.rlmflow-detail .msg.kind-resume .role { color: #56d4dd; }

.rlmflow-detail .msg.kind-result {
    border-left-color: #56d364; background: #0d2114;
}
.rlmflow-detail .msg.kind-result .role { color: #56d364; }

.rlmflow-detail .msg.kind-error {
    border-left-color: #ff7b72; background: #2a1010;
}
.rlmflow-detail .msg.kind-error .role { color: #ff7b72; }

.rlmflow-detail .msg.kind-system {
    border-left-color: #6e7681; background: #0d1117;
    color: #8b949e;
}
.rlmflow-detail .msg.kind-system .role { color: #8b949e; }
.rlmflow-detail details.agent-block {
    border: 1px solid #30363d; border-radius: 6px;
    background: #0a0d12; margin: 8px 0; padding: 0;
}
.rlmflow-detail details.agent-block > summary {
    cursor: pointer; padding: 8px 12px; user-select: none;
    list-style: none; font-size: 12px; color: #e6edf3;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.rlmflow-detail details.agent-block > summary::-webkit-details-marker { display: none; }
.rlmflow-detail details.agent-block > summary::before {
    content: "▸"; color: #8b949e; font-size: 10px; transition: transform 0.1s;
    display: inline-block; width: 10px;
}
.rlmflow-detail details.agent-block[open] > summary::before {
    transform: rotate(90deg);
}
.rlmflow-detail details.agent-block > .agent-body {
    padding: 8px 12px 12px 24px;
    border-top: 1px solid #30363d;
}
.rlmflow-detail .child-agents { margin-bottom: 12px; }
.rlmflow-detail h6.msg-header {
    margin: 12px 0 6px 0; color: #8b949e; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.05em;
    font-family: -apple-system, system-ui, sans-serif; font-weight: 600;
}
.rlmflow-detail details.agent-block > summary:hover { background: #161b22; }

.rlmflow-detail details.payload-block {
    border: 1px solid #30363d; border-radius: 6px;
    background: #0a0d12; margin: 6px 0; padding: 0;
}
.rlmflow-detail details.payload-block > summary {
    cursor: pointer; padding: 6px 12px; user-select: none;
    list-style: none; font-size: 11px; color: #e6edf3;
}
.rlmflow-detail details.payload-block > summary::-webkit-details-marker { display: none; }
.rlmflow-detail details.payload-block > summary::before {
    content: "▸"; color: #8b949e; font-size: 10px; margin-right: 6px;
    display: inline-block; transition: transform 0.1s;
}
.rlmflow-detail details.payload-block[open] > summary::before {
    transform: rotate(90deg);
}
.rlmflow-detail details.payload-block > pre {
    margin: 0 12px 12px 12px;
}
""".strip()

_GRAPH_JS_ON_LOAD = r"""
(async () => {
    const plot = element.querySelector('.rlmflow-plot');
    const detail = element.querySelector('.rlmflow-detail');
    const bootstrap = element.querySelector('.rlmflow-bootstrap');
    const shell = element.querySelector('.rlmflow-shell');
    let currentStep = parseInt(shell?.dataset.initialStep || '0', 10) || 0;

    async function ensurePlotly() {
        if (typeof window.Plotly !== 'undefined') return;
        await new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
            s.onload = resolve;
            s.onerror = () => reject(new Error('failed to load Plotly'));
            document.head.appendChild(s);
        });
    }

    function decodeFig(raw) {
        try { return JSON.parse(raw); } catch (e) { return null; }
    }

    async function drawFig(figJson, useReact) {
        const fig = decodeFig(figJson);
        if (!fig || !fig.data) return;
        if (useReact && plot.data) {
            await window.Plotly.react(plot, fig.data, fig.layout, { responsive: true });
        } else {
            await window.Plotly.newPlot(plot, fig.data, fig.layout, {
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            });
            plot.on('plotly_click', async (e) => {
                if (!e.points || !e.points.length) return;
                const nid = e.points[0].customdata;
                if (!nid) return;
                detail.innerHTML = '<i style="color:#8b949e">loading…</i>';
                try {
                    detail.innerHTML = await server.get_node_detail(currentStep, nid);
                } catch (err) {
                    detail.innerHTML =
                        '<span style="color:#f85149">error: ' + err.message + '</span>';
                }
            });
        }
    }

    await ensurePlotly();
    const initial = bootstrap ? bootstrap.textContent : '{}';
    await drawFig(initial, false);

    window.addEventListener('rlmflow-step-change', async (ev) => {
        const step = parseInt(ev.detail, 10);
        if (Number.isNaN(step)) return;
        try {
            currentStep = step;
            const figJson = await server.get_step_fig(step);
            await drawFig(figJson, true);
        } catch (err) {
            console.error('rlmflow step refresh failed', err);
        }
    });
})();
""".strip()


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _node_detail_html(node: Node, all_nodes: list[Node]) -> str:
    """Render the full payload of a single node + its agent's conversation as HTML."""
    parts: list[str] = []

    pill_type = f'<span class="pill type-{_esc(node.type)}">{_esc(node.type)}</span>'
    tin = node.total_input_tokens or 0
    tout = node.total_output_tokens or 0
    parts.append(
        f"""
<div class="header">
  <div>
    <h4>{_esc(node.agent_id or "root")} {pill_type}</h4>
    <div>
      <span class="pill">id {_esc(node.id)}</span>
      <span class="pill">model {_esc(node.model_label)}</span>
      <span class="pill">depth {node.depth or 0}</span>
      <span class="pill">{tin:,} in / {tout:,} out tokens</span>
      {'<span class="pill" style="color:#7ee787;border-color:#7ee787">terminal</span>' if node.terminal else ''}
    </div>
  </div>
</div>
"""
    )

    # Conversation tree first — start at the clicked node's agent and
    # walk that agent's chain in time order, inlining every spawned
    # sub-agent at the exact delegation point. Sub-agents render as
    # collapsible blocks so the parent's flow stays readable.
    nodes_by_id = {n.id: n for n in all_nodes}
    seen_agents: set[str] = {node.agent_id}
    parts.append("<h5>conversation tree</h5>")
    parts.append(
        _agent_block_html(node.agent_id, nodes_by_id, seen_agents, open_top=True)
    )

    # Then the raw payload fields. Each gets its own collapsible <details>
    # so a giant `code` or `output` field doesn't push everything else off
    # screen.
    fields: list[tuple[str, str]] = []
    for label in ("query", "reply", "code", "output", "content", "result", "error"):
        val = getattr(node, label, None)
        if val:
            fields.append((label, str(val)))
    waiting_on = getattr(node, "waiting_on", None) or []
    if waiting_on:
        fields.append(("waiting_on", "\n".join(waiting_on)))

    if fields:
        parts.append("<h5>raw node payload</h5>")
        for label, val in fields:
            preview = val.strip().splitlines()[0][:80] if val.strip() else ""
            preview_html = (
                f" <span style='color:#8b949e'>{_esc(preview)}</span>"
                if preview
                else ""
            )
            # First field opens by default; the rest stay collapsed.
            open_attr = " open" if label == fields[0][0] else ""
            parts.append(
                f"<details class='payload-block'{open_attr}>"
                f"<summary><strong>{_esc(label)}</strong>{preview_html}</summary>"
                f"<pre>{_esc(val)}</pre>"
                "</details>"
            )

    return "".join(parts)


def _emit_msg(
    out: list[str], role: str, html_body: str, *, kind: str | None = None
) -> None:
    """Append one message bubble. `kind` is the originating node type
    (query / action / supervising / observation / resume / result / error)
    so CSS can color-code by graph node type, matching the legend in the
    graph above."""
    cls = f"msg {_esc(role)}"
    if kind:
        cls += f" kind-{_esc(kind)}"
    label = kind or role
    out.append(
        f'<div class="{cls}">'
        f'<span class="role">{_esc(label)}</span>'
        f"<div>{html_body}</div>"
        "</div>"
    )


def _emit_chain_into(
    agent_id: str,
    chain: list[Node],
    nodes_by_id: dict[str, Node],
    seen_agents: set[str],
    out: list[str],
) -> None:
    """Walk one agent's chain in time order, emitting messages and inlining
    every spawned sub-agent's block at the node that spawned it."""
    if not chain:
        _emit_msg(
            out,
            "tool",
            "<i style='color:#8b949e'>(no nodes recorded for this agent)</i>",
            kind="system",
        )
        return

    sysp = (getattr(chain[0], "system_prompt", "") or "").strip()
    if sysp:
        _emit_msg(
            out,
            "system",
            "<details><summary>system prompt</summary><pre>"
            + _esc(sysp)
            + "</pre></details>",
            kind="system",
        )

    pending_obs: list[tuple[str, str]] = []  # (kind, content)
    prev_action: Node | None = None

    out.append('<div class="messages">')

    def flush_pending() -> None:
        if not pending_obs:
            return
        # If everything was the same kind, color the bubble that kind;
        # otherwise just call it "observation" (the common case).
        kinds = {k for k, _ in pending_obs}
        kind = next(iter(kinds)) if len(kinds) == 1 else "observation"
        body = "".join(
            f"<details><summary>note</summary><pre>{_esc(c[:4000])}</pre></details>"
            for _, c in pending_obs
        )
        _emit_msg(out, "user", body, kind=kind)
        pending_obs.clear()

    def inject_spawned(node: Node) -> None:
        spawned: list[str] = []
        for cid in _node_children_ids(node):
            child_node = nodes_by_id.get(cid)
            if child_node is None:
                continue
            if child_node.agent_id == agent_id:
                continue
            if child_node.agent_id in seen_agents:
                continue
            seen_agents.add(child_node.agent_id)
            spawned.append(child_node.agent_id)
        if not spawned:
            return
        out.append("</div>")  # close current messages block
        out.append("<div class='spawned-agents'>")
        for sub_agent_id in spawned:
            out.append(_agent_block_html(sub_agent_id, nodes_by_id, seen_agents))
        out.append("</div>")
        out.append('<div class="messages">')  # reopen messages

    for node in chain:
        if node.type == "query":
            q = (getattr(node, "query", "") or "").strip()
            if q:
                _emit_msg(out, "user", _render_md_to_html(q), kind="query")
        elif node.type in ("observation", "resume"):
            content = (getattr(node, "content", "") or "").strip()
            if content and not _looks_like_continue_ping(content):
                pending_obs.append((node.type, content))
        else:
            flush_pending()
            msg = _render_action_message(node, prev=prev_action)
            if msg is not None:
                _emit_msg(
                    out,
                    msg["role"],
                    _render_md_to_html(msg["content"]),
                    kind=node.type,
                )
            if node.type in ("action", "supervising"):
                prev_action = node

        inject_spawned(node)

    flush_pending()
    out.append("</div>")


def _agent_block_html(
    agent_id: str,
    nodes_by_id: dict[str, Node],
    seen_agents: set[str],
    *,
    open_top: bool = False,
) -> str:
    """Collapsible block for one agent. Nested sub-agents render inline at
    the delegation point inside this agent's body, not bunched at the top
    or bottom — natural conversational order."""
    all_nodes = list(nodes_by_id.values())
    chain = _agent_chain(agent_id, all_nodes)
    in_tokens = sum((n.total_input_tokens or 0) for n in chain)
    out_tokens = sum((n.total_output_tokens or 0) for n in chain)
    marker, _kind = _agent_status(chain)
    result_preview = _agent_result_preview(chain, limit=80)

    summary_html = (
        f"<span class='pill'>{_esc(marker)}</span>"
        f"<strong>{_esc(agent_id or 'root')}</strong>"
        f"<span class='pill'>{(in_tokens + out_tokens):,} tok</span>"
        f"<span style='color:#8b949e;margin-left:8px;font-style:italic'>"
        f"{_esc(result_preview)}</span>"
    )

    body_parts: list[str] = []
    _emit_chain_into(agent_id, chain, nodes_by_id, seen_agents, body_parts)

    open_attr = " open" if open_top else ""
    return (
        f"<details class='agent-block'{open_attr}>"
        f"<summary>{summary_html}</summary>"
        f"<div class='agent-body'>{''.join(body_parts)}</div>"
        "</details>"
    )


def _render_md_to_html(text: str) -> str:
    """Lightweight markdown-ish to HTML for the in-detail message view.

    We don't want to pull in a full markdown library for the viewer, but
    we DO want fenced code blocks and basic line breaks to read cleanly.
    """
    text = _esc(text)
    out: list[str] = []
    in_code = False
    lang = ""
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if in_code:
                out.append("</pre>")
                in_code = False
            else:
                lang = stripped[3:].strip()
                out.append(f'<pre data-lang="{_esc(lang) if lang else "txt"}">')
                in_code = True
            continue
        if in_code:
            out.append(line + "\n")
        else:
            if not line.strip():
                out.append("<br>")
            else:
                out.append(line + "<br>")
    if in_code:
        out.append("</pre>")
    return "".join(out)


__all__ = ["node_plot", "node_plot_html", "node_transcript", "open_viewer"]
