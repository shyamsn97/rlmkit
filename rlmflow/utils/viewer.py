"""Visualization, transcripts, and the Gradio viewer for RLMFlow graphs.

Public viewer/export functions accept a workspace, workspace path, graph, or
graph list. The viewer + HTML stepper + image / GIF / step-image exporters all
share the same Plotly figure builder so the on-screen and offline renders match
pixel-for-pixel.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from html import escape as _esc_html
from pathlib import Path
from typing import Any, TypeAlias

from rlmflow.graph import (
    Graph,
    Node,
    is_done,
    is_errored,
    is_exec_output,
    is_llm_output,
    is_resumed,
    is_supervising,
    is_user_query,
)
from rlmflow.utils.export import _kind as _display_kind  # re-use one mapping
from rlmflow.workspace import BaseWorkspace, Workspace

# ``gradio`` is intentionally NOT imported at module load. It's a heavy
# dependency (~3s import on CPython) that's only needed by ``open_viewer``,
# and pulling it into every ``import rlmflow`` made each spawned REPL
# subprocess pay that cost. The viewer entrypoint imports it lazily.


# An ActionNode is "bookkeeping" when its successor in the same
# agent is one of these observations — the observation already
# represents the step, so the action is hidden in figures and in
# deduper sigs.
#
# Engine writes ``(action, observation)`` atomically per
# ``apply_one`` call (one obs-to-obs transition), so the action
# is always paired with its observation in any persisted snapshot.
# The viewer collapses each pair into the single observation
# node, except for the meaningful exec/resume *outcomes* —
# ``done_output`` and ``error_output`` keep their predecessor
# action visible so the figure reads as ``... → exec → done`` or
# ``... → resume → errored``.
HIDDEN_ACTION_PAIRS: dict[str, set[str]] = {
    "llm_action": {"llm_output"},
    "exec_action": {"exec_output", "supervising_output"},
    "resume_action": {"exec_output", "supervising_output"},
}

# Vertical multiplier applied to the depth-based Y coordinate when laying
# out the graph figure. The figure no longer renders per-node state
# labels (``llm`` / ``exec`` / ``done`` / …) because the legend already
# explains them via colour and symbol — only agent-name annotations
# remain. With state labels gone, rows can sit closer together; 1.4
# leaves comfortable space for the agent annotations above each
# agent-entry marker.
_Y_SPACING = 1.4

# Distinct color for the "agent name" annotation attached to the first
# visible node of each agent.
_AGENT_LABEL_COLOR = "#3fb950"

# Pixel offset between an agent-entry marker and its annotation label.
# Plotly annotations apply ``yshift`` in screen pixels regardless of the
# figure's data-coordinate range, so this is a stable visual gap.
_AGENT_LABEL_YSHIFT = 10
_DENSE_NODE_THRESHOLD = 24
_DENSE_LABEL_NODE_THRESHOLD = 80
_DENSE_LABEL_AGENT_THRESHOLD = 24
_DENSE_MIN_MARKER_SIZE = 12
_DENSE_MAX_MARKER_SIZE = 18
_DENSE_MAX_MARKER_LINE_WIDTH = 1.2
_DENSE_MAX_AGENT_LABEL_SIZE = 8
_MIN_COLUMN_GAP = 0.75


def is_bookkeeping(state: Node, successor: Node | None) -> bool:
    """``True`` when ``state`` is an action collapsed into ``successor``.

    Two surfaces use this: the figure builder hides such nodes; the
    viz-frame deduper treats two snapshots that differ only in
    bookkeeping states as one frame.
    """
    paired = HIDDEN_ACTION_PAIRS.get(state.type)
    if paired is None or successor is None:
        return False
    return successor.type in paired


ViewSource: TypeAlias = BaseWorkspace | str | Path | Graph | Iterable[Graph]


def _looks_like_graph_dump(data: Any) -> bool:
    return isinstance(data, dict) and "agent_id" in data and "states" in data


def _load_graphs_from_path(path: str | Path) -> list[Graph]:
    p = Path(path)
    if p.is_dir():
        if Workspace.check_path(p):
            return Workspace.open_path(p).load_steps()

        graph_json = p / "graph.json"
        if not graph_json.exists():
            raise ValueError(
                f"{p} is not a workspace or graph directory (missing graph.json)"
            )

        try:
            data = json.loads(graph_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{graph_json} is not valid JSON: {exc}") from exc

        if _looks_like_graph_dump(data):
            return [Graph.from_dict(data)]

        raise ValueError(
            f"{p} has graph.json, but it is not a workspace manifest "
            "or standalone Graph dump"
        )

    if not p.is_file():
        raise ValueError(f"no such file or directory: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc

    if _looks_like_graph_dump(data):
        return [Graph.from_dict(data)]
    if isinstance(data, list) and all(_looks_like_graph_dump(item) for item in data):
        return [Graph.from_dict(item) for item in data]
    raise ValueError(f"{p} does not look like a workspace or graph dump")


def _visible_signature(graph: Graph) -> tuple:
    """Per-agent display-kind fingerprint for a graph snapshot.

    Two snapshots that produce the same fingerprint render to the
    same figure under the action-collapse rule — for example a tick
    that adds an ``llm_action`` and a tick that replaces it with the
    paired ``llm_output`` both show one ``llm`` node per agent.
    Used to dedupe consecutive viz frames where only bookkeeping
    states changed.
    """
    sig: list[tuple[str, tuple[str, ...]]] = []
    for sub in graph.walk():
        next_in: dict[str, Node] = {}
        for prev, nxt in zip(sub.states[:-1], sub.states[1:]):
            next_in[prev.id] = nxt
        kinds = tuple(
            _display_kind(n)
            for n in sub.states
            if not is_bookkeeping(n, next_in.get(n.id))
        )
        sig.append((sub.agent_id, kinds))
    return tuple(sorted(sig))


def _dedupe_by_visible_signature(graphs: list[Graph]) -> list[Graph]:
    """Drop consecutive snapshots whose visible figure is unchanged."""
    out: list[Graph] = []
    last_sig: tuple | None = None
    for g in graphs:
        sig = _visible_signature(g)
        if sig == last_sig:
            continue
        out.append(g)
        last_sig = sig
    return out


def resolve_graphs(source: ViewSource) -> list[Graph]:
    """Resolve a viewer/export source into graph snapshots.

    ``source`` can be a workspace, workspace path, standalone graph, graph list,
    or standalone graph JSON path.
    """
    if isinstance(source, BaseWorkspace):
        return source.load_steps()
    if isinstance(source, Graph):
        return [source]
    if isinstance(source, (str, Path)):
        return _load_graphs_from_path(source)
    graphs = list(source)
    if not all(isinstance(graph, Graph) for graph in graphs):
        raise TypeError("expected a Graph, Workspace, path, or iterable of Graphs")
    return graphs


def _resolve_latest_graph(source: ViewSource) -> Graph:
    graphs = resolve_graphs(source)
    if not graphs:
        raise ValueError("expected at least one Graph")
    return graphs[-1]


# ── transcripts ──────────────────────────────────────────────────────


def agent_transcript(source: ViewSource, *, include_system: bool = True) -> str:
    """Render one agent's sub-:class:`Graph` as a chat-log transcript.

    Pulls the system prompt + original query off the :class:`Graph`,
    then walks the per-agent state log in order. Used by both the public
    API (``graph[aid].transcript()``) and
    :class:`~rlmflow.workspace.SessionVariable`.
    """
    graph = _resolve_latest_graph(source)
    parts: list[str] = []
    if include_system and graph.system_prompt:
        parts.append(f"--- system ---\n{graph.system_prompt.strip()}")
    if graph.query:
        parts.append(f"--- query ---\n{graph.query.strip()}")
    for state in graph.states:
        rendered = _render_state_transcript(state)
        if rendered is not None:
            parts.append(rendered)
    return "\n\n".join(parts)


def graph_session(source: ViewSource, *, include_system: bool = False) -> str:
    """Render every agent's trajectory in graph order — flat chat-log view.

    Agents appear in spawn order; each agent's states render in their
    own block. ``include_system`` emits the agent's system prompt once
    on first appearance.
    """
    graph = _resolve_latest_graph(source)
    parts: list[str] = []
    for aid in graph.agents:
        sub = graph.agents[aid]
        if include_system and sub.system_prompt:
            parts.append(f"--- [{aid}] system ---\n{sub.system_prompt.strip()}")
        if sub.query:
            parts.append(f"--- [{aid}] query ---\n{sub.query.strip()}")
        for state in sub.states:
            rendered = _render_state_transcript(state, agent_id=aid)
            if rendered is not None:
                parts.append(rendered)
    return "\n\n".join(parts)


def _render_state_transcript(state: Node, *, agent_id: str | None = None) -> str | None:
    prefix = f"[{agent_id}] " if agent_id else ""
    if is_user_query(state):
        body = (state.content or "").strip()
        return f"--- {prefix}query/turn ---\n{body}" if body else None
    if is_llm_output(state):
        body = (state.reply or "").strip()
        return f"--- {prefix}assistant ---\n{body}" if body else None
    if is_supervising(state):
        wait_on = ", ".join(state.waiting_on or [])
        return f"--- {prefix}supervising ---\nwaiting on: {wait_on}"
    if is_done(state):
        body = (state.result or "").strip()
        return f"--- {prefix}result ---\n{body}"
    if is_errored(state):
        body = (state.content or "").strip()
        return f"--- {prefix}error ({state.error}) ---\n{body}"
    if is_resumed(state) and is_exec_output(state):
        resumed_from = ", ".join(state.resumed_from or [])
        body = f"resumed from: {resumed_from or '(none)'}"
        output = (state.output or "").strip()
        if output:
            body += f"\noutput:\n{output}"
        return f"--- {prefix}resume ---\n{body}"
    if is_exec_output(state):
        body = (state.content or state.output or "").strip()
        return f"--- {prefix}observation ---\n{body}" if body else None
    # ActionNode bookkeeping (LLMAction / ExecAction / ResumeAction):
    # not part of a chat-log transcript.
    return None


# ── tree rendering (text) ────────────────────────────────────────────


def _short(text: str, n: int) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _label(s: Node) -> str:
    t = _display_kind(s)
    if is_supervising(s) and s.waiting_on:
        return f"{t} waiting_on={s.waiting_on}"
    if is_done(s) and s.result:
        return f"{t} -> {_short(s.result, 60)}"
    if is_errored(s):
        return f"{t} ({s.error or 'error'})"
    if is_llm_output(s) and s.code:
        return f"{t} code={_short(s.code, 40)}"
    if is_exec_output(s) and (s.content or s.output):
        return f"{t} {_short(s.content or s.output, 60)}"
    return t


def graph_tree(source: ViewSource) -> str:
    """Render a :class:`Graph` as a nested text tree.

    Children are attached visually under the supervising state that
    awaited them when possible, falling back to ``parent_node_id``.
    """
    graph = _resolve_latest_graph(source)
    lines: list[str] = []

    def walk(g: Graph, indent: str) -> None:
        head = f"{indent}● {g.agent_id} ({g.model_label})"
        if g.query:
            head += f" — {_short(g.query, 60)}"
        lines.append(head)

        state_ids = {s.id for s in g.states}
        sup_for_agent: dict[str, str] = {}
        for s in g.states:
            if is_supervising(s):
                for aid in s.waiting_on:
                    sup_for_agent[aid] = s.id

        attach_at: dict[str, list[Graph]] = {}
        unplaced: list[Graph] = []
        for child in g.children.values():
            key = sup_for_agent.get(child.agent_id) or (
                child.parent_node_id if child.parent_node_id in state_ids else None
            )
            if key:
                attach_at.setdefault(key, []).append(child)
            else:
                unplaced.append(child)

        for s in g.states:
            lines.append(f"{indent}  - [{s.seq:>2}] {_label(s)}")
            for child in attach_at.get(s.id, []):
                walk(child, indent + "    ")
        for child in unplaced:
            walk(child, indent + "    ")

    walk(graph, "")
    return "\n".join(lines)


# ── plot palette ─────────────────────────────────────────────────────


_NODE_COLORS: dict[str, str] = {
    "query": "#58a6ff",
    "llm_call": "#a98a2a",
    "llm": "#bc8cff",
    "exec_call": "#b87650",
    "exec": "#ff9e64",
    "supervising": "#ffd33d",
    "resume_call": "#3a8a5d",
    "resume": "#56d4dd",
    "done": "#56d364",
    "errored": "#ff7b72",
}

_NODE_SYMBOLS: dict[str, str] = {
    "query": "circle",
    "llm_call": "diamond-open",
    "llm": "diamond",
    "exec_call": "square-open",
    "exec": "square",
    "supervising": "star",
    "resume_call": "triangle-right-open",
    "resume": "triangle-right",
    "done": "hexagon",
    "errored": "x",
}


# ── plot helpers ─────────────────────────────────────────────────────


def _state_hover_text(state: Node, agent: Graph) -> str:
    rows = [
        f"<b>{_esc_html(agent.agent_id or 'root')}</b>",
        f"<i>{_display_kind(state)}</i> · depth {agent.depth} · seq {state.seq}",
    ]
    if agent.model_label:
        rows.append(f"model: {_esc_html(agent.model_label)}")
    inp, out = agent.tokens()
    if inp or out:
        rows.append(f"agent tokens: {inp + out:,} (in {inp:,} / out {out:,})")
    return "<br>".join(rows)


def _agent_display_label(agent: Graph, *, limit: int = 22) -> str:
    """Short label for an agent's column. Used only for agent-entry
    annotations; the figure does not render any per-state labels."""
    label = agent.agent_id or ""
    if label.startswith("root."):
        label = label[len("root.") :]
    if agent.depth >= 2 and "." in label:
        label = label.rsplit(".", 1)[-1]
    if len(label) > limit:
        label = label[: limit - 1] + "…"
    return label


def _visible_node_count(fig: Any) -> int:
    """Return the plotted state-node count from the marker trace."""

    best = 0
    for trace in getattr(fig, "data", ()):
        customdata = getattr(trace, "customdata", None)
        if customdata is None:
            continue
        try:
            best = max(best, len(customdata))
        except TypeError:
            continue
    return best


def _dense_marker_cap(fig: Any) -> float:
    """Cap marker size using the rendered pixel spacing between rows."""

    layout = getattr(fig, "layout", None)
    if layout is None:
        return float(_DENSE_MAX_MARKER_SIZE)

    height = getattr(layout, "height", None) or 420
    margin = getattr(layout, "margin", None)
    margin_top = getattr(margin, "t", 0) or 0
    margin_bottom = getattr(margin, "b", 0) or 0
    plot_height = max(120.0, float(height) - float(margin_top) - float(margin_bottom))

    yaxis = getattr(layout, "yaxis", None)
    yrange = getattr(yaxis, "range", None)
    if yrange is None or len(yrange) < 2:
        return float(_DENSE_MAX_MARKER_SIZE)
    y_span = abs(float(yrange[1]) - float(yrange[0]))
    if y_span <= 0:
        return float(_DENSE_MAX_MARKER_SIZE)

    row_px = plot_height * _Y_SPACING / y_span
    return max(_DENSE_MIN_MARKER_SIZE, min(_DENSE_MAX_MARKER_SIZE, row_px * 0.6))


def _build_graph_figure(
    graph: Graph,
    *,
    height: int = 360,
    title: str = "execution graph",
):
    """Plotly figure showing every node, colored by type, edges from
    :attr:`Graph.edges`.

    Lay-out is a tidy tree from explicit edges: a node sits centered
    above its children, leaf count determines horizontal slot width.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover - optional dep
        return None

    # An ActionNode is hidden when its successor "absorbs" it — see
    # :func:`is_bookkeeping` above. Same rule that ``retrace_steps`` uses to
    # merge bookkeeping states into the previous tick, so the figure and the
    # slider stay in sync.
    next_in_agent: dict[str, Node] = {}
    for sub in graph.walk():
        states = sub.states
        for prev, nxt in zip(states[:-1], states[1:]):
            next_in_agent[prev.id] = nxt

    nodes = [n for n in graph.nodes if not is_bookkeeping(n, next_in_agent.get(n.id))]
    by_id: dict[str, Node] = {n.id: n for n in nodes}
    # Two kinds of outgoing edges per node:
    #   chain_child[n]: at most one — the next state of the SAME agent (flows_to).
    #   spawn_children[n]: zero or more — first state of each child agent (spawns).
    # Visual layout: the chain child sits directly below its parent (same column),
    # spawn children fan out to either side so the parent's trajectory stays
    # readable as a vertical spine.
    #
    # We deliberately re-root spawn edges from the spawning *action* onto the
    # matching ``SupervisingOutput``. Reason: spawned children only "live" during
    # the supervising state (the parent is yielded waiting on them). Visually
    # attaching them to supervising makes the action → supervising → resume
    # spine readable as a column, with the children fanning out *from* the
    # state that owns their lifecycle. When no supervising is waiting on a
    # child agent (mid-run, before the wait is committed) we fall back to
    # the action that physically spawned it.
    chain_child: dict[str, str] = {}
    spawn_children: dict[str, list[str]] = {n.id: [] for n in nodes}
    parent_of: dict[str, str] = {}

    # agent_id -> id of the SupervisingOutput that waits on it (first wins).
    sup_for_child_agent: dict[str, str] = {}
    for node in nodes:
        if is_supervising(node):
            for aid in node.waiting_on:
                sup_for_child_agent.setdefault(aid, node.id)

    # Rebuild flows_to from each agent's visible states in seq order so
    # collapsed action nodes don't punch holes in the chain.
    for sub in graph.walk():
        prev_id: str | None = None
        for s in sub.states:
            if s.id not in by_id:
                continue
            if prev_id is not None:
                chain_child[prev_id] = s.id
                parent_of[s.id] = prev_id
            prev_id = s.id

    # When a spawning edge points at a hidden action, walk the parent
    # agent's trajectory to find the nearest preceding visible state.
    visible_predecessor: dict[str, str | None] = {}
    for sub in graph.walk():
        prev_visible: str | None = None
        for s in sub.states:
            visible_predecessor[s.id] = prev_visible
            if s.id in by_id:
                prev_visible = s.id

    for edge in graph.edges:
        if edge.kind != "spawns":
            continue
        if edge.to not in by_id or edge.to in parent_of:
            continue
        child_agent_id = by_id[edge.to].agent_id
        attach_to = sup_for_child_agent.get(child_agent_id)
        if attach_to is None:
            attach_to = (
                edge.from_
                if edge.from_ in by_id
                else visible_predecessor.get(edge.from_)
            )
        if attach_to is None:
            continue
        parent_of[edge.to] = attach_to
        spawn_children.setdefault(attach_to, []).append(edge.to)

    def all_children(eid: str) -> list[str]:
        kids = list(spawn_children.get(eid, []))
        if eid in chain_child:
            # Insert the chain child at the middle so the trajectory column
            # stays visually centered under its parent.
            mid = len(kids) // 2
            kids.insert(mid, chain_child[eid])
        return kids

    roots = [eid for eid in by_id if eid not in parent_of]
    roots.sort(key=lambda eid: (graph.agents[by_id[eid].agent_id].depth, eid))

    pos: dict[str, tuple[float, float]] = {}

    def leaf_count(eid: str, seen: set[str]) -> int:
        if eid in seen:
            return 1
        seen.add(eid)
        kids = all_children(eid)
        if not kids:
            return 1
        return sum(leaf_count(k, seen) for k in kids)

    def place(eid: str, left: float, right: float, depth: int, seen: set[str]) -> None:
        if eid in seen:
            return
        seen.add(eid)
        center_x = (left + right) / 2
        pos[eid] = (center_x, -float(depth) * _Y_SPACING)
        kids = all_children(eid)
        if not kids:
            return
        # Tidy tree: partition ``[left, right]`` into one disjoint sub-interval
        # per child, sized by that child's leaf count, and recurse one row down.
        # ``all_children`` already slots the same-agent chain child in the
        # middle, so the agent's own trajectory keeps a near-vertical spine
        # while spawned child agents fan out to either side. Because every
        # subtree (including the chain child's future fanouts, which its leaf
        # count already accounts for) owns a disjoint horizontal band, no two
        # columns can land on top of each other.
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
    for eid in by_id:
        if eid not in pos:
            cursor_left += 1.0
            pos[eid] = (cursor_left, 0.0)

    def shift_agent_subtree(agent_id: str, dx: float) -> None:
        """Shift an agent column and any descendant columns together."""

        prefix = f"{agent_id}."
        for nid, node in by_id.items():
            if nid not in pos:
                continue
            if node.agent_id != agent_id and not node.agent_id.startswith(prefix):
                continue
            x, y = pos[nid]
            pos[nid] = (x + dx, y)

    # Repeated parent fanouts can put a later child column nearly on top of an
    # earlier still-descending child column. Preserve the tree layout, but run a
    # small same-row spacing pass so same-level markers do not visually collide.
    for _ in range(6):
        moved = False
        rows: dict[float, list[str]] = {}
        for nid, (_x, y) in pos.items():
            rows.setdefault(round(y, 6), []).append(nid)
        for row_ids in rows.values():
            row_ids.sort(key=lambda nid: pos[nid][0])
            for left_id, right_id in zip(row_ids, row_ids[1:]):
                left_x = pos[left_id][0]
                right_x = pos[right_id][0]
                gap = right_x - left_x
                if gap >= _MIN_COLUMN_GAP:
                    continue
                left_agent = by_id[left_id].agent_id
                right_agent = by_id[right_id].agent_id
                if left_agent == right_agent:
                    continue
                shift_agent_subtree(right_agent, _MIN_COLUMN_GAP - gap)
                moved = True
        if not moved:
            break

    chain_edge_x: list[float | None] = []
    chain_edge_y: list[float | None] = []
    spawn_edge_x: list[float | None] = []
    spawn_edge_y: list[float | None] = []
    for eid in by_id:
        if eid not in pos:
            continue
        x0, y0 = pos[eid]
        kids = all_children(eid)
        for cid in kids:
            if cid not in pos:
                continue
            x1, y1 = pos[cid]
            if chain_child.get(eid) == cid:
                chain_edge_x.extend([x0, x1, None])
                chain_edge_y.extend([y0, y1, None])
            else:
                spawn_edge_x.extend([x0, x1, None])
                spawn_edge_y.extend([y0, y1, None])

    chain_edge_trace = go.Scatter(
        x=chain_edge_x,
        y=chain_edge_y,
        mode="lines",
        line={"color": "#30363d", "width": 1},
        hoverinfo="skip",
        showlegend=False,
    )
    spawn_edge_trace = go.Scatter(
        x=spawn_edge_x,
        y=spawn_edge_y,
        mode="lines",
        line={"color": "rgba(48,54,61,0.55)", "width": 0.6},
        hoverinfo="skip",
        showlegend=False,
    )

    ordered = [by_id[eid] for eid in pos]
    node_x = [pos[n.id][0] for n in ordered]
    node_y = [pos[n.id][1] for n in ordered]
    agent_counts: dict[str, int] = {}
    for node in ordered:
        agent_counts[node.agent_id] = agent_counts.get(node.agent_id, 0) + 1
    agent_first_id: dict[str, str] = {}
    for node in ordered:
        agent_first_id.setdefault(node.agent_id, node.id)
    is_agent_entry = [agent_first_id.get(n.agent_id) == n.id for n in ordered]
    colors = [_NODE_COLORS.get(_display_kind(n), "#8b949e") for n in ordered]
    symbols = [_NODE_SYMBOLS.get(_display_kind(n), "circle") for n in ordered]
    sizes = [24 for _ in ordered]
    hover = [_state_hover_text(n, graph.agents[n.agent_id]) for n in ordered]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=hover,
        hoverinfo="text",
        customdata=[n.id for n in ordered],
        marker={
            "color": colors,
            "symbol": symbols,
            "size": sizes,
            "line": {"color": "#0d1117", "width": 2.0},
        },
        cliponaxis=False,
        showlegend=False,
    )

    fig = go.Figure(data=[spawn_edge_trace, chain_edge_trace, node_trace])

    # Agent-name headers are emitted as Plotly annotations rather than
    # part of the marker text so we can offset them by a fixed pixel
    # amount (``yshift``) regardless of zoom or data range. The label
    # always floats a stable pixel distance above each agent's first
    # marker — never overlapping the marker, never colliding with the
    # parent's terminal / supervising marker that may sit directly
    # above it. State kinds are conveyed by the legend (color +
    # symbol), so no per-state text is rendered.
    entry_nodes = [node for node, entry in zip(ordered, is_agent_entry) if entry]
    dense_labels = (
        len(ordered) >= _DENSE_LABEL_NODE_THRESHOLD
        or len(entry_nodes) >= _DENSE_LABEL_AGENT_THRESHOLD
    )
    label_limit = 9 if dense_labels else 22
    labeled_bounds_by_row: dict[float, list[tuple[float, float]]] = {}

    for node in entry_nodes:
        x, y = pos[node.id]
        label = _agent_display_label(graph.agents[node.agent_id], limit=label_limit)
        # Dense runs can have dozens of sibling agents on the same row. Keep
        # labels where they fit, and rely on hover text for omitted agent names.
        if dense_labels and node.agent_id != graph.root_agent_id:
            row = round(y, 3)
            half_width = max(0.25, min(1.0, len(label) * 0.06))
            bounds = (x - half_width, x + half_width)
            occupied = labeled_bounds_by_row.setdefault(row, [])
            if any(bounds[0] < hi and bounds[1] > lo for lo, hi in occupied):
                continue
            occupied.append(bounds)
        fig.add_annotation(
            x=x,
            y=y,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            yshift=14 if dense_labels else _AGENT_LABEL_YSHIFT,
            font={
                "color": _AGENT_LABEL_COLOR,
                "family": "ui-monospace, SFMono-Regular, Menlo, monospace",
                "size": 9 if dense_labels else 11,
            },
        )
    visible_kinds = {_display_kind(n) for n in ordered}
    for ntype, color in _NODE_COLORS.items():
        if ntype not in visible_kinds:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "color": color,
                    "symbol": _NODE_SYMBOLS.get(ntype, "circle"),
                    "size": 14,
                    "line": {"color": "#0d1117", "width": 1},
                },
                name=ntype,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    n_edges = sum(1 for e in chain_edge_x + spawn_edge_x if e is None)
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
            "text": f"<b>{title}</b> · {len(ordered)} states · {n_edges} edges",
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


def _scale_figure_elements(
    fig: Any,
    marker_mult: float,
    text_mult: float | None = None,
) -> None:
    """Multiply marker and font pixel sizes in-place."""
    if text_mult is None:
        text_mult = marker_mult
    if marker_mult == 1.0 and text_mult == 1.0:
        return
    if fig is None:
        return
    node_count = _visible_node_count(fig)
    dense_graph = node_count >= _DENSE_NODE_THRESHOLD
    dense_marker_cap = _dense_marker_cap(fig) if dense_graph else None

    def scale_marker_size(size: Any) -> Any:
        scaled = size * marker_mult
        if dense_marker_cap is not None:
            return min(scaled, dense_marker_cap)
        return scaled

    for trace in getattr(fig, "data", ()):
        marker = getattr(trace, "marker", None)
        if marker is not None:
            size = getattr(marker, "size", None)
            if isinstance(size, (list, tuple)):
                marker.size = [scale_marker_size(s) for s in size]
            elif isinstance(size, (int, float)):
                marker.size = scale_marker_size(size)
            mline = getattr(marker, "line", None)
            if mline is not None and getattr(mline, "width", None) is not None:
                scaled = mline.width * marker_mult
                mline.width = (
                    min(scaled, _DENSE_MAX_MARKER_LINE_WIDTH) if dense_graph else scaled
                )
        textfont = getattr(trace, "textfont", None)
        if textfont is not None and getattr(textfont, "size", None) is not None:
            textfont.size = textfont.size * text_mult
    layout = getattr(fig, "layout", None)
    if layout is None:
        return
    if layout.title and layout.title.font and layout.title.font.size:
        layout.title.font.size = layout.title.font.size * text_mult
    if layout.legend and layout.legend.font and layout.legend.font.size:
        layout.legend.font.size = layout.legend.font.size * text_mult
    if layout.font and layout.font.size:
        layout.font.size = layout.font.size * text_mult
    # Agent-name labels live in annotations now, so ``text_mult`` has
    # to scale those too.
    for ann in getattr(layout, "annotations", ()) or ():
        font = getattr(ann, "font", None)
        if font is not None and getattr(font, "size", None) is not None:
            scaled = font.size * text_mult
            font.size = (
                min(scaled, _DENSE_MAX_AGENT_LABEL_SIZE) if dense_graph else scaled
            )


def _normalize_label_positions(fig: Any) -> None:
    """Force every node label to ``bottom center``."""
    if fig is None:
        return
    for trace in getattr(fig, "data", ()):
        mode = getattr(trace, "mode", None) or ""
        if "text" not in mode:
            continue
        positions = getattr(trace, "textposition", None)
        if positions is None:
            continue
        if isinstance(positions, str):
            trace.textposition = "bottom center"
        else:
            trace.textposition = tuple(
                "bottom center" if (p or "").startswith("top") else p for p in positions
            )


# ── plot public API ──────────────────────────────────────────────────


def graph_plot(
    source: ViewSource,
    kind: str = "graph",
    *,
    height: int = 420,
    title: str | None = None,
    include_results: bool = True,
    element_mult: float = 1.0,
    marker_mult: float | None = None,
    text_mult: float | None = None,
    normalize_labels: bool = False,
):
    """Render a workspace, path, graph, or graph list in one of several formats.

    ``kind`` may be ``"graph"`` / ``"plotly"`` (Plotly figure),
    ``"mermaid"`` / ``"flowchart"`` / ``"sequence"`` / ``"dot"`` / ``"d2"``
    (string formats), ``"tree"`` (plain text), or ``"gantt"`` (HTML).
    """
    fmt = kind.lower().replace("-", "_")
    if fmt in {"gantt", "swimlane", "swimlanes"}:
        from rlmflow.utils.viz import gantt_html

        return gantt_html(resolve_graphs(source), title=title or "rlmflow gantt")

    graph = _resolve_latest_graph(source)
    if fmt in {"mermaid", "state", "state_diagram"}:
        from rlmflow.utils.export import to_mermaid

        return to_mermaid(graph, include_results=include_results)
    if fmt in {"flow", "flowchart", "mermaid_flowchart"}:
        from rlmflow.utils.export import to_mermaid_flowchart

        return to_mermaid_flowchart(graph, include_results=include_results)
    if fmt in {"sequence", "sequence_diagram"}:
        from rlmflow.utils.export import to_mermaid_sequence

        return to_mermaid_sequence(graph)
    if fmt == "dot":
        from rlmflow.utils.export import to_dot

        return to_dot(graph, include_results=include_results)
    if fmt == "d2":
        from rlmflow.utils.export import to_d2

        return to_d2(graph, include_results=include_results)
    if fmt in {"tree", "ascii"}:
        return graph.tree()
    if fmt not in {"graph", "plotly", "viewer"}:
        raise ValueError(
            f"Unknown plot kind {kind!r}. Supported: "
            "graph, mermaid, flowchart, sequence, dot, d2, tree, gantt."
        )

    fig = _build_graph_figure(
        graph,
        height=height,
        title=title
        or f"{graph.root_agent_id} · {graph.current().type if graph.current() else 'empty'}",
    )
    if fig is None:
        raise ImportError(
            "Graph.plot() requires the viewer extra: `pip install rlmflow[viewer]`."
        )
    if normalize_labels:
        _normalize_label_positions(fig)
    resolved_marker = marker_mult if marker_mult is not None else element_mult
    resolved_text = text_mult if text_mult is not None else element_mult
    _scale_figure_elements(fig, resolved_marker, resolved_text)
    return fig


def graph_plot_html(
    source: ViewSource,
    kind: str = "graph",
    *,
    height: int = 420,
    title: str | None = None,
    include_results: bool = True,
    include_plotlyjs: str | bool = "cdn",
    element_mult: float = 1.0,
    marker_mult: float | None = None,
    text_mult: float | None = None,
    normalize_labels: bool = False,
) -> str:
    rendered = graph_plot(
        source,
        kind,
        height=height,
        title=title,
        include_results=include_results,
        element_mult=element_mult,
        marker_mult=marker_mult,
        text_mult=text_mult,
        normalize_labels=normalize_labels,
    )
    if hasattr(rendered, "to_html"):
        return rendered.to_html(include_plotlyjs=include_plotlyjs, full_html=False)
    if kind.lower().replace("-", "_") in {"gantt", "swimlane", "swimlanes"}:
        return str(rendered)
    return (
        '<pre style="background:#0d1117;color:#c9d1d9;padding:12px;'
        'border-radius:6px;overflow:auto;">'
        f"{_esc_html(str(rendered))}</pre>"
    )


# ── exports ──────────────────────────────────────────────────────────


_DEFAULT_EXPORT_MARGIN = {"l": 24, "r": 24, "t": 80, "b": 120}


def save_image(
    source: ViewSource,
    path: str | Path,
    *,
    width: int = 1800,
    height: int = 1350,
    scale: float = 2.0,
    element_mult: float = 1.0,
    marker_mult: float | None = None,
    text_mult: float | None = None,
    normalize_labels: bool = True,
    margin: dict | None = None,
    title: str | None = None,
) -> Path:
    """Render ``source`` to a PNG/SVG/PDF file. Requires ``kaleido``."""
    graph = _resolve_latest_graph(source)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = graph_plot(
        graph,
        "graph",
        height=height,
        title=title,
        element_mult=element_mult,
        marker_mult=marker_mult,
        text_mult=text_mult,
        normalize_labels=normalize_labels,
    )
    fig.update_layout(margin=margin if margin is not None else _DEFAULT_EXPORT_MARGIN)
    try:
        fig.write_image(out, width=width, height=height, scale=scale)
    except (ImportError, ValueError) as exc:
        raise ImportError(
            "save_image() requires the `kaleido` package: `pip install kaleido`."
        ) from exc
    return out


def save_steps(
    source: ViewSource,
    out_dir: str | Path,
    *,
    fmt: str = "png",
    width: int = 1800,
    height: int = 1350,
    scale: float = 2.0,
    element_mult: float = 1.0,
    marker_mult: float | None = None,
    text_mult: float | None = None,
    normalize_labels: bool = True,
    margin: dict | None = None,
    title_template: str = "step {i} / {total}: {agent_id} [{type}]",
) -> Path:
    """Save one image per snapshot in ``source`` under ``out_dir``.

    Consecutive snapshots that produce an identical visible figure
    (same set of non-bookkeeping nodes after the action-collapse rule)
    are deduplicated — only the first is written. This keeps the
    frame count tied to *visible* progress, not raw state-appends.
    """
    graphs = resolve_graphs(source)
    graphs = _dedupe_by_visible_signature(graphs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not graphs:
        return out
    total = len(graphs) - 1
    for i, graph in enumerate(graphs):
        current = graph.current()
        title = title_template.format(
            i=i,
            total=total,
            agent_id=graph.root_agent_id,
            type=current.type if current else "empty",
        )
        save_image(
            graph,
            out / f"step_{i:02d}.{fmt}",
            width=width,
            height=height,
            scale=scale,
            element_mult=element_mult,
            marker_mult=marker_mult,
            text_mult=text_mult,
            normalize_labels=normalize_labels,
            margin=margin,
            title=title,
        )
    return out


# ── HTML stepper ─────────────────────────────────────────────────────


_HTML_STYLES = """
:root {
  color-scheme: dark;
  --bg: #0d1117;
  --panel: #161b22;
  --border: #30363d;
  --text: #c9d1d9;
  --muted: #8b949e;
  --blue: #58a6ff;
  --green: #3fb950;
  --purple: #bc8cff;
  --yellow: #d29922;
  --orange: #ff9e64;
  --red: #ff7b72;
}
body {
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
  padding: 2rem;
}
main { margin: 0 auto; max-width: 1400px; }
h1, h2, h3 { margin-top: 0; }
.slide {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  display: none;
  margin: 1rem 0;
  padding: 1rem;
}
.slide.active { display: block; }
.slide-head {
  align-items: baseline;
  display: flex;
  gap: 0.75rem;
  justify-content: space-between;
}
.step { color: var(--muted); white-space: nowrap; }
.viewer-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: minmax(0, 1.7fr) minmax(300px, 0.7fr);
}
.graph-card, .detail-card { min-width: 0; }
pre {
  background: #0d1117;
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: auto;
  padding: 1rem;
}
table { border-collapse: collapse; width: 100%; }
th, td {
  border-bottom: 1px solid var(--border);
  padding: 0.45rem;
  text-align: left;
  vertical-align: top;
}
th { color: var(--muted); font-weight: 600; }
code { color: var(--blue); }
.pill {
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted);
  display: inline-block;
  padding: 0.1rem 0.45rem;
}
.kind-done { color: var(--green); border-color: var(--green); }
.kind-supervising { color: var(--yellow); border-color: var(--yellow); }
.kind-llm { color: var(--purple); border-color: var(--purple); }
.kind-llm_call { color: var(--purple); border-color: var(--purple); opacity: 0.7; }
.kind-resume { color: var(--blue); border-color: var(--blue); }
.kind-resume_call { color: var(--green); border-color: var(--green); opacity: 0.7; }
.kind-errored { color: var(--red); border-color: var(--red); }
.kind-query { color: var(--blue); border-color: var(--blue); }
.kind-exec { color: var(--orange); border-color: var(--orange); }
.kind-exec_call { color: var(--orange); border-color: var(--orange); opacity: 0.7; }
.nav {
  align-items: center;
  display: flex;
  gap: 0.75rem;
  justify-content: center;
  margin-top: 1rem;
}
.arrow, .dot { cursor: pointer; }
.arrow {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--text);
  height: 2.2rem;
  width: 2.2rem;
}
.dot {
  background: #30363d;
  border: 1px solid #484f58;
  border-radius: 999px;
  height: 0.7rem;
  width: 0.7rem;
}
.dot.active { background: var(--blue); border-color: var(--blue); }
@media (max-width: 900px) {
  .viewer-grid { grid-template-columns: 1fr; }
  body { padding: 1rem; }
}
""".strip()

_HTML_SCRIPT = """
const slides = [...document.querySelectorAll(".slide")];
const dots = [...document.querySelectorAll(".dot")];
let current = 0;
function show(index) {
  current = (index + slides.length) % slides.length;
  slides.forEach((s, i) => s.classList.toggle("active", i === current));
  dots.forEach((d, i) => d.classList.toggle("active", i === current));
}
document.getElementById("prev").addEventListener("click", () => show(current - 1));
document.getElementById("next").addEventListener("click", () => show(current + 1));
dots.forEach((dot, idx) => dot.addEventListener("click", () => show(idx)));
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") show(current - 1);
  if (event.key === "ArrowRight") show(current + 1);
});
""".strip()


def _state_table_html(graph: Graph) -> str:
    rows: list[str] = []
    for node in graph.nodes:
        detail = (
            getattr(node, "result", "")
            or ", ".join(getattr(node, "waiting_on", []) or [])
            or graph.agents[node.agent_id].query
            or getattr(node, "content", "")
            or ""
        )
        rows.append(
            "<tr>"
            f"<td><code>{_esc_html(node.agent_id)}</code></td>"
            f"<td><span class='pill kind-{_esc_html(_display_kind(node))}'>"
            f"{_esc_html(_display_kind(node))}</span></td>"
            f"<td>{_esc_html(str(detail)[:120])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_html(
    source: ViewSource,
    *,
    title: str = "rlmflow run",
    height: int = 720,
    include_plotlyjs: str | bool = "cdn",
    element_mult: float = 1.0,
    marker_mult: float | None = None,
    text_mult: float | None = None,
    normalize_labels: bool = True,
) -> str:
    """Render ``source`` as a single self-contained HTML stepper."""
    graphs = _dedupe_by_visible_signature(resolve_graphs(source))
    if not graphs:
        raise ValueError("render_html() needs at least one graph")

    slides: list[str] = []
    dots: list[str] = []
    total = len(graphs)
    for idx, graph in enumerate(graphs, start=1):
        active = " active" if idx == 1 else ""
        plotly_inc = include_plotlyjs if idx == 1 else False
        graph_html = graph_plot_html(
            graph,
            "graph",
            height=height,
            title=f"step {idx} / {total}",
            include_plotlyjs=plotly_inc,
            element_mult=element_mult,
            marker_mult=marker_mult,
            text_mult=text_mult,
            normalize_labels=normalize_labels,
        )
        current = graph.current()
        head_title = (
            f"{graph.root_agent_id} \u00b7 {current.type if current else 'empty'}"
        )
        slide_body = (
            f'<section class="slide{active}" data-step="{idx}">'
            '<div class="slide-head">'
            f'<span class="step">Step {idx} / {total}</span>'
            f"<h2>{_esc_html(head_title)}</h2>"
            "</div>"
            '<div class="viewer-grid">'
            f'<div class="graph-card">{graph_html}</div>'
            '<aside class="detail-card">'
            f"<h3>Transcript: <code>{_esc_html(graph.root_agent_id)}</code></h3>"
            f"<pre>{_esc_html(graph.transcript(include_system=False))}</pre>"
            "<h3>States</h3>"
            "<table>"
            "<thead><tr><th>Agent</th><th>Type</th><th>Detail</th></tr></thead>"
            f"<tbody>{_state_table_html(graph)}</tbody>"
            "</table>"
            "</aside>"
            "</div>"
            "</section>"
        )
        slides.append(slide_body)
        dots.append(
            f'<button class="dot{active}" data-step="{idx}" '
            f'aria-label="Step {idx}"></button>'
        )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc_html(title)}</title>\n"
        f"<style>{_HTML_STYLES}</style>\n"
        "</head>\n"
        "<body>\n"
        "<main>\n"
        f"<h1>{_esc_html(title)}</h1>\n" + "".join(slides) + '\n<div class="nav">\n'
        '<button class="arrow" id="prev" aria-label="Previous step">&larr;</button>\n'
        + "".join(dots)
        + '\n<button class="arrow" id="next" aria-label="Next step">&rarr;</button>\n'
        "</div>\n"
        "</main>\n"
        f"<script>{_HTML_SCRIPT}</script>\n"
        "</body>\n"
        "</html>\n"
    )


def save_html(
    source: ViewSource,
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write :func:`render_html` output to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(source, **kwargs), encoding="utf-8")
    return out


def save_gif(
    source: ViewSource,
    path: str | Path,
    *,
    duration: int = 600,
    loop: int = 0,
    width: int = 1200,
    height: int = 900,
    scale: float = 1.0,
    element_mult: float = 1.0,
    marker_mult: float | None = None,
    text_mult: float | None = None,
    normalize_labels: bool = True,
    margin: dict | None = None,
    title_template: str = "step {i} / {total}: {agent_id} [{type}]",
) -> Path:
    """Stitch every graph snapshot into an animated GIF."""
    graphs = _dedupe_by_visible_signature(resolve_graphs(source))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not graphs:
        raise ValueError("save_gif() needs at least one graph")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("save_gif() requires Pillow: `pip install pillow`.") from exc

    import io

    frames: list[Image.Image] = []
    total = len(graphs) - 1
    for i, graph in enumerate(graphs):
        current = graph.current()
        title = title_template.format(
            i=i,
            total=total,
            agent_id=graph.root_agent_id,
            type=current.type if current else "empty",
        )
        fig = graph_plot(
            graph,
            "graph",
            height=height,
            title=title,
            element_mult=element_mult,
            marker_mult=marker_mult,
            text_mult=text_mult,
            normalize_labels=normalize_labels,
        )
        fig.update_layout(
            margin=margin if margin is not None else _DEFAULT_EXPORT_MARGIN
        )
        try:
            png_bytes = fig.to_image(
                format="png",
                width=width,
                height=height,
                scale=scale,
            )
        except (ImportError, ValueError) as exc:
            raise ImportError(
                "save_gif() needs `kaleido` for PNG rendering: "
                "`pip install kaleido`."
            ) from exc
        frames.append(
            Image.open(io.BytesIO(png_bytes)).convert("P", palette=Image.ADAPTIVE)
        )

    head, *tail = frames
    head.save(
        out,
        format="GIF",
        save_all=True,
        append_images=tail,
        duration=duration,
        loop=loop,
        optimize=True,
        disposal=2,
    )
    return out


# ── Gradio viewer ────────────────────────────────────────────────────


def open_viewer(source: ViewSource, **launch_kwargs: Any):
    """Open the Gradio stepper over a workspace, path, graph, or graph list."""
    import gradio as gr

    graphs = _dedupe_by_visible_signature(resolve_graphs(source))
    if not graphs:
        raise ValueError("open_viewer needs at least one Graph")

    n_steps = len(graphs)

    def _coerce_step(value: Any) -> int:
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        return max(0, min(int(value), n_steps - 1))

    def _fig_json_for_step(step: int) -> str:
        step = _coerce_step(step)
        graph = graphs[step]
        fig = _build_graph_figure(
            graph,
            height=420,
            title=f"step {step + 1} / {n_steps}",
        )
        return fig.to_json() if fig is not None else "{}"

    def get_step_fig(*args: Any) -> str:
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
        graph = graphs[step]
        node = graph.nodes.find(node_id)
        if node is None:
            return "<i style='color:#8b949e'>(node not visible at this step)</i>"
        return _state_detail_html(node, graph)

    initial_fig_json = _fig_json_for_step(n_steps - 1)
    initial_detail = (
        "<i style='color:#8b949e'>Click any node above to see its full payload.</i>"
    )

    total_nodes = sum(len(g.nodes) for g in graphs)

    with gr.Blocks(title="RLMFlow Viewer", fill_height=True) as demo:
        gr.Markdown(
            f"### RLMFlow Viewer · {n_steps} steps · "
            f"{total_nodes} nodes\n"
            "Drag the slider to scrub the graph through time. "
            "Click any node to see its step-local payload + conversation."
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
            js_on_load=_GRAPH_JS_ON_LOAD,
            server_functions=[get_step_fig, get_node_detail],
            initial_fig_json=initial_fig_json,
            initial_detail=initial_detail,
            initial_step=n_steps - 1,
            min_height=720,
        )

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


# ── viewer HTML/CSS/JS templates ─────────────────────────────────────


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
.rlmflow-detail .pill.type-query       { color: #58a6ff; border-color: #58a6ff; }
.rlmflow-detail .pill.type-llm         { color: #bc8cff; border-color: #bc8cff; }
.rlmflow-detail .pill.type-exec        { color: #ff9e64; border-color: #ff9e64; }
.rlmflow-detail .pill.type-supervising { color: #ffd33d; border-color: #ffd33d; }
.rlmflow-detail .pill.type-resume      { color: #56d4dd; border-color: #56d4dd; }
.rlmflow-detail .pill.type-done        { color: #56d364; border-color: #56d364; }
.rlmflow-detail .pill.type-errored     { color: #ff7b72; border-color: #ff7b72; }
.rlmflow-detail .payload-block {
    margin: 8px 0; padding: 8px 10px;
    background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
}
.rlmflow-detail .payload-block pre {
    margin-top: 6px; background: #161b22;
}
.rlmflow-detail .payload-label {
    color: #8b949e; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-family: -apple-system, system-ui, sans-serif;
}
.rlmflow-detail .state-blocks {
    display: flex; flex-direction: column; gap: 10px;
}
.rlmflow-detail .state-block {
    border: 1px solid #30363d; border-left: 3px solid #30363d;
    border-radius: 6px;
    background: #0d1117; padding: 10px 12px;
}
.rlmflow-detail .state-block-query       { border-left-color: #58a6ff; background: #58a6ff0a; }
.rlmflow-detail .state-block-llm         { border-left-color: #bc8cff; background: #bc8cff0a; }
.rlmflow-detail .state-block-llm_call    { border-left-color: #a98a2a; background: #a98a2a0a; }
.rlmflow-detail .state-block-exec        { border-left-color: #ff9e64; background: #ff9e640a; }
.rlmflow-detail .state-block-exec_call   { border-left-color: #b87650; background: #b876500a; }
.rlmflow-detail .state-block-supervising { border-left-color: #ffd33d; background: #ffd33d0a; }
.rlmflow-detail .state-block-resume      { border-left-color: #56d4dd; background: #56d4dd0a; }
.rlmflow-detail .state-block-resume_call { border-left-color: #3a8a5d; background: #3a8a5d0a; }
.rlmflow-detail .state-block-done        { border-left-color: #56d364; background: #56d3640a; }
.rlmflow-detail .state-block-errored     { border-left-color: #ff7b72; background: #ff7b720a; }
.rlmflow-detail .state-block-selected {
    box-shadow: 0 0 0 1px currentColor;
    border-top-color: currentColor;
    border-right-color: currentColor;
    border-bottom-color: currentColor;
}
.rlmflow-detail .state-block-query.state-block-selected       { color: #58a6ff; }
.rlmflow-detail .state-block-llm.state-block-selected         { color: #bc8cff; }
.rlmflow-detail .state-block-llm_call.state-block-selected    { color: #a98a2a; }
.rlmflow-detail .state-block-exec.state-block-selected        { color: #ff9e64; }
.rlmflow-detail .state-block-exec_call.state-block-selected   { color: #b87650; }
.rlmflow-detail .state-block-supervising.state-block-selected { color: #ffd33d; }
.rlmflow-detail .state-block-resume.state-block-selected      { color: #56d4dd; }
.rlmflow-detail .state-block-resume_call.state-block-selected { color: #3a8a5d; }
.rlmflow-detail .state-block-done.state-block-selected        { color: #56d364; }
.rlmflow-detail .state-block-errored.state-block-selected     { color: #ff7b72; }
.rlmflow-detail .state-block-head {
    display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px;
}
.rlmflow-detail .state-block .payload-block {
    margin: 6px 0 0; background: #161b22;
}
.rlmflow-detail .payload-empty {
    color: #6e7681; font-style: italic; font-size: 11px;
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
                    const sel = detail.querySelector('.state-block-selected');
                    if (sel) sel.scrollIntoView({block: 'nearest'});
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


# ── state detail HTML (clicked-node panel) ───────────────────────────


# Per-node-type ordered field list. Each tuple is (attr, heading).
# Empty values are skipped at render time.
_DETAIL_FIELDS: dict[str, list[tuple[str, str]]] = {
    "user_query": [("content", "query")],
    "llm_action": [("model", "model")],
    "llm_output": [("reply", "reply"), ("code", "code")],
    "exec_action": [("code", "code")],
    "exec_output": [("output", "output"), ("content", "rendered")],
    "supervising_output": [("waiting_on", "waiting on"), ("output", "output")],
    "error_output": [
        ("error", "error kind"),
        ("content", "retry message"),
        ("output", "raw output"),
    ],
    "done_output": [("result", "result"), ("output", "output")],
    "resume_action": [
        ("resumed_from", "resumed from"),
        ("code", "code"),
    ],
}


def _render_state_block(state: Node, *, selected: bool) -> str:
    """Render a single state as a labeled block of its typed fields."""
    fields = _DETAIL_FIELDS.get(state.type, [])
    body_parts: list[str] = []
    for attr, heading in fields:
        val = getattr(state, attr, None)
        if val is None or val == "" or val == []:
            continue
        if isinstance(val, list):
            text = "\n".join(str(x) for x in val)
        else:
            text = str(val)
        if not text.strip():
            continue
        body_parts.append(
            f"<div class='payload-block'>"
            f"<div class='payload-label'>{_esc_html(heading)}</div>"
            f"<pre>{_esc_html(text)}</pre>"
            f"</div>"
        )
    if not body_parts:
        body_parts.append("<div class='payload-block payload-empty'>(no payload)</div>")

    kind = _display_kind(state)
    pills = (
        f"<span class='pill kind-{_esc_html(kind)}'>{_esc_html(kind)}</span>"
        f"<span class='pill'>seq {state.seq}</span>"
        f"<span class='pill'>{_esc_html(state.type)}</span>"
    )
    sel_class = " state-block-selected" if selected else ""
    kind_class = f" state-block-{_esc_html(kind)}"
    return (
        f"<section class='state-block{kind_class}{sel_class}' "
        f"id='state-block-{_esc_html(state.id)}'>"
        f"<header class='state-block-head'>{pills}</header>"
        f"{''.join(body_parts)}"
        f"</section>"
    )


def _state_detail_html(state: Node, graph: Graph) -> str:
    agent = graph.agents[state.agent_id]
    parts: list[str] = []
    kind = _display_kind(state)
    pill = f'<span class="pill type-{_esc_html(kind)}">{_esc_html(kind)}</span>'
    inp, out = agent.tokens()
    parts.append(
        f"""
<div class="header">
  <div>
    <h4>{_esc_html(agent.agent_id or "root")} {pill}</h4>
    <div>
      <span class="pill">type {_esc_html(state.type)}</span>
      <span class="pill">id {_esc_html(state.id)}</span>
      <span class="pill">model {_esc_html(agent.model_label)}</span>
      <span class="pill">depth {agent.depth}</span>
      <span class="pill">seq {state.seq}</span>
      <span class="pill">agent tokens {inp + out:,}</span>
      {'<span class="pill" style="color:#7ee787;border-color:#7ee787">terminal</span>' if state.terminal else ''}
    </div>
  </div>
</div>
"""
    )

    if agent.query:
        parts.append("<h5>agent query</h5>")
        parts.append(f"<pre>{_esc_html(agent.query)}</pre>")

    parts.append(f"<h5>states ({len(agent.states)})</h5>" "<div class='state-blocks'>")
    for s in agent.states:
        parts.append(_render_state_block(s, selected=s.id == state.id))
    parts.append("</div>")

    return "".join(parts)


__all__ = [
    "agent_transcript",
    "graph_plot",
    "graph_plot_html",
    "graph_session",
    "open_viewer",
    "render_html",
    "resolve_graphs",
    "save_gif",
    "save_html",
    "save_image",
    "save_steps",
]
