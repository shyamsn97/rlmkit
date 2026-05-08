"""Live tree visualization and simple swimlanes for RLMFlow nodes."""

from __future__ import annotations

import difflib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from rlmflow.node import Node, parse_node_json
from rlmflow.workspace.session import FileSession, Session


def _node_label(state: Node):
    from rich.text import Text

    active = not state.finished
    info = Text()
    info.append("• " if active else "ok ", style="magenta" if active else "green")
    info.append(state.agent_id or "root", style="bold")
    info.append(f" [{state.type}]", style="magenta" if active else "green")
    info.append(f" {{{state.model_label}}}", style="cyan")
    result = getattr(state, "result", None)
    if result:
        info.append(f" -> {str(result)[:80].replace(chr(10), ' ')}", style="dim green")
    return info


def _build_tree(state: Node):
    from rich.tree import Tree

    tree = Tree(_node_label(state), guide_style="dim")
    for child in state.child_nodes():
        tree.add(_build_tree(child))
    return tree


class LiveView:
    """Live-updating Rich tree of an RLMFlow state. Use as a context manager.

    Decouples display from the step loop — you own the agent loop, this just
    re-renders whichever node you hand it.

    Usage::

        state = agent.start(query)
        states = [state]
        with live_view() as view:
            view(state)
            while not state.finished:
                state = agent.step(state)
                states.append(state)
                view(state)
    """

    def __init__(self, *, console: Any = None) -> None:
        from rich.console import Console

        self._console = console or Console()
        self._live: Any = None

    def __enter__(self) -> LiveView:
        from rich.live import Live

        # auto_refresh=False keeps rich.Live from publishing background frames in
        # Jupyter — we already repaint exactly once per view(state) call. Combined
        # with no animated renderables (no Spinner) inside the tree, this avoids
        # the publish-display → stderr → file_proxy recursion in IPython.
        self._live = Live(
            console=self._console,
            vertical_overflow="visible",
            auto_refresh=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    def __call__(self, state: Node | Iterable[Node]) -> None:
        """Re-render the tree. Accepts a single Node or an iterable (uses the last)."""
        if self._live is None:
            raise RuntimeError("live_view used outside of context")
        node = state if isinstance(state, Node) else list(state)[-1]
        self._live.update(_build_tree(node), refresh=True)


def live_view(**kwargs: Any) -> LiveView:
    """Construct a :class:`LiveView`. Use as ``with live_view() as view:``."""
    return LiveView(**kwargs)


def live(agent: Any, state: Node) -> list[Node]:
    """Run ``agent``'s step loop while streaming a live tree. Returns every state.

    Convenience wrapper around :class:`LiveView`. For more control over the
    loop, use :func:`live_view` directly inside your own ``while not finished``.
    """
    states = [state]
    with LiveView() as view:
        view(state)
        while not state.finished:
            state = agent.step(state)
            states.append(state)
            view(state)
    return states


def _flatten_state(state: Node) -> list[Node]:
    out = [state]
    for child in state.child_nodes():
        out.extend(_flatten_state(child))
    return out


def gantt_matrix(states: list[Node]) -> tuple[list[str], list[list[str | None]]]:
    order: list[str] = []
    seen: set[str] = set()
    per_step: list[dict[str, str]] = []
    for state in states:
        step_map: dict[str, str] = {}
        for node in _flatten_state(state):
            aid = node.agent_id or "root"
            step_map[aid] = f"{node.type} ({node.model_label})"
            if aid not in seen:
                seen.add(aid)
                order.append(aid)
        per_step.append(step_map)
    return order, [[step.get(aid) for step in per_step] for aid in order]


_TYPE_CELL = {
    "query": ("Q", "blue"),
    "observation": ("O", "blue"),
    "action": ("A", "yellow"),
    "supervising": ("S", "magenta"),
    "resume": ("R", "green"),
    "error": ("E", "red"),
    "result": ("F", "green"),
}

_TYPE_HTML = {
    "query": "#58a6ff",
    "observation": "#58a6ff",
    "action": "#d29922",
    "supervising": "#bc8cff",
    "resume": "#7ee787",
    "error": "#f85149",
    "result": "#3fb950",
}


def gantt(states: list[Node]) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    agents, rows = gantt_matrix(states)
    table = Table(
        show_header=True,
        header_style="dim",
        show_lines=False,
        pad_edge=False,
        padding=(0, 0),
    )
    table.add_column("agent", style="bold", no_wrap=True)
    for i in range(len(states)):
        table.add_column(str(i), justify="center", no_wrap=True)
    for aid, row in zip(agents, rows):
        cells = [Text(aid)]
        for kind in row:
            if kind is None:
                cells.append(Text(" ", style="dim"))
            else:
                node_type = kind.split(" ", 1)[0]
                glyph, color = _TYPE_CELL.get(node_type, ("?", "dim"))
                cells.append(Text(glyph, style=color))
        table.add_row(*cells)
    Console().print(table)


def gantt_html(states: list[Node], *, title: str = "rlmflow gantt") -> str:
    agents, rows = gantt_matrix(states)
    n_steps = len(states)
    cells_html: list[str] = []
    for aid, row in zip(agents, rows):
        cells_html.append(
            f'<div class="row"><div class="name">{aid}</div>'
            f'<div class="bars" style="grid-template-columns: repeat({n_steps}, 1fr)">'
        )
        for kind in row:
            if kind is None:
                cells_html.append('<div class="cell empty"></div>')
            else:
                node_type = kind.split(" ", 1)[0]
                color = _TYPE_HTML.get(node_type, "#8b949e")
                cells_html.append(
                    f'<div class="cell" style="background:{color}" title="{kind}"></div>'
                )
        cells_html.append("</div></div>")
    legend = "".join(
        f'<span><i style="background:{c}"></i>{s}</span>' for s, c in _TYPE_HTML.items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0d1117; color: #e6edf3; margin: 24px; }}
h1 {{ font-size: 14px; color: #8b949e; font-weight: 500; margin: 0 0 12px; }}
.row {{ display: grid; grid-template-columns: 220px 1fr; gap: 8px; align-items: center; margin: 2px 0; }}
.name {{ font-family: monospace; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bars {{ display: grid; gap: 1px; height: 18px; background: #161b22; border: 1px solid #30363d; border-radius: 3px; overflow: hidden; }}
.cell {{ height: 100%; }} .cell.empty {{ background: transparent; }}
.legend {{ margin-top: 16px; font-size: 12px; color: #8b949e; }} .legend span {{ margin-right: 16px; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; margin-right: 4px; border-radius: 2px; vertical-align: middle; }}
</style></head><body>
<h1>{title} - {n_steps} steps, {len(agents)} agents</h1>
{"".join(cells_html)}
<div class="legend">{legend}</div>
</body></html>"""


# ── per-node detail ──────────────────────────────────────────────────


def error_summary(state: Node) -> str:
    """Group every ``ErrorNode`` in the subtree by ``error`` kind."""
    errors = state.errors()
    if not errors:
        return "(no errors)"
    by_kind: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for node in errors:
        kind = getattr(node, "error", "") or "(unknown)"
        by_kind[kind] += 1
        if kind not in samples:
            samples[kind] = (
                (getattr(node, "content", "") or "").strip().splitlines()[0:1]
            )
            samples[kind] = samples[kind][0] if samples[kind] else ""
    lines = [f"{len(errors)} error(s) across {len(by_kind)} kind(s):"]
    for kind, count in by_kind.most_common():
        lines.append(f"  {kind}: {count}")
        if samples.get(kind):
            lines.append(f"    └─ {samples[kind][:120]}")
    return "\n".join(lines)


def code_log(
    source: Node | list[Node] | Iterable[Node],
    agent_id: str | None = None,
) -> str:
    """Render every code block executed in the run, paired with its output.

    Pass a step history (``list[Node]`` from the step loop), a single
    state, or any iterable of nodes (e.g. ``agent.session.load().values()``).
    Observations, supervising nodes, and error nodes carry the code that
    produced them, so this works directly on the visible state graph as
    well as on a session dump.
    """
    if isinstance(source, Node):
        all_nodes: list[Node] = source.walk()
    else:
        items = list(source)
        # If we got states (each with subtrees), flatten and dedupe by id.
        if items and any(isinstance(n, Node) and n.child_nodes() for n in items):
            seen: set[str] = set()
            ordered: list[Node] = []
            for state in items:
                for n in state.walk():
                    if n.id not in seen:
                        seen.add(n.id)
                        ordered.append(n)
            all_nodes = ordered
        else:
            all_nodes = items

    if agent_id:
        all_nodes = [n for n in all_nodes if n.agent_id == agent_id]

    out: list[str] = []
    for node in all_nodes:
        code = getattr(node, "code", None)
        if not code:
            continue
        out.append(f"# [{node.agent_id}] {node.type}")
        out.append(code.strip())
        output = getattr(node, "output", None) or ""
        if not output:
            content = getattr(node, "content", None) or ""
            if content and node.type in ("error", "observation", "resume"):
                output = content
        if output:
            out.append("→ " + output.strip()[:240])
        out.append("")
    return "\n".join(out).rstrip() or "(no code blocks)"


def message_stream(agent_id: str, session: Any) -> str:
    """Reconstruct the chat-log view for one agent from its session.

    Renders ``system / query / assistant.code / observation / ...`` for the
    agent, in execution order. ``session`` is any object with
    ``chain_to(node)`` and ``load()`` (i.e., a `Session`).
    """
    from rlmflow.workspace.session import _NODE_TERMINALITY

    nodes = list(session.load().values())
    nodes = [n for n in nodes if n.agent_id == agent_id]
    if not nodes:
        return f"(no nodes for agent {agent_id!r})"

    # Terminal-first: prefer a result/error/observation node so the chain
    # walk reaches the full conversation rather than stopping at the query.
    leaf = min(nodes, key=lambda n: _NODE_TERMINALITY.get(n.type, 99))
    chain = session.chain_to(leaf)

    parts: list[str] = []
    if chain and chain[0].system_prompt:
        parts.append(f"--- system ---\n{chain[0].system_prompt.strip()}")
    for node in chain:
        if node.type == "query":
            parts.append(f"--- query ---\n{node.query.strip()}")
        elif node.type == "action":
            parts.append(f"--- assistant ---\n{getattr(node, 'reply', '').strip()}")
        elif node.type == "observation":
            parts.append(f"--- observation ---\n{getattr(node, 'content', '').strip()}")
        elif node.type == "supervising":
            wait_on = ", ".join(getattr(node, "waiting_on", []) or [])
            parts.append(f"--- supervising ---\nwaiting on: {wait_on}")
        elif node.type == "resume":
            parts.append(f"--- resume ---\n{getattr(node, 'content', '').strip()}")
        elif node.type == "error":
            parts.append(
                f"--- error ({getattr(node, 'error', '')}) ---\n"
                f"{getattr(node, 'content', '').strip()}"
            )
        elif node.type == "result":
            parts.append(f"--- result ---\n{getattr(node, 'result', '').strip()}")
    return "\n\n".join(parts)


def diff_system_prompts(node_a: Node, node_b: Node) -> str:
    """Unified diff of two ``system_prompt`` strings."""
    a = (node_a.system_prompt or "").splitlines(keepends=True)
    b = (node_b.system_prompt or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        a,
        b,
        fromfile=node_a.agent_id or "a",
        tofile=node_b.agent_id or "b",
        n=3,
    )
    return "".join(diff) or "(prompts identical)"


# ── cost & tokens ────────────────────────────────────────────────────


def token_sparkline(states: list[Node], width: int = 40) -> str:
    """One-line ASCII sparkline of cumulative tokens across steps."""
    if not states:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    cum = [s.tree_tokens for s in states]
    span = max(cum) or 1
    if width <= 0 or len(cum) <= width:
        sample = cum
    else:
        step = len(cum) / width
        sample = [cum[min(int(i * step), len(cum) - 1)] for i in range(width)]
    bars = "".join(
        blocks[min(len(blocks) - 1, int((v / span) * (len(blocks) - 1)))]
        for v in sample
    )
    return f"{bars}  {cum[-1]:>6} tok over {len(states)} steps"


def budget_burndown(
    states: list[Node],
    max_budget: int | None = None,
    *,
    width: int = 40,
) -> str:
    """Cumulative tokens vs ``max_budget``. Falls back to peak if no budget."""
    if not states:
        return ""
    cum = [s.tree_tokens for s in states]
    final = cum[-1]
    target = max_budget or max(cum) or 1
    pct = min(1.0, final / target)
    filled = int(round(pct * width))
    bar = "█" * filled + "·" * (width - filled)
    label = f"{final}/{target}" if max_budget else f"{final} tok (peak)"
    return f"[{bar}] {pct * 100:5.1f}%  {label}"


# ── reports ──────────────────────────────────────────────────────────


def report_md(
    states: list[Node],
    *,
    title: str = "rlmflow run",
    max_budget: int | None = None,
) -> str:
    """Render a Markdown summary of a run: tree + cost + final result."""
    if not states:
        return f"# {title}\n\n(empty trace)\n"

    final = states[-1]
    inp, out = final.tree_usage()
    parts: list[str] = [f"# {title}", ""]

    parts.extend(
        [
            f"**Steps:** {len(states)}",
            f"**Agents:** {len({n.agent_id for n in final.walk()})}",
        ]
    )
    parts.append(f"**Tokens:** {inp + out:,} ({inp:,} in, {out:,} out)")
    if max_budget is not None:
        parts.append(f"**Budget:** {budget_burndown(states, max_budget)}")
    current = final.current()
    parts.append(f"**Outcome:** {current.type}")
    if final.errors():
        parts.append(f"**Errors:** {len(final.errors())}")

    parts.extend(["", "## Tree", "", "```", final.tree(), "```"])

    parts.extend(
        ["", "## Cumulative tokens", "", "```", token_sparkline(states), "```"]
    )

    if final.errors():
        parts.extend(["", "## Errors", "", "```", error_summary(final), "```"])

    result = getattr(current, "result", None)
    if result:
        parts.extend(["", "## Result", "", "```", str(result), "```"])

    return "\n".join(parts) + "\n"


# ── comparison ───────────────────────────────────────────────────────


def bench_table(
    traces: dict[str, list[Node]],
    *,
    pricing: Callable[[Node], float] | None = None,
) -> str:
    """Aggregate one row per labeled trace: outcome, steps, agents, tokens, errors."""
    if not traces:
        return "(no traces)"
    header = ["label", "steps", "agents", "outcome", "tokens", "errors"]
    if pricing is not None:
        header.append("cost")
    rows: list[list[str]] = [header]

    for label, states in traces.items():
        if not states:
            rows.append([label, "0", "0", "(empty)", "0", "0"])
            continue
        final = states[-1]
        agents = len({n.agent_id for n in final.walk()})
        tokens = final.tree_tokens
        errors = len(final.errors())
        outcome = (
            "result"
            if final.type == "result"
            else "error" if final.type == "error" else "open"
        )
        row = [
            label,
            str(len(states)),
            str(agents),
            outcome,
            f"{tokens:,}",
            str(errors),
        ]
        if pricing is not None:
            row.append(f"${pricing(final):.4f}")
        rows.append(row)

    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    out: list[str] = []
    for i, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


# ── streaming utilities ──────────────────────────────────────────────


def tee(stream: Iterable[Node], *sinks: Callable[[Node], Any]) -> Iterator[Node]:
    """Fan a step iterator out to multiple sinks while still yielding each node.

    ``for node in viz.tee(agent_steps, print, save_jsonl_writer): ...``
    """
    for node in stream:
        for sink in sinks:
            try:
                sink(node)
            except Exception:
                pass
        yield node


def _webhook_payload(state: Node, *, title: str) -> dict[str, str]:
    inp, out = state.tree_usage()
    body = (
        f"*{title}*\n"
        f"agents: {len({n.agent_id for n in state.walk()})}  "
        f"tokens: {inp + out:,}  "
        f"errors: {len(state.errors())}\n"
        f"```\n{state.tree()}\n```"
    )
    return {"text": body}


def slack_webhook(url: str, state: Node, *, title: str = "rlmflow run") -> int:
    """POST a tree summary to a Slack incoming webhook. Returns HTTP status."""
    return _post_json(url, _webhook_payload(state, title=title))


def discord_webhook(url: str, state: Node, *, title: str = "rlmflow run") -> int:
    """POST a tree summary to a Discord webhook. Returns HTTP status."""
    payload = _webhook_payload(state, title=title)
    return _post_json(url, {"content": payload["text"]})


def _post_json(url: str, payload: dict) -> int:
    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


# ── ascii boxes ──────────────────────────────────────────────────────


def ascii_boxes(state: Node) -> str:
    """Boxed-tree variant of ``Node.tree()`` using Rich panels."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree

    def _label(node: Node) -> Panel:
        head = Text()
        head.append(node.agent_id or "root", style="bold")
        head.append(
            f"  [{node.type}]", style="magenta" if not node.terminal else "green"
        )
        head.append(f"  {{{node.model_label}}}", style="cyan")
        result = getattr(node, "result", None)
        body = ""
        if result:
            body = str(result)[:160].replace("\n", " ")
        return Panel(
            Text(body) if body else Text(""),
            title=head,
            border_style="dim",
            padding=(0, 1),
        )

    def build(node: Node) -> Tree:
        tree = Tree(_label(node), guide_style="dim")
        for child in node.child_nodes():
            tree.add(build(child))
        return tree

    con = Console(record=True, width=120)
    con.print(build(state))
    return con.export_text()


GraphMode = Literal["events", "snapshot"]
EdgeKind = Literal["next", "spawn", "contains"]


@dataclass(frozen=True)
class VizNode:
    id: str
    type: str
    agent_id: str
    depth: int
    label: str
    current: bool
    payload: Node


@dataclass(frozen=True)
class VizEdge:
    source: str
    target: str
    kind: EdgeKind


@dataclass(frozen=True)
class VizGraph:
    nodes: list[VizNode]
    edges: list[VizEdge]
    mode: GraphMode
    step: int | None = None

    @property
    def payloads(self) -> list[Node]:
        return [node.payload for node in self.nodes]

    @property
    def nodes_by_id(self) -> dict[str, Node]:
        return {node.id: node.payload for node in self.nodes}


def session_events(session: Session | str | Path | None) -> list[Node]:
    """Return persisted node events in append order when available."""
    if session is None:
        return []
    if isinstance(session, (str, Path)):
        session = FileSession(session)
    nodes_path = getattr(session, "nodes_path", None)
    if nodes_path is not None:
        path = Path(nodes_path)
        if not path.exists():
            return []
        return [
            parse_node_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return list(session.load().values())


def trace_events(path: str | Path) -> list[Node]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [parse_node_json(json.dumps(event)) for event in data.get("events", [])]


def build_viz_graph(
    *,
    states: list[Node] | None = None,
    session: Session | str | Path | None = None,
    events: list[Node] | None = None,
    step: int | None = None,
    mode: GraphMode = "events",
) -> VizGraph:
    """Build the step-local graph that all visualization surfaces should share."""
    if mode == "snapshot":
        state = _state_for_step(states, step)
        nodes = (
            list(state.walk()) if state is not None else _latest_unique(events or [])
        )
        return _graph_from_nodes(nodes, mode="snapshot", step=step)

    event_nodes = list(events or session_events(session))
    if not event_nodes:
        raise ValueError("mode='events' requires `events=` or a persisted `session=`.")

    visible = _events_visible_at_step(event_nodes, states, step)
    return _graph_from_nodes(visible, mode="events", step=step)


def node_tree(graph: VizGraph) -> str:
    """Render a node-event tree from a VizGraph."""
    by_id = graph.nodes_by_id
    children: dict[str, list[tuple[str, EdgeKind]]] = {
        node.id: [] for node in graph.nodes
    }
    parented: set[str] = set()
    for edge in graph.edges:
        if edge.source not in by_id or edge.target not in by_id:
            continue
        children.setdefault(edge.source, []).append((edge.target, edge.kind))
        parented.add(edge.target)

    roots = [node.id for node in graph.nodes if node.id not in parented]
    roots.sort(key=lambda node_id: _viz_node_sort_key(by_id[node_id]))

    lines: list[str] = []

    def label(node: Node) -> str:
        suffix = _agent_suffix(node.agent_id)
        result = getattr(node, "result", "")
        preview = f" -> {str(result).strip()[:80]}" if result else ""
        return f"{suffix} [{node.type}] {{{node.model_label}}}{preview}"

    def walk(
        node_id: str, prefix: str, is_last: bool, edge_kind: EdgeKind | None
    ) -> None:
        node = by_id[node_id]
        connector = (
            "" if not prefix and edge_kind is None else ("└── " if is_last else "├── ")
        )
        edge_label = f"{edge_kind}: " if edge_kind else ""
        lines.append(f"{prefix}{connector}{edge_label}{label(node)}")
        next_prefix = prefix + ("    " if is_last else "│   ") if connector else ""
        kids = children.get(node_id, [])
        kids.sort(
            key=lambda item: (item[1] != "next", _viz_node_sort_key(by_id[item[0]]))
        )
        for idx, (child_id, kind) in enumerate(kids):
            walk(child_id, next_prefix, idx == len(kids) - 1, kind)

    for idx, root in enumerate(roots):
        walk(root, "", idx == len(roots) - 1, None)
    return "\n".join(lines)


def _state_for_step(states: list[Node] | None, step: int | None) -> Node | None:
    if not states:
        return None
    idx = len(states) - 1 if step is None else max(0, min(int(step), len(states) - 1))
    return states[idx]


def _events_visible_at_step(
    events: list[Node], states: list[Node] | None, step: int | None
) -> list[Node]:
    if not states:
        if step is None:
            cutoff = len(events) - 1
        else:
            cutoff = max(0, min(int(step), len(events) - 1))
        return _latest_unique(events[: cutoff + 1])

    state = _state_for_step(states, step)
    if state is None:
        return []

    visible_ids = {node.id for node in state.walk()}
    cutoff = max(
        (
            _event_position(events, node)
            for node in state.walk()
            if node.id in visible_ids
        ),
        default=-1,
    )
    if cutoff < 0:
        return list(state.walk())
    return _latest_unique(events[: cutoff + 1])


def _latest_unique(events: list[Node]) -> list[Node]:
    latest: dict[str, Node] = {}
    order: list[str] = []
    for event in events:
        if event.id not in latest:
            order.append(event.id)
        latest[event.id] = event
    return [latest[node_id] for node_id in order]


def _event_position(events: list[Node], target: Node) -> int:
    target_sig = _node_signature(target)
    fallback = -1
    for idx, event in enumerate(events):
        if event.id != target.id:
            continue
        if fallback < 0:
            fallback = idx
        if _node_signature(event) == target_sig:
            return idx
    return fallback


def _node_signature(node: Node) -> tuple[str, str, tuple[str, ...]]:
    return (node.id, node.type, tuple(_child_ids(node)))


def _graph_from_nodes(
    nodes: list[Node], *, mode: GraphMode, step: int | None = None
) -> VizGraph:
    by_id = {node.id: node for node in nodes}
    edges: list[VizEdge] = []
    seen_edges: set[tuple[str, str, EdgeKind]] = set()

    def add_edge(source: str, target: str, kind: EdgeKind) -> None:
        if source == target or source not in by_id or target not in by_id:
            return
        key = (source, target, kind)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(VizEdge(source=source, target=target, kind=kind))

    first_by_agent: dict[str, Node] = {}
    latest_by_agent: dict[str, Node] = {}
    for node in nodes:
        first_by_agent.setdefault(node.agent_id, node)
        latest_by_agent[node.agent_id] = node

    agent_of_id = {node.id: node.agent_id for node in nodes}
    for node in nodes:
        for child_id in _child_ids(node):
            child_agent = agent_of_id.get(child_id)
            if child_agent is None:
                continue
            if child_agent == node.agent_id:
                add_edge(node.id, child_id, "next")
            else:
                first = first_by_agent.get(child_agent)
                if first is not None:
                    add_edge(node.id, first.id, "spawn")

    viz_nodes = [
        VizNode(
            id=node.id,
            type=node.type,
            agent_id=node.agent_id,
            depth=node.depth or 0,
            label=_viz_node_label(node),
            current=latest_by_agent.get(node.agent_id) is node,
            payload=node,
        )
        for node in nodes
    ]
    return VizGraph(nodes=viz_nodes, edges=edges, mode=mode, step=step)


def _child_ids(node: Node) -> list[str]:
    return [
        child.id if isinstance(child, Node) else str(child) for child in node.children
    ]


def _agent_suffix(agent_id: str) -> str:
    if not agent_id or agent_id == "root":
        return "root"
    return agent_id.rsplit(".", 1)[-1]


def _viz_node_label(node: Node) -> str:
    suffix = _agent_suffix(node.agent_id)
    return f"{suffix}:{node.type}"


def _viz_node_sort_key(node: Node) -> tuple[int, int, str]:
    type_order = {
        "query": 0,
        "action": 1,
        "supervising": 2,
        "observation": 3,
        "resume": 4,
        "result": 5,
        "error": 6,
    }
    return (node.depth or 0, type_order.get(node.type, 99), node.id)
