"""Rich terminal visualization for RLM step traces.

Requires: ``pip install rich``

Usage::

    from rlmkit.utils.viz import live, record, save, load

    # Live mode — runs agent with real-time terminal display
    states = live(agent, agent.start("task"))

    # Batch mode
    states, final = record(agent, agent.start("task"))
    save(states, "trace.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..rlm import RLM
    from ..state import RLMState

from ..state import ChildStep, CodeExec, LLMReply, NoCodeBlock

# ── serialization (for save / load) ──────────────────────────────


def _serialize_event(event) -> dict | None:
    if event is None:
        return None
    base = {
        "type": type(event).__name__,
        "agent_id": event.agent_id,
        "iteration": event.iteration,
    }
    if isinstance(event, LLMReply):
        base["code"] = event.code
        base["text"] = event.text[:4000] if event.text else None
    elif isinstance(event, CodeExec):
        base["code"] = event.code
        base["output"] = event.output[:4000] if event.output else ""
        base["suspended"] = event.suspended
    elif isinstance(event, ChildStep):
        base["all_done"] = event.all_done
        base["agent_finished"] = event.agent_finished
        base["exec_output"] = event.exec_output[:4000] if event.exec_output else None
        base["child_events"] = [_serialize_event(ce) for ce in event.child_events]
    elif isinstance(event, NoCodeBlock):
        base["text"] = event.text[:4000] if event.text else ""
    return base


def _serialize_tree(state: RLMState) -> dict:
    label = (
        state.agent_id.rsplit(".", 1)[-1] if "." in state.agent_id else state.agent_id
    )
    return {
        "id": state.agent_id,
        "label": label,
        "status": state.status.value,
        "task": state.task[:800] if state.task else "",
        "iteration": state.iteration,
        "result": state.result[:2000] if state.result else None,
        "context": state.context[:1000] if state.context else None,
        "children": [_serialize_tree(c) for c in state.children],
    }


def _collect_node_events(steps: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}

    def walk(ev, step_num):
        if not ev:
            return
        key = ev["agent_id"]
        index.setdefault(key, []).append({"step": step_num, **ev})
        for ce in ev.get("child_events") or []:
            if ce:
                walk(ce, step_num)

    for s in steps:
        walk(s["event"], s["step"])
    return index


def _build_payload(states: list[RLMState]) -> dict:
    steps = []
    for i, state in enumerate(states):
        steps.append(
            {
                "step": i + 1,
                "event": _serialize_event(state.event),
                "tree": _serialize_tree(state),
            }
        )
    return {
        "steps": steps,
        "node_events": _collect_node_events(steps),
    }


# ── rich rendering helpers ───────────────────────────────────────

STATUS_STYLE = {
    "waiting": "blue",
    "has_reply": "yellow",
    "supervising": "magenta",
    "finished": "green",
}


def _format_event(event, *, indent: int = 0):
    """Return a ``rich.console.Group`` for a single step event."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    parts: list = []
    pad = "  " * indent

    if isinstance(event, LLMReply):
        parts.append(
            Text.from_markup(
                f"{pad}[blue bold]{event.agent_id}[/] — "
                f"LLM reply [dim](iter {event.iteration})[/]"
            )
        )
        if event.code:
            parts.append(
                Syntax(
                    event.code,
                    "python",
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                    padding=(0, 2),
                )
            )
        elif event.text:
            parts.append(Text(event.text[:300], style="dim"))

    elif isinstance(event, CodeExec):
        out = (event.output or "").strip()
        has_error = out and any(
            kw in out for kw in ("Error:", "Traceback", "SyntaxError")
        )
        if has_error:
            tag, style = "error", "red bold"
        elif event.suspended:
            tag, style = "suspended", "yellow bold"
        else:
            tag, style = "done", "green bold"
        parts.append(
            Text.from_markup(
                f"{pad}[{style}]{event.agent_id}[/] — "
                f"exec → [{style}]{tag}[/] [dim](iter {event.iteration})[/]"
            )
        )
        if event.code:
            parts.append(
                Syntax(
                    event.code,
                    "python",
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                    padding=(0, 2),
                )
            )
        if out:
            out_style = "red" if has_error else "dim"
            parts.append(
                Panel(
                    Text(out[:800], style=out_style),
                    title="output",
                    title_align="left",
                    border_style=out_style,
                    expand=False,
                    padding=(0, 1),
                )
            )

    elif isinstance(event, ChildStep):
        n = len(event.child_events)
        if event.agent_finished:
            tag, style = "agent finished", "green bold"
        elif event.all_done:
            tag, style = "all done", "green bold"
        else:
            tag, style = f"in progress ({n} events)", "magenta"
        parts.append(
            Text.from_markup(
                f"{pad}[{style}]{event.agent_id}[/] — children [{style}]{tag}[/]"
            )
        )
        if event.exec_output and event.exec_output.strip():
            parts.append(
                Panel(
                    Text(event.exec_output.strip()[:400], style="dim"),
                    title="resume output",
                    title_align="left",
                    border_style="magenta dim",
                    expand=False,
                    padding=(0, 1),
                )
            )
        for ce in event.child_events[:12]:
            parts.append(_event_oneliner(ce, indent=indent + 1))
        if n > 12:
            parts.append(Text(f"{pad}  ... and {n - 12} more", style="dim"))

    elif isinstance(event, NoCodeBlock):
        parts.append(
            Text.from_markup(f"{pad}[yellow]{event.agent_id}[/] — no code block")
        )
        if event.text:
            parts.append(Text(event.text[:200], style="dim"))

    return Group(*parts) if parts else Text("")


def _event_oneliner(event, *, indent: int = 0):
    """Compact one-line summary for child event listings."""
    from rich.text import Text

    pad = "  " * indent
    if isinstance(event, LLMReply):
        return Text.from_markup(
            f"{pad}[dim]├[/] [blue]{event.agent_id}[/] "
            f"[dim]LLM reply (iter {event.iteration})[/]"
        )
    if isinstance(event, CodeExec):
        tag = "suspended" if event.suspended else "done"
        c = "yellow" if event.suspended else "green"
        out = (event.output or "").strip()
        has_err = out and any(
            kw in out for kw in ("Error:", "Traceback", "SyntaxError")
        )
        if has_err:
            tag, c = "error", "red"
        snippet = ""
        if out and len(out) < 80:
            snippet = f" [dim]→ {out}[/]"
        return Text.from_markup(
            f"{pad}[dim]├[/] [{c}]{event.agent_id}[/] "
            f"[dim]exec →[/] [{c}]{tag}[/]{snippet}"
        )
    if isinstance(event, ChildStep):
        n = len(event.child_events)
        return Text.from_markup(
            f"{pad}[dim]├[/] [magenta]{event.agent_id}[/] "
            f"[dim]children ({n} events)[/]"
        )
    if isinstance(event, NoCodeBlock):
        return Text.from_markup(
            f"{pad}[dim]├[/] [yellow]{event.agent_id}[/] [dim]no code[/]"
        )
    return Text(f"{pad}├ {type(event).__name__}", style="dim")


def _count_agents(state: RLMState) -> dict[str, int]:
    counts: dict[str, int] = {}

    def walk(s: RLMState):
        key = s.status.value
        counts[key] = counts.get(key, 0) + 1
        for c in s.children:
            walk(c)

    walk(state)
    return counts


def _find_state(state: RLMState, agent_id: str) -> RLMState | None:
    if state.agent_id == agent_id:
        return state
    for child in state.children:
        found = _find_state(child, agent_id)
        if found:
            return found
    return None


def _find_event_for_node(event, agent_id: str):
    """Walk an event tree and return the sub-event matching *agent_id*."""
    if event is None:
        return None
    if event.agent_id == agent_id:
        return event
    if isinstance(event, ChildStep):
        for ce in event.child_events:
            found = _find_event_for_node(ce, agent_id)
            if found:
                return found
    return None


# ── plotly graph (optional dep) ──────────────────────────────────

STATUS_COLORS = {
    "waiting": "#58a6ff",
    "has_reply": "#d29922",
    "supervising": "#bc8cff",
    "finished": "#3fb950",
}


def _tree_positions(tree: dict, x_gap: float = 1.0) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}

    def layout(node, x_start, depth):
        children = node.get("children", [])
        if not children:
            positions[node["id"]] = (x_start, -depth)
            return x_start + x_gap
        child_start = x_start
        for child in children:
            child_start = layout(child, child_start, depth + 1)
        xs = [positions[c["id"]][0] for c in children]
        positions[node["id"]] = ((min(xs) + max(xs)) / 2, -depth)
        return child_start

    layout(tree, 0, 0)
    return positions


def _build_plotly_figure(tree: dict):
    import plotly.graph_objects as go

    pos = _tree_positions(tree)
    edge_x, edge_y = [], []
    node_x, node_y, labels, colors, hovers = [], [], [], [], []

    def walk(node):
        px, py = pos[node["id"]]
        node_x.append(px)
        node_y.append(py)
        labels.append(node["label"])
        colors.append(STATUS_COLORS.get(node["status"], "#8b949e"))
        hover = (
            f"<b>{node['id']}</b><br>"
            f"Status: {node['status']}<br>"
            f"Iter: {node['iteration']}"
        )
        if node.get("result"):
            hover += f"<br>Result: {node['result'][:120]}"
        hovers.append(hover)
        for child in node.get("children", []):
            cx, cy = pos[child["id"]]
            edge_x.extend([px, cx, None])
            edge_y.extend([py, cy, None])
            walk(child)

    walk(tree)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#888", width=1.2),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=10),
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(size=18, color=colors, line=dict(width=1.5, color="#fff")),
        )
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(node_x) * 30),
    )
    return fig


# ── rich tree helper ──────────────────────────────────────────────


def _rich_tree(state: RLMState):
    """Build a ``rich.tree.Tree`` from an ``RLMState``."""
    from rich.text import Text
    from rich.tree import Tree

    style = STATUS_STYLE.get(state.status.value, "dim")
    label = Text()
    label.append(state.agent_id, style="bold")
    label.append(f" [{state.status.value}]", style=style)
    if state.result:
        preview = state.result[:80].replace("\n", " ")
        label.append(f"  → {preview}", style="green dim")

    tree = Tree(label, guide_style="dim")
    for child in state.children:
        tree.add(_rich_tree(child))
    return tree


def _stats_line(state: RLMState, step: int) -> str:
    counts = _count_agents(state)
    total = sum(counts.values())
    parts = [f"step {step}", f"{total} agents"]
    for key, label in [
        ("finished", "done"),
        ("waiting", "waiting"),
        ("supervising", "supervising"),
    ]:
        if counts.get(key):
            parts.append(f"{counts[key]} {label}")
    return " · ".join(parts)


# ── public API ───────────────────────────────────────────────────


def record(
    agent: RLM, state: RLMState, *, limit: int = 500
) -> tuple[list[RLMState], RLMState]:
    """Run an agent to completion, collecting every step's state."""
    states: list[RLMState] = []
    while not state.finished:
        state = agent.step(state)
        states.append(state)
        if len(states) > limit:
            break
    return states, state


def save(states: list[RLMState], path: str | Path) -> None:
    """Save a step trace to JSON."""
    payload = _build_payload(states)
    Path(path).write_text(json.dumps(payload, indent=2))


def load(path: str | Path) -> dict:
    """Load a saved step trace from JSON."""
    return json.loads(Path(path).read_text())


def plot(state: RLMState):
    """Return a Plotly figure of the agent tree for a single state."""
    tree = _serialize_tree(state)
    return _build_plotly_figure(tree)


def live(
    agent: RLM,
    state: RLMState,
    *,
    console: object | None = None,
) -> list[RLMState]:
    """Run the agent with a live-updating rich terminal display.

    Each step refreshes the tree and prints the latest event with
    syntax-highlighted code and color-coded output.

    Returns the collected list of states.

    Usage::

        state = agent.start("task")
        states = live(agent, state)
    """
    from rich.console import Console, Group
    from rich.live import Live
    from rich.rule import Rule
    from rich.text import Text

    con: Console = console or Console()  # type: ignore[assignment]
    states: list[RLMState] = []
    step = 0

    with Live(
        console=con, refresh_per_second=4, vertical_overflow="visible"
    ) as display:
        while not state.finished:
            state = agent.step(state)
            step += 1
            states.append(state)

            tree_widget = _rich_tree(state)
            stats = Text(_stats_line(state, step), style="dim")
            event_widget = _format_event(state.event) if state.event else Text("")

            display.update(
                Group(
                    Rule(f"[bold]Step {step}[/]", style="blue"),
                    stats,
                    Text(""),
                    tree_widget,
                    Text(""),
                    event_widget,
                )
            )

    con.print()
    con.print(Rule("[bold green]Done[/bold green]", style="green"))
    con.print(Text(f"Completed in {step} steps", style="green bold"))
    con.print()
    con.print(_rich_tree(state))
    if state.result:
        con.print()
        con.print(f"[bold green]Result:[/bold green] {state.result}")

    return states
