"""Live tree visualization and Gantt swimlane for RLM agents.

Requires: ``pip install rich``

Usage::

    from rlmkit.utils.viz import live, gantt, gantt_html

    states = live(agent, agent.start("task"))

    gantt(states)                       # print colored swimlane to the terminal
    Path("run.html").write_text(gantt_html(states))
"""

from __future__ import annotations

from rlmkit.rlm import RLM
from rlmkit.state import RLMState, Status


def _node_label(state: RLMState):
    from rich.console import Group
    from rich.spinner import Spinner
    from rich.text import Text

    active = not state.finished

    model = state.config.get("model")

    info = Text()
    info.append(state.agent_id or "root", style="bold")
    if model:
        info.append(f" ({model})", style="dim")
    info.append(f" [{state.status.value}]", style="magenta" if active else "green")
    if state.iteration:
        info.append(f" iter {state.iteration}", style="dim")
    if state.finished and state.result is not None:
        preview = state.result[:80].replace("\n", " ")
        info.append(f" → {preview}", style="dim green")

    if active:
        return Group(Spinner("dots", text=info))
    return Text.assemble("✓ ", info, style="green" if state.finished else "")


def _build_tree(state: RLMState):
    from rich.tree import Tree

    tree = Tree(_node_label(state), guide_style="dim")
    for child in state.children:
        tree.add(_build_tree(child))
    return tree


def live(agent: RLM, state: RLMState) -> list[RLMState]:
    """Run the agent with a live-updating tree display. Returns all states."""
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text

    con = Console()
    states: list[RLMState] = [state]
    step = 0

    with Live(
        console=con, refresh_per_second=12, vertical_overflow="visible"
    ) as display:
        while not state.finished:
            display.update(
                Group(
                    Text(f"step {step}", style="dim"),
                    _build_tree(state),
                )
            )
            state = agent.step(state)
            step += 1
            states.append(state)

    con.print()
    con.print(Text(f"✓ done in {step} steps", style="bold green"))
    con.print(_build_tree(state))
    if state.result:
        con.print()
        con.print(f"[bold]result:[/] {state.result[:500]}")

    return states


# ── Gantt / swimlane ─────────────────────────────────────────────────

_STATUS_ORDER = {
    Status.READY: 0,
    Status.EXECUTING: 1,
    Status.SUPERVISING: 2,
    Status.FINISHED: 3,
}

_STATUS_CELL = {
    Status.READY: ("░", "blue"),
    Status.EXECUTING: ("█", "yellow"),
    Status.SUPERVISING: ("▓", "magenta"),
    Status.FINISHED: ("█", "green"),
}

_STATUS_HTML = {
    "ready": "#58a6ff",
    "executing": "#d29922",
    "supervising": "#bc8cff",
    "finished": "#3fb950",
}


def _flatten_state(state: RLMState) -> list[RLMState]:
    out = [state]
    for c in state.children:
        out.extend(_flatten_state(c))
    return out


def gantt_matrix(states: list[RLMState]) -> tuple[list[str], list[list[Status | None]]]:
    """Build the per-agent-per-step status matrix.

    Returns ``(agent_ids, rows)`` where ``rows[i][j]`` is the status of
    ``agent_ids[i]`` at step ``j``, or ``None`` if that agent did not exist yet.

    Agent order is stable: each agent is placed under its parent in the order
    it first appeared.
    """
    order: list[str] = []
    seen: set[str] = set()
    per_step: list[dict[str, Status]] = []

    for state in states:
        step_map: dict[str, Status] = {}
        for node in _flatten_state(state):
            aid = node.agent_id or "root"
            step_map[aid] = node.status
            if aid not in seen:
                seen.add(aid)
                order.append(aid)
        per_step.append(step_map)

    rows = [[step.get(aid) for step in per_step] for aid in order]
    return order, rows


def gantt(states: list[RLMState]) -> None:
    """Print a colored swimlane to stdout using Rich.

    One row per agent, one column per step, colored by status.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    agents, rows = gantt_matrix(states)
    con = Console()

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
        for status in row:
            if status is None:
                cells.append(Text(" ", style="dim"))
            else:
                glyph, color = _STATUS_CELL[status]
                cells.append(Text(glyph, style=color))
        table.add_row(*cells)

    con.print(table)
    con.print(
        Text.assemble(
            ("legend: ", "dim"),
            ("█ ready", "blue"),
            ("  ", ""),
            ("█ executing", "yellow"),
            ("  ", ""),
            ("█ supervising", "magenta"),
            ("  ", ""),
            ("█ finished", "green"),
        )
    )


def gantt_html(states: list[RLMState], *, title: str = "rlmkit gantt") -> str:
    """Render the swimlane as a self-contained HTML string."""
    agents, rows = gantt_matrix(states)
    n_steps = len(states)

    cells_html: list[str] = []
    for aid, row in zip(agents, rows):
        cells_html.append(
            f'<div class="row"><div class="name">{aid}</div>'
            f'<div class="bars" style="grid-template-columns: repeat({n_steps}, 1fr)">'
        )
        for status in row:
            if status is None:
                cells_html.append('<div class="cell empty"></div>')
            else:
                color = _STATUS_HTML[status.value]
                cells_html.append(
                    f'<div class="cell" style="background:{color}" '
                    f'title="{status.value}"></div>'
                )
        cells_html.append("</div></div>")

    legend = "".join(
        f'<span><i style="background:{c}"></i>{s}</span>'
        for s, c in _STATUS_HTML.items()
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0d1117;
        color: #e6edf3; margin: 24px; }}
h1 {{ font-size: 14px; color: #8b949e; font-weight: 500; margin: 0 0 12px; }}
.row {{ display: grid; grid-template-columns: 220px 1fr; gap: 8px;
        align-items: center; margin: 2px 0; }}
.name {{ font-family: 'SF Mono', 'Menlo', monospace; font-size: 12px;
        color: #e6edf3; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis; }}
.bars {{ display: grid; gap: 1px; height: 18px;
        background: #161b22; border: 1px solid #30363d; border-radius: 3px;
        overflow: hidden; }}
.cell {{ height: 100%; }}
.cell.empty {{ background: transparent; }}
.legend {{ margin-top: 16px; font-size: 12px; color: #8b949e; }}
.legend span {{ margin-right: 16px; }}
.legend i {{ display: inline-block; width: 10px; height: 10px;
        margin-right: 4px; border-radius: 2px; vertical-align: middle; }}
</style></head><body>
<h1>{title} — {n_steps} steps, {len(agents)} agents</h1>
{"".join(cells_html)}
<div class="legend">{legend}</div>
</body></html>
"""
