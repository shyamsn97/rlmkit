"""Live tree visualization and simple swimlanes for RLMFlow nodes."""

from __future__ import annotations

from rlmkit.node import Node
from rlmkit.rlm import RLMFlow


def _node_label(state: Node):
    from rich.console import Group
    from rich.spinner import Spinner
    from rich.text import Text

    active = not state.finished
    info = Text()
    info.append(state.agent_id or "root", style="bold")
    info.append(f" [{state.type}]", style="magenta" if active else "green")
    result = getattr(state, "result", None)
    if result:
        info.append(f" -> {str(result)[:80].replace(chr(10), ' ')}", style="dim green")
    if active:
        return Group(Spinner("dots", text=info))
    return Text.assemble("ok ", info, style="green" if state.finished else "")


def _build_tree(state: Node):
    from rich.tree import Tree

    tree = Tree(_node_label(state), guide_style="dim")
    for child in state.child_nodes():
        tree.add(_build_tree(child))
    return tree


def live(agent: RLMFlow, state: Node) -> list[Node]:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text

    con = Console()
    states = [state]
    step = 0
    with Live(
        console=con, refresh_per_second=12, vertical_overflow="visible"
    ) as display:
        while not state.finished:
            display.update(Group(Text(f"step {step}", style="dim"), _build_tree(state)))
            state = agent.step(state)
            step += 1
            states.append(state)
    con.print()
    con.print(Text(f"done in {step} steps", style="bold green"))
    con.print(_build_tree(state))
    result = getattr(state, "result", None)
    if result:
        con.print()
        con.print(f"[bold]result:[/] {str(result)[:500]}")
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
            step_map[aid] = node.type
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
                glyph, color = _TYPE_CELL.get(kind, ("?", "dim"))
                cells.append(Text(glyph, style=color))
        table.add_row(*cells)
    Console().print(table)


def gantt_html(states: list[Node], *, title: str = "rlmkit gantt") -> str:
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
                color = _TYPE_HTML.get(kind, "#8b949e")
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
