"""Live tree visualization for RLM agents.

Requires: ``pip install rich``

Usage::

    from viz import live

    state = agent.start("task")
    states = live(agent, state)
"""

from __future__ import annotations

from rlmkit.rlm import RLM
from rlmkit.state import RLMState

STATUS_STYLE = {
    "waiting": "blue",
    "has_reply": "yellow",
    "supervising": "magenta",
    "finished": "green",
}

STATUS_ICON = {
    "waiting": "◌",
    "has_reply": "◐",
    "supervising": "◑",
    "finished": "✓",
}


def _build_tree(state: RLMState):
    """Build a rich Tree from an RLMState, recursively."""
    from rich.text import Text
    from rich.tree import Tree

    style = STATUS_STYLE.get(state.status.value, "dim")
    icon = STATUS_ICON.get(state.status.value, "·")

    label = Text()
    label.append(f"{icon} ", style=style)
    label.append(state.agent_id or "root", style="bold")
    label.append(f" [{state.status.value}]", style=style)
    label.append(f" iter {state.iteration}", style="dim")
    if state.finished and state.result is not None:
        preview = state.result[:80].replace("\n", " ")
        label.append(f"  → {preview}", style="green dim")

    tree = Tree(label, guide_style="dim")
    for child in state.children:
        tree.add(_build_tree(child))
    return tree


def _stats(state: RLMState) -> str:
    """Count agents by status across the full tree."""
    counts: dict[str, int] = {}

    def _walk(s: RLMState) -> None:
        counts[s.status.value] = counts.get(s.status.value, 0) + 1
        for c in s.children:
            _walk(c)

    _walk(state)
    total = sum(counts.values())
    parts = [f"{total} agents"]
    for key in ("finished", "waiting", "supervising"):
        if counts.get(key):
            parts.append(f"{counts[key]} {key}")
    return " · ".join(parts)


def live(agent: RLM, state: RLMState) -> list[RLMState]:
    """Run the agent with a live-updating tree display.

    Returns the collected list of states.
    """
    from rich.console import Console, Group
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    con = Console()
    states: list[RLMState] = []
    step = 0

    with Live(console=con, refresh_per_second=8, vertical_overflow="visible") as display:
        while not state.finished:
            display.update(Group(
                Group(Spinner("dots", text=Text(f" step {step + 1}...", style="dim"))),
                Text(_stats(state), style="dim"),
                Text(""),
                _build_tree(state),
            ))

            state = agent.step(state)
            step += 1
            states.append(state)

            display.update(Group(
                Text(f"step {step}", style="bold blue"),
                Text(_stats(state), style="dim"),
                Text(""),
                _build_tree(state),
            ))

    con.print()
    con.print(Text(f"✓ Done in {step} steps", style="bold green"))
    con.print()
    con.print(_build_tree(state))
    if state.result:
        con.print()
        con.print(f"[bold green]Result:[/] {state.result}")

    return states
