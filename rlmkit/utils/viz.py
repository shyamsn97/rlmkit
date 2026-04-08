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
