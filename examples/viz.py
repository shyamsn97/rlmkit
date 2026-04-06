"""Lightweight live terminal visualization for RLM agents.

Requires: ``pip install rich``

Usage::

    from viz import live

    state = agent.start("task")
    states = live(agent, state)
"""

from __future__ import annotations

from rlmkit.rlm import RLM
from rlmkit.state import ChildStep, CodeExec, LLMReply, NoCodeBlock, RLMState

STATUS_STYLE = {
    "waiting": "blue",
    "has_reply": "yellow",
    "supervising": "magenta",
    "finished": "green",
}


def _rich_tree(state: RLMState):
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


def _format_event(event):
    from rich.console import Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    parts: list = []

    if isinstance(event, LLMReply):
        parts.append(
            Text.from_markup(
                f"[blue bold]{event.agent_id}[/] — "
                f"LLM reply [dim](iter {event.iteration})[/]"
            )
        )
        if event.code:
            parts.append(Syntax(event.code, "python", theme="monokai", line_numbers=False, word_wrap=True))
        elif event.text:
            parts.append(Text(event.text[:300], style="dim"))

    elif isinstance(event, CodeExec):
        out = (event.output or "").strip()
        has_error = out and any(kw in out for kw in ("Error:", "Traceback", "SyntaxError"))
        if has_error:
            tag, style = "error", "red bold"
        elif event.suspended:
            tag, style = "suspended", "yellow bold"
        else:
            tag, style = "done", "green bold"
        parts.append(
            Text.from_markup(
                f"[{style}]{event.agent_id}[/] — exec → [{style}]{tag}[/] [dim](iter {event.iteration})[/]"
            )
        )
        if event.code:
            parts.append(Syntax(event.code, "python", theme="monokai", line_numbers=False, word_wrap=True))
        if out:
            out_style = "red" if has_error else "dim"
            parts.append(Panel(Text(out[:800], style=out_style), title="output", title_align="left", border_style=out_style, expand=False))

    elif isinstance(event, ChildStep):
        n = len(event.child_events)
        if event.agent_finished:
            tag, style = "agent finished", "green bold"
        elif event.all_done:
            tag, style = "all done", "green bold"
        else:
            tag, style = f"in progress ({n} events)", "magenta"
        parts.append(
            Text.from_markup(f"[{style}]{event.agent_id}[/] — children [{style}]{tag}[/]")
        )
        for ce in event.child_events[:12]:
            parts.append(_oneliner(ce))
        if n > 12:
            parts.append(Text(f"  ... and {n - 12} more", style="dim"))

    elif isinstance(event, NoCodeBlock):
        parts.append(Text.from_markup(f"[yellow]{event.agent_id}[/] — no code block"))

    return Group(*parts) if parts else Text("")


def _oneliner(event):
    from rich.text import Text

    if isinstance(event, LLMReply):
        return Text.from_markup(f"  [dim]├[/] [blue]{event.agent_id}[/] [dim]LLM reply (iter {event.iteration})[/]")
    if isinstance(event, CodeExec):
        tag = "suspended" if event.suspended else "done"
        c = "yellow" if event.suspended else "green"
        out = (event.output or "").strip()
        if out and any(kw in out for kw in ("Error:", "Traceback", "SyntaxError")):
            tag, c = "error", "red"
        snippet = f" [dim]→ {out}[/]" if out and len(out) < 80 else ""
        return Text.from_markup(f"  [dim]├[/] [{c}]{event.agent_id}[/] [dim]exec →[/] [{c}]{tag}[/]{snippet}")
    if isinstance(event, ChildStep):
        return Text.from_markup(f"  [dim]├[/] [magenta]{event.agent_id}[/] [dim]children ({len(event.child_events)} events)[/]")
    return Text.from_markup(f"  [dim]├[/] [yellow]{event.agent_id}[/] [dim]no code[/]")


def live(agent: RLM, state: RLMState) -> list[RLMState]:
    """Run the agent with a live-updating rich terminal display.

    Returns the collected list of states.
    """
    from rich.console import Console, Group
    from rich.live import Live
    from rich.rule import Rule
    from rich.text import Text

    con = Console()
    states: list[RLMState] = []
    step = 0

    with Live(console=con, refresh_per_second=4, vertical_overflow="visible") as display:
        while not state.finished:
            state = agent.step(state)
            step += 1
            states.append(state)

            counts = {}
            def _count(s):
                counts[s.status.value] = counts.get(s.status.value, 0) + 1
                for c in s.children:
                    _count(c)
            _count(state)

            total = sum(counts.values())
            stats_parts = [f"step {step}", f"{total} agents"]
            for key in ("finished", "waiting", "supervising"):
                if counts.get(key):
                    stats_parts.append(f"{counts[key]} {key}")

            display.update(
                Group(
                    Rule(f"[bold]Step {step}[/]", style="blue"),
                    Text(" · ".join(stats_parts), style="dim"),
                    Text(""),
                    _rich_tree(state),
                    Text(""),
                    _format_event(state.event) if state.event else Text(""),
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
