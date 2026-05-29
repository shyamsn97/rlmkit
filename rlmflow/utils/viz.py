"""Live tree view, Gantt swimlanes, message logs, and webhook helpers.

Public visualization helpers accept the same source shapes as the HTML viewer:
a :class:`~rlmflow.workspace.Workspace`, workspace path, standalone graph path,
in-memory :class:`~rlmflow.graph.Graph`, or graph list. Trace operations use all
snapshots; single-snapshot operations use the latest graph.
"""

from __future__ import annotations

import difflib
import json
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable

from rlmflow.graph import Graph, is_done, is_errored, is_supervising
from rlmflow.utils.export import _kind
from rlmflow.utils.viewer import ViewSource, resolve_graphs


def _resolve_graphs(source: ViewSource) -> list[Graph]:
    return resolve_graphs(source)


def _resolve_latest_graph(source: ViewSource) -> Graph:
    graphs = _resolve_graphs(source)
    if not graphs:
        raise ValueError("expected at least one Graph")
    return graphs[-1]


# ── live tree ────────────────────────────────────────────────────────


class LiveView:
    """Live-updating Rich tree of an RLMFlow run."""

    def __init__(self, *, console: Any = None) -> None:
        from rich.console import Console

        self._console = console or Console()
        self._live: Any = None

    def __enter__(self) -> LiveView:
        from rich.live import Live

        self._live = Live(
            console=self._console,
            vertical_overflow="visible",
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    def __call__(self, source: ViewSource) -> None:
        if self._live is None:
            raise RuntimeError("live_view used outside of context")
        latest = _resolve_latest_graph(source)
        self._live.update(_render_rich_tree(latest), refresh=True)


def _render_rich_tree(graph: Graph):
    from rich.text import Text
    from rich.tree import Tree

    def is_settled(aid: str) -> bool:
        sub = graph.agents.get(aid)
        cur = sub.current() if sub is not None else None
        return bool(cur and (is_done(cur) or is_errored(cur)))

    def running_children(sub: Graph) -> tuple[int, int] | None:
        cur = sub.current()
        if not is_supervising(cur):
            return None
        waiting_on = list(cur.waiting_on or [])
        active = sum(1 for child_id in waiting_on if not is_settled(child_id))
        return active, len(waiting_on)

    def label_for(aid: str) -> Text:
        sub = graph.agents[aid]
        cur = sub.current()
        info = Text()
        info.append(aid, style="bold")
        info.append(f" [{sub.model_label}]", style="cyan")
        if cur is not None:
            info.append(
                f" [{cur.type}]", style="magenta" if not cur.terminal else "green"
            )
        counts = running_children(sub)
        if counts is not None:
            active, total = counts
            info.append(f" | children running {active}/{total}", style="cyan")
        return info

    def populate(tree: Tree, aid: str) -> None:
        for child_aid in graph.agents[aid].children:
            child = tree.add(label_for(child_aid), guide_style="dim")
            populate(child, child_aid)

    def build(aid: str) -> Tree:
        tree = Tree(label_for(aid), guide_style="dim")
        populate(tree, aid)
        return tree

    return build(graph.root_agent_id)


def live_view(**kwargs: Any) -> LiveView:
    return LiveView(**kwargs)


def live(agent: Any, source: ViewSource) -> list[Graph]:
    """Run ``agent``'s step loop while streaming a live tree."""
    graph = _resolve_latest_graph(source)
    graphs = [graph]
    with LiveView() as live:
        live(graph)
        while not graph.finished:
            graph = agent.step(graph)
            graphs.append(graph)
            live(graph)
    return graphs


# ── gantt swimlanes ──────────────────────────────────────────────────


def _gantt_per_step(graphs: list[Graph]) -> tuple[list[str], list[list[str | None]]]:
    order: list[str] = []
    seen: set[str] = set()
    per_step: list[dict[str, str]] = []
    for graph in graphs:
        step_map: dict[str, str] = {}
        for aid, sub in graph.agents.items():
            cur = sub.current()
            if cur is None:
                continue
            step_map[aid] = f"{_kind(cur)} ({sub.model_label})"
            if aid not in seen:
                seen.add(aid)
                order.append(aid)
        per_step.append(step_map)
    return order, [[step.get(aid) for step in per_step] for aid in order]


_TYPE_CELL = {
    "query": ("Q", "blue"),
    "llm_call": ("a", "yellow"),
    "llm": ("A", "yellow"),
    "exec_call": ("o", "blue"),
    "exec": ("O", "blue"),
    "supervising": ("S", "magenta"),
    "resume_call": ("r", "green"),
    "resume": ("R", "green"),
    "errored": ("E", "red"),
    "done": ("F", "green"),
}

_TYPE_HTML = {
    "query": "#58a6ff",
    "llm_call": "#a98a2a",
    "llm": "#d29922",
    "exec_call": "#b87650",
    "exec": "#ff9e64",
    "supervising": "#bc8cff",
    "resume_call": "#5fa067",
    "resume": "#7ee787",
    "errored": "#f85149",
    "done": "#3fb950",
}


def gantt(source: ViewSource) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    graphs = _resolve_graphs(source)
    agents, rows = _gantt_per_step(graphs)
    table = Table(
        show_header=True,
        header_style="dim",
        show_lines=False,
        pad_edge=False,
        padding=(0, 0),
    )
    table.add_column("agent", style="bold", no_wrap=True)
    for i in range(len(graphs)):
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


def gantt_html(source: ViewSource, *, title: str = "rlmflow gantt") -> str:
    graphs = _resolve_graphs(source)
    agents, rows = _gantt_per_step(graphs)
    n_steps = len(graphs)
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


# ── error / code log ─────────────────────────────────────────────────


def error_summary(source: ViewSource) -> str:
    """Group every :class:`ErrorOutput` in ``source`` by ``error`` kind."""
    graph = _resolve_latest_graph(source)
    errors = [e for e in graph.nodes if is_errored(e)]
    if not errors:
        return "(no errors)"
    by_kind: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for err in errors:
        kind = err.error or "(unknown)"
        by_kind[kind] += 1
        if kind not in samples:
            head = (err.content or "").strip().splitlines()
            samples[kind] = head[0] if head else ""
    lines = [f"{len(errors)} error(s) across {len(by_kind)} kind(s):"]
    for kind, count in by_kind.most_common():
        lines.append(f"  {kind}: {count}")
        if samples.get(kind):
            lines.append(f"    └─ {samples[kind][:120]}")
    return "\n".join(lines)


def code_log(
    source: ViewSource,
    agent_id: str | None = None,
) -> str:
    """Render every code block executed in the run, paired with its output."""
    graphs = _resolve_graphs(source)
    if not graphs:
        return "(no code blocks)"
    graph = graphs[-1]

    nodes = list(graph.nodes)
    if agent_id:
        nodes = [n for n in nodes if n.agent_id == agent_id]

    # Pair each ExecAction / ResumeAction with the CodeObservation that
    # immediately follows it in the same agent's trajectory.
    by_agent: dict[str, list] = {}
    for n in nodes:
        by_agent.setdefault(n.agent_id, []).append(n)

    out: list[str] = []
    for aid, states in by_agent.items():
        for i, node in enumerate(states):
            if node.type not in ("exec_action", "resume_action"):
                continue
            code = getattr(node, "code", "") or ""
            if not code:
                continue
            out.append(f"# [{aid}] {node.type}")
            out.append(code.strip())
            obs = states[i + 1] if i + 1 < len(states) else None
            output = ""
            if obs is not None:
                output = (
                    getattr(obs, "content", "")
                    or getattr(obs, "output", "")
                    or getattr(obs, "result", "")
                    or ""
                )
            if output:
                out.append("→ " + output.strip()[:240])
            out.append("")
    return "\n".join(out).rstrip() or "(no code blocks)"


def message_stream(agent_id: str, source: ViewSource) -> str:
    """Render the chat-log transcript for one agent of a :class:`Graph`."""
    graph = _resolve_latest_graph(source)
    if agent_id not in graph.agents:
        return f"(no nodes for agent {agent_id!r})"
    return graph.agents[agent_id].transcript(include_system=True)


def diff_system_prompts(
    source_a: ViewSource,
    source_b: ViewSource,
    aid: str = "root",
) -> str:
    """Unified diff of two ``system_prompt`` strings for agent ``aid``."""
    graph_a = _resolve_latest_graph(source_a)
    graph_b = _resolve_latest_graph(source_b)
    a = (graph_a.agents[aid].system_prompt if aid in graph_a.agents else "").splitlines(
        keepends=True
    )
    b = (graph_b.agents[aid].system_prompt if aid in graph_b.agents else "").splitlines(
        keepends=True
    )
    diff = difflib.unified_diff(a, b, fromfile=aid + " (a)", tofile=aid + " (b)", n=3)
    return "".join(diff) or "(prompts identical)"


# ── cost & tokens ────────────────────────────────────────────────────


def token_sparkline(source: ViewSource, width: int = 40) -> str:
    """One-line ASCII sparkline of cumulative tokens across steps."""
    graphs = _resolve_graphs(source)
    if not graphs:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    cum = [g.total_tokens() for g in graphs]
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
    return f"{bars}  {cum[-1]:>6} tok over {len(graphs)} steps"


def budget_burndown(
    source: ViewSource,
    max_budget: int | None = None,
    *,
    width: int = 40,
) -> str:
    """Cumulative tokens vs ``max_budget``. Falls back to peak if no budget."""
    graphs = _resolve_graphs(source)
    if not graphs:
        return ""
    cum = [g.total_tokens() for g in graphs]
    final = cum[-1]
    target = max_budget or max(cum) or 1
    pct = min(1.0, final / target)
    filled = int(round(pct * width))
    bar = "█" * filled + "·" * (width - filled)
    label = f"{final}/{target}" if max_budget else f"{final} tok (peak)"
    return f"[{bar}] {pct * 100:5.1f}%  {label}"


# ── reports ──────────────────────────────────────────────────────────


def report_md(
    source: ViewSource,
    *,
    title: str = "rlmflow run",
    max_budget: int | None = None,
) -> str:
    """Render a Markdown summary of a run: tree + cost + final result."""
    graphs = _resolve_graphs(source)
    if not graphs:
        return f"# {title}\n\n(empty trace)\n"

    final = graphs[-1]
    inp, out = final.tokens()
    parts: list[str] = [f"# {title}", ""]

    parts.append(f"**Steps:** {len(graphs)}")
    parts.append(f"**Agents:** {len(final.agents)}")
    parts.append(f"**Tokens:** {inp + out:,} ({inp:,} in, {out:,} out)")
    if max_budget is not None:
        parts.append(f"**Budget:** {budget_burndown(graphs, max_budget)}")
    current = final.current()
    parts.append(f"**Outcome:** {current.type if current else 'empty'}")
    errors = [e for e in final.nodes if is_errored(e)]
    if errors:
        parts.append(f"**Errors:** {len(errors)}")

    parts.extend(["", "## Tree", "", "```", final.tree(), "```"])
    parts.extend(
        ["", "## Cumulative tokens", "", "```", token_sparkline(graphs), "```"]
    )

    if errors:
        parts.extend(["", "## Errors", "", "```", error_summary(final), "```"])

    result = final.result()
    if result:
        parts.extend(["", "## Result", "", "```", str(result), "```"])

    return "\n".join(parts) + "\n"


# ── comparison ───────────────────────────────────────────────────────


def bench_table(
    sources: dict[str, ViewSource],
    *,
    pricing: Callable[[Graph], float] | None = None,
) -> str:
    """One row per labeled source: outcome, steps, agents, tokens, errors."""
    if not sources:
        return "(no sources)"
    header = ["label", "steps", "agents", "outcome", "tokens", "errors"]
    if pricing is not None:
        header.append("cost")
    rows: list[list[str]] = [header]

    for label, source in sources.items():
        graphs = _resolve_graphs(source)
        if not graphs:
            rows.append([label, "0", "0", "(empty)", "0", "0"])
            continue
        final = graphs[-1]
        agents = len(final.agents)
        tokens = final.total_tokens()
        errors = sum(1 for e in final.nodes if is_errored(e))
        cur = final.current()
        outcome = (
            "done"
            if cur and is_done(cur)
            else "errored" if cur and is_errored(cur) else "open"
        )
        row = [
            label,
            str(len(graphs)),
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


def tee(
    source: Iterable[Graph] | Graph | str | Path,
    *sinks: Callable[[Graph], Any],
) -> Iterator[Graph]:
    """Fan a step iterator out to multiple sinks while still yielding each graph."""
    if isinstance(source, (Graph, str, Path)) or hasattr(source, "load_steps"):
        graph_source: Iterable[Graph] = _resolve_graphs(source)
    else:
        graph_source = source
    for graph in graph_source:
        for sink in sinks:
            try:
                sink(graph)
            except (
                Exception
            ) as exc:  # noqa: BLE001 — one bad sink shouldn't stop the stream
                warnings.warn(
                    f"tee sink {getattr(sink, '__name__', sink)!r} failed: "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
        yield graph


def _webhook_payload(graph: Graph, *, title: str) -> dict[str, str]:
    inp, out = graph.tokens()
    errors = sum(1 for e in graph.nodes if is_errored(e))
    body = (
        f"*{title}*\n"
        f"agents: {len(graph.agents)}  "
        f"tokens: {inp + out:,}  "
        f"errors: {errors}\n"
        f"```\n{graph.tree()}\n```"
    )
    return {"text": body}


def slack_webhook(url: str, source: ViewSource, *, title: str = "rlmflow run") -> int:
    graph = _resolve_latest_graph(source)
    return _post_json(url, _webhook_payload(graph, title=title))


def discord_webhook(url: str, source: ViewSource, *, title: str = "rlmflow run") -> int:
    graph = _resolve_latest_graph(source)
    payload = _webhook_payload(graph, title=title)
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


def ascii_boxes(source: ViewSource) -> str:
    """Boxed-tree variant of :meth:`Graph.tree` using Rich panels."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree

    graph = _resolve_latest_graph(source)

    def _agent_label(aid: str) -> Panel:
        sub = graph.agents[aid]
        cur = sub.current()
        head = Text()
        head.append(aid, style="bold")
        if cur is not None:
            head.append(
                f"  [{cur.type}]",
                style="magenta" if not cur.terminal else "green",
            )
        head.append(f"  {{{sub.model_label}}}", style="cyan")
        result = sub.result()
        body = ""
        if result:
            body = str(result)[:160].replace("\n", " ")
        return Panel(
            Text(body) if body else Text(""),
            title=head,
            border_style="dim",
            padding=(0, 1),
        )

    def build(aid: str) -> Tree:
        tree = Tree(_agent_label(aid), guide_style="dim")
        for child_aid in graph.agents[aid].children:
            tree.add(build(child_aid))
        return tree

    con = Console(record=True, width=120)
    con.print(build(graph.root_agent_id))
    return con.export_text()


__all__ = [
    "LiveView",
    "ascii_boxes",
    "bench_table",
    "budget_burndown",
    "code_log",
    "diff_system_prompts",
    "discord_webhook",
    "error_summary",
    "gantt",
    "gantt_html",
    "live",
    "live_view",
    "message_stream",
    "report_md",
    "slack_webhook",
    "tee",
    "token_sparkline",
]
