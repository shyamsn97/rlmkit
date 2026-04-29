"""Live multi-task viz for the OOLONG runner.

Renders a Rich ``Live`` board with one panel per active worker (mini RLM
tree per task) plus a recent-completions table at the bottom. Falls
back to a no-op reporter when ``rich`` isn't installed or stdout
isn't a TTY.

Wired into ``run.py`` via the :class:`Reporter` context manager: every
worker thread calls ``reporter.start(task_id)`` once, then
``reporter.update(task_id, state)`` after every ``agent.step``, then
``reporter.complete(task_id, row)`` when the task finishes. The board
itself owns all locking — workers never touch the renderer.
"""

from __future__ import annotations

import sys
import time
from threading import Lock
from typing import Any

from rlmkit.node import RLMNode, Status


# ── Mini-tree renderer (compact, no spinners — many panels share screen) ──


def render_node(state: RLMNode):
    from rich.text import Text

    aid = state.agent_id or "root"
    info = Text()
    info.append(aid, style="bold")

    model = state.config.get("model")
    if model:
        info.append(f" ({model})", style="dim")

    status_color = {
        Status.READY: "blue",
        Status.EXECUTING: "yellow",
        Status.SUPERVISING: "magenta",
        Status.FINISHED: "green",
    }.get(state.status, "white")
    info.append(f"  [{state.status.value}]", style=status_color)

    if state.iteration:
        info.append(f"  iter {state.iteration}", style="dim")

    inp, out = state.tree_usage()
    if inp or out:
        info.append(f"  in={inp} out={out}", style="dim")

    if state.finished and state.result:
        preview = state.result.replace("\n", " ⏎ ")[:60]
        info.append(f"  → {preview}", style="dim green")

    return info


def render_tree(state: RLMNode):
    from rich.tree import Tree

    tree = Tree(render_node(state), guide_style="dim")
    for child in state.children:
        tree.add(render_subtree(child))
    return tree


def render_subtree(state: RLMNode):
    from rich.tree import Tree

    sub = Tree(render_node(state), guide_style="dim")
    for child in state.children:
        sub.add(render_subtree(child))
    return sub


# ── Thread-safe state holder + Rich renderable ────────────────────────


class TaskBoard:
    """Snapshot of all in-flight + completed tasks.

    Workers call :meth:`start` / :meth:`update` / :meth:`complete`
    from arbitrary threads. The Rich live thread reads via
    ``__rich__`` — every read takes a consistent snapshot under
    ``self.lock`` so we never tear a render mid-update.
    """

    def __init__(
        self,
        *,
        total: int,
        mode: str,
        model: str,
        fast_model: str | None,
        workers: int,
        max_concurrency: int,
    ) -> None:
        self.total = total
        self.mode = mode
        self.model = model
        self.fast_model = fast_model
        self.workers = workers
        self.max_concurrency = max_concurrency

        self.t_start = time.time()
        self.lock = Lock()
        self.in_flight: dict[str, dict[str, Any]] = {}
        self.completed: list[dict[str, Any]] = []
        self.errors = 0

    def start(self, task_id: str) -> None:
        with self.lock:
            self.in_flight[task_id] = {
                "state": None,
                "started_at": time.time(),
            }

    def update(self, task_id: str, state: RLMNode) -> None:
        with self.lock:
            slot = self.in_flight.get(task_id)
            if slot is not None:
                slot["state"] = state

    def complete(self, task_id: str, row: dict[str, Any]) -> None:
        with self.lock:
            self.in_flight.pop(task_id, None)
            self.completed.append(row)
            if row.get("error"):
                self.errors += 1

    def snapshot(self):
        with self.lock:
            return (
                {k: dict(v) for k, v in self.in_flight.items()},
                list(self.completed),
                self.errors,
                time.time() - self.t_start,
            )

    # ── Rich rendering ────────────────────────────────────────────────

    def __rich__(self):
        from rich.columns import Columns
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        in_flight, completed, errors, elapsed = self.snapshot()
        n_done = len(completed)
        rate = n_done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - n_done) / rate if rate > 0 else 0.0

        # Header line
        header = Text()
        header.append("OOLONG", style="bold cyan")
        header.append("  ·  mode=")
        header.append(self.mode, style="bold")
        header.append("  ·  model=")
        header.append(self.model, style="bold")
        if self.fast_model:
            header.append("  ·  fast=")
            header.append(self.fast_model, style="bold")
        header.append(
            f"  ·  workers={self.workers}  ·  max_concurrency={self.max_concurrency}",
            style="dim",
        )

        # Progress line
        progress = Text()
        progress.append(f"{n_done}", style="bold green")
        progress.append(f"/{self.total} tasks  ·  ")
        if errors:
            progress.append(f"{errors} errors", style="bold red")
            progress.append("  ·  ")
        progress.append(f"rate {rate:.2f}/s", style="dim")
        progress.append("  ·  ")
        progress.append(f"ETA {eta:.0f}s", style="dim")
        progress.append("  ·  ")
        progress.append(f"elapsed {elapsed:.0f}s", style="dim")

        # Active task panels — one per worker currently running
        panels = []
        for task_id in sorted(in_flight):
            slot = in_flight[task_id]
            state = slot["state"]
            t = time.time() - slot["started_at"]

            if state is None:
                body = Text(f"starting...  ({t:.1f}s)", style="dim")
            else:
                body = render_tree(state)

            title = Text()
            title.append(task_id, style="bold")
            title.append(f"  ·  {t:.1f}s", style="dim")
            panels.append(
                Panel(body, title=title, border_style="blue", padding=(0, 1))
            )

        if panels:
            active_block = Columns(panels, expand=True, equal=False)
        else:
            active_block = Text("(no tasks running)", style="dim")

        # Recent completions — last 10
        recent = Table(
            show_header=True,
            header_style="dim",
            show_lines=False,
            expand=False,
            pad_edge=False,
        )
        recent.add_column("task", style="dim")
        recent.add_column("score", justify="right")
        recent.add_column("EM", justify="right")
        recent.add_column("type", style="cyan")
        recent.add_column("in", justify="right")
        recent.add_column("out", justify="right")
        recent.add_column("time", justify="right")
        recent.add_column("tree", style="dim")

        for r in completed[-10:]:
            tr = r.get("tree")
            tree_str = (
                f"d{tr['depth']} n{tr['nodes']} i{tr['total_iterations']}"
                if tr
                else ""
            )
            err = r.get("error")
            row_style = "red" if err else None
            recent.add_row(
                r.get("task_id", "?"),
                f"{r.get('score', 0):.2f}",
                f"{r.get('exact_match', 0):.2f}",
                str(r.get("answer_type", "?")),
                f"{r.get('input_tokens', 0)}",
                f"{r.get('output_tokens', 0)}",
                f"{r.get('elapsed_s', 0):.1f}s",
                tree_str,
                style=row_style,
            )

        return Group(
            header,
            progress,
            Text(""),
            Text("Active:", style="dim bold"),
            active_block,
            Text(""),
            Text("Recent:", style="dim bold"),
            recent,
        )


# ── Reporter — pluggable progress sink ────────────────────────────────


class Reporter:
    """Either a Rich live board (``viz=True``) or plain ``print()`` lines.

    Use as a context manager around the executor loop. Workers and the
    main thread call :meth:`start` / :meth:`update` / :meth:`complete`
    interchangeably; the no-viz path falls through to one progress
    line per completion (same shape as the pre-viz output) so log
    pipelines keep working.
    """

    def __init__(
        self,
        *,
        viz: bool,
        total: int,
        mode: str,
        model: str,
        fast_model: str | None,
        workers: int,
        max_concurrency: int,
    ) -> None:
        if viz:
            try:
                import rich  # noqa: F401
            except ImportError:
                print(
                    "!!! --viz requested but `rich` is not installed; "
                    "falling back to plain progress. `pip install rich`.",
                    file=sys.stderr,
                )
                viz = False

        self.viz = viz
        self.total = total
        self.t_start = time.time()
        self.completed_count = 0
        self.error_count = 0

        self.board: TaskBoard | None = None
        self.live = None

        if self.viz:
            self.board = TaskBoard(
                total=total,
                mode=mode,
                model=model,
                fast_model=fast_model,
                workers=workers,
                max_concurrency=max_concurrency,
            )

    def __enter__(self) -> Reporter:
        if self.viz and self.board is not None:
            from rich.console import Console
            from rich.live import Live

            self.console = Console()
            self.live = Live(
                self.board,
                console=self.console,
                refresh_per_second=4,
                vertical_overflow="visible",
                transient=False,
            )
            self.live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self.live is not None:
            self.live.__exit__(*exc)
            self.live = None

    def start(self, task_id: str) -> None:
        if self.board is not None:
            self.board.start(task_id)

    def update(self, task_id: str, state: RLMNode) -> None:
        if self.board is not None:
            self.board.update(task_id, state)

    def complete(self, task_id: str, row: dict[str, Any]) -> None:
        self.completed_count += 1
        if row.get("error"):
            self.error_count += 1

        if self.board is not None:
            self.board.complete(task_id, row)
            return

        elapsed = time.time() - self.t_start
        rate = self.completed_count / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.completed_count) / rate if rate > 0 else 0.0
        tr = row.get("tree")
        tree_str = (
            f" depth={tr['depth']} nodes={tr['nodes']} iter={tr['total_iterations']}"
            if tr
            else ""
        )
        print(
            f">>> [{self.completed_count}/{self.total}] {row.get('task_id', '?')}  "
            f"score={row.get('score', 0):.2f} "
            f"EM={row.get('exact_match', 0):.2f} "
            f"type={row.get('answer_type', '?')} "
            f"in={row.get('input_tokens', 0)} "
            f"out={row.get('output_tokens', 0)} "
            f"{row.get('elapsed_s', 0)}s{tree_str}  "
            f"(rate {rate:.2f}/s, ETA {eta:.0f}s)"
        )
        if row.get("error"):
            print(f"    ERROR: {row['error'].splitlines()[0]}")


__all__ = ["Reporter", "TaskBoard", "render_node", "render_tree"]
