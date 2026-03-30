"""Rich-powered logger — colorful streaming output with panels and syntax highlighting."""

from __future__ import annotations

from .base import Logger

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


class RichLogger(Logger):
    """Pretty agent output using Rich.

    Requires ``pip install rich``. Falls back to no-op ``Logger`` if
    rich is not installed.
    """

    def __init__(self, console: Console | None = None) -> None:
        if not _HAS_RICH:
            raise ImportError(
                "RichLogger requires `rich`. Install with: pip install rich"
            )
        self.console = console or Console()
        self._live: Live | None = None
        self._token_buf: list[str] = []

    def on_run_start(self, agent_id: str, task: str, max_iterations: int) -> None:
        self.console.rule(
            f"[bold cyan]{agent_id}[/] — {max_iterations} iters max", style="cyan"
        )
        self.console.print(f"  [dim]{task[:120]}[/dim]")

    def on_run_end(self, agent_id: str, result: str, iterations_used: int) -> None:
        self.console.rule(
            f"[bold green]{agent_id}[/] done ({iterations_used} iters)", style="green"
        )

    def on_iter_start(self, agent_id: str, iteration: int) -> None:
        self.console.print()
        self.console.rule(f"[bold]{agent_id}[/] iter {iteration}", style="dim")

    def on_llm_start(self, agent_id: str, iteration: int, num_messages: int) -> None:
        self._token_buf.clear()

    def on_llm_token(self, agent_id: str, token: str) -> None:
        self._token_buf.append(token)
        self.console.print(token, end="", highlight=False)

    def on_llm_end(self, agent_id: str, text: str) -> None:
        self.console.print()

    def on_exec_start(self, agent_id: str, code: str) -> None:
        self.console.print()
        self.console.print(
            Panel(
                Syntax(code, "python", theme="monokai", line_numbers=True),
                title="[bold yellow]repl[/]",
                border_style="yellow",
                expand=False,
            )
        )

    def on_exec_end(self, agent_id: str, output: str) -> None:
        style = "green" if "Error" not in output else "red"
        self.console.print(
            Panel(
                Text(output),
                title=f"[bold {style}]output[/]",
                border_style=style,
                expand=False,
            )
        )

    def on_no_code_block(self, agent_id: str, iteration: int) -> None:
        self.console.print(f"  [yellow]⚠ no code block (iter {iteration})[/]")

    def on_stall(self, agent_id: str, iteration: int) -> None:
        self.console.print(f"  [yellow]⚠ stall detected (iter {iteration})[/]")

    def on_done(self, agent_id: str, message: str) -> None:
        if message:
            self.console.print(f"  [bold green]✓ done:[/] {message[:120]}")
        else:
            self.console.print("  [bold green]✓ done[/]")

    def on_delegate(self, agent_id: str, child_id: str, task: str, wait: bool) -> None:
        mode = "[bold]sync[/]" if wait else "[dim]async[/]"
        self.console.print(
            f"  [blue]→ delegate[/] [{mode}] [cyan]{child_id}[/]: {task[:80]}"
        )

    def on_delegate_refused(self, agent_id: str, max_depth: int) -> None:
        self.console.print(
            f"  [bold red]✗ delegation refused[/] (max depth {max_depth})"
        )
