"""Local Python runtime — runs code in-process via exec()."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from .runtime import Runtime


class LocalRuntime(Runtime):
    """Execute Python code in a persistent local namespace.

    Comes with all the builtin file/shell tools registered by default.
    """

    def __init__(
        self,
        workspace: Path | str = ".",
    ) -> None:
        super().__init__(workspace=workspace)
        self._namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self.register_builtins()

    def execute(self, code: str, timeout: float | None = None) -> str:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self._namespace)
        except Exception as exc:
            stderr_buf.write(f"{type(exc).__name__}: {exc}\n")

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        parts = [p for p in (out, err) if p.strip()]
        return "\n".join(parts) if parts else "(no output)"

    def inject(self, name: str, value: Any) -> None:
        self._namespace[name] = value
