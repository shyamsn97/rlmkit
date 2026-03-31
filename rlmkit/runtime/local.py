"""Local Python runtime — runs code in-process via exec()."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from .runtime import Runtime

_builtin_print = print


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
        for mod in self.available_modules():
            self._namespace[mod] = __import__(mod)
        self.register_builtins()

    def execute(self, code: str, timeout: float | None = None) -> str:
        buf = io.StringIO()
        self._namespace["print"] = lambda *a, **kw: _builtin_print(
            *a, **{**kw, "file": buf}
        )
        try:
            exec(code, self._namespace)
        except Exception as exc:
            buf.write(f"\n{type(exc).__name__}: {exc}")
        out = buf.getvalue().strip()
        return out

    def inject(self, name: str, value: Any) -> None:
        self._namespace[name] = value

    def available_modules(self) -> list[str]:
        return ["re", "os", "json", "math", "collections", "itertools", "functools"]
