"""Local Python runtime — runs code in-process via exec()."""

from __future__ import annotations

import io
import textwrap
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
        self._gen = None
        self._buf = None
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

    # ── generator-based execution ─────────────────────────────────────

    def start_code(self, code: str) -> tuple[bool, object]:
        self._buf = io.StringIO()
        self._namespace["print"] = lambda *a, **kw: _builtin_print(
            *a, **{**kw, "file": self._buf}
        )
        indented = textwrap.indent(code, "    ")
        exec(f"def __rlm_gen__():\n{indented}\n", self._namespace)
        self._gen = self._namespace["__rlm_gen__"]()
        return self._drive()

    def resume_code(self, send_value=None) -> tuple[bool, object]:
        return self._drive(send_value=send_value)

    def _drive(self, send_value=None):
        try:
            request = self._gen.send(send_value)
            return True, request
        except StopIteration:
            return False, self._buf.getvalue().strip()
        except Exception as exc:
            self._buf.write(f"\n{type(exc).__name__}: {exc}")
            return False, self._buf.getvalue().strip()

    def available_modules(self) -> list[str]:
        return ["re", "os", "json", "math", "collections", "itertools", "functools"]
