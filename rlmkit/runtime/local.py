"""Local Python runtime — runs code in-process via REPL."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rlmkit.runtime.repl import REPL
from rlmkit.runtime.runtime import DEFAULT_MODULES, Runtime


class LocalRuntime(Runtime):
    """Execute Python code in a persistent local namespace.

    The ``workspace`` is entered only for the duration of each ``execute`` /
    ``start_code`` / ``resume_code`` call, then the caller's CWD is restored.
    This keeps relative paths in the main script (session dirs, trace dirs,
    etc.) resolving against the user's CWD — not against the workspace.
    """

    def __init__(self, workspace: str | Path = ".") -> None:
        super().__init__(workspace=workspace)
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self.repl = REPL(self.namespace)
        for mod in self.available_modules():
            self.namespace[mod] = __import__(mod)

    @contextmanager
    def _in_workspace(self):
        prev = os.getcwd()
        os.chdir(self.workspace)
        try:
            yield
        finally:
            os.chdir(prev)

    def execute(self, code: str, timeout: float | None = None) -> str:
        with self._in_workspace():
            return self.repl.execute(code)

    def inject(self, name: str, value: Any) -> None:
        self.namespace[name] = value

    def start_code(self, code: str) -> tuple[bool, object]:
        with self._in_workspace():
            return self.repl.start(code)

    def resume_code(self, send_value=None) -> tuple[bool, object]:
        with self._in_workspace():
            return self.repl.resume(send_value)

    def available_modules(self) -> list[str]:
        return DEFAULT_MODULES
