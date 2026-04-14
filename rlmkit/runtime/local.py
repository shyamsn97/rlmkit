"""Local Python runtime — runs code in-process via REPL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rlmkit.runtime.repl import REPL
from rlmkit.runtime.runtime import Runtime


class LocalRuntime(Runtime):
    """Execute Python code in a persistent local namespace."""

    def __init__(self, workspace: str | Path = ".") -> None:
        super().__init__(workspace=workspace)
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self.repl = REPL(self.namespace)
        for mod in self.available_modules():
            self.namespace[mod] = __import__(mod)
        os.chdir(self.workspace)

    def execute(self, code: str, timeout: float | None = None) -> str:
        return self.repl.execute(code)

    def inject(self, name: str, value: Any) -> None:
        self.namespace[name] = value

    def start_code(self, code: str) -> tuple[bool, object]:
        return self.repl.start(code)

    def resume_code(self, send_value=None) -> tuple[bool, object]:
        return self.repl.resume(send_value)

    def available_modules(self) -> list[str]:
        return ["re", "os", "json", "math", "collections", "itertools", "functools"]
