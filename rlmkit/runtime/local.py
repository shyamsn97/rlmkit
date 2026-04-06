"""Local Python runtime — runs code in-process via Sandbox."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .runtime import Runtime
from .sandbox import Sandbox


class LocalRuntime(Runtime):
    """Execute Python code in a persistent local namespace.

    Comes with all the builtin file/shell tools registered by default.
    Sets cwd to the workspace so ``open("file.txt")`` works naturally.
    """

    def __init__(
        self,
        workspace: Path | str = ".",
    ) -> None:
        super().__init__(workspace=workspace)
        os.chdir(self.workspace)
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self.sandbox = Sandbox(self.namespace)
        for mod in self.available_modules():
            self.namespace[mod] = __import__(mod)
        self.register_builtins()

    def execute(self, code: str, timeout: float | None = None) -> str:
        return self.sandbox.execute(code)

    def inject(self, name: str, value: Any) -> None:
        self.namespace[name] = value

    def start_code(self, code: str) -> tuple[bool, object]:
        return self.sandbox.start(code)

    def resume_code(self, send_value=None) -> tuple[bool, object]:
        return self.sandbox.resume(send_value)

    def available_modules(self) -> list[str]:
        return ["re", "os", "json", "math", "collections", "itertools", "functools"]
