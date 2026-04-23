"""In-process Python runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlmkit.runtime.repl import REPL
from rlmkit.runtime.runtime import DEFAULT_MODULES, Runtime


class LocalRuntime(Runtime):
    """Execute agent code in the current Python process.

    ``send``/``recv`` stash a message and dispatch it to an in-process
    :class:`REPL` — no wire.  ``inject`` is overridden to bind values
    directly into the REPL namespace, so the proxy branch of
    :meth:`Runtime.call` never fires for local use.

    The workspace is entered only for the duration of each dispatched
    message so that relative paths in the host script still resolve
    against the caller's CWD.
    """

    def __init__(self, workspace: str | Path = ".") -> None:
        super().__init__(workspace=workspace)
        self.repl = REPL()
        for mod in DEFAULT_MODULES:
            self.repl.namespace[mod] = __import__(mod)
        self.pending: dict | None = None

    def send(self, msg: dict) -> None:
        self.pending = msg

    def recv(self) -> dict:
        assert self.pending is not None
        msg, self.pending = self.pending, None
        with self._in_workspace():
            return self.repl.handle(msg)

    def inject(self, name: str, value: Any) -> None:
        # In-process: bind directly, skip the proxy protocol entirely.
        self.repl.namespace[name] = value

    def available_modules(self) -> list[str]:
        return DEFAULT_MODULES
