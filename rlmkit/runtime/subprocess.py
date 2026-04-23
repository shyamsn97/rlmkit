"""Generic subprocess runtime — spawn any argv that runs the REPL server.

The command on the other end just needs to be
``python -m rlmkit.runtime.repl`` (possibly inside a sandbox, container,
or SSH session).  This covers the vast majority of "run agent code
somewhere else" use cases with a single line of config::

    SubprocessRuntime(["python", "-m", "rlmkit.runtime.repl"])

    SubprocessRuntime([
        "docker", "exec", "-i", "my-container",
        "python", "-m", "rlmkit.runtime.repl",
    ])

    SubprocessRuntime([
        "kubectl", "exec", "-i", "my-pod", "--",
        "python", "-m", "rlmkit.runtime.repl",
    ])

    SubprocessRuntime(["ssh", "prod-box", "python", "-m", "rlmkit.runtime.repl"])
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path

from rlmkit.runtime.runtime import DEFAULT_MODULES, Runtime


class SubprocessRuntime(Runtime):
    """Run the REPL server as a subprocess and talk to it over stdio."""

    def __init__(self, argv: list[str], *, workspace: str | Path = ".") -> None:
        super().__init__(workspace=workspace)
        self.argv = list(argv)
        self.proc: sp.Popen | None = None

    def send(self, msg: dict) -> None:
        if self.proc is None:
            self.proc = sp.Popen(
                self.argv,
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                bufsize=0,
            )
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def recv(self) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            err = b""
            if self.proc.stderr is not None:
                try:
                    err = self.proc.stderr.read() or b""
                except Exception:
                    pass
            raise RuntimeError(
                f"REPL subprocess {self.argv!r} exited unexpectedly. "
                f"stderr: {err.decode(errors='replace')}"
            )
        return json.loads(line)

    def terminate(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

    def clone(self) -> SubprocessRuntime:
        new = type(self)(self.argv, workspace=self.workspace)
        for name, td in self.tools.items():
            new.tools[name] = td
            new.inject(name, td.fn)
        return new

    def factory(self) -> SubprocessRuntime:
        """Use as ``runtime_factory=runtime.factory`` on the RLM engine."""
        return self.clone()

    def available_modules(self) -> list[str]:
        return DEFAULT_MODULES
