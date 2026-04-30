"""Generic subprocess runtime — spawn any argv that runs the REPL server.

The command on the other end just needs to be
``python -m rlmflow.runtime.repl`` (possibly inside a sandbox, container,
or SSH session).  This covers the vast majority of "run agent code
somewhere else" use cases with a single line of config::

    SubprocessRuntime(["python", "-m", "rlmflow.runtime.repl"])

    SubprocessRuntime([
        "docker", "exec", "-i", "my-container",
        "python", "-m", "rlmflow.runtime.repl",
    ])

    SubprocessRuntime([
        "kubectl", "exec", "-i", "my-pod", "--",
        "python", "-m", "rlmflow.runtime.repl",
    ])

    SubprocessRuntime(["ssh", "prod-box", "python", "-m", "rlmflow.runtime.repl"])
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path
from typing import Any

from rlmflow.runtime.runtime import Runtime


class SubprocessRuntime(Runtime):
    """Run the REPL server as a subprocess and talk to it over stdio."""

    def __init__(self, argv: list[str], *, workspace: str | Path | Any = ".") -> None:
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
                cwd=self.workspace,
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

    def close(self) -> None:
        """Tear down the REPL subprocess and release its pipe FDs.

        Closes ``stdin`` first — the ``serve()`` loop in ``rlmflow.runtime.repl``
        reads until EOF, so that's enough for a graceful shutdown in the
        common case (the REPL exits, the container's ``--rm`` flag wipes
        it). We only escalate to ``terminate()``/``kill()`` if the child
        is still alive after a short wait, then close the remaining
        pipes and reap the process so its FDs aren't left behind for
        the GC to clean up at some unspecified later time.
        """
        proc, self.proc = self.proc, None
        if proc is None:
            return

        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass

        try:
            proc.wait(timeout=2)
        except sp.TimeoutExpired:
            for action in (proc.terminate, proc.kill):
                try:
                    action()
                    proc.wait(timeout=2)
                    break
                except Exception:
                    continue
        except Exception:
            pass

        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception:
                pass

    def clone(self, workspace: str | Path | None = None) -> SubprocessRuntime:
        new = self.__class__(self.argv, workspace=workspace or self.workspace)
        for name, td in self.tools.items():
            new.tools[name] = td
            new.inject(name, td.fn)
        return new
