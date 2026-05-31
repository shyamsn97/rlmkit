"""Fake sandbox providers and fast in-process REPL transport for tests."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

from rlmflow.llm import LLMClient
from rlmflow.runtime.sandbox.remote import RemoteFileRuntime


_REPL_DETECT = "from rlmflow.runtime.repl import main"
_REMOTE_DIR_RE = re.compile(r"mkdir -p (/[^\s]*rlmflow-[a-f0-9]+)\s+(\S+)")
_KILL_PID_RE = re.compile(r"kill \$\(cat ([^)]+)/pid\)")

_inproc_sessions: dict[str, "_InProcessReplSession"] = {}


class _InProcessReplSession:
    """Hold a REPL instance and process queued input synchronously."""

    def __init__(self, remote_dir: str, workdir: str | None = None) -> None:
        from rlmflow.runtime.repl import REPL

        self.remote_dir = Path(remote_dir)
        self.workdir = Path(workdir) if workdir else None
        self.input_path = self.remote_dir / "in.jsonl"
        self.output_path = self.remote_dir / "out.jsonl"
        self.stderr_path = self.remote_dir / "stderr.log"
        self.remote_dir.mkdir(parents=True, exist_ok=True)
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
        for path in (self.input_path, self.output_path, self.stderr_path):
            path.write_text("")
        (self.remote_dir / "pid").write_text("0")
        self.repl = REPL()
        self.offset = 0

    def process_pending(self) -> None:
        if not self.input_path.exists():
            return
        text = self.input_path.read_text()

        prev_cwd = os.getcwd() if self.workdir is not None else None
        try:
            if self.workdir is not None:
                os.chdir(self.workdir)
            while True:
                nl = text.find("\n", self.offset)
                if nl < 0:
                    return
                line = text[self.offset : nl]
                self.offset = nl + 1
                try:
                    msg = json.loads(line)
                    resp = self.repl.handle(msg)
                except Exception as exc:  # noqa: BLE001 - mirror REPL error semantics.
                    resp = {"error": f"{type(exc).__name__}: {exc}"}
                with self.output_path.open("a") as f:
                    f.write(json.dumps(resp) + "\n")
        finally:
            if prev_cwd is not None:
                os.chdir(prev_cwd)


def _drain_all_sessions() -> None:
    for session in _inproc_sessions.values():
        session.process_pending()


def _maybe_handle_repl_lifecycle(command: str) -> tuple[int, str, str] | None:
    if _REPL_DETECT in command:
        match = _REMOTE_DIR_RE.search(command)
        if match is None:
            return None
        remote_dir = match.group(1)
        remote_workdir = match.group(2)
        if remote_dir not in _inproc_sessions:
            _inproc_sessions[remote_dir] = _InProcessReplSession(
                remote_dir,
                workdir=remote_workdir,
            )
        return 0, "", ""
    match = _KILL_PID_RE.search(command)
    if match is not None:
        _inproc_sessions.pop(match.group(1), None)
        return 0, "", ""
    return None


def _maybe_eval_python_dash_c(command: str) -> tuple[int, str, str] | None:
    """Run remote transport `python -c` snippets in-process when possible."""

    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if len(parts) < 3 or parts[0] not in ("python", "python3"):
        return None
    if "-c" not in parts:
        return None
    c_idx = parts.index("-c")
    if c_idx + 1 >= len(parts):
        return None
    script = parts[c_idx + 1]
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(compile(script, "<fake-sandbox>", "exec"), {"__name__": "__main__"})
        except SystemExit as exc:
            code = (
                int(exc.code)
                if isinstance(exc.code, int)
                else 0
                if exc.code is None
                else 1
            )
            if exc.code and not isinstance(exc.code, int):
                stderr.write(str(exc.code))
            return code, stdout.getvalue(), stderr.getvalue()
        except BaseException as exc:  # noqa: BLE001 - mirror subprocess semantics.
            import traceback

            traceback.print_exc(file=stderr)
            del exc
            return 1, stdout.getvalue(), stderr.getvalue()
    return 0, stdout.getvalue(), stderr.getvalue()


def run_local(command: str, *, timeout: float | None = None) -> tuple[int, str, str]:
    repl_lifecycle = _maybe_handle_repl_lifecycle(command)
    if repl_lifecycle is not None:
        return repl_lifecycle
    fast = _maybe_eval_python_dash_c(command)
    if fast is not None:
        _drain_all_sessions()
        return fast
    proc = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


class FakeE2BCommands:
    def run(self, command: str, timeout: float | None = None):
        code, stdout, stderr = run_local(command, timeout=timeout)
        return SimpleNamespace(exit_code=code, stdout=stdout, stderr=stderr)


class FakeE2BSandbox:
    def __init__(self):
        self.commands = FakeE2BCommands()
        self.killed = False

    def kill(self):
        self.killed = True


class FakeE2BSandboxFactory:
    created: list[FakeE2BSandbox] = []

    @classmethod
    def create(cls, **kwargs):
        sandbox = FakeE2BSandbox()
        cls.created.append(sandbox)
        return sandbox


class FakeDaytonaProcess:
    def exec(self, command: str, env=None, timeout: int | None = None):
        code, stdout, stderr = run_local(command, timeout=timeout)
        return SimpleNamespace(
            exit_code=code,
            artifacts=SimpleNamespace(stdout=stdout),
            stderr=stderr,
        )


class FakeDaytonaSandbox:
    def __init__(self):
        self.process = FakeDaytonaProcess()
        self.deleted = False

    def delete(self):
        self.deleted = True


class FakeDaytonaClient:
    def __init__(self):
        self.created: list[FakeDaytonaSandbox] = []

    def create(self, *args, **kwargs):
        sandbox = FakeDaytonaSandbox()
        self.created.append(sandbox)
        return sandbox


class NoopLLM(LLMClient):
    def chat(self, messages, *args, **kwargs) -> str:
        del messages, args, kwargs
        return '```repl\ndone("ok")\n```'


class NoStartRemoteRuntime(RemoteFileRuntime):
    def __init__(self, workspace):
        super().__init__(workspace=workspace, remote_workdir="/workspace")
        self.touched_remote = False

    def send(self, msg: dict) -> None:
        del msg
        self.touched_remote = True
        raise AssertionError("runtime should not start during child spawn")

    def recv(self) -> dict:
        self.touched_remote = True
        raise AssertionError("runtime should not start during child spawn")

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        del command, timeout
        self.touched_remote = True
        raise AssertionError("runtime should not exec during child spawn")

    def list_files(self, remote_root: str) -> list[str]:
        del remote_root
        return []


__all__ = [
    "FakeDaytonaClient",
    "FakeE2BSandboxFactory",
    "NoopLLM",
    "NoStartRemoteRuntime",
]
