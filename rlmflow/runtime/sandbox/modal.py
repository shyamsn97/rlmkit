"""Modal runtime — runs agent code in a remote Modal container.

Requires ``modal`` to be installed (``pip install rlmflow[modal]``).

Usage::

    import modal
    from rlmflow.runtime.sandbox.modal import ModalRuntime

    runtime = ModalRuntime(
        app_name="my-rlm-app",
        image=modal.Image.debian_slim().pip_install("rlmflow"),
    )
    agent = RLMFlow(llm_client=llm, runtime=runtime, runtime_factory=runtime.clone)
"""

from __future__ import annotations

import json
import posixpath
import shlex
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue

from rlmflow.runtime.sandbox.remote import RemoteFileRuntime
from rlmflow.workspace import BaseWorkspace


class ModalRuntime(RemoteFileRuntime):
    """Execute agent code inside a Modal Sandbox.

    The Modal Sandbox itself runs one long-lived ``rlmflow.runtime.repl``
    process as its entrypoint. The runtime talks to that process through
    Modal's documented Sandbox ``stdin`` / ``stdout`` streams. ``Sandbox.exec``
    is only used for short shell commands and workspace file operations.

    Parameters
    ----------
    app_name : str
        Modal app name (created if missing).
    image : modal.Image | None
        Container image.  Defaults to ``debian_slim + rlmflow``.
    timeout : int
        Container lifetime in seconds (default 1 hour).
    **container_kwargs
        Extra kwargs forwarded to ``modal.Sandbox.create()``
        (e.g. ``gpu``, ``secrets``, ``volumes``).
    """

    def __init__(
        self,
        app_name: str = "rlmflow",
        *,
        workspace: BaseWorkspace | str | Path = ".",
        remote_workdir: str = "/workspace",
        image=None,
        timeout: int = 3600,
        repl_timeout: float = 30,
        verbose: bool = False,
        trace: bool = False,
        **container_kwargs,
    ) -> None:
        super().__init__(
            workspace=workspace,
            remote_workdir=remote_workdir,
            repl_timeout=repl_timeout,
        )
        self.app_name = app_name
        self.image = image
        self.timeout = timeout
        self.verbose = verbose
        self.trace = trace
        self.container_kwargs = container_kwargs
        self.container = None
        self._stdout_queue: Queue[str | None] | None = None
        self._stdout_error: str | None = None
        self._stderr_tail: deque[str] = deque(maxlen=40)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[ModalRuntime] {message}", flush=True)

    def _trace(self, message: str) -> None:
        if self.trace:
            print(f"[ModalRuntime] {message}", flush=True)

    def _ensure_sandbox(self) -> None:
        if self.container is not None:
            return

        try:
            import modal
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise ModuleNotFoundError(
                "ModalRuntime requires the optional `modal` dependency. "
                "Install it with `pip install rlmflow[modal]`."
            ) from exc

        self._log(f"looking up Modal app {self.app_name!r}")
        app = modal.App.lookup(self.app_name, create_if_missing=True)
        image = self.image or modal.Image.debian_slim().pip_install("rlmflow")
        self._log("creating Modal sandbox")
        entrypoint = (
            f"mkdir -p {shlex.quote(self.remote_workdir)} && "
            "exec python -u -c "
            f"{shlex.quote('from rlmflow.runtime.repl import main; main()')} "
            f"--workdir {shlex.quote(self.remote_workdir)}"
        )
        self.container = modal.Sandbox.create(
            "sh",
            "-lc",
            entrypoint,
            app=app,
            image=image,
            timeout=self.timeout,
            **self.container_kwargs,
        )
        self._stdout_queue = Queue()
        self._start_stream_reader(self.container.stdout, self._stdout_queue)
        self._start_stderr_reader(self.container.stderr)
        self._started = True

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        self._ensure_sandbox()
        assert self.container is not None
        self._trace("executing sandbox command")
        command_timeout = float(timeout or self.repl_timeout)
        try:
            from modal.stream_type import StreamType

            process = _call_with_timeout(
                lambda: self.container.exec(
                    "sh",
                    "-lc",
                    command,
                    timeout=max(1, int(command_timeout)),
                    stdout=StreamType.PIPE,
                    stderr=StreamType.PIPE,
                ),
                timeout=command_timeout,
                label="Modal Sandbox.exec()",
            )
            stdout = _call_with_timeout(
                lambda: _read_stream(process.stdout),
                timeout=command_timeout,
                label="Modal command stdout.read()",
            )
            stderr = _call_with_timeout(
                lambda: _read_stream(process.stderr),
                timeout=command_timeout,
                label="Modal command stderr.read()",
            )
            exit_code = _call_with_timeout(
                process.wait,
                timeout=command_timeout,
                label="Modal command wait()",
            )
        except Exception as exc:
            self._started = False
            if not self._is_sandbox_gone(exc):
                raise RuntimeError(
                    "Modal command did not complete cleanly before "
                    f"timeout={command_timeout}s: {type(exc).__name__}: {exc}"
                ) from exc
            self.container = None
            raise RuntimeError(
                "Modal sandbox disappeared while executing a command. "
                "This usually means the sandbox lifetime expired; increase "
                f"`ModalRuntime(timeout=...)` or `--sandbox-timeout` "
                f"(current timeout: {self.timeout}s)."
            ) from exc
        if exit_code:
            raise RuntimeError(
                f"Modal command failed ({exit_code}): {stderr or stdout}"
            )
        return stdout

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._log("starting Modal sandbox REPL entrypoint")
        self._ensure_sandbox()

    def _start_stream_reader(
        self,
        stream: object,
        output: Queue[str | None],
    ) -> None:
        def read_stream() -> None:
            pending = ""
            try:
                for chunk in stream:
                    text = _to_text(chunk)
                    pending += text
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        if line:
                            output.put(line)
                if pending:
                    output.put(pending)
            except Exception as exc:
                self._stdout_error = f"{type(exc).__name__}: {exc}"
            finally:
                output.put(None)

        threading.Thread(target=read_stream, daemon=True).start()

    def _start_stderr_reader(self, stream: object) -> None:
        def read_stderr() -> None:
            try:
                for line in stream:
                    self._stderr_tail.append(str(line))
            except Exception:
                return

        threading.Thread(target=read_stderr, daemon=True).start()

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        self._ensure_sandbox()
        assert self.container is not None
        self.exec(f"mkdir -p {shlex.quote(posixpath.dirname(remote_path))}")
        _call_with_timeout(
            lambda: self.container.filesystem.copy_from_local(local_path, remote_path),
            timeout=max(self.repl_timeout, 120),
            label="Modal filesystem.copy_from_local()",
        )

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        self._ensure_sandbox()
        assert self.container is not None
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        _call_with_timeout(
            lambda: self.container.filesystem.copy_to_local(remote_path, local_path),
            timeout=max(self.repl_timeout, 120),
            label="Modal filesystem.copy_to_local()",
        )

    def remove_path(self, remote_path: str, *, recursive: bool = False) -> None:
        if recursive and posixpath.normpath(remote_path) == posixpath.normpath(
            self.remote_workdir
        ):
            quoted = shlex.quote(remote_path)
            self.exec(
                f"mkdir -p {quoted} && "
                f"find {quoted} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +"
            )
            return
        flag = "-rf" if recursive else "-f"
        self.exec(f"rm {flag} -- {shlex.quote(remote_path)}")

    def send(self, msg: dict) -> None:
        self._trace(f"sending REPL command: {msg.get('cmd', '<proxy-response>')}")
        self._ensure_started()
        assert self.container is not None
        try:
            line = json.dumps(msg) + "\n"
            _call_with_timeout(
                lambda: self.container.stdin.write(line),
                timeout=self.repl_timeout,
                label="Modal sandbox stdin.write()",
            )
            _call_with_timeout(
                self.container.stdin.drain,
                timeout=self.repl_timeout,
                label="Modal sandbox stdin.drain()",
            )
        except Exception as exc:
            if not self._is_sandbox_gone(exc):
                raise
            self.container = None
            self._started = False
            raise RuntimeError(
                "Modal sandbox disappeared while writing to the rlmflow REPL. "
                "Increase `ModalRuntime(timeout=...)` or `--sandbox-timeout` "
                f"(current timeout: {self.timeout}s)."
            ) from exc

    def recv(self) -> dict:
        self._ensure_started()
        assert self.container is not None
        assert self._stdout_queue is not None
        try:
            line = self._stdout_queue.get(timeout=self.repl_timeout + 5)
        except Empty as exc:
            exit_code = self.container.poll()
            stderr = "".join(self._stderr_tail).strip()
            stdout_error = (
                f"; stdout_error={self._stdout_error}" if self._stdout_error else ""
            )
            raise RuntimeError(
                "Modal rlmflow REPL did not return a response before "
                f"repl_timeout={self.repl_timeout}s. "
                f"sandbox_exit_code={exit_code}; stderr: {stderr or '<empty>'}"
                f"{stdout_error}"
            ) from exc
        if line is None:
            exit_code = self.container.poll()
            stderr = "".join(self._stderr_tail).strip()
            stdout_error = (
                f"; stdout_error={self._stdout_error}" if self._stdout_error else ""
            )
            raise RuntimeError(
                "Modal rlmflow REPL exited before returning a response. "
                f"exit_code={exit_code}; stderr: {stderr or '<empty>'}"
                f"{stdout_error}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Modal rlmflow REPL returned invalid JSON: {line!r}"
            ) from exc

    def _close_sandbox(self) -> None:
        container, self.container = self.container, None
        self._stdout_queue = None
        self._stdout_error = None
        self._stderr_tail.clear()
        if container is None:
            return
        self._log("terminating Modal sandbox")
        try:
            _call_with_timeout(
                container.terminate,
                timeout=max(self.repl_timeout, 30),
                label="Modal sandbox terminate()",
            )
        except Exception as exc:
            if not self._is_sandbox_gone(exc):
                raise

    def _is_sandbox_gone(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}"
        lower_text = text.lower()
        return (
            type(exc).__name__ == "NotFoundError"
            and "sandbox" in lower_text
            and ("not found" in lower_text or "already shut down" in lower_text)
        )

    def _on_workspace_pull_skipped(self, exc: Exception) -> None:
        self._log(
            "skipping workspace pull because Modal sandbox is already gone: "
            f"{type(exc).__name__}: {exc}"
        )

    def clone(
        self, workspace: BaseWorkspace | str | Path | None = None
    ) -> ModalRuntime:
        new = ModalRuntime(
            self.app_name,
            workspace=workspace or self.workspace_obj,
            remote_workdir=self.remote_workdir,
            image=self.image,
            timeout=self.timeout,
            repl_timeout=self.repl_timeout,
            verbose=self.verbose,
            trace=self.trace,
            **self.container_kwargs,
        )
        self._copy_tools_to(new)
        return new

    def fork(self, new_workspace: BaseWorkspace | str | Path) -> ModalRuntime:
        return super().fork(new_workspace)


def _read_stream(stream: object) -> str:
    data = stream.read()
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data or ""


def _to_text(data: object) -> str:
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return str(data)


def _read_available(stream: object) -> str:
    try:
        return _read_stream(stream)
    except Exception:
        return ""


def _call_with_timeout(
    fn: Callable[[], object], *, timeout: float, label: str
) -> object:
    done: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def run() -> None:
        try:
            done.put((True, fn()))
        except Exception as exc:
            done.put((False, exc))

    threading.Thread(target=run, daemon=True).start()
    try:
        ok, value = done.get(timeout=timeout)
    except Empty as exc:
        raise TimeoutError(f"{label} did not finish within {timeout}s") from exc
    if ok:
        return value
    raise value
