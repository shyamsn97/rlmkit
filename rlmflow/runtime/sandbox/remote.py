"""Public base classes for provider-backed sandbox runtimes.

Some sandbox providers make it easy to run shell commands but do not expose a
portable persistent stdin/stdout process handle. ``RemoteFileRuntime`` keeps one
remote ``rlmflow.runtime.repl`` process alive and speaks the JSON-line protocol
through remote files:

- append outbound messages to an input file;
- ``tail -f`` feeds that file into the REPL process;
- poll the output file for the next response line.
"""

from __future__ import annotations

import json
import shlex
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

from rlmflow.graph import WaitRequest
from rlmflow.runtime.runtime import Runtime, ToolDef
from rlmflow.tools import get_tool_metadata
from rlmflow.tools.registry import CONTROL_PROXY_TOOLS, LAUNCHER_TOOLS, SHOW_VARS_NAME
from rlmflow.workspace import BaseWorkspace, ContextVariable, SessionVariable

_REPL_ENTRYPOINT = "from rlmflow.runtime.repl import main; main()"


class RemoteFileRuntime(Runtime):
    """Base class for provider SDKs with ``exec(command) -> stdout`` semantics."""

    def __init__(
        self,
        *,
        workspace: BaseWorkspace | str | Path = ".",
        remote_workdir: str = "/workspace",
        repl_timeout: float = 30,
    ) -> None:
        super().__init__(workspace=workspace)
        self.remote_workdir = remote_workdir
        self.repl_timeout = repl_timeout
        self._run_id = uuid.uuid4().hex
        self._remote_dir = f"/tmp/rlmflow-{self._run_id}"
        self._input_path = f"{self._remote_dir}/in.jsonl"
        self._output_path = f"{self._remote_dir}/out.jsonl"
        self._stderr_path = f"{self._remote_dir}/stderr.log"
        self._pid_path = f"{self._remote_dir}/pid"
        self._output_offset = 0
        self._started = False
        self._workspace_pushed = False
        self._pending_wait_ack = False
        # Provider adapters that pip-install rlmflow on first use override
        # these (see ``_run_setup`` / ``_resolve_setup_commands``). Modal,
        # which bakes rlmflow into its image, leaves them empty.
        self.setup_commands: list[str] = []
        self._setup_done = False

    #: Default first-boot commands for providers that don't ship rlmflow.
    DEFAULT_SETUP_COMMANDS = ("python -m pip install -q rlmflow",)

    @classmethod
    def _resolve_setup_commands(cls, setup_commands: list[str] | None) -> list[str]:
        """Normalize a caller-supplied ``setup_commands`` list.

        ``None`` means "use the default install command"; an explicit
        (possibly empty) list is taken verbatim — so ``setup_commands=[]``
        really means "run nothing".
        """

        if setup_commands is not None:
            return list(setup_commands)
        return list(cls.DEFAULT_SETUP_COMMANDS)

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        """Execute ``command`` in the provider sandbox and return stdout."""

        raise NotImplementedError

    def _provider_prepare(self) -> None:
        """Bring the provider sandbox up and run setup before launching the REPL.

        Default is a no-op. File-based adapters (E2B, Daytona) override this
        to create the sandbox and pip-install rlmflow on first use.
        """

    def _run_setup(self) -> None:
        """Run ``self.setup_commands`` once, before the REPL launches."""

        if self._setup_done:
            return
        for command in self.setup_commands:
            self.exec(command, timeout=self.repl_timeout)
        self._setup_done = True

    def _ensure_started(self) -> None:
        if self._started:
            return

        self._provider_prepare()

        remote_dir = shlex.quote(self._remote_dir)
        input_path = shlex.quote(self._input_path)
        output_path = shlex.quote(self._output_path)
        stderr_path = shlex.quote(self._stderr_path)
        pid_path = shlex.quote(self._pid_path)
        remote_workdir = shlex.quote(self.remote_workdir)

        command = " && ".join(
            [
                f"mkdir -p {remote_dir} {remote_workdir}",
                f": > {input_path}",
                f": > {output_path}",
                f": > {stderr_path}",
                (
                    "(nohup sh -lc "
                    + shlex.quote(
                        f"tail -n +1 -f {input_path} | "
                        "python -u -c "
                        f"{shlex.quote(_REPL_ENTRYPOINT)} "
                        f"--workdir {remote_workdir} "
                        f"> {output_path} 2> {stderr_path}"
                    )
                    + f" >/dev/null 2>&1 & echo $! > {pid_path})"
                ),
            ]
        )
        self.exec(command, timeout=self.repl_timeout)
        self._started = True

    def sync_workspace_to_runtime(self) -> None:
        """Push the materialized workspace into this runtime if needed."""

        if self._workspace_pushed:
            return
        self.workspace_obj.push_to(
            self,
            self.remote_workdir,
            replace=not self._started,
        )
        self._workspace_pushed = True

    def sync_workspace_from_runtime(self, *, merge: bool = True) -> None:
        """Pull runtime filesystem changes back into the workspace."""

        if not self._started:
            return
        try:
            self.workspace_obj.pull_from(
                self,
                self.remote_workdir,
                merge=merge,
                skip_engine_state=True,
            )
        except Exception as exc:
            if self._is_sandbox_gone(exc):
                self._on_workspace_pull_skipped(exc)
                return
            raise

    def after_execution_transition(self, runtimes: Iterable[Runtime] = ()) -> None:
        """Pull artifacts after this remote runtime finishes an exec/resume."""

        self.sync_workspace_from_runtime(merge=True)
        seen: set[int] = set()
        for runtime in runtimes:
            if runtime is self:
                continue
            ident = id(runtime)
            if ident in seen:
                continue
            seen.add(ident)
            runtime.on_workspace_changed()

    def on_workspace_changed(self) -> None:
        """Require a fresh host-to-runtime sync before this runtime executes."""

        self.mark_workspace_stale()

    def mark_workspace_stale(self) -> None:
        """Require a fresh host-to-runtime sync before the next execution."""

        self._workspace_pushed = False

    def prepare_for_execution(self) -> None:
        """Sync workspace and install tools before touching the remote REPL."""

        self._consume_pending_wait_ack()
        self.sync_workspace_to_runtime()
        self.install_registered_tools()

    def prepare_for_resume(self) -> None:
        """Sync workspace without sending ``run`` commands into a suspended REPL."""

        self._consume_pending_wait_ack()
        self.sync_workspace_to_runtime()

    def call(self, msg: dict) -> dict:
        """Run the REPL call loop, returning immediately for proxied waits."""

        self.send(msg)
        while True:
            resp = self.recv()
            if "proxy" not in resp:
                return resp

            result = self.handle_proxy_call(resp)
            if isinstance(result, WaitRequest):
                self._pending_wait_ack = True
                return {"suspended": True, "agent_ids": result.agent_ids}

    def _consume_pending_wait_ack(self) -> None:
        if not self._pending_wait_ack:
            return
        resp = self.recv()
        if not resp.get("suspended"):
            raise RuntimeError(f"Expected remote wait acknowledgement, got: {resp}")
        self._pending_wait_ack = False

    def resume_code(self, send_value=None) -> tuple[bool, object, bool]:
        self._consume_pending_wait_ack()
        return super().resume_code(send_value)

    def send(self, msg: dict) -> None:
        self._ensure_started()
        line = json.dumps(msg) + "\n"
        script = (
            "from pathlib import Path\n"
            f"Path({self._input_path!r}).open('a').write({line!r})\n"
        )
        self.exec(f"python -c {shlex.quote(script)}", timeout=self.repl_timeout)

    def recv(self) -> dict:
        self._ensure_started()
        script = (
            "from pathlib import Path\n"
            "import sys, time\n"
            f"out = Path({self._output_path!r})\n"
            f"err = Path({self._stderr_path!r})\n"
            f"offset = {self._output_offset}\n"
            f"deadline = time.time() + {self.repl_timeout!r}\n"
            "while time.time() < deadline:\n"
            "    text = out.read_text() if out.exists() else ''\n"
            "    idx = text.find('\\n', offset)\n"
            "    if idx >= 0:\n"
            "        print(text[offset:idx])\n"
            "        sys.exit(0)\n"
            "    time.sleep(0.05)\n"
            "stderr = err.read_text()[-4000:] if err.exists() else ''\n"
            "raise SystemExit('timed out waiting for rlmflow REPL output\\n' + stderr)\n"
        )
        line = self.exec(
            f"python -c {shlex.quote(script)}", timeout=self.repl_timeout + 5
        )
        line = line.strip()
        self._output_offset += len(line) + 1
        return json.loads(line)

    def inject(self, name: str, value: object) -> None:
        if isinstance(value, ContextVariable):
            self.inject_remote_context(name, value)
            return
        if isinstance(value, SessionVariable):
            self.inject_remote_session(name, value)
            return
        super().inject(name, value)

    def register_tool(
        self,
        fn: Callable[..., object],
        description: str | None = None,
        *,
        core: bool = False,
        hidden: bool = False,
    ) -> None:
        """Register a remote tool without starting the sandbox immediately."""

        td = ToolDef.from_fn(fn, description, core=core, hidden=hidden)
        self.tools[td.name] = td

    def _install_tool(self, td: ToolDef) -> None:
        if td.name in self._installed_tools:
            return
        if td.name == SHOW_VARS_NAME:
            self.inject_show_vars()
            self._installed_tools.add(td.name)
            return
        if td.name in LAUNCHER_TOOLS:
            self.inject_launcher(td.name)
            self._installed_tools.add(td.name)
            return
        if (
            td.fn is not None
            and td.name not in CONTROL_PROXY_TOOLS
            and self._inject_importable_tool(td.name, td.fn)
        ):
            self._installed_tools.add(td.name)
            return
        if td.fn is None:
            return
        self.inject(td.name, td.fn)
        self._installed_tools.add(td.name)

    def inject_code(self, code: str) -> None:
        resp = self.call({"cmd": "run", "code": code})
        if resp.get("errored"):
            raise RuntimeError(resp.get("output", "remote injection failed"))

    def inject_remote_context(self, name: str, value: ContextVariable) -> None:
        self.inject_code(
            "from rlmflow.workspace import ContextVariable, FileContext\n"
            f"{name} = ContextVariable("
            f"FileContext({self.remote_workdir!r}), "
            f"agent_id={value.agent_id!r}, key={value.key!r})\n"
        )

    def inject_remote_session(self, name: str, value: SessionVariable) -> None:
        self.inject_code(
            "from rlmflow.workspace import SessionVariable, FileSession\n"
            f"{name} = SessionVariable("
            f"FileSession({self.remote_workdir!r}), "
            f"agent_id={value.agent_id!r}, "
            f"node_id={value.node_id!r}, "
            f"branch_id={value.branch_id!r})\n"
        )

    def _inject_importable_tool(self, name: str, fn: Callable[..., object]) -> bool:
        module = getattr(fn, "__module__", "")
        qualname = getattr(fn, "__qualname__", "")
        if not module or "<locals>" in qualname or "." in qualname:
            return False
        if not module.startswith("rlmflow."):
            return False
        meta = get_tool_metadata(fn)
        if meta is None:
            return False
        self.inject_code(f"from {module} import {qualname} as {name}\n")
        return True

    def close(self) -> None:
        if not self._started:
            self._close_sandbox()
            return

        sync_error: Exception | None = None
        try:
            self.sync_workspace_from_runtime(merge=True)
        except Exception as exc:
            sync_error = exc

        try:
            self.exec(
                "sh -lc "
                + shlex.quote(
                    f"kill $(cat {shlex.quote(self._pid_path)}) 2>/dev/null || true; "
                    f"rm -rf {shlex.quote(self._remote_dir)}"
                ),
                timeout=5,
            )
        except Exception:
            pass
        finally:
            self._started = False
            self._close_sandbox()
        if sync_error is not None:
            raise sync_error

    def remove_path(self, remote_path: str, *, recursive: bool = False) -> None:
        """Delete ``remote_path`` in the sandbox via ``rm``."""

        flag = "-rf" if recursive else "-f"
        self.exec(f"rm {flag} -- {shlex.quote(remote_path)}")

    def list_files(self, remote_root: str) -> list[str]:
        """Return sandbox file paths under ``remote_root``, relative to it."""

        output = self.exec(
            f"find {shlex.quote(remote_root)} -type f -print 2>/dev/null || true"
        )
        prefix = remote_root.rstrip("/") + "/"
        return sorted(line.removeprefix(prefix) for line in output.splitlines() if line)

    def _copy_tools_to(self, new: Runtime) -> None:
        for name, td in self.tools.items():
            if td.core:
                continue
            new.tools[name] = td

    @staticmethod
    def _close_with_methods(sandbox: object, method_names: Iterable[str]) -> None:
        """Call the first available teardown method on a provider sandbox."""

        for name in method_names:
            method = getattr(sandbox, name, None)
            if callable(method):
                method()
                return

    def _close_sandbox(self) -> None:
        """Provider-specific resource cleanup hook."""

    def _is_sandbox_gone(self, exc: Exception) -> bool:
        """Return True when provider state says the sandbox already disappeared."""

        return False

    def _on_workspace_pull_skipped(self, exc: Exception) -> None:
        """Hook called when close cannot pull because the sandbox is already gone."""


__all__ = ["RemoteFileRuntime"]
