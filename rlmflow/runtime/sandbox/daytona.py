"""Daytona runtime — run agent code inside a Daytona Sandbox.

Requires ``daytona-sdk`` to be installed (``pip install rlmflow[daytona]``)
and Daytona credentials configured for the SDK.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rlmflow.runtime.sandbox.remote import RemoteFileRuntime
from rlmflow.workspace import BaseWorkspace


class DaytonaRuntime(RemoteFileRuntime):
    """Execute agent code inside a Daytona Sandbox."""

    def __init__(
        self,
        *,
        workspace: BaseWorkspace | str | Path = ".",
        create_params: object = None,
        create_timeout: float = 60,
        remote_workdir: str = "/workspace",
        repl_timeout: float = 30,
        env: dict[str, str] | None = None,
        setup_commands: list[str] | None = None,
        daytona: object = None,
    ) -> None:
        super().__init__(
            workspace=workspace,
            remote_workdir=remote_workdir,
            repl_timeout=repl_timeout,
        )
        self.create_params = create_params
        self.create_timeout = create_timeout
        self.env = env
        self.setup_commands = self._resolve_setup_commands(setup_commands)
        self.daytona = daytona
        self.sandbox = None

    def _ensure_sandbox(self) -> None:
        if self.sandbox is not None:
            return

        if self.daytona is None:
            try:
                from daytona import Daytona
            except (
                ModuleNotFoundError
            ) as exc:  # pragma: no cover - optional dependency.
                raise ModuleNotFoundError(
                    "DaytonaRuntime requires the optional `daytona-sdk` dependency. "
                    "Install it with `pip install rlmflow[daytona]`."
                ) from exc
            self.daytona = Daytona()

        if self.create_params is None:
            self.sandbox = self.daytona.create(timeout=self.create_timeout)
        else:
            self.sandbox = self.daytona.create(
                self.create_params, timeout=self.create_timeout
            )

    def _provider_prepare(self) -> None:
        self._ensure_sandbox()
        self._run_setup()

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        self._ensure_sandbox()
        assert self.sandbox is not None
        result = self.sandbox.process.exec(
            command,
            env=self.env,
            timeout=int(timeout or self.repl_timeout),
        )
        exit_code = getattr(result, "exit_code", 0)
        stdout = _stdout(result)
        stderr = getattr(result, "stderr", "") or getattr(result, "error", "")
        if exit_code:
            raise RuntimeError(
                f"Daytona command failed ({exit_code}): {stderr or stdout}"
            )
        return stdout

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        self._ensure_sandbox()
        assert self.sandbox is not None
        fs = getattr(self.sandbox, "fs", None) or getattr(
            self.sandbox, "filesystem", None
        )
        if fs is not None and hasattr(fs, "upload_file"):
            fs.upload_file(local_path, remote_path)
            return
        dst = Path(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        self._ensure_sandbox()
        assert self.sandbox is not None
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        fs = getattr(self.sandbox, "fs", None) or getattr(
            self.sandbox, "filesystem", None
        )
        if fs is not None and hasattr(fs, "download_file"):
            fs.download_file(remote_path, local_path)
            return
        shutil.copy2(remote_path, dst)

    def clone(
        self, workspace: BaseWorkspace | str | Path | None = None
    ) -> DaytonaRuntime:
        new = DaytonaRuntime(
            workspace=workspace or self.workspace_obj,
            create_params=self.create_params,
            create_timeout=self.create_timeout,
            remote_workdir=self.remote_workdir,
            repl_timeout=self.repl_timeout,
            env=self.env,
            setup_commands=self.setup_commands,
            daytona=self.daytona,
        )
        self._copy_tools_to(new)
        return new

    def fork(self, new_workspace: BaseWorkspace | str | Path) -> DaytonaRuntime:
        return super().fork(new_workspace)

    def _close_sandbox(self) -> None:
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        self._close_with_methods(sandbox, ("delete", "stop", "close"))


def _stdout(result: object) -> str:
    artifacts = getattr(result, "artifacts", None)
    if artifacts is not None and getattr(artifacts, "stdout", None) is not None:
        return artifacts.stdout
    for attr in ("stdout", "result", "output"):
        value = getattr(result, attr, None)
        if value is not None:
            return value
    return ""


__all__ = ["DaytonaRuntime"]
