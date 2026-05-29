"""E2B runtime — run agent code inside an E2B Sandbox.

Requires ``e2b`` to be installed (``pip install rlmflow[e2b]``) and an
``E2B_API_KEY`` environment variable, unless you pass SDK auth options through
``sandbox_kwargs``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rlmflow.runtime.sandbox.remote import RemoteFileRuntime
from rlmflow.workspace import BaseWorkspace


class E2BRuntime(RemoteFileRuntime):
    """Execute agent code inside an E2B Sandbox."""

    def __init__(
        self,
        *,
        workspace: BaseWorkspace | str | Path = ".",
        template: str | None = None,
        timeout: int = 300,
        envs: dict[str, str] | None = None,
        remote_workdir: str = "/workspace",
        repl_timeout: float = 30,
        setup_commands: list[str] | None = None,
        sandbox_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            workspace=workspace,
            remote_workdir=remote_workdir,
            repl_timeout=repl_timeout,
        )
        self.template = template
        self.timeout = timeout
        self.envs = envs
        self.setup_commands = self._resolve_setup_commands(setup_commands)
        self.sandbox_kwargs = dict(sandbox_kwargs or {})
        self.sandbox = None

    def _ensure_sandbox(self) -> None:
        if self.sandbox is not None:
            return

        try:
            from e2b import Sandbox
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise ModuleNotFoundError(
                "E2BRuntime requires the optional `e2b` dependency. "
                "Install it with `pip install rlmflow[e2b]`."
            ) from exc

        self.sandbox = Sandbox.create(
            template=self.template,
            timeout=self.timeout,
            envs=self.envs,
            **self.sandbox_kwargs,
        )

    def _provider_prepare(self) -> None:
        self._ensure_sandbox()
        self._run_setup()

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        self._ensure_sandbox()
        assert self.sandbox is not None
        result = self.sandbox.commands.run(
            command, timeout=timeout or self.repl_timeout
        )
        exit_code = getattr(result, "exit_code", 0)
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
        if exit_code:
            raise RuntimeError(f"E2B command failed ({exit_code}): {stderr or stdout}")
        return stdout

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        self._ensure_sandbox()
        assert self.sandbox is not None
        files = getattr(self.sandbox, "files", None)
        if files is not None and hasattr(files, "write"):
            files.write(remote_path, Path(local_path).read_bytes())
            return
        dst = Path(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        self._ensure_sandbox()
        assert self.sandbox is not None
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        files = getattr(self.sandbox, "files", None)
        if files is not None and hasattr(files, "read"):
            data = files.read(remote_path)
            if isinstance(data, str):
                dst.write_text(data)
            else:
                dst.write_bytes(data)
            return
        shutil.copy2(remote_path, dst)

    def clone(self, workspace: BaseWorkspace | str | Path | None = None) -> E2BRuntime:
        new = E2BRuntime(
            workspace=workspace or self.workspace_obj,
            template=self.template,
            timeout=self.timeout,
            envs=self.envs,
            remote_workdir=self.remote_workdir,
            repl_timeout=self.repl_timeout,
            setup_commands=self.setup_commands,
            sandbox_kwargs=self.sandbox_kwargs,
        )
        self._copy_tools_to(new)
        return new

    def fork(self, new_workspace: BaseWorkspace | str | Path) -> E2BRuntime:
        return super().fork(new_workspace)

    def _close_sandbox(self) -> None:
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        self._close_with_methods(sandbox, ("kill", "close", "disconnect"))


__all__ = ["E2BRuntime"]
