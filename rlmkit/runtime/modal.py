"""Modal runtime — runs agent code in a remote Modal container.

Requires ``modal`` to be installed (``pip install modal``).

Usage::

    import modal
    from rlmkit.runtime.modal import ModalRuntime

    runtime = ModalRuntime(
        app_name="my-rlm-app",
        image=modal.Image.debian_slim().pip_install("rlmkit"),
    )
    agent = RLMFlow(llm_client=llm, runtime=runtime, runtime_factory=runtime.clone)
"""

from __future__ import annotations

import json

from rlmkit.runtime.runtime import Runtime


class ModalRuntime(Runtime):
    """Execute agent code inside a Modal Sandbox.

    The Modal Sandbox runs ``python -m rlmkit.runtime.repl``; this class
    just ships JSON messages to its stdin and reads responses from
    stdout.  All of the REPL protocol (proxy loop, inject, suspend /
    resume) is inherited from :class:`Runtime`.

    Parameters
    ----------
    app_name : str
        Modal app name (created if missing).
    image : modal.Image | None
        Container image.  Defaults to ``debian_slim + rlmkit``.
    timeout : int
        Container lifetime in seconds (default 5 min).
    **container_kwargs
        Extra kwargs forwarded to ``modal.Sandbox.create()``
        (e.g. ``gpu``, ``secrets``, ``volumes``).
    """

    def __init__(
        self,
        app_name: str = "rlmkit",
        *,
        image=None,
        timeout: int = 300,
        **container_kwargs,
    ) -> None:
        super().__init__(workspace=".")
        self.app_name = app_name
        self.image = image
        self.timeout = timeout
        self.container_kwargs = container_kwargs
        self.container = None
        self.process = None

    def send(self, msg: dict) -> None:
        if self.process is None:
            import modal

            app = modal.App.lookup(self.app_name, create_if_missing=True)
            image = self.image or modal.Image.debian_slim().pip_install("rlmkit")
            self.container = modal.Sandbox.create(
                app=app,
                image=image,
                timeout=self.timeout,
                **self.container_kwargs,
            )
            self.process = self.container.exec("python", "-m", "rlmkit.runtime.repl")
        self.process.stdin.write((json.dumps(msg) + "\n").encode())
        self.process.stdin.write_eof()
        self.process.stdin.drain()

    def recv(self) -> dict:
        return json.loads(next(iter(self.process.stdout)))

    def close(self) -> None:
        """Shut down the Modal container."""
        if self.container:
            self.container.terminate()
            self.container = None
            self.process = None

    def clone(self, workspace: object | None = None) -> ModalRuntime:
        # ModalRuntime manages its own workspace inside the container,
        # so the ``workspace`` argument is accepted for API consistency
        # with the base ``Runtime.clone`` but otherwise unused.
        del workspace
        new = ModalRuntime(
            self.app_name,
            image=self.image,
            timeout=self.timeout,
            **self.container_kwargs,
        )
        new.tools = dict(self.tools)
        return new

    def fork(self, new_workspace: object) -> ModalRuntime:
        # No host filesystem to copy — Modal manages its own workspace
        # inside the container. Use container snapshots / Volume forks
        # for true isolation if you need it.
        del new_workspace
        return self.clone()
