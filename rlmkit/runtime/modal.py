"""Modal runtime — runs agent code in a remote Modal container.

Requires ``modal`` to be installed (``pip install modal``).

Usage::

    import modal
    from rlmkit.runtime.modal import ModalRuntime

    runtime = ModalRuntime(
        app_name="my-rlm-app",
        image=modal.Image.debian_slim().pip_install("rlmkit"),
    )
    agent = RLM(llm_client=llm, runtime=runtime, runtime_factory=runtime.factory)
"""

from __future__ import annotations

import json
from typing import Any

from rlmkit.runtime.repl import deserialize, serialize
from rlmkit.runtime.runtime import DEFAULT_MODULES, Runtime
from rlmkit.state import WaitRequest


class ModalRuntime(Runtime):
    """Execute agent code inside a Modal container.

    Tool functions (``done``, ``delegate``, ``wait``, etc.) are
    transparently proxied — when the container code calls a tool,
    the call is forwarded to the engine process and the result sent
    back.

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
        self.proxied: dict[str, Any] = {}

    # ── lazy container startup ────────────────────────────────────────

    def ensure_started(self):
        if self.process is not None:
            return
        import modal

        app = modal.App.lookup(self.app_name, create_if_missing=True)
        image = self.image or modal.Image.debian_slim().pip_install("rlmkit")
        self.container = modal.Sandbox.create(
            app=app,
            image=image,
            timeout=self.timeout,
            **self.container_kwargs,
        )
        self.process = self.container.exec(
            "python",
            "-m",
            "rlmkit.runtime.repl",
        )

    # ── low-level protocol ────────────────────────────────────────────

    def send(self, msg: dict):
        self.process.stdin.write((json.dumps(msg) + "\n").encode())
        self.process.stdin.write_eof()
        self.process.stdin.drain()

    def recv(self) -> dict:
        line = next(iter(self.process.stdout))
        return json.loads(line)

    def call(self, msg: dict) -> dict:
        """Send a command and handle proxy calls until a real response arrives."""
        self.send(msg)
        while True:
            resp = self.recv()
            if "proxy" not in resp:
                return resp
            fn = self.proxied[resp["proxy"]]
            args = [deserialize(a) for a in resp.get("args", [])]
            kwargs = {k: deserialize(v) for k, v in resp.get("kwargs", {}).items()}
            result = fn(*args, **kwargs)
            self.send({"value": serialize(result)})

    def parse_exec_response(self, resp: dict) -> tuple[bool, object]:
        if resp.get("suspended"):
            request = WaitRequest(agent_ids=resp["agent_ids"])
            pre_output = resp.get("pre_output", "")
            return True, (request, pre_output)
        return False, resp.get("output", "")

    # ── Runtime interface ─────────────────────────────────────────────

    def execute(self, code: str, timeout: float | None = None) -> str:
        self.ensure_started()
        resp = self.call({"cmd": "run", "code": code})
        return resp.get("output", "")

    def inject(self, name: str, value: Any) -> None:
        self.ensure_started()
        if callable(value):
            self.proxied[name] = value
            self.call({"cmd": "inject_proxy", "name": name})
        else:
            self.call({"cmd": "inject", "name": name, "value": repr(value)})

    def start_code(self, code: str) -> tuple[bool, object]:
        self.ensure_started()
        resp = self.call({"cmd": "run", "code": code})
        return self.parse_exec_response(resp)

    def resume_code(self, send_value=None) -> tuple[bool, object]:
        resp = self.call({"cmd": "resume", "value": send_value})
        return self.parse_exec_response(resp)

    def clone(self) -> ModalRuntime:
        new = ModalRuntime(
            self.app_name,
            image=self.image,
            timeout=self.timeout,
            **self.container_kwargs,
        )
        new.tools = dict(self.tools)
        return new

    def factory(self) -> ModalRuntime:
        """Use as ``runtime_factory=runtime.factory`` on the RLM engine."""
        return self.clone()

    def terminate(self):
        """Shut down the Modal container."""
        if self.container:
            self.container.terminate()
            self.container = None
            self.process = None

    def available_modules(self) -> list[str]:
        return DEFAULT_MODULES
