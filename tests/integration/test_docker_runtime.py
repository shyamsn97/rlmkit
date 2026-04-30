"""Integration test: DockerRuntime end-to-end against the shipped image.

Covers the three smoke paths that regressed during the 0.1.0 cycle:

1. ``inject`` on a non-literal object exposes its methods over the wire.
2. ``from X import *`` works inside a code block that also ``yield``s.
3. Proxied tool writes resolve to the host workspace directory.

Gated on ``RLMKIT_DOCKER_TEST=1`` plus a running docker daemon.  Build
the image once before running:

    docker build -t rlmflow:local .
    RLMKIT_DOCKER_TEST=1 pytest tests/integration/test_docker_runtime.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rlmflow.runtime.docker import DockerRuntime
from rlmflow.workspace import FileContext
from rlmflow.node import ChildHandle, WaitRequest


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, check=False
        )
        return r.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    os.environ.get("RLMKIT_DOCKER_TEST") != "1" or not _docker_available(),
    reason="set RLMKIT_DOCKER_TEST=1 with a working docker daemon",
)


IMAGE = os.environ.get("RLMKIT_DOCKER_TEST_IMAGE", "rlmflow:local")


@pytest.fixture
def runtime(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    rt = DockerRuntime(
        IMAGE,
        mounts={str(ws): "/workspace"},
        workdir="/workspace",
        workspace=ws,
        network="none",
    )
    try:
        yield rt, ws
    finally:
        rt.close()


def test_object_proxy_round_trips_file_context_methods(runtime, tmp_path: Path):
    """Injected objects expose public methods as callable proxies."""
    rt, _ = runtime
    context = FileContext(tmp_path / "context")
    rt.inject("STORE", context)

    rt.execute("STORE.write('context', 'hello from container')")
    assert context.read("context") == "hello from container"

    out = rt.execute("print(STORE.read('context'))")
    assert out == "hello from container"


def test_star_import_with_yield(runtime):
    rt, _ = runtime

    def delegate(prompt: str) -> ChildHandle:
        return ChildHandle(agent_id="c")

    def wait(handles):
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    rt.inject("delegate", delegate)
    rt.inject("wait", wait)

    suspended, _ = rt.start_code(
        "from math import *\n"
        "h = delegate('q')\n"
        "yield wait([h])\n"
        "print(int(pi * 100))\n"
    )
    assert suspended is True
    suspended, out = rt.resume_code({"c": "done"})
    assert suspended is False
    assert "314" in out


def test_proxied_writes_land_in_host_workspace(runtime):
    """A host-side tool writing a relative path must resolve inside the workspace."""
    rt, ws = runtime

    def write_rel(path: str, content: str) -> str:
        Path(path).write_text(content)
        return "ok"

    rt.inject("write_rel", write_rel)
    rt.execute("write_rel('hello.txt', 'hi')")

    assert (ws / "hello.txt").read_text() == "hi"


def test_end_to_end_delegate_wait():
    """Spawn a fresh container; exercise execute, inject, and generator suspension."""
    rt = DockerRuntime(IMAGE, network="none")
    try:
        assert rt.execute("print('hi from container')") == "hi from container"

        def delegate(prompt: str) -> ChildHandle:
            return ChildHandle(agent_id=f"child-{prompt}")

        def wait(handles):
            return WaitRequest(agent_ids=[h.agent_id for h in handles])

        rt.inject("delegate", delegate)
        rt.inject("wait", wait)

        suspended, payload = rt.start_code(
            "h = delegate('q1')\n"
            "results = yield wait([h])\n"
            "print('after:', results)\n"
        )
        assert suspended is True
        request, _ = payload
        assert request.agent_ids == ["child-q1"]

        suspended, out = rt.resume_code({"child-q1": "answer"})
        assert suspended is False
        assert "after: {'child-q1': 'answer'}" in out
    finally:
        rt.close()
