"""Integration test: DockerRuntime end-to-end against the shipped image.

Covers the three smoke paths that regressed during the 0.1.0 cycle:

1. ``inject`` on a non-literal object exposes its methods over the wire.
2. ``from X import *`` works inside a code block that also ``yield``s.
3. Proxied tool writes resolve to the host workspace directory.

Gated on ``RLMKIT_DOCKER_TEST=1`` plus a running docker daemon.  Build
the image once before running:

    docker build -t rlmkit:local .
    RLMKIT_DOCKER_TEST=1 pytest tests/integration/test_docker_runtime.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rlmkit.runtime.docker import DockerRuntime
from rlmkit.session import FileSession
from rlmkit.state import ChildHandle, WaitRequest


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


IMAGE = os.environ.get("RLMKIT_DOCKER_TEST_IMAGE", "rlmkit:local")


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
        rt.terminate()


def test_object_proxy_round_trips_session_methods(runtime, tmp_path: Path):
    """Injecting a FileSession exposes its public methods as callable proxies."""
    rt, _ = runtime
    session = FileSession(tmp_path / "sessions")
    rt.inject("SESSION", session)

    rt.execute(
        "SESSION.write('root', [{'role': 'user', 'content': 'hello from container'}])"
    )
    assert session.read("root") == [{"role": "user", "content": "hello from container"}]

    out = rt.execute("print(SESSION.read('root')[0]['content'])")
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
