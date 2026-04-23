"""Tests for DockerRuntime.

Unit tests cover argv construction and clone semantics and run everywhere.
The end-to-end test spawns an actual container and is skipped unless
``docker`` is on PATH and ``RLMKIT_DOCKER_TEST=1`` is set.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from rlmkit.runtime.docker import DockerRuntime
from rlmkit.state import ChildHandle, WaitRequest


# ── unit tests (no docker required) ───────────────────────────────────


def test_argv_defaults_to_python_repl_entrypoint():
    rt = DockerRuntime("python:3.12-slim")
    assert rt.argv[:4] == ["docker", "run", "-i", "--rm"]
    assert rt.argv[-4:] == ["python:3.12-slim", "python", "-m", "rlmkit.runtime.repl"]


def test_argv_includes_mounts_env_network_limits(tmp_path):
    host_dir = tmp_path / "hostdata"
    host_dir.mkdir()
    rt = DockerRuntime(
        "myimage",
        mounts={str(host_dir): "/data"},
        env={"FOO": "bar", "BAZ": "qux"},
        network="none",
        cpus=1.5,
        memory="512m",
        user="1000:1000",
        workdir="/workspace",
    )
    argv = rt.argv
    assert "-v" in argv
    vi = argv.index("-v")
    assert argv[vi + 1] == f"{host_dir.resolve()}:/data"
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--cpus" in argv and argv[argv.index("--cpus") + 1] == "1.5"
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == "512m"
    assert "--user" in argv and argv[argv.index("--user") + 1] == "1000:1000"
    assert "--workdir" in argv and argv[argv.index("--workdir") + 1] == "/workspace"
    for pair in ("FOO=bar", "BAZ=qux"):
        assert pair in argv


def test_custom_docker_bin_and_entrypoint():
    rt = DockerRuntime(
        "myimage",
        docker_bin="podman",
        entrypoint_argv=["/opt/venv/bin/python", "-m", "rlmkit.runtime.repl"],
    )
    assert rt.argv[0] == "podman"
    assert rt.argv[-3:] == ["/opt/venv/bin/python", "-m", "rlmkit.runtime.repl"]


def test_extra_args_are_spliced_before_image():
    rt = DockerRuntime(
        "myimage",
        extra_args=["--read-only", "--security-opt", "no-new-privileges"],
    )
    i = rt.argv.index("myimage")
    assert rt.argv[i - 3 : i] == ["--read-only", "--security-opt", "no-new-privileges"]


def test_clone_preserves_config():
    rt = DockerRuntime(
        "myimage",
        mounts={"/tmp/host": "/data"},
        env={"FOO": "bar"},
        network="none",
    )
    twin = rt.clone()
    assert twin.argv == rt.argv
    assert twin.image == rt.image
    assert twin.options == rt.options


# ── end-to-end (requires docker) ──────────────────────────────────────


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


pytestmark_e2e = pytest.mark.skipif(
    os.environ.get("RLMKIT_DOCKER_TEST") != "1" or not _docker_available(),
    reason="set RLMKIT_DOCKER_TEST=1 with a working docker daemon to run",
)


@pytestmark_e2e
def test_docker_end_to_end_delegate_wait():
    """Spawn a real container; exercise execute, inject, and generator suspension.

    Uses an image that has rlmkit installed.  Override with
    ``RLMKIT_DOCKER_TEST_IMAGE``; defaults to ``rlmkit-test:latest``.
    """
    image = os.environ.get("RLMKIT_DOCKER_TEST_IMAGE", "rlmkit-test:latest")
    rt = DockerRuntime(image, network="none")
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
        rt.terminate()
