"""Unit tests for ``DockerRuntime`` argv construction and clone semantics.

These run everywhere — they don't shell out to ``docker``. The
end-to-end tests live in ``tests/integration/test_docker_runtime.py``
and are gated on ``RLMKIT_DOCKER_TEST=1`` with a working daemon.
"""

from __future__ import annotations

from rlmkit.runtime.docker import DockerRuntime
from rlmkit.workspace import Workspace


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


def test_workspace_mount_is_writable_workdir(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    rt = DockerRuntime("myimage", workspace=workspace)

    assert "-v" in rt.argv
    assert f"{workspace.root}:/workspace" in rt.argv
    assert "--workdir" in rt.argv
    assert rt.argv[rt.argv.index("--workdir") + 1] == "/workspace"

    child = rt.clone()
    assert child.argv == rt.argv
