"""Docker runtime — run agent code inside a fresh container each session.

Each :class:`DockerRuntime` instance spawns one ``docker run -i --rm ...``
subprocess; the container must have ``rlmkit`` installed so it can run
``python -m rlmkit.runtime.repl``.  All REPL I/O happens over stdin/stdout
of the container, so this is just :class:`SubprocessRuntime` with an
ergonomic argv builder on top.

Example::

    from rlmkit.runtime.docker import DockerRuntime

    runtime = DockerRuntime(
        image="myorg/rlmkit-sandbox:latest",
        mounts={"./data": "/workspace"},
        env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
        network="none",       # air-gap the container
        cpus=1.0,
        memory="512m",
    )
    agent = RLMFlow(llm_client=llm, runtime=runtime, runtime_factory=runtime.clone)

Prerequisites:

1. ``docker`` is on ``PATH``.
2. The image has Python + ``rlmkit`` installed.  The repo ships a ready
   ``Dockerfile`` at its root — build it once with::

       docker build -t rlmkit:local .

   and pass ``image="rlmkit:local"``.  Any image whose ``CMD`` (or your
   ``entrypoint_argv``) runs ``python -m rlmkit.runtime.repl`` works.

If you need to run the server under a different interpreter or path, set
``entrypoint_argv`` (defaults to ``["python", "-m", "rlmkit.runtime.repl"]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlmkit.runtime.runtime import workspace_path
from rlmkit.runtime.subprocess import SubprocessRuntime


class DockerRuntime(SubprocessRuntime):
    """Run the REPL server inside an isolated Docker container.

    Parameters
    ----------
    image : str
        Docker image to run.  Must have ``rlmkit`` installed.
    workspace : str
        Host-side workspace path (affects the engine's CWD handling, not
        the container's).  The container's CWD is controlled by the
        image's ``WORKDIR`` or the ``workdir`` argument.
    mounts : dict[str, str] | None
        ``{host_path: container_path}`` bind mounts.  Host paths are
        resolved to absolute paths.
    env : dict[str, str] | None
        Environment variables to set inside the container.
    network : str | None
        Docker network mode (``"none"``, ``"host"``, a network name, ...).
        When unset, Docker's default bridge network is used.
    cpus : float | None
        CPU quota (``--cpus`` flag).
    memory : str | None
        Memory limit (``--memory`` flag, e.g. ``"512m"``, ``"2g"``).
    user : str | None
        User to run as (``--user`` flag).
    workdir : str | None
        Working directory inside the container (``--workdir`` flag).
    extra_args : list[str] | None
        Raw arguments spliced into ``docker run`` before the image.
    docker_bin : str
        Path to the docker binary (default ``"docker"``).  Use this to
        point at ``podman``, ``nerdctl``, or a full path.
    entrypoint_argv : list[str] | None
        Command to run inside the container.  Defaults to
        ``["python", "-m", "rlmkit.runtime.repl"]``.
    """

    def __init__(
        self,
        image: str,
        *,
        workspace: str | Path | Any = ".",
        mounts: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        network: str | None = None,
        cpus: float | None = None,
        memory: str | None = None,
        user: str | None = None,
        workdir: str | None = None,
        extra_args: list[str] | None = None,
        docker_bin: str = "docker",
        entrypoint_argv: list[str] | None = None,
    ) -> None:
        runtime_workspace = workspace_path(workspace)
        is_workspace = not isinstance(workspace, str | Path) and hasattr(
            workspace, "root"
        )
        if mounts is None and is_workspace:
            mounts = {str(runtime_workspace): "/workspace"}
        if workdir is None and is_workspace:
            workdir = "/workspace"

        self.image = image
        self.options = dict(
            mounts=mounts,
            env=env,
            network=network,
            cpus=cpus,
            memory=memory,
            user=user,
            workdir=workdir,
            extra_args=extra_args,
            docker_bin=docker_bin,
            entrypoint_argv=entrypoint_argv,
        )
        super().__init__(build_argv(image, **self.options), workspace=runtime_workspace)

    def clone(self, workspace: str | Path | None = None) -> DockerRuntime:
        new = self.__class__(
            self.image, workspace=workspace or self.workspace, **self.options
        )
        for name, td in self.tools.items():
            new.tools[name] = td
            new.inject(name, td.fn)
        return new


def build_argv(
    image: str,
    *,
    mounts: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    network: str | None = None,
    cpus: float | None = None,
    memory: str | None = None,
    user: str | None = None,
    workdir: str | None = None,
    extra_args: list[str] | None = None,
    docker_bin: str = "docker",
    entrypoint_argv: list[str] | None = None,
) -> list[str]:
    """Build the ``docker run ...`` argv for :class:`DockerRuntime`."""
    argv: list[str] = [docker_bin, "run", "-i", "--rm"]
    for host, container in (mounts or {}).items():
        argv += ["-v", f"{Path(host).resolve()}:{container}"]
    for k, v in (env or {}).items():
        argv += ["-e", f"{k}={v}"]
    if network is not None:
        argv += ["--network", network]
    if cpus is not None:
        argv += ["--cpus", str(cpus)]
    if memory is not None:
        argv += ["--memory", str(memory)]
    if user is not None:
        argv += ["--user", user]
    if workdir is not None:
        argv += ["--workdir", workdir]
    argv += list(extra_args or [])
    argv += [image]
    argv += list(entrypoint_argv or ["python", "-m", "rlmkit.runtime.repl"])
    return argv
