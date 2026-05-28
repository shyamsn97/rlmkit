"""Docker runtime — run agent code inside a fresh container each session.

Each :class:`DockerRuntime` instance spawns one ``docker run -i --rm ...``
subprocess; the container must have ``rlmflow`` installed so it can run
``python -m rlmflow.runtime.repl``.  All REPL I/O happens over stdin/stdout
of the container.

Example::

    from rlmflow.runtime.docker import DockerRuntime

    runtime = DockerRuntime(
        image="myorg/rlmflow-sandbox:latest",
        mounts={"./data": "/workspace"},
        env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
        network="none",       # air-gap the container
        cpus=1.0,
        memory="512m",
    )
    agent = RLMFlow(llm_client=llm, runtime=runtime, runtime_factory=runtime.clone)

Prerequisites:

1. ``docker`` is on ``PATH``.
2. The image has Python + ``rlmflow`` installed.  The repo ships a ready
   ``Dockerfile`` at its root — build it once with::

       docker build -t rlmflow:local .

   and pass ``image="rlmflow:local"``.  Any image whose ``CMD`` (or your
   ``entrypoint_argv``) runs ``python -m rlmflow.runtime.repl`` works.

If you need to run the server under a different interpreter or path, set
``entrypoint_argv`` (defaults to ``["python", "-m", "rlmflow.runtime.repl"]``).
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path

from rlmflow.runtime.runtime import Runtime, workspace_path
from rlmflow.workspace import BaseWorkspace


class DockerRuntime(Runtime):
    """Run the REPL server inside an isolated Docker container.

    Talks to a long-running ``docker run -i --rm <image> python -m
    rlmflow.runtime.repl`` subprocess over its stdin/stdout pipes using
    the JSON-line protocol from :mod:`rlmflow.runtime.repl`.

    Parameters
    ----------
    image : str
        Docker image to run.  Must have ``rlmflow`` installed.
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
        ``["python", "-m", "rlmflow.runtime.repl"]``.
    """

    def __init__(
        self,
        image: str,
        *,
        workspace: BaseWorkspace | str | Path = ".",
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
        super().__init__(workspace=workspace)
        runtime_workspace = workspace_path(workspace)
        is_workspace = isinstance(workspace, BaseWorkspace)
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
        # Remember the caller-supplied workspace form so ``clone()`` can replay
        # it. The base ``Runtime`` always wraps non-``BaseWorkspace`` inputs
        # into a ``Workspace`` object, which would otherwise cause clones to
        # re-trigger the auto-mount / auto-workdir branches above and diverge
        # from the original argv.
        self._workspace_arg = workspace
        self.argv = build_argv(image, **self.options)
        self.proc: sp.Popen | None = None

    # ── REPL stdio transport ───────────────────────────────────────────

    def send(self, msg: dict) -> None:
        if self.proc is None:
            self.proc = sp.Popen(
                self.argv,
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                cwd=self.workspace,
                bufsize=0,
            )
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def recv(self) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            err = b""
            if self.proc.stderr is not None:
                try:
                    err = self.proc.stderr.read() or b""
                except Exception:
                    pass
            raise RuntimeError(
                f"REPL subprocess {self.argv!r} exited unexpectedly. "
                f"stderr: {err.decode(errors='replace')}"
            )
        return json.loads(line)

    def close(self) -> None:
        """Tear down the container subprocess and release its pipe FDs.

        Closes ``stdin`` first — the ``serve()`` loop in ``rlmflow.runtime.repl``
        reads until EOF, so that's enough for a graceful shutdown in the
        common case (the REPL exits, the container's ``--rm`` flag wipes
        it). We only escalate to ``terminate()``/``kill()`` if the child
        is still alive after a short wait, then close the remaining
        pipes and reap the process so its FDs aren't left behind for
        the GC to clean up at some unspecified later time.
        """
        proc, self.proc = self.proc, None
        if proc is None:
            return

        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass

        try:
            proc.wait(timeout=2)
        except sp.TimeoutExpired:
            for action in (proc.terminate, proc.kill):
                try:
                    action()
                    proc.wait(timeout=2)
                    break
                except Exception:
                    continue
        except Exception:
            pass

        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception:
                pass

    def clone(
        self, workspace: BaseWorkspace | str | Path | None = None
    ) -> DockerRuntime:
        new = self.__class__(
            self.image,
            workspace=workspace if workspace is not None else self._workspace_arg,
            **self.options,
        )
        for name, td in self.tools.items():
            if td.core:
                continue
            new.tools[name] = td
            if td.fn is not None:
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
    argv += list(entrypoint_argv or ["python", "-m", "rlmflow.runtime.repl"])
    return argv
