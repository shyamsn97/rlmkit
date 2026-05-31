import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from rlmflow.graph import ChildHandle, ExecAction, Graph, WaitRequest
from rlmflow.rlm import RLMConfig, RLMFlow
from rlmflow.runtime.sandbox.remote import RemoteFileRuntime
from rlmflow.tools import FILE_TOOLS
from rlmflow.tools.builtins import SHOW_VARS
from rlmflow.workspace import ContextVariable, SessionVariable, Workspace
from tests.fakes.sandbox import (
    FakeDaytonaClient,
    FakeE2BSandboxFactory,
    NoopLLM,
    NoStartRemoteRuntime,
)

def test_e2b_runtime_executes_repl_protocol_with_sdk_shape(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rlmflow.runtime.sandbox.e2b import E2BRuntime

    runtime = E2BRuntime(
        workspace=tmp_path / "host",
        remote_workdir=str(tmp_path / "remote"),
        setup_commands=[],
    )
    try:
        assert runtime.execute("print('hello from e2b')") == "hello from e2b"
    finally:
        runtime.close()


def test_e2b_runtime_exposes_public_exec(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rlmflow.runtime.sandbox.e2b import E2BRuntime

    runtime = E2BRuntime(
        workspace=tmp_path / "host",
        remote_workdir=str(tmp_path / "remote"),
        setup_commands=[],
    )
    try:
        assert runtime.exec("printf public") == "public"
    finally:
        runtime.close()


def test_e2b_runtime_syncs_workspace_on_start_and_close(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rlmflow.runtime.sandbox.e2b import E2BRuntime

    host = tmp_path / "host"
    remote = tmp_path / "remote"
    host.mkdir()
    (host / "input.txt").write_text("from host")

    runtime = E2BRuntime(
        workspace=host,
        remote_workdir=str(remote),
        setup_commands=[],
    )
    try:
        runtime.execute(
            "from pathlib import Path\n"
            "assert Path('input.txt').read_text() == 'from host'\n"
            "Path('remote.txt').write_text('from sandbox')\n"
        )
    finally:
        runtime.close()

    assert (host / "remote.txt").read_text() == "from sandbox"


def test_daytona_runtime_executes_repl_protocol_with_sdk_shape(tmp_path):
    from rlmflow.runtime.sandbox.daytona import DaytonaRuntime

    client = FakeDaytonaClient()
    runtime = DaytonaRuntime(
        workspace=tmp_path / "host",
        remote_workdir=str(tmp_path / "remote"),
        setup_commands=[],
        daytona=client,
    )
    try:
        assert runtime.execute("print('hello from daytona')") == "hello from daytona"
    finally:
        runtime.close()


def test_remote_file_runtime_is_publicly_exported():
    from rlmflow.runtime import DaytonaRuntime, E2BRuntime, RemoteFileRuntime
    from rlmflow.runtime.sandbox.remote import (
        RemoteFileRuntime as DirectRemoteFileRuntime,
    )

    assert RemoteFileRuntime is DirectRemoteFileRuntime
    assert issubclass(E2BRuntime, RemoteFileRuntime)
    assert issubclass(DaytonaRuntime, RemoteFileRuntime)


def test_remote_runtime_clone_does_not_copy_core_tools(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rlmflow.runtime.sandbox.e2b import E2BRuntime

    def custom_tool() -> str:
        return "ok"

    runtime = E2BRuntime(
        workspace=tmp_path / "host",
        remote_workdir=str(tmp_path / "remote"),
        setup_commands=[],
    )
    runtime.register_tool(SHOW_VARS, core=True)
    runtime.register_tool(custom_tool)

    clone = runtime.clone(workspace=tmp_path / "child")

    assert "SHOW_VARS" not in clone.tools
    assert "custom_tool" in clone.tools


def test_remote_child_spawn_does_not_start_child_runtime(tmp_path):
    root_runtime = NoStartRemoteRuntime(tmp_path / "root")
    child_runtime = NoStartRemoteRuntime(tmp_path / "child")
    agent = RLMFlow(
        NoopLLM(),
        runtime=root_runtime,
        runtime_factory=lambda: child_runtime,
        config=RLMConfig(max_depth=1),
    )
    graph = agent.start("parent")
    parent_node = graph.agents["root"].current()

    handle = agent.spawn_child(
        "root",
        parent_node.id,
        "child",
        "do child work",
        "shared context",
    )

    assert isinstance(handle, ChildHandle)
    assert not root_runtime.touched_remote
    assert not child_runtime.touched_remote
    assert handle.agent_id in agent.session.load_graph().agents


def test_e2b_runtime_injects_workspace_handles_remotely(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rlmflow.runtime.sandbox.e2b import E2BRuntime

    workspace = Workspace.create(tmp_path / "host")
    workspace.context.write("context", "remote context", agent_id="root")
    workspace.session.write_agent(Graph(agent_id="root", query="hello"))
    runtime = E2BRuntime(
        workspace=workspace,
        remote_workdir=str(tmp_path / "remote"),
        setup_commands=[],
    )

    try:
        runtime.inject("CONTEXT", ContextVariable(workspace.context, agent_id="root"))
        runtime.inject(
            "SESSION",
            SessionVariable(workspace.session, agent_id="root", branch_id="main"),
        )
        assert "CONTEXT.read" not in runtime.proxied
        assert "SESSION.tree" not in runtime.proxied
        assert runtime.execute("print(CONTEXT.read())") == "remote context"
        assert "root" in runtime.execute("print(SESSION.tree())")
    finally:
        runtime.close()


def test_e2b_runtime_imports_file_tools_remotely(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rlmflow.runtime.sandbox.e2b import E2BRuntime

    runtime = E2BRuntime(
        workspace=tmp_path / "host",
        remote_workdir=str(tmp_path / "remote"),
        setup_commands=[],
    )
    try:
        runtime.register_tools(FILE_TOOLS)
        assert "write_file" not in runtime.proxied
        assert runtime.execute("write_file('x.txt', 'hello')\nprint(read_file('x.txt'))") == "hello"
    finally:
        runtime.close()


def test_e2b_runtime_keeps_control_tools_as_proxies(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rlmflow.runtime.sandbox.e2b import E2BRuntime
    from rlmflow.tools.builtins import make_delegate, make_done, make_wait

    runtime = E2BRuntime(
        workspace=tmp_path / "host",
        remote_workdir=str(tmp_path / "remote"),
        setup_commands=[],
    )
    env = {"AGENT_ID": "root", "PARENT_NODE_ID": "n0"}
    try:
        runtime.register_tool(make_done(env), core=True)
        runtime.register_tool(make_wait(), core=True)
        runtime.register_tool(make_delegate(lambda *args, **kwargs: "child", env), core=True)

        assert runtime.proxied == {}
        runtime.install_registered_tools()
        assert "done" in runtime.proxied
        assert "rlm_wait" in runtime.proxied
        assert "rlm_delegate" in runtime.proxied
    finally:
        runtime.close()


def _launcher_repl():
    """A fresh REPL (the same class the sandbox runs) with stub
    launchers plus ``rlm_delegate`` / ``rlm_wait`` bound by name.

    Remotely the primitives are stdio proxies; here they are plain closures.
    Either way the registered launchers resolve them from the executing REPL
    frame at call time, which is the sandbox-side guarantee.
    """
    from rlmflow.runtime.repl import REPL
    from rlmflow.tools.builtins import make_delegate, make_wait

    env = {"AGENT_ID": "root", "PARENT_NODE_ID": "n0"}

    def spawn_child(parent_agent_id, parent_node_id, name, query, context, **kwargs):
        return ChildHandle(f"root.{name}")

    repl = REPL()
    repl.namespace["rlm_delegate"] = make_delegate(spawn_child, env)
    repl.namespace["rlm_wait"] = make_wait()
    repl.handle({"cmd": "inject_launcher", "name": "launch_subagent"})
    repl.handle({"cmd": "inject_launcher", "name": "launch_subagents"})
    return repl


def test_repl_launch_subagents_suspends_and_resumes():
    """`await launch_subagents([...])` spawns
    each child via ``rlm_delegate``, suspends on one ``WaitRequest`` with every
    id, and on resume returns the children's answers in order."""

    repl = _launcher_repl()
    suspended, payload = repl.start(
        'results = await launch_subagents('
        '[{"name": "a", "query": "x"}, {"name": "b", "query": "y"}])'
    )
    assert suspended is True
    request, _ = payload
    assert request.agent_ids == ["root.a", "root.b"]

    suspended_again, _ = repl.resume(["A", "B"])
    assert suspended_again is False
    assert repl.namespace["results"] == ["A", "B"]


def test_repl_launch_subagent_single_suspends_and_unwraps_result():
    """`await launch_subagent(...)` suspends on a single-child ``WaitRequest``
    and returns the child's answer string (not a one-element list)."""

    repl = _launcher_repl()
    suspended, payload = repl.start(
        'answer = await launch_subagent("solve it", name="solver")'
    )
    assert suspended is True
    request, _ = payload
    assert request.agent_ids == ["root.solver"]

    suspended_again, _ = repl.resume(["the answer"])
    assert suspended_again is False
    assert repl.namespace["answer"] == "the answer"


def test_internal_delegation_primitives_are_hidden_from_public_tools(tmp_path):
    from rlmflow.runtime.local import LocalRuntime

    runtime = LocalRuntime(workspace=tmp_path / "workspace")
    RLMFlow(NoopLLM(), runtime=runtime, config=RLMConfig(max_depth=1))

    visible = {td.name for td in runtime.get_tool_defs()}
    assert "launch_subagent" in visible
    assert "launch_subagents" in visible
    assert "rlm_delegate" not in visible
    assert "rlm_wait" not in visible

    internal = {td.name for td in runtime.get_tool_defs(include_hidden=True)}
    assert {"rlm_delegate", "rlm_wait"} <= internal

    show_vars = runtime.repl._show_vars()
    assert "launch_subagent" in show_vars
    assert "launch_subagents" in show_vars
    assert "rlm_delegate" not in show_vars
    assert "rlm_wait" not in show_vars


def test_host_proxied_tool_can_call_visible_tool_via_context(tmp_path):
    from rlmflow.tools import get_repl_tools, tool
    from rlmflow.runtime.sandbox.remote import RemoteFileRuntime

    class InMemoryRemote(RemoteFileRuntime):
        def __init__(self):
            super().__init__(workspace=tmp_path / "host", remote_workdir="/workspace")
            self.sent: list[dict] = []

        def send(self, msg: dict) -> None:
            self.sent.append(msg)

        def recv(self) -> dict:
            raise AssertionError("recv should not be called")

        def exec(self, command: str, *, timeout: float | None = None) -> str:
            del command, timeout
            return ""

    @tool("Echo text.")
    def echo(text: str) -> str:
        return f"echo:{text}"

    @tool("Call another tool.")
    def call_echo(text: str) -> str:
        return get_repl_tools()["echo"](text)

    runtime = InMemoryRemote()
    runtime.register_tool(echo)
    runtime.register_tool(call_echo)
    runtime.proxied["call_echo"] = call_echo

    runtime.handle_proxy_call({"proxy": "call_echo", "args": ["hi"]})

    assert runtime.sent == [{"value": "echo:hi"}]


def test_remote_runtime_short_circuits_proxied_wait_and_consumes_ack(tmp_path):
    from rlmflow.runtime.sandbox.remote import RemoteFileRuntime

    class InMemoryRemote(RemoteFileRuntime):
        def __init__(self):
            super().__init__(workspace=tmp_path / "host", remote_workdir="/workspace")
            self.sent: list[dict] = []
            self.responses: queue.Queue[dict] = queue.Queue()

        def prepare_for_execution(self) -> None:
            pass

        def send(self, msg: dict) -> None:
            self.sent.append(msg)

        def recv(self) -> dict:
            return self.responses.get_nowait()

        def exec(self, command: str, *, timeout: float | None = None) -> str:
            del command, timeout
            return ""

        def list_files(self, remote_root: str) -> list[str]:
            del remote_root
            return []

    runtime = InMemoryRemote()
    handle = ChildHandle("root.child")
    runtime.proxied["rlm_wait"] = lambda *handles: WaitRequest(
        [h.agent_id for h in handles]
    )

    runtime.responses.put({"proxy": "rlm_wait", "args": [handle.to_dict()]})
    response = runtime.call({"cmd": "run", "code": "await rlm_wait(h)"})

    assert response == {"suspended": True, "agent_ids": ["root.child"]}
    assert runtime.sent == [
        {"cmd": "run", "code": "await rlm_wait(h)"},
        {"value": {"wait_request": ["root.child"]}},
    ]
    assert runtime._pending_wait_ack

    runtime.responses.put({"suspended": True, "agent_ids": ["root.child"]})
    runtime.responses.put({"suspended": False, "output": "resumed", "errored": False})

    suspended, payload, errored = runtime.resume_code(["child result"])

    assert runtime.sent[-1] == {
        "cmd": "resume",
        "value": ["child result"],
        "tool_context": {"visible": [], "hidden": []},
    }
    assert (suspended, payload, errored) == (False, "resumed", False)
    assert not runtime._pending_wait_ack


def test_remote_runtime_consumes_wait_ack_before_prepare_side_effects(tmp_path):
    from rlmflow.runtime.sandbox.remote import RemoteFileRuntime

    class InMemoryRemote(RemoteFileRuntime):
        def __init__(self):
            super().__init__(workspace=tmp_path / "host", remote_workdir="/workspace")
            self.events: list[str] = []
            self.responses: queue.Queue[dict] = queue.Queue()

        def recv(self) -> dict:
            self.events.append("recv")
            return self.responses.get_nowait()

        def send(self, msg: dict) -> None:
            del msg
            self.events.append("send")

        def sync_workspace_to_runtime(self) -> None:
            self.events.append("sync")

        def install_registered_tools(self) -> None:
            self.events.append("install")

        def exec(self, command: str, *, timeout: float | None = None) -> str:
            del command, timeout
            return ""

        def list_files(self, remote_root: str) -> list[str]:
            del remote_root
            return []

    runtime = InMemoryRemote()
    runtime._pending_wait_ack = True
    runtime.responses.put({"suspended": True, "agent_ids": ["root.child"]})

    runtime.prepare_for_execution()

    assert runtime.events == ["recv", "sync", "install"]
    assert not runtime._pending_wait_ack


def test_inject_env_preserves_suspended_remote_repl(tmp_path):
    from rlmflow.graph import RuntimeRef
    from rlmflow.runtime.sandbox.remote import RemoteFileRuntime

    class SuspendedRemote(RemoteFileRuntime):
        def __init__(self):
            super().__init__(workspace=tmp_path / "host", remote_workdir="/workspace")
            self.suspended = True
            self.events: list[object] = []
            self.responses: queue.Queue[dict] = queue.Queue()

        def prepare_for_resume(self) -> None:
            self.events.append("prepare_for_resume")

        def prepare_for_execution(self) -> None:
            raise AssertionError("resume setup must not do full execution prep")

        def send(self, msg: dict) -> None:
            self.events.append(msg)

        def recv(self) -> dict:
            return self.responses.get_nowait()

        def exec(self, command: str, *, timeout: float | None = None) -> str:
            del command, timeout
            return ""

        def list_files(self, remote_root: str) -> list[str]:
            del remote_root
            return []

    runtime = SuspendedRemote()
    runtime.responses.put({"ok": True})
    runtime.responses.put({"ok": True})
    runtime.responses.put({"ok": True})
    runtime.responses.put({"ok": True})
    agent = RLMFlow(
        NoopLLM(),
        runtime=runtime,
        config=RLMConfig(max_depth=1),
    )
    graph = Graph(agent_id="root", runtime=RuntimeRef(id="root"))
    node = ExecAction(agent_id="root", seq=1, code="")

    assert agent.inject_env(graph, node) is runtime

    assert runtime.events[0] == "prepare_for_resume"
    sent = [event for event in runtime.events[1:] if isinstance(event, dict)]
    assert [msg["cmd"] for msg in sent] == ["inject", "inject", "inject", "inject"]
    assert {msg["name"] for msg in sent} == {
        "AGENT_ID",
        "DEPTH",
        "MAX_DEPTH",
        "PARENT_NODE_ID",
    }


def test_e2b_runtime_proxies_llm_query_batched_to_engine(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rlmflow.runtime.sandbox.e2b import E2BRuntime

    runtime = E2BRuntime(
        workspace=tmp_path / "host",
        remote_workdir=str(tmp_path / "remote"),
        setup_commands=[],
    )
    try:
        RLMFlow(NoopLLM(), runtime=runtime, config=RLMConfig(max_depth=1))
        assert runtime.execute("print(callable(llm_query_batched))") == "True"
        assert "llm_query_batched" in runtime.proxied
    finally:
        runtime.close()


def test_remote_close_ignores_already_gone_sandbox(tmp_path):
    from rlmflow.runtime.sandbox.remote import RemoteFileRuntime

    class GoneSandboxError(Exception):
        pass

    class GoneRuntime(RemoteFileRuntime):
        def __init__(self):
            super().__init__(workspace=tmp_path / "host", remote_workdir="/workspace")
            self._started = True
            self.closed = False

        def exec(self, command: str, *, timeout: float | None = None) -> str:
            del command, timeout
            raise GoneSandboxError("sandbox already shut down")

        def list_files(self, remote_root: str) -> list[str]:
            del remote_root
            raise GoneSandboxError("sandbox already shut down")

        def _is_sandbox_gone(self, exc: Exception) -> bool:
            return isinstance(exc, GoneSandboxError)

        def _close_sandbox(self) -> None:
            self.closed = True

    runtime = GoneRuntime()
    runtime.close()

    assert runtime.closed


def test_modal_runtime_reports_gone_sandbox_with_timeout_hint(monkeypatch, tmp_path):
    from rlmflow.runtime.sandbox.modal import ModalRuntime

    class NotFoundError(Exception):
        pass

    class GoneModalSandbox:
        def exec(self, *args, **kwargs):
            del args, kwargs
            raise NotFoundError(
                "Modal Sandbox with container ID ta-test not found. "
                "This means this Sandbox has already shut down."
            )

    monkeypatch.setitem(
        sys.modules,
        "modal.stream_type",
        SimpleNamespace(StreamType=SimpleNamespace(PIPE="PIPE")),
    )

    runtime = ModalRuntime(workspace=tmp_path / "host", timeout=123)
    runtime.container = GoneModalSandbox()
    runtime._started = True

    try:
        runtime.exec("printf hello")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "Modal sandbox disappeared" in message
    assert "timeout: 123s" in message
    assert runtime.container is None
    assert not runtime._started


def test_modal_runtime_exec_times_out_on_stuck_stream(monkeypatch, tmp_path):
    from rlmflow.runtime.sandbox.modal import ModalRuntime

    class HangingStream:
        def read(self):
            time.sleep(1)
            return ""

    class HangingProcess:
        stdout = HangingStream()
        stderr = HangingStream()

        def wait(self):
            return 0

    class HangingModalSandbox:
        def exec(self, *args, **kwargs):
            del args, kwargs
            return HangingProcess()

    monkeypatch.setitem(
        sys.modules,
        "modal.stream_type",
        SimpleNamespace(StreamType=SimpleNamespace(PIPE="PIPE")),
    )

    runtime = ModalRuntime(workspace=tmp_path / "host", repl_timeout=0.05)
    runtime.container = HangingModalSandbox()
    runtime._started = True

    try:
        runtime.exec("printf hello", timeout=0.05)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "Modal command did not complete cleanly" in message
    assert "stdout.read" in message
    assert not runtime._started


def test_modal_runtime_starts_repl_as_sandbox_entrypoint(monkeypatch, tmp_path):
    from rlmflow.runtime.sandbox.modal import ModalRuntime

    class FakeStdin:
        def __init__(self):
            self.writes: list[str] = []

        def write(self, data: str) -> None:
            self.writes.append(data)

        def drain(self) -> None:
            pass

    class FakeSandbox:
        created_args: tuple[str, ...] | None = None
        created_kwargs: dict | None = None
        instance = None

        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = iter(())
            self.stderr = iter(())

        @classmethod
        def create(cls, *args, **kwargs):
            cls.created_args = args
            cls.created_kwargs = kwargs
            cls.instance = cls()
            return cls.instance

        def poll(self):
            return None

    class FakeApp:
        @staticmethod
        def lookup(name, create_if_missing=False):
            return {"name": name, "create": create_if_missing}

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(App=FakeApp, Sandbox=FakeSandbox),
    )
    monkeypatch.setitem(
        sys.modules,
        "modal.stream_type",
        SimpleNamespace(StreamType=SimpleNamespace(PIPE="PIPE")),
    )

    image = object()
    runtime = ModalRuntime(
        app_name="test-app",
        workspace=tmp_path / "host",
        remote_workdir="/workspace",
        image=image,
    )

    runtime.send({"cmd": "ping"})

    assert FakeSandbox.created_args[:2] == ("sh", "-lc")
    assert "rlmflow.runtime.repl" in FakeSandbox.created_args[2]
    assert "--workdir /workspace" in FakeSandbox.created_args[2]
    assert FakeSandbox.created_kwargs["app"] == {"name": "test-app", "create": True}
    assert FakeSandbox.created_kwargs["image"] is image
    assert "stdout" not in FakeSandbox.created_kwargs
    assert "stderr" not in FakeSandbox.created_kwargs
    assert FakeSandbox.instance.stdin.writes == ['{"cmd": "ping"}\n']


def test_modal_runtime_clears_workspace_contents_not_root(tmp_path):
    from rlmflow.runtime.sandbox.modal import ModalRuntime

    class RecordingModalRuntime(ModalRuntime):
        def __init__(self):
            super().__init__(workspace=tmp_path / "host", remote_workdir="/workspace")
            self.commands: list[str] = []

        def exec(self, command: str, *, timeout: float | None = None) -> str:
            del timeout
            self.commands.append(command)
            return ""

    runtime = RecordingModalRuntime()
    runtime.remove_path("/workspace", recursive=True)

    assert runtime.commands
    assert "find /workspace -mindepth 1 -maxdepth 1" in runtime.commands[0]
    assert "rm -rf -- /workspace" not in runtime.commands[0]


def test_modal_runtime_exec_uses_stream_type_pipe(monkeypatch, tmp_path):
    from rlmflow.runtime.sandbox.modal import ModalRuntime

    class FakeStream:
        def read(self):
            return ""

    class FakeProcess:
        stdout = FakeStream()
        stderr = FakeStream()

        def wait(self):
            return 0

    class FakeSandbox:
        def __init__(self):
            self.exec_kwargs: dict | None = None

        def exec(self, *args, **kwargs):
            del args
            self.exec_kwargs = kwargs
            return FakeProcess()

    monkeypatch.setitem(
        sys.modules,
        "modal.stream_type",
        SimpleNamespace(StreamType=SimpleNamespace(PIPE="PIPE")),
    )

    runtime = ModalRuntime(workspace=tmp_path / "host")
    runtime.container = FakeSandbox()

    assert runtime.exec("printf ok") == ""
    assert runtime.container.exec_kwargs["stdout"] == "PIPE"
    assert runtime.container.exec_kwargs["stderr"] == "PIPE"


def test_modal_runtime_uses_direct_repl_streams(tmp_path):
    from rlmflow.runtime.sandbox.modal import ModalRuntime

    class FakeStdin:
        def __init__(self):
            self.writes: list[str] = []
            self.drained = False

        def write(self, data: str) -> None:
            self.writes.append(data)

        def drain(self) -> None:
            self.drained = True

    class FakeSandbox:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = iter(['{"ok": true}\n'])
            self.stderr = iter(())

        def poll(self):
            return None

    runtime = ModalRuntime(workspace=tmp_path / "host")
    runtime.container = FakeSandbox()
    runtime._stdout_queue = queue.Queue()
    runtime._stdout_queue.put('{"ok": true}\n')
    runtime._started = True

    runtime.send({"cmd": "ping"})
    assert runtime.container.stdin.writes == ['{"cmd": "ping"}\n']
    assert runtime.container.stdin.drained
    assert runtime.recv() == {"ok": True}


def test_modal_runtime_splits_stdout_chunks_into_json_lines(tmp_path):
    from rlmflow.runtime.sandbox.modal import ModalRuntime

    runtime = ModalRuntime(workspace=tmp_path / "host")
    runtime._stdout_queue = queue.Queue()
    runtime._start_stream_reader(
        iter(['{"ok": true}\n{"suspended": false, "output": ""}\n']),
        runtime._stdout_queue,
    )

    assert runtime._stdout_queue.get(timeout=1) == '{"ok": true}'
    assert (
        runtime._stdout_queue.get(timeout=1)
        == '{"suspended": false, "output": ""}'
    )
