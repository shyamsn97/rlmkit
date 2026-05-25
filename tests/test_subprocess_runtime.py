"""End-to-end tests for SubprocessRuntime.

Spawns a real ``python -m rlmflow.runtime.repl`` subprocess and exercises
the full JSON-over-stdio protocol: execute, inject (value + proxy), and
await-based suspension with proxied tool calls during resume.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from rlmflow.graph import ChildHandle, WaitRequest
from rlmflow.runtime.subprocess import SubprocessRuntime


def _argv() -> list[str]:
    return [sys.executable, "-m", "rlmflow.runtime.repl"]


@pytest.fixture
def runtime():
    rt = SubprocessRuntime(_argv())
    try:
        yield rt
    finally:
        rt.close()


def test_execute_returns_stdout(runtime):
    assert runtime.execute("print('hello')") == "hello"


def test_execute_multiple_statements_share_namespace(runtime):
    runtime.execute("x = 7")
    assert runtime.execute("print(x * 2)") == "14"


def test_inject_value(runtime):
    runtime.inject("TOKEN", "abc123")
    assert runtime.execute("print(TOKEN)") == "abc123"


def test_proxied_tool_error_does_not_kill_runtime(runtime):
    """If a proxied tool raises, agent code sees a Python exception and
    the REPL stays alive for the next execute() call."""

    def boom(x):
        raise ValueError(f"bad input: {x!r}")

    runtime.inject("boom", boom)

    out = runtime.execute("boom('hi')")
    assert "ValueError" in out
    assert "bad input" in out

    # Runtime still works after the error.
    assert runtime.execute("print(1 + 1)") == "2"


def test_inject_object_proxies_methods(runtime):
    """Non-callable, non-literal objects are exposed via a method proxy."""

    class Store:
        def __init__(self) -> None:
            self.data: dict[str, str] = {}

        def write(self, key: str, value: str) -> None:
            self.data[key] = value

        def read(self, key: str) -> str:
            return self.data[key]

    store = Store()
    runtime.inject("STORE", store)

    runtime.execute("STORE.write('k', 'v')")
    assert store.data == {"k": "v"}
    assert runtime.execute("print(STORE.read('k'))") == "v"


def test_inject_proxy_callable_roundtrip(runtime):
    calls = []

    def add(a, b):
        calls.append((a, b))
        return a + b

    runtime.inject("add", add)
    out = runtime.execute("print(add(2, 3))")
    assert out == "5"
    assert calls == [(2, 3)]


def test_start_code_no_await(runtime):
    suspended, result, errored = runtime.start_code("print('ran')")
    assert suspended is False
    assert errored is False
    assert result == "ran"


def test_start_code_exception_sets_errored(runtime):
    suspended, result, errored = runtime.start_code("raise ValueError('boom')")
    assert suspended is False
    assert errored is True
    assert "ValueError: boom" in result


def test_start_code_syntax_error_sets_errored(runtime):
    suspended, result, errored = runtime.start_code("def (")
    assert suspended is False
    assert errored is True
    assert "SyntaxError" in result


def test_start_code_with_await_and_resume(runtime):
    """Exercise the full delegate/wait cycle over the wire."""

    def delegate(prompt: str) -> ChildHandle:
        return ChildHandle(agent_id=f"child-{prompt}")

    def rlm_wait(handles):
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    runtime.inject("delegate", delegate)
    runtime.inject("rlm_wait", rlm_wait)

    code = (
        "print('before')\n"
        "h = delegate('q1')\n"
        "results = await rlm_wait([h])\n"
        "print('after:', results)\n"
    )
    suspended, payload, errored = runtime.start_code(code)
    assert suspended is True
    assert errored is False
    request, pre_output = payload
    assert isinstance(request, WaitRequest)
    assert request.agent_ids == ["child-q1"]
    assert pre_output == "before"

    suspended, result, errored = runtime.resume_code({"child-q1": "answer"})
    assert suspended is False
    assert errored is False
    assert result == "after: {'child-q1': 'answer'}"


def test_star_import_inside_awaiting_block(runtime):
    """Top-level await keeps `from X import *` legal."""

    def delegate(prompt: str) -> ChildHandle:
        return ChildHandle(agent_id="c")

    def rlm_wait(handles):
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    runtime.inject("delegate", delegate)
    runtime.inject("rlm_wait", rlm_wait)

    code = (
        "from math import *\n"
        "h = delegate('q')\n"
        "await rlm_wait([h])\n"
        "print(int(pi * 100))\n"
    )
    suspended, _, _ = runtime.start_code(code)
    assert suspended is True
    suspended, out, _ = runtime.resume_code({"c": "done"})
    assert suspended is False
    assert out == "314"


def test_proxied_tool_sees_workspace_as_cwd(tmp_path, monkeypatch):
    """Proxied tools run with the host CWD chdir'd into the workspace
    so relative paths land inside it, not in the caller's CWD."""
    workspace = tmp_path / "ws"
    caller_cwd = tmp_path / "caller"
    caller_cwd.mkdir()
    monkeypatch.chdir(caller_cwd)

    rt = SubprocessRuntime(_argv(), workspace=workspace)
    try:
        def write_rel(path: str, content: str) -> str:
            Path(path).write_text(content)
            return "ok"

        rt.inject("write_rel", write_rel)
        rt.execute("write_rel('hello.txt', 'hi')")

        assert (workspace / "hello.txt").read_text() == "hi"
        assert not (caller_cwd / "hello.txt").exists()
        # Caller's CWD is restored after each proxy invocation.
        assert Path(os.getcwd()) == caller_cwd
    finally:
        rt.close()


def test_repl_code_sees_workspace_as_cwd(tmp_path, monkeypatch):
    """Relative writes in agent code should land in the runtime workspace."""
    workspace = tmp_path / "ws"
    caller_cwd = tmp_path / "caller"
    caller_cwd.mkdir()
    monkeypatch.chdir(caller_cwd)

    rt = SubprocessRuntime(_argv(), workspace=workspace)
    try:
        rt.execute("from pathlib import Path\nPath('style.css').write_text('body{}')")

        assert (workspace / "style.css").read_text() == "body{}"
        assert not (caller_cwd / "style.css").exists()
    finally:
        rt.close()


def test_annotated_assignment_inside_awaiting_block(runtime):
    """Top-level await should preserve ordinary annotation semantics."""

    def delegate(prompt: str) -> ChildHandle:
        return ChildHandle(agent_id="c")

    def rlm_wait(handles):
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    runtime.inject("delegate", delegate)
    runtime.inject("rlm_wait", rlm_wait)

    code = (
        "errors: list[str] = []\n"
        "count: int\n"
        "for i in range(2):\n"
        "    tag: str = f'e{i}'\n"
        "    errors.append(tag)\n"
        "h = delegate('q')\n"
        "await rlm_wait([h])\n"
        "count = len(errors)\n"
        "print(count, errors)\n"
    )
    suspended, _, _ = runtime.start_code(code)
    assert suspended is True
    suspended, out, _ = runtime.resume_code({"c": "done"})
    assert suspended is False
    assert out == "2 ['e0', 'e1']"


def test_clone_is_independent_process(runtime):
    runtime.execute("x = 1")
    twin = runtime.clone()
    try:
        twin.execute("x = 99")
        assert runtime.execute("print(x)") == "1"
        assert twin.execute("print(x)") == "99"
    finally:
        twin.close()
