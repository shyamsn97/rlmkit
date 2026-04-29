"""End-to-end tests for SubprocessRuntime.

Spawns a real ``python -m rlmkit.runtime.repl`` subprocess and exercises
the full JSON-over-stdio protocol: execute, inject (value + proxy), and
generator-based suspension with proxied tool calls during yield/resume.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from rlmkit.runtime.subprocess import SubprocessRuntime
from rlmkit.node import ChildHandle, WaitRequest


def _argv() -> list[str]:
    return [sys.executable, "-m", "rlmkit.runtime.repl"]


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


def test_start_code_no_yield(runtime):
    suspended, result = runtime.start_code("print('ran')")
    assert suspended is False
    assert result == "ran"


def test_start_code_with_yield_and_resume(runtime):
    """Exercise the full delegate/wait cycle over the wire."""

    def delegate(prompt: str) -> ChildHandle:
        return ChildHandle(agent_id=f"child-{prompt}")

    def wait(handles):
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    runtime.inject("delegate", delegate)
    runtime.inject("wait", wait)

    code = (
        "print('before')\n"
        "h = delegate('q1')\n"
        "results = yield wait([h])\n"
        "print('after:', results)\n"
    )
    suspended, payload = runtime.start_code(code)
    assert suspended is True
    request, pre_output = payload
    assert isinstance(request, WaitRequest)
    assert request.agent_ids == ["child-q1"]
    assert pre_output == "before"

    suspended, result = runtime.resume_code({"child-q1": "answer"})
    assert suspended is False
    assert result == "after: {'child-q1': 'answer'}"


def test_star_import_inside_yielding_block(runtime):
    """`from X import *` is illegal inside functions, but the generator
    wrapper hoists it out so agent code with ``yield`` can still use it."""

    def delegate(prompt: str) -> ChildHandle:
        return ChildHandle(agent_id="c")

    def wait(handles):
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    runtime.inject("delegate", delegate)
    runtime.inject("wait", wait)

    code = (
        "from math import *\n"
        "h = delegate('q')\n"
        "yield wait([h])\n"
        "print(int(pi * 100))\n"
    )
    suspended, _ = runtime.start_code(code)
    assert suspended is True
    suspended, out = runtime.resume_code({"c": "done"})
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


def test_annotated_assignment_inside_yielding_block(runtime):
    """``x: T = v`` must not clash with the generator wrapper's implicit
    ``global x``. Python raises ``SyntaxError: annotated name 'x' can't be
    global`` unless the wrapper strips the annotation first.
    """

    def delegate(prompt: str) -> ChildHandle:
        return ChildHandle(agent_id="c")

    def wait(handles):
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    runtime.inject("delegate", delegate)
    runtime.inject("wait", wait)

    code = (
        "errors: list[str] = []\n"
        "count: int\n"
        "for i in range(2):\n"
        "    tag: str = f'e{i}'\n"
        "    errors.append(tag)\n"
        "h = delegate('q')\n"
        "yield wait([h])\n"
        "count = len(errors)\n"
        "print(count, errors)\n"
    )
    suspended, _ = runtime.start_code(code)
    assert suspended is True
    suspended, out = runtime.resume_code({"c": "done"})
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
