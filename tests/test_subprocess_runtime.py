"""End-to-end tests for SubprocessRuntime.

Spawns a real ``python -m rlmkit.runtime.repl`` subprocess and exercises
the full JSON-over-stdio protocol: execute, inject (value + proxy), and
generator-based suspension with proxied tool calls during yield/resume.
"""

from __future__ import annotations

import sys

import pytest

from rlmkit.runtime.subprocess import SubprocessRuntime
from rlmkit.state import ChildHandle, WaitRequest


def _argv() -> list[str]:
    return [sys.executable, "-m", "rlmkit.runtime.repl"]


@pytest.fixture
def runtime():
    rt = SubprocessRuntime(_argv())
    try:
        yield rt
    finally:
        rt.terminate()


def test_execute_returns_stdout(runtime):
    assert runtime.execute("print('hello')") == "hello"


def test_execute_multiple_statements_share_namespace(runtime):
    runtime.execute("x = 7")
    assert runtime.execute("print(x * 2)") == "14"


def test_inject_value(runtime):
    runtime.inject("TOKEN", "abc123")
    assert runtime.execute("print(TOKEN)") == "abc123"


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


def test_clone_is_independent_process(runtime):
    runtime.execute("x = 1")
    twin = runtime.clone()
    try:
        twin.execute("x = 99")
        assert runtime.execute("print(x)") == "1"
        assert twin.execute("print(x)") == "99"
    finally:
        twin.terminate()
