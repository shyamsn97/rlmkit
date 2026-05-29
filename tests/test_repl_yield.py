"""REPL await protocol — only `await rlm_wait(...)` suspends.

Covers the REPL yield protocol matrix in `docs/internals.md`. The engine must:

1. Suspend (and surface ``WaitRequest.agent_ids``) only when the
   block does ``await rlm_wait(...)`` at top level.
2. Reject top-level ``yield``; it is no longer part of the action language.
3. Leave generic generator code alone: helpers defined inside the
   block, generator expressions, and ``yield from`` inside helpers
   must not trigger await handling or any engine intervention.
"""

from __future__ import annotations

from rlmflow.graph import ChildHandle, WaitRequest
from rlmflow.runtime.local import LocalRuntime
from rlmflow.runtime.repl import REPL, _has_top_level_await
from rlmflow.tools.builtins import SHOW_VARS


# ── _has_top_level_await ─────────────────────────────────────────────


def _awaits(code: str) -> bool:
    import ast

    return _has_top_level_await(ast.parse(code))


def test_top_level_await_detected():
    assert _awaits("await rlm_wait(h)")
    assert _awaits("x = await rlm_wait(h)")
    assert _awaits("if cond:\n    await rlm_wait(h)\nelse:\n    pass")
    assert _awaits("for h in handles:\n    await rlm_wait(h)")


def test_await_inside_nested_function_is_not_top_level():
    assert not _awaits("async def f():\n    await other()")
    assert not _awaits(
        "class C:\n    async def run(self):\n        await other()\n"
    )


def test_await_inside_comprehension_is_not_top_level():
    assert not _awaits("xs = [await f(i) for i in range(10)]")


def test_no_await_anywhere():
    assert not _awaits("x = 1\nprint(x)")


# ── REPL.start / advance ─────────────────────────────────────────────


def _new_repl():
    return REPL(namespace={"__builtins__": __builtins__})


def test_block_with_no_yield_runs_to_completion():
    r = _new_repl()
    suspended, out = r.start("print('hi')\nx = 2 + 2\nprint(x)")
    assert suspended is False
    assert "hi" in out and "4" in out


def test_show_vars_is_registered_as_builtin_tool():
    runtime = LocalRuntime()
    runtime.register_tool(SHOW_VARS, core=True)

    out = runtime.execute(
        "small = 4\n"
        "huge = 'x' * 1000\n"
        "_private = 'hidden'\n"
        "print(SHOW_VARS())\n"
    )
    assert "SHOW_VARS" in runtime.tools
    assert runtime.tools["SHOW_VARS"].core is True
    assert "'small': 'int'" in out
    assert "'huge': 'str'" in out
    assert "_private" not in out
    assert "xxxxxxxx" not in out


def test_helper_generator_defined_and_consumed_does_not_suspend():
    r = _new_repl()
    code = (
        "def squares(n):\n"
        "    for i in range(n):\n"
        "        yield i * i\n"
        "print(list(squares(5)))\n"
        "print(sum(squares(100)))\n"
    )
    suspended, out = r.start(code)
    assert suspended is False
    assert "[0, 1, 4, 9, 16]" in out
    assert "328350" in out


def test_generator_expression_does_not_suspend():
    r = _new_repl()
    suspended, out = r.start("print(sum(x*x for x in range(10)))")
    assert suspended is False
    assert "285" in out


def test_yield_from_inside_helper_does_not_suspend():
    r = _new_repl()
    code = (
        "def chained():\n"
        "    yield from range(3)\n"
        "    yield from (10, 20)\n"
        "print(list(chained()))\n"
    )
    suspended, out = r.start(code)
    assert suspended is False
    assert "[0, 1, 2, 10, 20]" in out


def test_top_level_bare_yield_is_rejected():
    r = _new_repl()
    code = "print('before')\nyield\nprint('after')\n"
    suspended, out = r.start(code)
    assert suspended is False
    assert r.errored is True
    assert "top-level `yield`" in out


def test_top_level_yield_with_value_is_rejected():
    r = _new_repl()
    code = "print('a')\nyield 42\nprint('b')\nyield 'hello'\nprint('c')\n"
    suspended, out = r.start(code)
    assert suspended is False
    assert r.errored is True
    assert "top-level `yield`" in out


def test_top_level_yield_handle_is_rejected():
    r = _new_repl()
    handle = ChildHandle(agent_id="root.kid")
    r.namespace["h"] = handle
    code = "print('before')\nyield h\nprint('after')\n"
    suspended, out = r.start(code)
    assert suspended is False
    assert r.errored is True
    assert "top-level `yield`" in out


# ── REPL.start / advance — only awaited WaitRequest suspends ─────────


def test_await_wait_suspends_with_correct_agent_ids():
    r = _new_repl()
    handle = ChildHandle(agent_id="root.kid")
    r.namespace["rlm_wait"] = lambda *hs: WaitRequest([h.agent_id for h in hs])
    r.namespace["h"] = handle
    suspended, payload = r.start("print('x')\nresult = await rlm_wait(h)\n")
    assert suspended is True
    request, pre = payload
    assert isinstance(request, WaitRequest)
    assert request.agent_ids == ["root.kid"]
    assert "x" in pre


def test_multiple_handles_in_one_wait():
    r = _new_repl()
    r.namespace["rlm_wait"] = lambda *hs: WaitRequest([h.agent_id for h in hs])
    r.namespace["h1"] = ChildHandle(agent_id="root.a")
    r.namespace["h2"] = ChildHandle(agent_id="root.b")
    suspended, (request, _) = r.start("await rlm_wait(h1, h2)")
    assert suspended is True
    assert request.agent_ids == ["root.a", "root.b"]


def test_unsupported_await_errors():
    r = _new_repl()
    code = "async def f():\n    return 1\nresult = await f()\n"
    suspended, out = r.start(code)
    assert suspended is False
    assert r.errored is True
    assert "only `await launch_subagent(...)`" in out


def test_system_exit_in_block_is_captured_not_propagated():
    """``raise SystemExit(...)`` inside agent code must NOT exit the host.

    SystemExit inherits from BaseException, so a naive
    ``except Exception`` lets it bubble all the way up and terminate
    the driver process. The REPL must catch it and turn it into an
    ordinary captured traceback like any other agent error.
    """
    r = _new_repl()
    code = (
        "print('FIRST')\n"
        "raise SystemExit('boom-msg')\n"
        "print('UNREACHED')\n"
    )
    suspended, out = r.start(code)
    assert suspended is False
    assert r.errored is True
    assert "FIRST" in out
    assert "SystemExit" in out and "boom-msg" in out
    # the line after the raise must not have run
    assert "UNREACHED" not in out


def test_sys_exit_call_in_block_is_captured():
    r = _new_repl()
    code = (
        "import sys\n"
        "print('FIRST')\n"
        "sys.exit('halt-msg')\n"
        "print('UNREACHED')\n"
    )
    suspended, out = r.start(code)
    assert suspended is False
    assert r.errored is True
    assert "FIRST" in out
    assert "UNREACHED" not in out
    assert "SystemExit" in out and "halt-msg" in out


def test_keyboard_interrupt_in_block_is_captured():
    r = _new_repl()
    code = "print('FIRST')\nraise KeyboardInterrupt('user halt')\n"
    suspended, out = r.start(code)
    assert suspended is False
    assert r.errored is True
    assert "FIRST" in out
    assert "KeyboardInterrupt" in out


def test_resume_returns_send_value_to_block():
    r = _new_repl()
    r.namespace["rlm_wait"] = lambda *hs: WaitRequest([h.agent_id for h in hs])
    r.namespace["h"] = ChildHandle(agent_id="root.kid")
    suspended, _ = r.start("result = await rlm_wait(h)\nprint('got', result)\n")
    assert suspended is True
    suspended, out = r.resume(send_value=["payload"])
    assert suspended is False
    assert "got ['payload']" in out
