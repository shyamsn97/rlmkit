"""REPL yield protocol — only `yield wait(...)` suspends.

Covers the matrix in `docs/repl_yield_protocol.md`. The engine must:

1. Suspend (and surface ``WaitRequest.agent_ids``) only when the
   block does ``yield wait(...)`` at top level.
2. Treat every other yield (bare ``yield``, ``yield 42``,
   ``yield handle``, ``yield delegate(...)`` without ``wait``) as a
   plain Python generator yield — discard, resume, never suspend.
3. Leave generic generator code alone: helpers defined inside the
   block, generator expressions, and ``yield from`` inside helpers
   must not trigger generator wrapping or any engine intervention.
"""

from __future__ import annotations

import ast

from rlmflow.graph import ChildHandle, WaitRequest
from rlmflow.runtime.repl import REPL, _has_top_level_yield


# ── _has_top_level_yield ─────────────────────────────────────────────


def _yields(code: str) -> bool:
    return _has_top_level_yield(ast.parse(code))


def test_top_level_yield_detected():
    assert _yields("yield wait(h)")
    assert _yields("x = yield wait(h)")
    assert _yields("yield from gen()")
    assert _yields("if cond:\n    yield wait(h)\nelse:\n    pass")
    assert _yields("for h in handles:\n    yield wait(h)")


def test_yield_inside_nested_function_is_not_top_level():
    assert not _yields("def squares(n):\n    for i in range(n):\n        yield i")
    assert not _yields("async def f():\n    yield 1")
    assert not _yields("f = lambda: (yield 1)")
    assert not _yields(
        "class C:\n    def __iter__(self):\n        yield 1\n"
    )


def test_yield_inside_genexp_is_not_top_level():
    # generator expressions are syntactically a hidden function;
    # their internal yields should not gate wrapping
    assert not _yields("xs = list(i*i for i in range(10))")


def test_no_yield_anywhere():
    assert not _yields("x = 1\nprint(x)")


# ── REPL.start / advance — non-Wait yields don't suspend ─────────────


def _new_repl():
    return REPL(namespace={"__builtins__": __builtins__})


def test_block_with_no_yield_runs_to_completion():
    r = _new_repl()
    suspended, out = r.start("print('hi')\nx = 2 + 2\nprint(x)")
    assert suspended is False
    assert "hi" in out and "4" in out


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


def test_top_level_bare_yield_pumps_through():
    r = _new_repl()
    code = "print('before')\nyield\nprint('after')\n"
    suspended, out = r.start(code)
    assert suspended is False
    assert "before" in out and "after" in out


def test_top_level_yield_with_value_pumps_through():
    r = _new_repl()
    code = "print('a')\nyield 42\nprint('b')\nyield 'hello'\nprint('c')\n"
    suspended, out = r.start(code)
    assert suspended is False
    for tok in ("a", "b", "c"):
        assert tok in out


def test_top_level_yield_handle_pumps_through_does_not_crash():
    """``yield delegate_handle`` (forgot to wrap in wait) must NOT crash
    the engine. It pumps through silently."""
    r = _new_repl()
    handle = ChildHandle(agent_id="root.kid")
    r.namespace["h"] = handle
    code = "print('before')\nyield h\nprint('after')\n"
    suspended, out = r.start(code)
    assert suspended is False
    assert "before" in out and "after" in out


# ── REPL.start / advance — only WaitRequest suspends ─────────────────


def test_yield_wait_suspends_with_correct_agent_ids():
    r = _new_repl()
    handle = ChildHandle(agent_id="root.kid")
    r.namespace["wait"] = lambda *hs: WaitRequest([h.agent_id for h in hs])
    r.namespace["h"] = handle
    suspended, payload = r.start("print('x')\nresult = yield wait(h)\n")
    assert suspended is True
    request, pre = payload
    assert isinstance(request, WaitRequest)
    assert request.agent_ids == ["root.kid"]
    assert "x" in pre


def test_multiple_handles_in_one_wait():
    r = _new_repl()
    r.namespace["wait"] = lambda *hs: WaitRequest([h.agent_id for h in hs])
    r.namespace["h1"] = ChildHandle(agent_id="root.a")
    r.namespace["h2"] = ChildHandle(agent_id="root.b")
    suspended, (request, _) = r.start("yield wait(h1, h2)")
    assert suspended is True
    assert request.agent_ids == ["root.a", "root.b"]


def test_non_wait_yields_before_wait_are_pumped_then_suspends():
    """A block that yields some junk and then yields a wait must
    still suspend on the wait; the engine pumps past the junk."""
    r = _new_repl()
    r.namespace["wait"] = lambda *hs: WaitRequest([h.agent_id for h in hs])
    r.namespace["h"] = ChildHandle(agent_id="root.kid")
    code = "yield 1\nyield 'noise'\nyield wait(h)\n"
    suspended, (request, _) = r.start(code)
    assert suspended is True
    assert request.agent_ids == ["root.kid"]


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
    r.namespace["wait"] = lambda *hs: WaitRequest([h.agent_id for h in hs])
    r.namespace["h"] = ChildHandle(agent_id="root.kid")
    suspended, _ = r.start("result = yield wait(h)\nprint('got', result)\n")
    assert suspended is True
    suspended, out = r.resume(send_value=["payload"])
    assert suspended is False
    assert "got ['payload']" in out
