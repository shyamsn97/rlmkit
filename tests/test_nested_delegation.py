"""Test nested delegation: root → child → grandchild → great-grandchild.

Uses a scripted FakeLLM that always returns the same ```repl``` block.
The code itself branches on DEPTH vs MAX_DEPTH at runtime — delegates
deeper if possible, otherwise does work directly and calls done().
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rlmkit.llm import LLMClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import ChildStep, CodeExec, LLMReply, RLMState, Status

# The REPL code every agent executes. It uses the injected DEPTH / MAX_DEPTH
# variables to decide whether to delegate or finish directly.
UNIVERSAL_REPLY = '''```repl
if int(DEPTH) < int(MAX_DEPTH):
    h = delegate("go deeper", wait=False)
    results = wait_all(h)
    done("from-" + AGENT_ID + ":" + results[0])
else:
    done("leaf-" + AGENT_ID)
```'''


class FakeLLM(LLMClient):
    """Always returns the same universal reply."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages, *args, **kwargs):
        self.call_count += 1
        return UNIVERSAL_REPLY


def run_to_completion(agent, state, limit=200):
    steps = []
    while not state.finished:
        state = agent.step(state)
        steps.append(state)
        if len(steps) > limit:
            raise RuntimeError("too many steps, likely stuck")
    return steps, state


def all_descendants(state):
    result = []
    for child in state.children:
        result.append(child)
        result.extend(all_descendants(child))
    return result


# ── Tests ─────────────────────────────────────────────────────────────

def test_two_levels():
    """root (depth=0) → child (depth=1, leaf). max_depth=1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        llm = FakeLLM()
        agent = RLM(llm_client=llm, runtime=rt, config=RLMConfig(max_depth=1))

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        assert "from-root:" in final.result
        assert "leaf-root.1" in final.result
        assert len(final.children) == 1
        assert final.children[0].finished
        assert final.children[0].result == "leaf-root.1"

        print(f"  {len(steps)} steps, {llm.call_count} LLM calls")
        print(f"  root result: {final.result}")
        print(f"  child result: {final.children[0].result}")


def test_three_levels():
    """root → child → grandchild. max_depth=2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        llm = FakeLLM()
        agent = RLM(llm_client=llm, runtime=rt, config=RLMConfig(max_depth=2))

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        descendants = all_descendants(final)
        assert len(descendants) == 2  # child + grandchild

        child = final.children[0]
        grandchild = child.children[0]
        assert grandchild.result == "leaf-root.1.1"
        assert "leaf-root.1.1" in child.result
        assert "leaf-root.1.1" in final.result

        print(f"  {len(steps)} steps, {llm.call_count} LLM calls")
        for d in [final] + descendants:
            print(f"  {d.agent_id}: {d.result}")


def test_four_levels():
    """root → child → grandchild → great-grandchild. max_depth=3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        llm = FakeLLM()
        agent = RLM(llm_client=llm, runtime=rt, config=RLMConfig(max_depth=3))

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        descendants = all_descendants(final)
        assert len(descendants) == 3

        deepest = descendants[-1]
        assert "leaf-" in deepest.result
        assert deepest.children == []

        # Result bubbles all the way up
        assert deepest.result in final.result

        print(f"  {len(steps)} steps, {llm.call_count} LLM calls")
        for d in [final] + descendants:
            print(f"  {d.agent_id}: {d.result}")


def test_child_events_during_supervision():
    """ChildStep events should contain sub-events from active children."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        llm = FakeLLM()
        agent = RLM(llm_client=llm, runtime=rt, config=RLMConfig(max_depth=2))

        steps, final = run_to_completion(agent, agent.start("test"))

        child_steps = [s for s in steps if isinstance(s.event, ChildStep)]
        assert len(child_steps) > 0

        has_sub_events = any(len(s.event.child_events) > 0 for s in child_steps)
        assert has_sub_events, "ChildStep events should contain child sub-events"

        event_types = {type(s.event).__name__ for s in steps}
        print(f"  event types: {event_types}")
        for s in child_steps:
            sub = [type(e).__name__ for e in s.event.child_events]
            print(f"  ChildStep all_done={s.event.all_done} sub_events={sub}")


def test_max_depth_zero():
    """At max_depth=0, root is already the leaf — no delegation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        llm = FakeLLM()
        agent = RLM(llm_client=llm, runtime=rt, config=RLMConfig(max_depth=0))

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        assert final.children == []
        assert final.result == "leaf-root"
        assert llm.call_count == 1

        print(f"  {len(steps)} steps, result: {final.result}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  PASSED\n")
    print("All tests passed.")
