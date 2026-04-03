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
    h = delegate("child", "go deeper")
    results = yield wait(h)
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
        assert "leaf-root.child" in final.result
        assert len(final.children) == 1
        assert final.children[0].finished
        assert final.children[0].result == "leaf-root.child"

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
        assert grandchild.result == "leaf-root.child.child"
        assert "leaf-root.child.child" in child.result
        assert "leaf-root.child.child" in final.result

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


def test_agent_finished_on_delegation_path():
    """ChildStep.agent_finished should be True when done() is called after children return."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        llm = FakeLLM()
        agent = RLM(llm_client=llm, runtime=rt, config=RLMConfig(max_depth=1))

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished

        # The final event should be a ChildStep with agent_finished=True
        assert isinstance(final.event, ChildStep)
        assert final.event.all_done
        assert final.event.agent_finished

        # In-progress ChildStep events should NOT have agent_finished
        in_progress = [
            s for s in steps
            if isinstance(s.event, ChildStep) and not s.event.all_done
        ]
        for s in in_progress:
            assert not s.event.agent_finished

        print(f"  final event: {type(final.event).__name__} agent_finished={final.event.agent_finished}")


def test_agent_finished_false_on_direct_path():
    """When done() is called directly (no delegation), there should be no ChildStep at all."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        llm = FakeLLM()
        agent = RLM(llm_client=llm, runtime=rt, config=RLMConfig(max_depth=0))

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        assert isinstance(final.event, CodeExec)
        assert not any(isinstance(s.event, ChildStep) for s in steps)

        print(f"  final event: {type(final.event).__name__}")


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


def test_named_delegate_collision():
    """Duplicate names get auto-suffixed: child, child_2, child_3."""
    reply = '''```repl
h1 = delegate("worker", "task A")
h2 = delegate("worker", "task B")
h3 = delegate("worker", "task C")
results = yield wait(h1, h2, h3)
done(" ".join(results))
```'''

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return reply
            return '```repl\ndone("leaf-" + AGENT_ID)\n```'

    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        agent = RLM(llm_client=ScriptedLLM(), runtime=rt, config=RLMConfig(max_depth=1))

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        child_ids = sorted(c.agent_id for c in final.children)
        assert child_ids == ["root.worker", "root.worker_2", "root.worker_3"]
        print(f"  child IDs: {child_ids}")


def test_llm_client_routing():
    """delegate(model='fast') should use the corresponding client."""
    reply = '''```repl
h = delegate("sub", "do it", model="fast")
[result] = yield wait(h)
done(result)
```'''

    class TaggedLLM(LLMClient):
        def __init__(self, tag: str):
            self.tag = tag
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            return f'```repl\ndone("used-{self.tag}")\n```'

    strong = TaggedLLM("strong")
    fast = TaggedLLM("fast")

    # Override strong's first reply to delegate with model="fast"
    original_chat = strong.chat
    def patched_chat(messages, *args, **kwargs):
        strong.call_count += 1
        if strong.call_count == 1:
            return reply
        return original_chat(messages, *args, **kwargs)
    strong.chat = patched_chat
    strong.call_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        agent = RLM(
            llm_client=strong,
            runtime=rt,
            config=RLMConfig(max_depth=1),
            llm_clients={
                "strong": {"model": strong},
                "fast": {"model": fast},
            },
        )

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        assert final.result == "used-fast"
        assert fast.call_count == 1
        print(f"  result: {final.result}, fast calls: {fast.call_count}")


def test_multi_yield():
    """Agent yields twice: first batch of children, resumes, delegates again."""
    root_reply = '''```repl
h1 = delegate("batch1_a", "search A")
h2 = delegate("batch1_b", "search B")
results1 = yield wait(h1, h2)

h3 = delegate("batch2_a", "search C")
h4 = delegate("batch2_b", "search D")
results2 = yield wait(h3, h4)

done(",".join(results1) + "|" + ",".join(results2))
```'''

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return root_reply
            return '```repl\ndone("leaf-" + AGENT_ID)\n```'

    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        agent = RLM(
            llm_client=ScriptedLLM(),
            runtime=rt,
            config=RLMConfig(max_depth=1),
        )

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        assert len(final.children) == 4
        child_ids = sorted(c.agent_id for c in final.children)
        assert child_ids == [
            "root.batch1_a", "root.batch1_b",
            "root.batch2_a", "root.batch2_b",
        ]
        assert "leaf-root.batch1_a" in final.result
        assert "leaf-root.batch2_a" in final.result
        print(f"  result: {final.result}")
        print(f"  children: {child_ids}")


def test_orphaned_delegates_error():
    """delegate() without wait() should surface an error, not silently orphan."""
    orphan_reply = '''```repl
h = delegate("worker", "do stuff")
print("forgot to wait")
```'''
    fix_reply = '''```repl
done("fixed")
```'''

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return orphan_reply
            return fix_reply

    with tempfile.TemporaryDirectory() as tmpdir:
        rt = LocalRuntime(workspace=Path(tmpdir))
        agent = RLM(llm_client=ScriptedLLM(), runtime=rt, config=RLMConfig(max_depth=1))

        steps, final = run_to_completion(agent, agent.start("test"))

        assert final.finished
        assert final.result == "fixed"
        # The orphan error should have appeared in a CodeExec event
        exec_events = [s for s in steps if isinstance(s.event, CodeExec)]
        orphan_output = exec_events[0].event.output
        assert "OrphanedDelegatesError" in orphan_output
        assert "wait" in orphan_output
        # The orphaned child should have been cleaned up
        assert len(final.children) == 0
        print(f"  error surfaced: {orphan_output.splitlines()[-1][:80]}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  PASSED\n")
    print("All tests passed.")
