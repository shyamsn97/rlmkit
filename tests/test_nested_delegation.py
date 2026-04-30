"""Test nested delegation: root -> child -> grandchild -> great-grandchild.

Uses a scripted FakeLLM that always returns the same ```repl``` block.
The code itself branches on DEPTH vs MAX_DEPTH at runtime — delegates
deeper if possible, otherwise does work directly and calls done().
"""

from __future__ import annotations

from rlmflow.llm import LLMClient
from rlmflow.rlm import RLM, RLMConfig
from rlmflow.runtime.local import LocalRuntime
from rlmflow.node import CodeExec, ResumeExec, Status

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
    """root (depth=0) -> child (depth=1, leaf). max_depth=1."""
    llm = FakeLLM()
    agent = RLM(llm_client=llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=1))

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
    """root -> child -> grandchild. max_depth=2."""
    llm = FakeLLM()
    agent = RLM(llm_client=llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=2))

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    descendants = all_descendants(final)
    assert len(descendants) == 2

    child = final.children[0]
    grandchild = child.children[0]
    assert grandchild.result == "leaf-root.child.child"
    assert "leaf-root.child.child" in child.result
    assert "leaf-root.child.child" in final.result

    print(f"  {len(steps)} steps, {llm.call_count} LLM calls")
    for d in [final] + descendants:
        print(f"  {d.agent_id}: {d.result}")


def test_four_levels():
    """root -> child -> grandchild -> great-grandchild. max_depth=3."""
    llm = FakeLLM()
    agent = RLM(llm_client=llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=3))

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    descendants = all_descendants(final)
    assert len(descendants) == 3

    deepest = descendants[-1]
    assert "leaf-" in deepest.result
    assert deepest.children == []
    assert deepest.result in final.result

    print(f"  {len(steps)} steps, {llm.call_count} LLM calls")
    for d in [final] + descendants:
        print(f"  {d.agent_id}: {d.result}")


def test_step_advances_one_level():
    """Each step() advances existing nodes once; newly spawned wait for next step."""
    llm = FakeLLM()
    agent = RLM(llm_client=llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=2))

    state = agent.start("test")
    state = agent.step(state)
    assert state.status == Status.SUPERVISING

    steps, final = run_to_completion(agent, state)
    assert final.finished
    assert "leaf-root.child.child" in final.result
    print(f"  completed in {len(steps) + 1} total steps, result: {final.result}")


def test_resume_exec_on_delegation_path():
    """ResumeExec should fire when root resumes after children return."""
    llm = FakeLLM()
    agent = RLM(llm_client=llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=1))

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    assert isinstance(final.event, ResumeExec)

    print(f"  final event: {type(final.event).__name__}")


def test_no_supervision_on_direct_path():
    """When done() is called directly (no delegation), no supervision events."""
    llm = FakeLLM()
    agent = RLM(llm_client=llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=0))

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    assert isinstance(final.event, CodeExec)
    assert not any(isinstance(s.event, ResumeExec) for s in steps)

    print(f"  final event: {type(final.event).__name__}")


def test_max_depth_zero():
    """At max_depth=0, root is already the leaf — no delegation."""
    llm = FakeLLM()
    agent = RLM(llm_client=llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=0))

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

    agent = RLM(llm_client=ScriptedLLM(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1))

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

    original_chat = strong.chat
    def patched_chat(messages, *args, **kwargs):
        strong.call_count += 1
        if strong.call_count == 1:
            return reply
        return original_chat(messages, *args, **kwargs)
    strong.chat = patched_chat
    strong.call_count = 0

    agent = RLM(
        llm_client=strong,
        runtime=LocalRuntime(),
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

    agent = RLM(
        llm_client=ScriptedLLM(),
        runtime=LocalRuntime(),
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

    agent = RLM(llm_client=ScriptedLLM(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1))

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    assert final.result == "fixed"
    exec_events = [s for s in steps if isinstance(s.event, CodeExec)]
    orphan_output = exec_events[0].event.output
    assert "OrphanedDelegatesError" in orphan_output
    assert "wait" in orphan_output
    assert len(final.children) == 0
    print(f"  error surfaced: {orphan_output.splitlines()[-1][:80]}")


def test_variable_persistence_no_yield():
    """Variables set in one code block persist in the next (no yield path)."""
    replies = [
        '```repl\nx = 42\nprint("set x")\n```',
        '```repl\nprint(f"x is {x}")\ndone(str(x))\n```',
    ]

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            return replies[min(self.call_count - 1, len(replies) - 1)]

    agent = RLM(llm_client=ScriptedLLM(), runtime=LocalRuntime(), config=RLMConfig(max_depth=0))

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    assert final.result == "42"
    print(f"  result: {final.result}")


def test_variable_persistence_with_yield():
    """Variables set before yield persist after resume."""
    root_reply = '''```repl
x = 100
h = delegate("worker", "compute")
[result] = yield wait(h)
done(f"{x}+{result}")
```'''

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return root_reply
            return '```repl\ndone("7")\n```'

    agent = RLM(llm_client=ScriptedLLM(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1))

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    assert final.result == "100+7"
    print(f"  result: {final.result}")


def test_duplicate_name_creates_new_child():
    """Delegating the same name twice creates two separate children."""
    root_reply = '''```repl
h = delegate("worker", "round 1")
[r1] = yield wait(h)

h = delegate("worker", "round 2")
[r2] = yield wait(h)

done(r1 + "|" + r2)
```'''

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return root_reply
            return '```repl\ndone("result-" + AGENT_ID)\n```'

    agent = RLM(llm_client=ScriptedLLM(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1))

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    child_ids = sorted(c.agent_id for c in final.children)
    assert child_ids == ["root.worker", "root.worker_2"], f"got {child_ids}"
    assert "result-root.worker" in final.result
    assert "result-root.worker_2" in final.result
    assert "|" in final.result
    print(f"  result: {final.result}")
    print(f"  children: {child_ids}")


def test_parallel_leaves_stepped_together():
    """Multiple leaves at the same depth should all run in one step."""
    root_reply = '''```repl
h1 = delegate("a", "task A")
h2 = delegate("b", "task B")
h3 = delegate("c", "task C")
results = yield wait(h1, h2, h3)
done(",".join(results))
```'''

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return root_reply
            return '```repl\ndone("leaf-" + AGENT_ID)\n```'

    agent = RLM(
        llm_client=ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1),
    )

    steps, final = run_to_completion(agent, agent.start("test"))
    assert final.finished
    assert "leaf-root.a" in final.result
    assert "leaf-root.b" in final.result
    assert "leaf-root.c" in final.result
    print(f"  result: {final.result}")


def test_second_delegation_after_resume():
    """Agent resumes, delegates again, re-suspends — new children picked up next step."""
    root_reply = '''```repl
h1 = delegate("batch1", "first")
[r1] = yield wait(h1)

h2 = delegate("batch2", "second")
[r2] = yield wait(h2)

done(r1 + "|" + r2)
```'''

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return root_reply
            return '```repl\ndone("result-" + AGENT_ID)\n```'

    agent = RLM(
        llm_client=ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1),
    )

    steps, final = run_to_completion(agent, agent.start("test"))

    assert final.finished
    assert len(final.children) == 2
    child_ids = sorted(c.agent_id for c in final.children)
    assert child_ids == ["root.batch1", "root.batch2"]
    assert "result-root.batch1" in final.result
    assert "result-root.batch2" in final.result
    print(f"  result: {final.result}")


def test_index_runtime_tree_and_rebuild_tree():
    """index_runtime_tree + rebuild_tree round-trip preserves the structure."""
    llm = FakeLLM()
    agent = RLM(llm_client=llm, runtime=LocalRuntime(), config=RLMConfig(max_depth=1))

    node = agent.start("test")
    node = agent.step_llm(node)
    node = agent.step_exec(node)
    assert node.status == Status.SUPERVISING

    snapshots, engines = {}, {}
    agent.index_runtime_tree(node, snapshots, engines)
    assert "root.child" in snapshots
    assert "root.child" in engines

    rebuilt = agent.rebuild_tree(node, snapshots)
    assert rebuilt.children[0].agent_id == node.children[0].agent_id
    assert rebuilt.children[0].status == node.children[0].status
    print(f"  indexed: {list(snapshots.keys())}")


def test_dynamic_child_creation():
    """After resume spawns a new child, next step picks it up."""
    root_reply = '''```repl
h = delegate("worker", "first")
[r1] = yield wait(h)
h2 = delegate("worker2", "second")
[r2] = yield wait(h2)
done(r1 + "|" + r2)
```'''

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return root_reply
            return '```repl\ndone("leaf-" + AGENT_ID)\n```'

    agent = RLM(
        llm_client=ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1),
    )

    steps, final = run_to_completion(agent, agent.start("test"))
    assert final.finished
    assert len(final.children) == 2
    child_ids = sorted(c.agent_id for c in final.children)
    assert child_ids == ["root.worker", "root.worker2"]
    assert "leaf-root.worker" in final.result
    assert "leaf-root.worker2" in final.result
    print(f"  result: {final.result}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  PASSED\n")
    print("All tests passed.")
