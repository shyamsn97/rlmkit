"""Integration tests for typed-node observability hooks."""

from __future__ import annotations

import json
from pathlib import Path

from rlmflow import LLMClient, LLMUsage, Node, RLMConfig, RLMFlow, ResultNode
from rlmflow.node import Node as BaseNode
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.trace import Trace, load_trace, save_trace


class DelegatingLLM(LLMClient):
    ROOT = (
        "```repl\n"
        "h = delegate('child', 'do the thing')\n"
        "results = yield wait(h)\n"
        "done(results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        for message in messages:
            if "do the thing" in (message.get("content") or ""):
                return self.CHILD
        return self.ROOT


def _run_to_completion(agent: RLMFlow, query: str) -> list[Node]:
    node = agent.start(query)
    states = [node]
    while not node.finished:
        node = agent.step(node)
        states.append(node)
        assert len(states) < 50
    return states


def _agent() -> RLMFlow:
    return RLMFlow(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )


def test_tree_render_contains_every_agent():
    rendered = _run_to_completion(_agent(), "obs-tree")[-1].tree(color=False)

    assert "root" in rendered
    assert "root.child" in rendered
    assert "[result]" in rendered


def test_tree_usage_sums_children_into_root():
    final = _run_to_completion(_agent(), "obs-usage")[-1]

    assert len(final.children) == 1
    child = final.children[0]
    root_in, root_out = final.total_input_tokens, final.total_output_tokens
    child_in, child_out = child.tree_usage()
    tree_in, tree_out = final.tree_usage()

    assert tree_in == root_in + child_in
    assert tree_out == root_out + child_out
    assert child_in > 0 and child_out > 0
    assert final.tree_tokens == tree_in + tree_out
    assert final.tree_tokens >= final.total_tokens


def test_steps_are_typed_nodes():
    states = _run_to_completion(_agent(), "obs-nodes")

    assert {state.type for state in states} >= {"query", "supervising", "result"}
    assert all(isinstance(state, BaseNode) for state in states)


def test_trace_save_and_load_round_trip(tmp_path: Path):
    states = _run_to_completion(_agent(), "obs-trace")
    out_dir = tmp_path / "trace"

    save_trace(states, out_dir, metadata={"kind": "test"})
    loaded = load_trace(out_dir)

    assert isinstance(loaded, Trace)
    assert loaded.metadata == {"kind": "test"}
    assert len(loaded.states) == len(states)
    assert loaded.states[0].agent_id == "root"
    assert loaded.states[0].query == "obs-trace"
    assert isinstance(loaded.states[-1], ResultNode)
    assert any(child.agent_id == "root.child" for child in loaded.states[-1].children)
    assert loaded.states[-1].tree(color=False) == states[-1].tree(color=False)


def test_state_save_load_round_trip(tmp_path: Path):
    final = _run_to_completion(_agent(), "obs-save")[-1]
    ckpt = tmp_path / "state.json"

    final.save(ckpt)
    restored = Node.load(ckpt)

    assert restored.tree(color=False) == final.tree(color=False)
    assert restored.tree_usage() == final.tree_usage()
    payload = json.loads(ckpt.read_text())
    assert payload["children"][0]["agent_id"] == "root.child"
