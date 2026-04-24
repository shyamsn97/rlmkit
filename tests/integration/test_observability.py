"""Integration test: a parent-plus-child run exposes every observability hook.

Checks that after a run we can read:
- ``state.tree()`` containing every agent in the tree
- ``state.tree_usage()`` summing child usage into the root
- a typed ``StepEvent`` on every step (LLMReply | CodeExec | ResumeExec | NoCodeBlock)
- ``save_trace`` → ``load_trace`` round-trip preserving the tree shape
"""

from __future__ import annotations

import json
from pathlib import Path

from rlmkit import (
    RLM,
    CodeExec,
    LLMClient,
    LLMReply,
    LLMUsage,
    NoCodeBlock,
    ResumeExec,
    RLMConfig,
    RLMState,
    StepEvent,
)
from rlmkit.runtime.local import LocalRuntime
from rlmkit.utils.viewer import load_trace, save_trace


class DelegatingLLM(LLMClient):
    """Root delegates once; child finishes immediately."""

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
        for m in messages:
            if "do the thing" in (m.get("content") or ""):
                return self.CHILD
        return self.ROOT


def _run_to_completion(agent: RLM, query: str) -> list[RLMState]:
    state = agent.start(query)
    states = [state]
    while not state.finished:
        state = agent.step(state)
        states.append(state)
        assert len(states) < 50, "agent failed to finish"
    return states


def test_tree_render_contains_every_agent():
    agent = RLM(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    states = _run_to_completion(agent, "obs-tree")
    rendered = states[-1].tree(color=False)

    assert "root" in rendered
    assert "root.child" in rendered
    assert "[finished]" in rendered


def test_tree_usage_sums_children_into_root():
    agent = RLM(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    final = _run_to_completion(agent, "obs-usage")[-1]

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


def test_every_step_has_a_typed_event():
    agent = RLM(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    states = _run_to_completion(agent, "obs-events")

    allowed = (LLMReply, CodeExec, ResumeExec, NoCodeBlock)
    events_seen = set()
    for s in states[1:]:
        assert s.event is not None
        assert isinstance(s.event, StepEvent)
        assert isinstance(s.event, allowed)
        events_seen.add(type(s.event).__name__)

    # READY+EXECUTING fold into one step, so LLMReply is immediately replaced
    # by CodeExec. ResumeExec shows up once the root's children finish.
    assert "CodeExec" in events_seen
    assert "ResumeExec" in events_seen


def test_trace_save_and_load_round_trip(tmp_path: Path):
    agent = RLM(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    states = _run_to_completion(agent, "obs-trace")

    out_dir = tmp_path / "trace"
    save_trace(states, out_dir, query="obs-trace", metadata={"kind": "test"})

    loaded_states, loaded_query, loaded_meta = load_trace(out_dir)
    assert loaded_query == "obs-trace"
    assert loaded_meta == {"kind": "test"}
    assert len(loaded_states) == len(states)

    first = loaded_states[0]
    last = loaded_states[-1]
    assert first["agent_id"] == "root"
    assert last["status"] == "finished"
    assert any(c["agent_id"] == "root.child" for c in last["children"])


def test_state_json_roundtrip_preserves_events(tmp_path: Path):
    agent = RLM(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    final = _run_to_completion(agent, "obs-json")[-1]

    dumped = final.model_dump_json()
    restored = RLMState.model_validate_json(dumped)

    assert restored.tree(color=False) == final.tree(color=False)
    assert restored.tree_usage() == final.tree_usage()

    payload = json.loads(dumped)
    assert payload["children"][0]["agent_id"] == "root.child"
