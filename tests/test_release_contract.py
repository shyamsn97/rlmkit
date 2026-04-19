"""Release-contract tests: chat smoke, budget, token propagation, checkpoint round-trip.

These exercise the pieces users interact with most directly:
- `RLM.chat(messages)` — the drop-in-LLM surface
- `RLMConfig.max_budget` — the budget cap
- `state.tree_usage()` — token propagation across the tree
- `RLMState.model_dump_json` / `model_validate_json` — checkpoint round-trip
"""

from __future__ import annotations

from rlmkit import (
    RLM,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMState,
    Status,
)
from rlmkit.runtime.local import LocalRuntime

# ── Scripted LLMs ────────────────────────────────────────────────────


class DoneLLM(LLMClient):
    """Emits a single ```repl``` block that calls done(text). One-shot."""

    def __init__(self, text: str = "hello", input_tokens: int = 10, output_tokens: int = 5):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_usage = LLMUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        return f'```repl\ndone({self.text!r})\n```'


class DelegatingLLM(LLMClient):
    """Root delegates to one child, then returns the child's result.

    Children call done() directly. Used to verify token propagation across
    a parent/child boundary.
    """

    ROOT_REPLY = '''```repl
h = delegate("child", "do the thing")
results = yield wait(h)
done(results[0])
```'''

    CHILD_REPLY = '''```repl
done("child-answer")
```'''

    def __init__(self, input_tokens: int = 7, output_tokens: int = 3):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_usage = LLMUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        is_root = any("MAX_DEPTH" in (m.get("content") or "") and "DEPTH=0" in (m.get("content") or "")
                      for m in messages)
        # Simple heuristic: if the latest user/system prompt mentions "delegate" in the query, act as root
        for m in messages:
            if m.get("role") == "user" and "do the thing" in (m.get("content") or ""):
                return self.CHILD_REPLY
        return self.ROOT_REPLY


class BudgetBusterLLM(LLMClient):
    """Every call burns a lot of tokens but never calls done(). Used for budget tests."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 100):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_usage = LLMUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        return '```repl\nprint("still going")\n```'


# ── Tests ────────────────────────────────────────────────────────────


def test_chat_smoke_returns_string_and_sets_last_usage():
    """RLM.chat() must return a string and populate last_usage."""
    llm = DoneLLM(text="hi there", input_tokens=12, output_tokens=4)
    agent = RLM(llm_client=llm, runtime=LocalRuntime())

    result = agent.chat([{"role": "user", "content": "say hi"}])

    assert isinstance(result, str)
    assert result == "hi there"
    assert agent.last_usage is not None
    assert agent.last_usage.input_tokens == 12
    assert agent.last_usage.output_tokens == 4


def test_max_budget_stops_the_loop():
    """When tree_usage exceeds max_budget, the agent finishes with a budget message."""
    llm = BudgetBusterLLM(input_tokens=50, output_tokens=50)
    agent = RLM(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_budget=150, max_iterations=100),
    )

    result = agent.run("loop forever")

    assert "budget exceeded" in result
    # At most 2 LLM calls (after the 2nd, tree_usage = 200 >= 150 → finish on next loop entry).
    assert llm.calls <= 3


def test_token_propagation_parent_sees_child_tokens():
    """A parent's tree_usage() should sum its own tokens plus all descendants."""
    llm = DelegatingLLM(input_tokens=10, output_tokens=5)
    agent = RLM(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    state = agent.start("root task")
    steps = 0
    while not state.finished:
        state = agent.step(state)
        steps += 1
        assert steps < 100, "agent failed to finish"

    # Root burned some tokens; child burned some tokens; tree_usage is the sum.
    assert len(state.children) >= 1
    child = state.children[0]
    assert child.total_input_tokens > 0
    assert child.total_output_tokens > 0

    tree_in, tree_out = state.tree_usage()
    root_in = state.total_input_tokens
    root_out = state.total_output_tokens
    child_in, child_out = child.tree_usage()

    assert tree_in == root_in + child_in
    assert tree_out == root_out + child_out
    assert tree_in > root_in  # child contributed


def test_checkpoint_round_trip_preserves_tree():
    """state.model_dump_json → RLMState.model_validate_json round-trips cleanly."""
    llm = DelegatingLLM()
    agent = RLM(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    state = agent.start("checkpoint me")
    while not state.finished:
        state = agent.step(state)

    serialized = state.model_dump_json()
    restored = RLMState.model_validate_json(serialized)

    assert restored.agent_id == state.agent_id
    assert restored.query == state.query
    assert restored.status == state.status == Status.FINISHED
    assert restored.result == state.result
    assert restored.total_input_tokens == state.total_input_tokens
    assert restored.total_output_tokens == state.total_output_tokens
    assert len(restored.children) == len(state.children)
    assert restored.tree_usage() == state.tree_usage()
    assert restored.messages == state.messages
