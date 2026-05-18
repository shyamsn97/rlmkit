"""Release-contract tests for the public RLMFlow surface."""

from __future__ import annotations

from rlmflow import (
    Graph,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMFlow,
    is_done,
)
from rlmflow.runtime.local import LocalRuntime


class DoneLLM(LLMClient):
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
    ROOT_REPLY = (
        "```repl\n"
        'h = delegate("child", "do the thing", "")\n'
        "results = yield wait(h)\n"
        "done(results[0])\n"
        "```"
    )
    CHILD_REPLY = '```repl\ndone("child-answer")\n```'

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
        for message in messages:
            if "do the thing" in (message.get("content") or ""):
                return self.CHILD_REPLY
        return self.ROOT_REPLY


class BudgetBusterLLM(LLMClient):
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


def _run(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


# ── chat-style API ────────────────────────────────────────────────


def test_chat_smoke_returns_string_and_sets_last_usage():
    llm = DoneLLM(text="hi there", input_tokens=12, output_tokens=4)
    agent = RLMFlow(llm_client=llm, runtime=LocalRuntime())

    result = agent.chat([{"role": "user", "content": "say hi"}])

    assert result == "hi there"
    assert agent.last_usage is not None
    assert agent.last_usage.input_tokens == 12
    assert agent.last_usage.output_tokens == 4


# ── budget enforcement ────────────────────────────────────────────


def test_max_budget_stops_the_loop():
    llm = BudgetBusterLLM(input_tokens=50, output_tokens=50)
    agent = RLMFlow(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_budget=150, max_iterations=100),
    )

    result = agent.run("loop forever")

    assert "budget exceeded" in result
    assert llm.calls <= 3


# ── token aggregation ─────────────────────────────────────────────


def test_graph_tokens_sum_parent_and_child_usage():
    agent = RLMFlow(
        llm_client=DelegatingLLM(input_tokens=10, output_tokens=5),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    final = _run(agent, agent.start("root task"))

    assert is_done(final.current())
    root_in, root_out = final.tokens(recursive=False)
    child_in, child_out = final["root.child"].tokens(recursive=False)
    tree_in, tree_out = final.tokens()

    assert tree_in == root_in + child_in
    assert tree_out == root_out + child_out
    assert child_in > 0 and child_out > 0


# ── graph save / load round-trip ──────────────────────────────────


def test_graph_save_load_round_trip(tmp_path):
    agent = RLMFlow(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    final = _run(agent, agent.start("checkpoint me"))

    ckpt = tmp_path / "graph.json"
    final.save(ckpt)
    restored = Graph.load(ckpt)

    assert restored.result() == final.result()
    assert list(restored.agents) == list(final.agents)
    assert restored.tree() == final.tree()
    assert restored.tokens() == final.tokens()


# ── tree rendering shows model labels ─────────────────────────────


def test_tree_displays_model_label_per_agent():
    class ModelAwareLLM(LLMClient):
        model = "gpt-strong"

        def chat(self, messages, *args, **kwargs):
            return (
                "```repl\n"
                'h = delegate("fast_worker", "use fast", "", model="fast")\n'
                "r = yield wait(h)\n"
                "done(r[0])\n"
                "```"
            )

    class FastLLM(LLMClient):
        model = "fast-mini"

        def chat(self, messages, *args, **kwargs):
            return '```repl\ndone("fast-result")\n```'

    agent = RLMFlow(
        llm_client=ModelAwareLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1),
        llm_clients={"fast": {"model": FastLLM(), "description": "tiny"}},
    )
    final = _run(agent, agent.start("test"))

    tree = final.tree()
    assert "root (default)" in tree
    assert "root.fast_worker (fast)" in tree
