"""Release-contract tests for the current RLMFlow public surface."""

from __future__ import annotations

from rlmflow import LLMClient, LLMUsage, RLMConfig, RLMFlow, ResultNode, SupervisingNode
from rlmflow.node import ActionNode, QueryNode, parse_node_json
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


def _run(agent: RLMFlow, node):
    while not node.finished:
        node = agent.step(node)
    return node


def test_chat_smoke_returns_string_and_sets_last_usage():
    llm = DoneLLM(text="hi there", input_tokens=12, output_tokens=4)
    agent = RLMFlow(llm_client=llm, runtime=LocalRuntime())

    result = agent.chat([{"role": "user", "content": "say hi"}])

    assert result == "hi there"
    assert agent.last_usage is not None
    assert agent.last_usage.input_tokens == 12
    assert agent.last_usage.output_tokens == 4


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


def test_token_propagation_parent_sees_child_tokens():
    agent = RLMFlow(
        llm_client=DelegatingLLM(input_tokens=10, output_tokens=5),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    node = agent.step(agent.start("root task"))
    assert isinstance(node, SupervisingNode)
    node = agent.step(node)
    final = agent.step(node)

    assert isinstance(final, ResultNode)
    assert len(final.children) == 1
    child = final.children[0]
    tree_in, tree_out = final.tree_usage()
    child_in, child_out = child.tree_usage()

    assert tree_in == final.total_input_tokens + child_in
    assert tree_out == final.total_output_tokens + child_out
    assert child_in > 0
    assert child_out > 0


def test_checkpoint_round_trip_preserves_tree():
    agent = RLMFlow(
        llm_client=DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    final = _run(agent, agent.start("checkpoint me"))
    restored = parse_node_json(final.model_dump_json())

    assert isinstance(restored, ResultNode)
    assert restored.result == final.result
    assert len(restored.children) == len(final.children)
    assert restored.tree_usage() == final.tree_usage()
    assert restored.tree(color=False) == final.tree(color=False)


def test_node_helpers_walk_find_replace_and_edit_immutably():
    leaf = QueryNode(agent_id="root.plan.search", content="old")
    plan = QueryNode(agent_id="root.plan", children=[leaf])
    verify = QueryNode(agent_id="root.verify")
    root = QueryNode(agent_id="root", children=[plan, verify])

    assert root.find("root.plan.search") is leaf
    assert [node.agent_id for node in root.walk()] == [
        "root",
        "root.plan",
        "root.plan.search",
        "root.verify",
    ]

    better_leaf = leaf.update(content="better")
    edited = root.replace_many({leaf.id: better_leaf})

    assert root.find("root.plan.search").content == "old"
    assert edited.find("root.plan.search").content == "better"


def test_tree_displays_action_model_label():
    child = ActionNode(
        agent_id="root.fast_worker",
        config={"model": "fast"},
        model="gpt-5-mini",
        code="done('ok')",
    )
    root = QueryNode(config={"model": "default"}, children=[child])

    tree = root.tree()

    assert "root [query] {default}" in tree
    assert "root.fast_worker [action] {fast:gpt-5-mini}" in tree
