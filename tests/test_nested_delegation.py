"""Nested delegation behavior for the typed RLMFlow graph."""

from __future__ import annotations

from rlmflow import LLMClient, RLMConfig, RLMFlow, ResultNode, SupervisingNode
from rlmflow.runtime.local import LocalRuntime


class RecursiveLLM(LLMClient):
    def __init__(self, *, max_child_depth: int) -> None:
        self.max_child_depth = max_child_depth
        self.calls: list[str] = []

    def chat(self, messages, *args, **kwargs):
        depth, max_depth = self._depth(messages)
        self.calls.append(f"depth:{depth}")
        if depth < max_depth and depth < self.max_child_depth:
            return (
                "```repl\n"
                'h = delegate("child", "go deeper")\n'
                "results = yield wait(h)\n"
                'done(AGENT_ID + "->" + results[0])\n'
                "```"
            )
        return '```repl\ndone("leaf:" + AGENT_ID)\n```'

    @staticmethod
    def _depth(messages: list[dict]) -> tuple[int, int]:
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        marker = "You are at recursion depth **"
        if marker not in system:
            return 0, 0
        rest = system.split(marker, 1)[1]
        depth_text, rest = rest.split("**", 1)
        max_text = rest.split("max **", 1)[1].split("**", 1)[0]
        return int(depth_text), int(max_text)


def _run(agent: RLMFlow, node):
    steps = [node]
    while not node.finished:
        node = agent.step(node)
        steps.append(node)
        assert len(steps) < 100
    return steps, node


def test_root_can_delegate_to_child_at_depth_one():
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=1),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1),
    )

    _steps, final = _run(agent, agent.start("test"))

    assert isinstance(final, ResultNode)
    assert final.result == "root->leaf:root.child"
    assert final.children[0].agent_id == "root.child"
    assert final.children[0].depth == 1


def test_nested_delegation_reaches_grandchild_at_depth_two():
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=2),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    _steps, final = _run(agent, agent.start("test"))

    assert isinstance(final, ResultNode)
    child = final.children[0]
    grandchild = child.children[0]
    assert child.agent_id == "root.child"
    assert grandchild.agent_id == "root.child.child"
    assert grandchild.depth == 2
    assert final.result == "root->root.child->leaf:root.child.child"


def test_max_depth_turns_delegate_into_direct_llm_work():
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=3),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=0, max_iterations=3),
    )

    _steps, final = _run(agent, agent.start("test"))

    assert isinstance(final, ResultNode)
    assert final.children == []
    assert final.result == "leaf:root"


def test_each_step_advances_existing_leaf_batch_once():
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=2),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    node = agent.step(agent.start("test"))
    assert isinstance(node, SupervisingNode)
    assert node.children[0].type == "query"

    node = agent.step(node)
    assert isinstance(node, SupervisingNode)
    assert node.children[0].type == "supervising"
    assert node.children[0].children[0].type == "query"

    node = agent.step(node)
    assert isinstance(node.children[0].children[0], ResultNode)


def test_model_routing_is_stored_on_child_nodes():
    class ModelAwareLLM(LLMClient):
        model = "strong-model"

        def chat(self, messages, *args, **kwargs):
            return (
                "```repl\n"
                'h = delegate("worker", "use fast", model="fast")\n'
                "results = yield wait(h)\n"
                "done(results[0])\n"
                "```"
            )

    class FastLLM(LLMClient):
        model = "fast-model"

        def chat(self, messages, *args, **kwargs):
            return '```repl\ndone("fast-result")\n```'

    agent = RLMFlow(
        llm_client=ModelAwareLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1),
        llm_clients={"fast": {"model": FastLLM(), "description": "quick worker"}},
    )

    node = agent.step(agent.start("test"))
    assert isinstance(node, SupervisingNode)
    assert node.children[0].config["model"] == "fast"

    node = agent.step(node)
    final = agent.step(node)

    assert final.result == "fast-result"
    assert final.children[0].model_label == "fast:fast-model"
