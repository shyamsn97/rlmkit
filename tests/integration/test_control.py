"""Integration tests for step-level control over typed nodes."""

from __future__ import annotations

from rlmflow import LLMClient, LLMUsage, RLMConfig, RLMFlow, ResultNode, SupervisingNode
from rlmflow.node import Node, parse_node_json
from rlmflow.runtime.local import LocalRuntime


class ScriptedLLM(LLMClient):
    def __init__(self, scripts: dict[str, list[str]]) -> None:
        self.scripts = {key: list(values) for key, values in scripts.items()}
        self.cursors = {key: 0 for key in scripts}

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=4, output_tokens=4)
        agent_id = self._agent_id(messages)
        script = self.scripts.get(agent_id)
        if not script:
            raise AssertionError(f"no script for {agent_id!r}")
        index = self.cursors[agent_id]
        if index >= len(script):
            raise AssertionError(f"script exhausted for {agent_id!r}")
        self.cursors[agent_id] = index + 1
        return script[index]

    @staticmethod
    def _agent_id(messages: list[dict]) -> str:
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        for line in system.splitlines():
            if line.startswith("AGENT_ID"):
                return line.split("=", 1)[1].strip()
        return "root"


def _done(text: str) -> str:
    return f"```repl\ndone({text!r})\n```"


def _delegate_one(child_name: str, query: str) -> str:
    return (
        "```repl\n"
        f"h = delegate({child_name!r}, {query!r})\n"
        "results = yield wait(h)\n"
        "done(results[0])\n"
        "```"
    )


def _run(agent: RLMFlow, node: Node) -> Node:
    while not node.finished:
        node = agent.step(node)
    return node


def test_step_loop_reaches_result_node():
    agent = RLMFlow(
        llm_client=ScriptedLLM({"root": [_done("direct-answer")]}),
        runtime=LocalRuntime(),
    )

    final = _run(agent, agent.start("control-step"))

    assert isinstance(final, ResultNode)
    assert final.result == "direct-answer"


def test_checkpoint_round_trip_resumes_cleanly():
    agent = RLMFlow(
        llm_client=ScriptedLLM({"root": [_done("checkpoint-me")]}),
        runtime=LocalRuntime(),
    )

    final = _run(agent, agent.start("control-ckpt"))
    restored = parse_node_json(final.model_dump_json())

    assert isinstance(restored, ResultNode)
    assert restored.result == final.result
    assert restored.tree_usage() == final.tree_usage()


def test_fork_from_midrun_node_diverges():
    agent_a = RLMFlow(
        llm_client=ScriptedLLM({"root": [_done("path-A")]}),
        runtime=LocalRuntime(),
    )
    agent_b = RLMFlow(
        llm_client=ScriptedLLM({"root": [_done("path-B")]}),
        runtime=LocalRuntime(),
    )
    mid = agent_a.start("control-fork")

    final_a = _run(agent_a, mid)
    final_b = _run(agent_b, mid)

    assert final_a.result == "path-A"
    assert final_b.result == "path-B"


def test_rewind_is_just_list_indexing():
    agent = RLMFlow(
        llm_client=ScriptedLLM({"root": [_done("rewind-me")]}),
        runtime=LocalRuntime(),
    )

    node = agent.start("control-rewind")
    states = [node]
    while not node.finished:
        node = agent.step(node)
        states.append(node)

    assert states[0].type == "query"
    assert states[-1].type == "result"
    assert states[0] is not states[-1]


def test_intervene_replaces_child_result_before_resume():
    agent = RLMFlow(
        llm_client=ScriptedLLM(
            {
                "root": [_delegate_one("child", "delegate-done")],
                "root.child": [_done("(will not finish)")],
            }
        ),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    node = agent.step(agent.start("control-intervene"))
    assert isinstance(node, SupervisingNode)
    assert node.waiting_on == ["root.child"]

    killed_child = node.children[0].successor(ResultNode, result="killed-by-supervisor")
    patched = node.update(children=[killed_child])
    final = _run(agent, patched)

    assert isinstance(final, ResultNode)
    assert final.result == "killed-by-supervisor"
