from __future__ import annotations

from rlmflow import QueryNode, ResultNode, RLMConfig, RLMFlow, SupervisingNode
from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.prompts.messages import FINAL_ANSWER_ACTION
from rlmflow.runtime.local import LocalRuntime


class StallingThenFinalLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.last_messages: list[dict] = []
        self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_messages = list(messages)
        if any("full iteration budget" in message.get("content", "") for message in messages):
            return '```repl\ndone("final answer")\n```'
        return "```repl\nx = 1\n```"


def run_to_completion(agent: RLMFlow):
    node = agent.start("answer the question")
    states = [node]
    while not node.finished:
        node = agent.step(node)
        states.append(node)
    return node, states


def test_exhaustion_marks_terminate_requested_and_runs_one_more_repl_turn():
    llm = StallingThenFinalLLM()
    agent = RLMFlow(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=1, max_depth=0),
    )

    node, _states = run_to_completion(agent)

    assert isinstance(node, ResultNode)
    assert node.result == "final answer"
    assert node.terminate_requested
    assert llm.calls == 2
    assert any(
        message.get("content") == FINAL_ANSWER_ACTION
        for message in llm.last_messages
        if message.get("role") == "user"
    )


def test_final_message_is_last_user_instruction_on_recovery_turn():
    llm = StallingThenFinalLLM()
    agent = RLMFlow(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=1, max_depth=0),
    )

    run_to_completion(agent)

    user_messages = [
        message["content"] for message in llm.last_messages if message.get("role") == "user"
    ]
    assert user_messages[-1] == FINAL_ANSWER_ACTION


def test_explicit_terminate_on_query_node_drives_one_final_turn():
    llm = StallingThenFinalLLM()
    agent = RLMFlow(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=10, max_depth=0),
    )

    node = agent.terminate(agent.start("answer the question"))

    assert node.terminate_requested
    assert llm.calls == 0

    while not node.finished:
        node = agent.step(node)

    assert node.result == "final answer"
    assert node.terminate_requested
    assert llm.calls == 1


def test_terminate_is_recursive_over_unfinished_children():
    agent = RLMFlow(
        llm_client=StallingThenFinalLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=10, max_depth=2),
    )
    finished_child = ResultNode(agent_id="root.search_0", depth=1, result="early result")
    pending_child = QueryNode(agent_id="root.search_1", depth=1)
    root = SupervisingNode(
        agent_id="root",
        waiting_on=["root.search_0", "root.search_1"],
        children=[finished_child, pending_child],
    )

    out = agent.terminate(root)

    assert out.terminate_requested
    assert not out.children[0].terminate_requested
    assert out.children[1].terminate_requested
    assert isinstance(out.children[0], ResultNode)
    assert isinstance(out.children[1], QueryNode)
