from __future__ import annotations

from rlmkit.llm import LLMClient, LLMUsage
from rlmkit.prompts.messages import FINAL_ANSWER_ACTION
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime


class StallingThenFinalLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.last_messages: list[dict] = []
        self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_messages = list(messages)
        if any("full iteration budget" in m.get("content", "") for m in messages):
            return '```repl\ndone("final answer")\n```'
        return "```repl\nx = 1\n```"


def run_to_completion(agent: RLM):
    state = agent.start("answer the question")
    states = [state]
    while not state.finished:
        state = agent.step(state)
        states.append(state)
    return state, states


def test_exhaustion_marks_terminate_requested_and_runs_one_more_repl_turn():
    llm = StallingThenFinalLLM()
    agent = RLM(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=1, max_depth=0),
    )

    state, states = run_to_completion(agent)

    assert state.result == "final answer"
    assert llm.calls == 2
    assert state.terminate_requested
    # Final-action prompt is transient — never stored in node.messages.
    assert not any(
        m.get("content") == FINAL_ANSWER_ACTION for m in state.messages
    )
    # …but it WAS injected into the last LLM input.
    assert any(
        "full iteration budget" in m.get("content", "")
        for m in llm.last_messages
        if m.get("role") == "user"
    )
    # The exhausted READY node was marked terminate_requested and got a
    # one-iteration bump so the next READY turn can run.
    assert states[-2].terminate_requested
    assert states[-2].config["max_iterations"] == 2


def test_final_message_is_last_user_instruction_on_recovery_turn():
    llm = StallingThenFinalLLM()
    agent = RLM(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=1, max_depth=0),
    )

    run_to_completion(agent)

    user_messages = [m["content"] for m in llm.last_messages if m.get("role") == "user"]
    assert "full iteration budget" in user_messages[-1]
    assert "Continue working on:" not in user_messages[-1]


def test_explicit_terminate_on_ready_node_drives_one_final_turn():
    llm = StallingThenFinalLLM()
    agent = RLM(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=10, max_depth=0),
    )

    state = agent.start("answer the question")
    state = agent.terminate(state)

    assert state.terminate_requested
    assert llm.calls == 0  # terminate is pure state edit

    while not state.finished:
        state = agent.step(state)

    assert state.result == "final answer"
    assert state.terminate_requested
    assert llm.calls == 1  # exactly one extra turn
    assert any(
        "full iteration budget" in m.get("content", "")
        for m in llm.last_messages
        if m.get("role") == "user"
    )


def test_terminate_is_recursive_over_unfinished_children():
    llm = StallingThenFinalLLM()
    agent = RLM(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=10, max_depth=2),
    )

    # Hand-built tree: root SUPERVISING with two children, one finished.
    from rlmkit.node import RLMNode, Status

    finished_child = RLMNode(
        agent_id="root.search_0",
        status=Status.FINISHED,
        result="early result",
    )
    pending_child = RLMNode(
        agent_id="root.search_1",
        status=Status.READY,
    )
    root = RLMNode(
        agent_id="root",
        status=Status.SUPERVISING,
        children=[finished_child, pending_child],
        waiting_on=["root.search_0", "root.search_1"],
    )

    out = agent.terminate(root)

    assert out.terminate_requested
    assert out.children[0].status == Status.FINISHED  # untouched
    assert not out.children[0].terminate_requested
    assert out.children[1].terminate_requested  # propagated
    assert out.children[1].status == Status.READY  # status preserved
