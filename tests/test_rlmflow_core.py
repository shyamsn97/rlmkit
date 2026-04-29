from __future__ import annotations

from rlmkit import RLMConfig, RLMFlow, ResultNode, SupervisingNode, Workspace
from rlmkit.llm import LLMClient
from rlmkit.runtime.local import LocalRuntime


class StaticLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, *args, **kwargs) -> str:
        return self.reply


def test_direct_done_returns_result_node():
    agent = RLMFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=2),
    )

    node = agent.step(agent.start("say ok"))

    assert isinstance(node, ResultNode)
    assert node.result == "ok"


def test_delegation_resumes_parent():
    class ScriptedLLM(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, *args, **kwargs) -> str:
            self.calls += 1
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone("child-result")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task")\n'
                "results = yield wait(h)\n"
                'done("parent:" + results[0])\n'
                "```"
            )

    agent = RLMFlow(
        ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )

    node = agent.step(agent.start("parent task"))
    assert isinstance(node, SupervisingNode)

    node = agent.step(node)
    assert isinstance(node, SupervisingNode)
    assert [child.type for child in node.children] == ["result"]

    node = agent.step(node)
    assert isinstance(node, ResultNode)
    assert node.result == "parent:child-result"


def test_workspace_context_persists_nodes(tmp_path):
    ws = Workspace.create(tmp_path / "workspace")
    agent = RLMFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        workspace=ws,
        config=RLMConfig(max_iterations=2),
    )

    node = agent.step(agent.start("say ok", context="hello"))

    assert isinstance(node, ResultNode)
    assert ws.context.read_context("context") == "hello"
    assert len(ws.context.load()) == 3


def test_child_scope_lives_on_node_not_child_flow():
    class ScriptedLLM(LLMClient):
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def chat(self, messages, *args, **kwargs) -> str:
            self.prompts.append(messages[-1]["content"])
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\nprint(AGENT_ID, DEPTH)\ndone("child")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task")\n'
                "results = yield wait(h)\n"
                'done(results[0])\n'
                "```"
            )

    agent = RLMFlow(
        ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )

    node = agent.step(agent.start("parent task"))
    assert isinstance(node, SupervisingNode)
    child = node.children[0]

    assert child.agent_id == "root.child"
    assert child.depth == 1
    assert child.runtime is not None
    assert child.runtime.id != "root"
    assert not hasattr(agent, "child_engines")

    node = agent.step(node)
    assert isinstance(node.children[0], ResultNode)
    assert node.children[0].result == "child"
