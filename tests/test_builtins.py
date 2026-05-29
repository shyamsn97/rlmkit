"""Tests for REPL builtins exposed to agent code."""

from __future__ import annotations

from inspect import Parameter, signature

import pytest

from rlmflow import LLMClient, RLMConfig, RLMFlow
from rlmflow.graph.handles import ChildHandle
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import get_repl_tools, tool
from rlmflow.tools.builtins import make_delegate


def test_rlm_delegate_is_keyword_only():
    spawned: list[tuple[str, str, str]] = []

    def spawn_child(parent_agent_id, parent_node_id, name, query, context, **kwargs):
        spawned.append((name, query, context))
        return ChildHandle(f"{parent_agent_id}.{name}")

    delegate = make_delegate(
        spawn_child,
        {"AGENT_ID": "root", "PARENT_NODE_ID": "node-1"},
    )

    params = signature(delegate).parameters
    assert params["name"].kind is Parameter.KEYWORD_ONLY
    assert params["query"].kind is Parameter.KEYWORD_ONLY
    assert params["context"].kind is Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        delegate("child", "task", "")

    handle = delegate(name="child", query="task", context="payload")
    assert handle.agent_id == "root.child"
    assert spawned == [("child", "task", "payload")]


class _EchoLLM(LLMClient):
    def chat(self, messages, *args, **kwargs) -> str:
        del args, kwargs
        return messages[-1]["content"].upper()


def test_llm_query_batched_validates_list_shape(tmp_path):
    agent = RLMFlow(
        _EchoLLM(),
        runtime=LocalRuntime(workspace=tmp_path / "workspace"),
        config=RLMConfig(max_concurrency=1),
    )

    params = signature(agent.llm_query_batched).parameters
    assert params["model"].kind is Parameter.KEYWORD_ONLY

    assert agent.llm_query_batched(["a", "b"]) == ["A", "B"]

    assert agent.llm_query_batched([]) == []

    with pytest.raises(TypeError):
        agent.llm_query_batched("not a list")
    with pytest.raises(TypeError):
        agent.llm_query_batched(["ok", 3])


def test_get_repl_tools_lets_local_tool_call_visible_tool(tmp_path):
    @tool("Return a greeting.")
    def greet(name: str) -> str:
        return f"hello {name}"

    @tool("Call another visible tool.")
    def call_greet(name: str) -> str:
        return get_repl_tools()["greet"](name)

    runtime = LocalRuntime(workspace=tmp_path / "workspace")
    runtime.register_tool(greet)
    runtime.register_tool(call_greet)

    assert runtime.execute("print(call_greet('rlm'))") == "hello rlm"


def test_get_repl_tools_hides_internal_primitives_by_default(tmp_path):
    runtime = LocalRuntime(workspace=tmp_path / "workspace")
    RLMFlow(_EchoLLM(), runtime=runtime, config=RLMConfig(max_depth=1))

    visible = runtime.execute(
        "from rlmflow.tools import get_repl_tools\n"
        "tools = get_repl_tools()\n"
        "print('rlm_delegate' in tools, 'rlm_wait' in tools, 'done' in tools)"
    )
    assert visible == "False False True"

    hidden = runtime.execute(
        "from rlmflow.tools import get_repl_tools\n"
        "tools = get_repl_tools(include_hidden=True)\n"
        "print('rlm_delegate' in tools, 'rlm_wait' in tools)"
    )
    assert hidden == "True True"


def test_get_repl_tools_requires_active_context():
    with pytest.raises(RuntimeError, match="No active RLMFlow tool context"):
        get_repl_tools()
