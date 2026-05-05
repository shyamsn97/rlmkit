from __future__ import annotations

from rlmflow import (
    ActionNode,
    ErrorNode,
    QueryNode,
    RLMConfig,
    RLMFlow,
    ResultNode,
    SupervisingNode,
    Workspace,
)
from rlmflow.llm import LLMClient
from rlmflow.runtime.local import LocalRuntime


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
                'h = delegate("child", "child task", "")\n'
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


def test_workspace_session_persists_nodes_and_context_payload(tmp_path):
    ws = Workspace.create(tmp_path / "workspace")
    agent = RLMFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        workspace=ws,
        config=RLMConfig(max_iterations=2),
    )

    node = agent.step(agent.start("say ok", context="hello"))

    assert isinstance(node, ResultNode)
    assert ws.context.read("context") == "hello"
    assert len(ws.session.load()) == 3


def test_tree_displays_model_labels():
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


def _sample_tree() -> QueryNode:
    """root supervising 3 children: one result, one error, one nested supervisor."""
    leaf_ok = ResultNode(agent_id="root.search.hit", depth=2, result="found it")
    leaf_err = ErrorNode(agent_id="root.search.miss", depth=2, error="no_code_block")
    nested = SupervisingNode(
        agent_id="root.search",
        depth=1,
        code="...",
        children=[leaf_ok, leaf_err],
    )
    sibling = ResultNode(agent_id="root.verify", depth=1, result="ok")
    return QueryNode(agent_id="root", depth=0, children=[nested, sibling])


def test_leaves_returns_every_node_with_no_children():
    root = _sample_tree()
    leaves = root.leaves()
    assert {n.agent_id for n in leaves} == {
        "root.search.hit",
        "root.search.miss",
        "root.verify",
    }


def test_leaves_on_solo_node_returns_self():
    solo = ResultNode(agent_id="root", result="ok")
    assert solo.leaves() == [solo]


def test_errors_finds_only_error_nodes():
    root = _sample_tree()
    errors = root.errors()
    assert [n.agent_id for n in errors] == ["root.search.miss"]
    assert all(n.type == "error" for n in errors)


def test_results_finds_only_result_nodes():
    root = _sample_tree()
    results = root.results()
    assert {n.agent_id for n in results} == {"root.search.hit", "root.verify"}
    assert all(n.type == "result" for n in results)


def test_where_filters_by_kwargs():
    root = _sample_tree()
    matches = root.where(type="result", depth=1)
    assert [n.agent_id for n in matches] == ["root.verify"]


def test_where_filters_by_predicate():
    root = _sample_tree()
    deep = root.where(lambda n: n.depth >= 2)
    assert {n.agent_id for n in deep} == {"root.search.hit", "root.search.miss"}


def test_where_combines_predicate_and_kwargs():
    root = _sample_tree()
    matches = root.where(lambda n: n.depth >= 1, type="result")
    assert {n.agent_id for n in matches} == {"root.search.hit", "root.verify"}


def test_where_kwargs_skip_nodes_missing_the_attribute():
    root = _sample_tree()
    by_error_kind = root.where(error="no_code_block")
    assert [n.agent_id for n in by_error_kind] == ["root.search.miss"]


def test_path_to_returns_root_to_node_chain():
    root = _sample_tree()
    path = root.path_to("root.search.hit")
    assert [n.agent_id for n in path] == ["root", "root.search", "root.search.hit"]


def test_path_to_returns_empty_when_not_found():
    root = _sample_tree()
    assert root.path_to("nope") == []


def test_diff_finds_added_and_removed_nodes():
    from rlmflow import ErrorNode, QueryNode, ResultNode

    a = QueryNode(
        agent_id="root",
        children=[ResultNode(agent_id="root.x", result="ok")],
    )
    b = QueryNode(
        agent_id="root",
        id=a.id,
        children=[
            *a.child_nodes(),
            ErrorNode(agent_id="root.y", error="boom"),
        ],
    )
    diff = b.diff(a)
    assert [n.agent_id for n in diff.added] == ["root.y"]
    assert diff.removed == []

    inverse = a.diff(b)
    assert [n.agent_id for n in inverse.removed] == ["root.y"]
    assert inverse.added == []


def test_repr_html_wraps_tree_in_pre():
    root = _sample_tree()
    html = root._repr_html_()
    assert "<pre" in html and "</pre>" in html
    assert "root" in html
    assert "&lt;" not in html or "[query]" in html


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
                'h = delegate("child", "child task", "")\n'
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


def test_delegate_can_pass_child_context_payload(tmp_path):
    class ScriptedLLM(LLMClient):
        def chat(self, messages, *args, **kwargs) -> str:
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone(CONTEXT.read())\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "child payload")\n'
                "results = yield wait(h)\n"
                'done(results[0])\n'
                "```"
            )

    workspace = Workspace.create(tmp_path / "workspace")
    agent = RLMFlow(
        ScriptedLLM(),
        workspace=workspace,
        config=RLMConfig(max_depth=1, max_iterations=5),
    )

    node = agent.step(agent.start("parent task", context="root payload"))
    node = agent.step(node)
    node = agent.step(node)

    assert isinstance(node, ResultNode)
    assert node.result == "child payload"
    assert (
        workspace.context.read("context", agent_id="root.child")
        == "child payload"
    )
