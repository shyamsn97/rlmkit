from __future__ import annotations

from importlib.util import find_spec

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
from rlmflow.utils.trace import load_trace, save_trace
from rlmflow.utils.viz import build_viz_graph, node_tree


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

    assert isinstance(node.current(), ResultNode)
    assert node.current().result == "ok"
    assert "root [query]" in node.tree()
    assert "root [action]" in node.tree()
    assert "root [result] {default} -> ok" in node.tree()


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
    assert isinstance(node.current(), SupervisingNode)

    node = agent.step(node)
    assert isinstance(node.current(), SupervisingNode)
    child = node.current().children[0]
    assert child.type == "query"
    assert child.current().type == "result"

    node = agent.step(node)
    assert isinstance(node.current(), ResultNode)
    assert node.current().result == "parent:child-result"


def test_resumed_parent_preserves_prior_children_when_delegating_again():
    class ScriptedLLM(LLMClient):
        def chat(self, messages, *args, **kwargs) -> str:
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone("child-result")\n```'
            if "verify task" in prompt:
                return '```repl\ndone("verified")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "child_results = yield wait(h)\n"
                'v = delegate("verify", "verify task", "")\n'
                "verdict = yield wait(v)\n"
                'done("parent:" + verdict[0])\n'
                "```"
            )

    agent = RLMFlow(
        ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )

    node = agent.step(agent.start("parent task"))
    assert isinstance(node.current(), SupervisingNode)

    node = agent.step(node)
    assert isinstance(node.current(), SupervisingNode)
    assert [child.agent_id for child in node.current().children] == ["root.child"]

    node = agent.step(node)
    assert isinstance(node.current(), SupervisingNode)
    assert {child.agent_id for child in node.current().children} == {
        "root.child",
        "root.verify",
    }

    node = agent.step(node)
    node = agent.step(node)
    assert isinstance(node.current(), ResultNode)
    assert node.current().result == "parent:verified"


def test_workspace_session_persists_nodes_and_context_payload(tmp_path):
    ws = Workspace.create(tmp_path / "workspace")
    agent = RLMFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        workspace=ws,
        config=RLMConfig(max_iterations=2),
    )

    node = agent.step(agent.start("say ok", context="hello"))

    assert isinstance(node.current(), ResultNode)
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


def test_node_transcript_renders_agent_chain():
    query = QueryNode(agent_id="root.worker", query="find the code")
    action = query.successor(ActionNode, reply='```repl\ndone("84721")\n```')
    result = action.successor(ResultNode, result="84721")
    root = QueryNode(
        agent_id="root",
        children=[
            query.update(children=[action.id]),
            action.update(children=[result.id]),
            result,
        ],
    )

    transcript = root.transcript("root.worker", include_system=False)

    assert "--- query ---\nfind the code" in transcript
    assert '--- assistant ---\n```repl\ndone("84721")\n```' in transcript
    assert "--- result ---\n84721" in transcript


def test_node_plot_returns_plotly_figure():
    if find_spec("plotly") is None:
        return

    fig = _sample_tree().plot(title="sample")

    assert fig.layout.title.text.startswith("<b>sample</b>")
    assert len(fig.data) >= 2


def test_node_plot_supports_static_formats():
    root = _sample_tree()

    assert root.plot("tree").startswith("root [query]")
    assert root.plot("mermaid").startswith("stateDiagram-v2")
    assert root.plot("flowchart").startswith("flowchart TD")
    assert root.plot("dot").startswith("digraph rlmflow")
    assert 'root.search' in root.plot("d2")


def test_node_plot_supports_gantt_html():
    root = _sample_tree()
    html = root.plot("gantt", states=[root], title="sample gantt")

    assert "<html>" in html
    assert "sample gantt" in html


def test_viz_graph_events_are_step_local():
    root_query = QueryNode(agent_id="root", query="parent")
    root_action = root_query.successor(ActionNode, code="delegate")
    child_query = QueryNode(agent_id="root.child", depth=1, query="child")
    root_supervising = root_action.successor(
        SupervisingNode,
        code="delegate",
        waiting_on=["root.child"],
        children=[child_query],
    )
    child_action = child_query.successor(ActionNode, code="done")
    child_result = child_action.successor(ResultNode, result="ok")
    events = [
        root_query.update(children=[root_action.id]),
        root_action.update(children=[root_supervising.id]),
        child_query.update(children=[child_action.id]),
        root_supervising,
        child_action.update(children=[child_result.id]),
        child_result,
    ]

    graph = build_viz_graph(
        states=[root_supervising],
        events=events,
        step=0,
        mode="events",
    )

    assert [node.type for node in graph.payloads] == [
        "query",
        "action",
        "query",
        "supervising",
    ]
    assert "result" not in {node.type for node in graph.payloads}
    assert ("next", "root", "root") in {
        (edge.kind, graph.nodes_by_id[edge.source].agent_id, graph.nodes_by_id[edge.target].agent_id)
        for edge in graph.edges
    }
    assert ("spawn", "root", "root.child") in {
        (edge.kind, graph.nodes_by_id[edge.source].agent_id, graph.nodes_by_id[edge.target].agent_id)
        for edge in graph.edges
    }


def test_tree_prefers_workspace_event_tree(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    agent = RLMFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        workspace=workspace,
        config=RLMConfig(max_iterations=2),
    )

    state = agent.start("say ok")
    state = agent.step(state)
    rendered = state.tree()

    assert "root [query]" in rendered
    assert "root [action]" in rendered
    assert "root [result]" in rendered


def test_node_tree_renders_node_lifecycle_not_agent_snapshot():
    query = QueryNode(agent_id="root.worker", query="find")
    action = query.successor(ActionNode, code="done")
    result = action.successor(ResultNode, result="84721")
    events = [
        query.update(children=[action.id]),
        action.update(children=[result.id]),
        result,
    ]

    graph = build_viz_graph(events=events, mode="events")
    text = node_tree(graph)

    assert "worker [query]" in text
    assert "next: worker [action]" in text
    assert "next: worker [result]" in text


def test_trace_can_persist_events_without_node_fields(tmp_path):
    query = QueryNode(agent_id="root", query="q")
    result = query.successor(ResultNode, result="ok")
    path = save_trace([result], tmp_path / "trace", events=[query, result])

    trace = load_trace(path)

    assert [node.type for node in trace.events] == ["query", "result"]
    assert not hasattr(trace.states[0], "plot_history")


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
    assert isinstance(node.current(), SupervisingNode)
    child = node.current().children[0]

    assert child.agent_id == "root.child"
    assert child.depth == 1
    assert child.runtime is not None
    assert child.runtime.id != "root"
    assert not hasattr(agent, "child_engines")

    node = agent.step(node)
    assert isinstance(node.current().children[0].current(), ResultNode)
    assert node.current().children[0].current().result == "child"


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

    assert isinstance(node.current(), ResultNode)
    assert node.current().result == "child payload"
    assert (
        workspace.context.read("context", agent_id="root.child")
        == "child payload"
    )
