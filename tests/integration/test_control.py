"""Step-level control over runs."""

from __future__ import annotations

from rlmflow import (
    Graph,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMFlow,
    is_done,
    is_user_query,
    is_supervising,
)
from rlmflow.runtime.local import LocalRuntime


class ScriptedByQueryLLM(LLMClient):
    """Picks a reply by matching keywords in any user message."""

    def __init__(self, scripts: dict[str, str]) -> None:
        # scripts is {keyword: reply}. First keyword that matches wins
        # (longest first so more specific keys beat shorter prefixes).
        self.scripts = scripts

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=4, output_tokens=4)
        haystack = "\n".join(
            m.get("content", "")
            for m in messages
            if m.get("role") == "user"
        )
        for key in sorted(self.scripts, key=len, reverse=True):
            if key and key in haystack:
                return self.scripts[key]
        if "" in self.scripts:
            return self.scripts[""]
        raise AssertionError(f"no script matched messages: {haystack[:200]!r}")


def _done(text: str) -> str:
    return f"```repl\ndone({text!r})\n```"


def _delegate_one(child_name: str, query: str) -> str:
    return (
        "```repl\n"
        f"h = delegate({child_name!r}, {query!r}, '')\n"
        "results = yield wait(h)\n"
        "done(results[0])\n"
        "```"
    )


def _run(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def test_step_loop_reaches_result_node():
    agent = RLMFlow(
        llm_client=ScriptedByQueryLLM({"control-step": _done("direct-answer")}),
        runtime=LocalRuntime(),
    )

    final = _run(agent, agent.start("control-step"))

    assert is_done(final.current())
    assert final.result() == "direct-answer"


def test_graph_save_load_resumes_cleanly(tmp_path):
    agent = RLMFlow(
        llm_client=ScriptedByQueryLLM({"control-ckpt": _done("checkpoint-me")}),
        runtime=LocalRuntime(),
    )

    final = _run(agent, agent.start("control-ckpt"))
    final.save(tmp_path / "graph.json")
    restored = Graph.load(tmp_path / "graph.json")

    assert is_done(restored.current())
    assert restored.result() == final.result()
    assert restored.tokens() == final.tokens()


def test_run_history_is_just_a_list_of_graphs():
    """Each ``step()`` returns a new immutable Graph — history is just a list."""
    agent = RLMFlow(
        llm_client=ScriptedByQueryLLM({"control-rewind": _done("rewind-me")}),
        runtime=LocalRuntime(),
    )

    graph = agent.start("control-rewind")
    history = [graph]
    while not graph.finished:
        graph = agent.step(graph)
        history.append(graph)

    assert is_user_query(history[0].current())
    assert is_done(history[-1].current())
    assert history[0] is not history[-1]


def test_supervising_state_lists_waiting_children():
    agent = RLMFlow(
        llm_client=ScriptedByQueryLLM(
            {
                "control-supervise": _delegate_one("child", "delegate-done"),
                "delegate-done": _done("child-answer"),
            }
        ),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    # Two ``step``s: the LLM half then the exec half (which yields,
    # producing the SupervisingOutput).
    graph = agent.start("control-supervise")
    graph = agent.step(graph)
    graph = agent.step(graph)
    assert is_supervising(graph.current())
    assert graph.current().waiting_on == ["root.child"]

    final = _run(agent, graph)
    assert is_done(final.current())
    assert final.result() == "child-answer"
