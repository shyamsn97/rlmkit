"""Inject typed nodes into a running graph.

This shows the controller workflow:

1. Build a new graph value with ``graph.inject(...)``.
2. Pass that graph to ``agent.step(graph)`` to materialize the injected node.
3. Let normal scheduling continue, or inject an ``ExecAction`` to finalize now.

Run:
    python examples/injections.py
"""

from __future__ import annotations

from rlmflow import ExecAction, ExecOutput, LLMClient, LLMUsage, RLMConfig, RLMFlow
from rlmflow.runtime.local import LocalRuntime

OBSERVATION = "Injected controller observation: finalize using this note."


class DemoLLM(LLMClient):
    """Deterministic model so the example runs offline."""

    def chat(self, messages, *args, **kwargs) -> str:
        del args, kwargs
        self.last_usage = LLMUsage(input_tokens=80, output_tokens=20)
        prompt = messages[-1]["content"]
        if "Injected controller observation" in prompt:
            return '```repl\ndone("used the injected controller observation")\n```'
        return '```repl\nprint("waiting for controller input")\n```'


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def state_types(graph) -> list[str]:
    return [state.type for state in graph.states]


def assert_types(graph, expected: list[str]) -> None:
    actual = state_types(graph)
    assert actual == expected, f"expected states {expected}, got {actual}"


def print_states(label: str, graph) -> None:
    print(f"\n{label}")
    print("state types:", " -> ".join(state_types(graph)))
    for state in graph.states:
        print(f"{state.seq}: {state.type}")


def observation_injection() -> None:
    banner("1. Inject an observation and let the LLM react")

    agent = RLMFlow(
        DemoLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=0, max_iterations=4),
    )
    graph = agent.start("Wait for a controller note, then finish.")
    assert_types(graph, ["user_query"])

    injected = graph.inject(
        target="root",
        node=ExecOutput(
            output=OBSERVATION,
            content=OBSERVATION,
        ),
    )

    assert injected is not graph
    assert_types(graph, ["user_query"])
    assert_types(injected, ["user_query", "exec_output"])

    extra = injected.states[-1]
    extra_keys = set(extra.to_dict())
    assert isinstance(extra, ExecOutput)
    assert "injected" not in extra_keys
    assert "injected_reason" not in extra_keys

    print_states("start(): original graph", graph)
    print_states("graph.inject(...): returned graph with one plain ExecOutput", injected)
    print("original graph is unchanged:", state_types(graph))
    print("extra node keys do not include injection metadata:", sorted(extra_keys))

    projected = agent.build_messages(injected)[-1]["content"]
    assert OBSERVATION in projected
    print("message projection contains the controller observation:", OBSERVATION)

    graph = agent.step(injected)  # materializes ExecOutput, then calls the LLM
    assert_types(graph, ["user_query", "exec_output", "llm_action", "llm_output"])
    print_states("agent.step(injected): committed observation and asked the LLM", graph)

    graph = agent.step(graph)  # executes the LLM's done(...) block
    assert_types(
        graph,
        [
            "user_query",
            "exec_output",
            "llm_action",
            "llm_output",
            "exec_action",
            "done_output",
        ],
    )
    assert graph.result() == "used the injected controller observation"

    print_states("agent.step(...): executed the LLM's done(...) block", graph)
    print(f"result={graph.result()!r}")


def action_injection() -> None:
    banner("2. Inject an ExecAction to finalize immediately")

    agent = RLMFlow(
        DemoLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=0, max_iterations=4),
    )
    graph = agent.start("This run will be stopped by the controller.")
    assert_types(graph, ["user_query"])

    injected = graph.inject(
        target="root",
        node=ExecAction(code='done("controller stopped the run")'),
    )
    assert injected is not graph
    assert_types(graph, ["user_query"])
    assert_types(injected, ["user_query", "exec_action"])

    print_states("start(): original graph", graph)
    print_states("graph.inject(...): returned graph with one plain ExecAction", injected)

    graph = agent.step(injected)  # persists the ExecAction and executes it directly
    assert_types(graph, ["user_query", "exec_action", "done_output"])
    assert graph.result() == "controller stopped the run"

    print_states("agent.step(injected): executed the appended action", graph)
    print(f"result={graph.result()!r}")


def main() -> None:
    observation_injection()
    action_injection()


if __name__ == "__main__":
    main()
