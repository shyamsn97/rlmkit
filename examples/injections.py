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


def print_states(graph) -> None:
    for state in graph.states:
        marker = " [injected]" if state.injected else ""
        print(f"{state.seq}: {state.type}{marker}")


def observation_injection() -> None:
    banner("1. Inject an observation and let the LLM react")

    agent = RLMFlow(
        DemoLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=0, max_iterations=4),
    )
    graph = agent.start("Wait for a controller note, then finish.")

    injected = graph.inject(
        target="root",
        node=ExecOutput(
            output="Injected controller observation: finalize using this note.",
            content="Injected controller observation: finalize using this note.",
        ),
        reason="demo controller note",
    )

    print("Original graph is unchanged:")
    print_states(graph)
    print("\nInjected graph has one extra observation:")
    print_states(injected)

    preview = agent.build_messages(injected)[-1]["content"]
    print("\nNext LLM user message contains the injected observation:")
    print(preview)

    graph = agent.step(injected)  # materializes ExecOutput, then calls the LLM
    graph = agent.step(graph)  # executes the LLM's done(...) block

    print("\nFinal graph:")
    print_states(graph)
    print(f"result={graph.result()!r}")


def action_injection() -> None:
    banner("2. Inject an ExecAction to finalize immediately")

    agent = RLMFlow(
        DemoLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=0, max_iterations=4),
    )
    graph = agent.start("This run will be stopped by the controller.")

    graph = graph.inject(
        target="root",
        node=ExecAction(code='done("controller stopped the run")'),
        reason="message budget exhausted",
    )
    graph = agent.step(graph)  # persists the ExecAction and executes it directly

    print_states(graph)
    print(f"result={graph.result()!r}")


def main() -> None:
    observation_injection()
    action_injection()


if __name__ == "__main__":
    main()
