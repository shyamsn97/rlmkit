"""Demonstrate `async_children=True` work-conserving child scheduling.

Run:
    python examples/async_children.py

The key thing to watch in the output:

- With `async_children=False`, child B's second LLM step starts only after
  child A's slow first LLM step completes.
- With `async_children=True`, child B's second LLM step starts while child A's
  slow first LLM step is still running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.rlm import RLMConfig, RLMFlow
from rlmflow.runtime.local import LocalRuntime


@dataclass
class TimelineLLM(LLMClient):
    started_at: float = field(default_factory=time.perf_counter)
    events: list[tuple[float, str]] = field(default_factory=list)

    def mark(self, label: str) -> None:
        self.events.append((time.perf_counter() - self.started_at, label))

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)
        prompt = messages[-1]["content"].lower()
        assistant_history = "\n".join(
            m["content"].lower()
            for m in messages
            if m.get("role") == "assistant"
        )

        if "child a slow task" in prompt:
            self.mark("childa.task_1 start")
            time.sleep(1.0)
            self.mark("childa.task_1 finish")
            return '```repl\ndone("A done")\n```'

        if "child b two-step task" in prompt:
            if "childb task_1 exec" not in assistant_history:
                self.mark("childb.task_1 start")
                self.mark("childb.task_1 finish")
                return '```repl\nprint("childb task_1 exec")\n```'
            self.mark("childb.task_2 start")
            self.mark("childb.task_2 finish")
            return '```repl\ndone("B done")\n```'

        return (
            "```repl\n"
            'a = rlm_delegate(name="childa", query="Child A slow task", context="")\n'
            'b = rlm_delegate(name="childb", query="Child B two-step task", context="")\n'
            "results = await rlm_wait(a, b)\n"
            'done(" | ".join(results))\n'
            "```"
        )


def run_case(*, async_children: bool) -> None:
    llm = TimelineLLM()
    agent = RLMFlow(
        llm,
        runtime=LocalRuntime(),
        config=RLMConfig(
            async_children=async_children,
            max_depth=1,
            max_iterations=8,
            max_concurrency=2,
        ),
    )

    graph = agent.start("Show async child scheduling.")
    steps = 0
    while not graph.finished:
        graph = agent.step(graph)
        steps += 1

    mode = "async_children=True" if async_children else "async_children=False"
    print(f"\n=== {mode} ===")
    print(f"outer step() calls: {steps}")
    for t, label in llm.events:
        print(f"{t:0.3f}s  {label}")
    print("result:", graph.result())


def main() -> None:
    run_case(async_children=False)
    run_case(async_children=True)


if __name__ == "__main__":
    main()
