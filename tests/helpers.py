"""Shared test scaffolding for engine-style tests."""

from __future__ import annotations

from rlmflow import Graph, LLMClient, RLMConfig, RLMFlow
from rlmflow.runtime.local import LocalRuntime


class StaticLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, *args, **kwargs) -> str:
        return self.reply


def run_to_completion(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def make_agent(reply: str = '```repl\ndone("ok")\n```', **config_kwargs) -> RLMFlow:
    config_kwargs.setdefault("max_iterations", 3)
    return RLMFlow(
        StaticLLM(reply),
        runtime=LocalRuntime(),
        config=RLMConfig(**config_kwargs),
    )


__all__ = ["StaticLLM", "make_agent", "run_to_completion"]
