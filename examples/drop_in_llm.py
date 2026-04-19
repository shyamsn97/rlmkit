"""RLM as a drop-in LLM.

Because `RLM` inherits from `LLMClient`, you can swap it in anywhere you'd
use a raw LLM. Calling `agent.chat(messages)` runs the full recursive agent
loop under the hood and returns a plain string — same signature as any other
LLM client.

This enables two patterns:

1. **Replace an LLM with an agent.** Any function that takes an `LLMClient`
   (e.g. a summarization helper, a router, a retrieval pipeline) gets agentic
   behavior for free — no code changes.

2. **Nest agents.** An outer `RLM` can use an inner `RLM` as its `llm_client`.
   The outer agent's every "LLM call" is itself a full recursive sub-agent run.

Run with:
    export OPENAI_API_KEY=...
    python examples/drop_in_llm.py
"""

from __future__ import annotations

from rlmkit import RLM, LLMClient, OpenAIClient, RLMConfig
from rlmkit.runtime.local import LocalRuntime


def ask(llm: LLMClient, question: str) -> str:
    """A generic helper that takes any LLMClient. Doesn't know or care
    whether it got a plain OpenAI client or a full recursive agent."""
    reply = llm.chat([{"role": "user", "content": question}])
    usage = llm.last_usage
    tokens = usage.input_tokens + usage.output_tokens if usage else 0
    print(f"[{type(llm).__name__}] tokens={tokens}")
    return reply


def demo_plain_llm():
    print("=== plain OpenAI client ===")
    llm = OpenAIClient(model="gpt-4o-mini")
    answer = ask(llm, "In one sentence: what is the capital of France?")
    print(answer, "\n")


def demo_rlm_as_llm():
    print("=== RLM as LLMClient (drop-in) ===")
    agent = RLM(
        llm_client=OpenAIClient(model="gpt-4o-mini"),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=5, max_budget=20_000),
    )
    answer = ask(agent, "Compute 17 * 23 using a ```repl``` block, then call done().")
    print(answer, "\n")


def demo_nested_rlm():
    print("=== nested RLM (outer agent uses inner agent as its LLM) ===")
    inner = RLM(
        llm_client=OpenAIClient(model="gpt-4o-mini"),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3),
    )
    outer = RLM(
        llm_client=inner,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3, max_budget=50_000),
    )
    answer = outer.run("What's the 7th Fibonacci number? Use ```repl``` to compute.")
    print(answer)
    print(f"outer.last_usage = {outer.last_usage}")


if __name__ == "__main__":
    demo_plain_llm()
    demo_rlm_as_llm()
    demo_nested_rlm()
