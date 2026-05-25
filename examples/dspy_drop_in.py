"""Use an RLMFlow agent as the LM behind a DSPy program.

Run with:
    export OPENAI_API_KEY=...
    pip install -e ".[openai,dspy]"
    python examples/dspy_drop_in.py
"""

from __future__ import annotations

from pathlib import Path

import dspy

from rlmflow import OpenAIClient, RLMConfig, RLMFlow, Workspace
from rlmflow.integrations.dspy import RLMFlowLM
from rlmflow.runtime.local import LocalRuntime


def main() -> None:
    workspace = Workspace.create(Path(__file__).parent / "example-workspaces" / "dspy-workspace")
    agent = RLMFlow(
        llm_client=OpenAIClient(model="gpt-4o-mini"),
        runtime=LocalRuntime(workspace=workspace),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )

    dspy.configure(lm=RLMFlowLM(agent, model="rlmflow/gpt-4o-mini"))

    qa = dspy.ChainOfThought("question -> answer")
    result = qa(question="What is 17 * 23? Show a short calculation.")
    print(result.answer)


if __name__ == "__main__":
    main()
