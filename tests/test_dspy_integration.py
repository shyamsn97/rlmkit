from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

from rlmflow import LLMClient, LLMUsage


class _FakeBaseLM:
    def __init__(self, model, model_type="chat", **kwargs):
        self.model = model
        self.model_type = model_type
        self.kwargs = kwargs

    def __call__(self, prompt=None, messages=None, **kwargs):
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        return [choice.message.content for choice in response.choices]


class _EchoClient(LLMClient):
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.kwargs: dict[str, Any] = {}
        self.last_usage = LLMUsage(input_tokens=3, output_tokens=4)

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        self.messages = messages
        self.kwargs = kwargs
        return f"answer: {messages[-1]['content']}"


def _import_with_fake_dspy(monkeypatch):
    fake_dspy = SimpleNamespace(BaseLM=_FakeBaseLM)
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    sys.modules.pop("rlmflow.integrations.dspy", None)
    return importlib.import_module("rlmflow.integrations.dspy")


def test_rlmflow_lm_adapts_prompt_to_dspy_lm(monkeypatch):
    integration = _import_with_fake_dspy(monkeypatch)
    client = _EchoClient()

    lm = integration.RLMFlowLM(client, model="rlmflow/test", max_tokens=100)

    assert lm("hello", temperature=0.2) == ["answer: hello"]
    assert client.messages == [{"role": "user", "content": "hello"}]
    assert client.kwargs == {"max_tokens": 100, "temperature": 0.2}


def test_rlmflow_lm_returns_openai_style_response(monkeypatch):
    integration = _import_with_fake_dspy(monkeypatch)
    client = _EchoClient()

    lm = integration.RLMFlowLM(client, model="rlmflow/test")
    response = lm.forward(messages=[{"role": "user", "content": "hi"}])

    assert response.model == "rlmflow/test"
    assert response.choices[0].message.content == "answer: hi"
    assert response.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
