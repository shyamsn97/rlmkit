from __future__ import annotations

import threading
import time

from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.llm_channel import LLMChannel


class _ObservedLLM(LLMClient):
    def __init__(self, *, thread_safe: bool, delay: float = 0.01) -> None:
        self.thread_safe = thread_safe
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def chat(self, messages, *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
        prompt = messages[-1]["content"]
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        if prompt == "slow":
            time.sleep(self.delay * 3)
        else:
            time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return prompt.upper(), LLMUsage(input_tokens=len(prompt), output_tokens=1)


def test_llm_channel_preserves_batch_order_when_calls_finish_out_of_order():
    client = _ObservedLLM(thread_safe=True, delay=0.01)
    channel = LLMChannel(
        {"default": client},
        max_concurrency=2,
    )
    try:
        pairs = channel.batch("default", ["slow", "fast"])
    finally:
        channel.shutdown()

    assert [text for text, _usage in pairs] == ["SLOW", "FAST"]
    assert client.max_active == 2


def test_llm_channel_global_cap_holds_across_nested_callers():
    client = _ObservedLLM(thread_safe=True, delay=0.01)
    channel = LLMChannel(
        {"default": client},
        max_concurrency=2,
    )
    outputs: list[list[str]] = []
    outputs_lock = threading.Lock()

    def run_batch(index: int) -> None:
        pairs = channel.batch(
            "default",
            [f"{index}-a", f"{index}-b", f"{index}-c"],
        )
        with outputs_lock:
            outputs.append([text for text, _usage in pairs])

    threads = [threading.Thread(target=run_batch, args=(i,)) for i in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        channel.shutdown()

    assert len(outputs) == 4
    assert client.max_active <= 2


def test_llm_channel_serializes_unsafe_clients():
    client = _ObservedLLM(thread_safe=False, delay=0.01)
    channel = LLMChannel(
        {"default": client},
        max_concurrency=4,
    )
    try:
        pairs = channel.batch("default", ["a", "b", "c", "d"])
    finally:
        channel.shutdown()

    assert [text for text, _usage in pairs] == ["A", "B", "C", "D"]
    assert client.max_active == 1


class _UsageLLM(LLMClient):
    thread_safe = True

    def chat(self, messages, *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
        prompt = messages[-1]["content"]
        usage = LLMUsage(input_tokens=int(prompt), output_tokens=1)
        self.last_usage = LLMUsage(input_tokens=999, output_tokens=999)
        return prompt, usage


def test_llm_channel_uses_per_request_usage_not_shared_last_usage():
    channel = LLMChannel(
        {"default": _UsageLLM()},
        max_concurrency=3,
    )
    try:
        pairs = channel.batch("default", ["1", "2", "3"])
    finally:
        channel.shutdown()

    assert [text for text, _usage in pairs] == ["1", "2", "3"]
    assert sum(usage.input_tokens for _text, usage in pairs) == 6
    assert sum(usage.output_tokens for _text, usage in pairs) == 3
