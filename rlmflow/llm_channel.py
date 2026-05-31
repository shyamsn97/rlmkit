"""Shared scheduling for model requests within one RLMFlow run."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from rlmflow.llm import LLMClient, LLMUsage


@dataclass
class LLMLane:
    client: LLMClient
    thread_safe: bool
    lock: threading.Lock


class LLMChannel:
    """Bounded request channel for all LLM calls in one engine instance."""

    def __init__(
        self,
        clients: dict[str, LLMClient],
        *,
        max_concurrency: int | None,
        thread_safe: dict[str, bool] | None = None,
    ) -> None:
        workers = max(1, max_concurrency or 1)
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._lanes = {
            model: LLMLane(
                client=client,
                thread_safe=(
                    thread_safe[model]
                    if thread_safe and model in thread_safe
                    else bool(getattr(client, "thread_safe", False))
                ),
                lock=threading.Lock(),
            )
            for model, client in clients.items()
        }
        self._closed = False

    def call(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> tuple[str, LLMUsage]:
        """Run one model request through the shared channel."""

        if self._closed:
            raise RuntimeError("LLMChannel is closed")
        lane = self._lane(model)
        future = self._executor.submit(self._call_lane, lane, messages)
        return future.result()

    def batch(
        self,
        model: str,
        prompts: list[str],
    ) -> list[tuple[str, LLMUsage]]:
        """Run prompts through the shared channel and preserve input order."""

        if self._closed:
            raise RuntimeError("LLMChannel is closed")
        if not prompts:
            return []

        lane = self._lane(model)
        futures = {
            self._executor.submit(
                self._call_lane,
                lane,
                [{"role": "user", "content": prompt}],
            ): index
            for index, prompt in enumerate(prompts)
        }
        pairs_by_index: dict[int, tuple[str, LLMUsage]] = {}
        for future in as_completed(futures):
            pairs_by_index[futures[future]] = future.result()
        return [pairs_by_index[index] for index in range(len(prompts))]

    def shutdown(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False)

    def _lane(self, model: str) -> LLMLane:
        try:
            return self._lanes[model]
        except KeyError as exc:
            keys = ", ".join(sorted(self._lanes))
            raise ValueError(f"unknown model {model!r}. available: {keys}") from exc

    @staticmethod
    def _call_lane(
        lane: LLMLane,
        messages: list[dict[str, str]],
    ) -> tuple[str, LLMUsage]:
        if lane.thread_safe:
            return lane.client.completion(messages)
        with lane.lock:
            return lane.client.completion(messages)


__all__ = ["LLMChannel"]
