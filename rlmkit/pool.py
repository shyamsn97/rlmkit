"""Execution pools for stepping agents in parallel.

A pool has one method: ``execute(items) -> results``, where *items* is a
list of ``(RLMState, RLM)`` pairs and *results* is a ``dict[str, RLMState]``
mapping agent IDs to their new states.

Pass a pool to ``RLM(pool=...)``.  If you pass a plain callable instead
of a Pool subclass, it gets wrapped in ``CallablePool`` automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class Pool(ABC):
    """Base class for execution pools."""

    @abstractmethod
    def execute(self, items: list[tuple[Any, Any]]) -> dict[str, Any]:
        """Step a batch of (state, engine) pairs. Returns {agent_id: new_state}."""


class ThreadPool(Pool):
    """Default pool — runs steps in a ThreadPoolExecutor."""

    def __init__(self, max_concurrency: int = 8) -> None:
        self.max_concurrency = max_concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

    def execute(self, items: list[tuple[Any, Any]]) -> dict[str, Any]:
        results = {}
        futures = {
            self.executor.submit(engine.step, cs): cs.agent_id for cs, engine in items
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return results

    def shutdown(self):
        self.executor.shutdown(wait=False)


class SequentialPool(Pool):
    """Runs everything one at a time — useful for testing and debugging."""

    def execute(self, items: list[tuple[Any, Any]]) -> dict[str, Any]:
        return {cs.agent_id: engine.step(cs) for cs, engine in items}


class CallablePool(Pool):
    """Wrap a plain function as a pool with .execute()."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def execute(self, items: list[tuple[Any, Any]]) -> dict[str, Any]:
        return self.fn(items)
