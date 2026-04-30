"""Execution pools for running tasks in parallel.

A pool has one method: ``execute(tasks) -> results``, where *tasks* is a
list of ``(id, callable)`` pairs and *results* is a ``dict[str, Any]``
mapping IDs to return values.

Pass a pool to ``RLMFlow(pool=...)``. If you pass a plain callable instead
of a Pool instance, it gets wrapped in ``CallablePool`` automatically.
If no pool is passed, ``RLMConfig.max_concurrency`` selects
``ThreadPool``; otherwise the engine uses ``SequentialPool``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class Pool(ABC):
    """Base class for execution pools."""

    @abstractmethod
    def execute(self, tasks: list[tuple[str, Callable[[], Any]]]) -> dict[str, Any]:
        """Run callables in parallel, keyed by ID."""


class ThreadPool(Pool):
    """Run steps concurrently in a ThreadPoolExecutor."""

    def __init__(self, max_concurrency: int = 8) -> None:
        self.max_concurrency = max_concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

    def execute(self, tasks: list[tuple[str, Callable[[], Any]]]) -> dict[str, Any]:
        futures = {self.executor.submit(fn): task_id for task_id, fn in tasks}
        return {futures[f]: f.result() for f in as_completed(futures)}

    def shutdown(self):
        self.executor.shutdown(wait=False)


class SequentialPool(Pool):
    """Runs everything one at a time — useful for testing and debugging."""

    def execute(self, tasks: list[tuple[str, Callable[[], Any]]]) -> dict[str, Any]:
        return {task_id: fn() for task_id, fn in tasks}


class CallablePool(Pool):
    """Wrap a plain function as a pool with .execute()."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def execute(self, tasks: list[tuple[str, Callable[[], Any]]]) -> dict[str, Any]:
        return self.fn(tasks)
