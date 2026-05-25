"""Execution pools for running tasks in parallel.

A pool has a barrier method, ``execute(tasks) -> results``, and a dynamic
method, ``run_until_idle(tasks, refill) -> results``. In both cases *tasks*
is a list of ``(id, callable)`` pairs and *results* is a ``dict[str, Any]``
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

Task = tuple[str, Callable[[], Any]]
Refill = Callable[[str, Any, set[str]], list[Task]]


class Pool(ABC):
    """Base class for execution pools."""

    @abstractmethod
    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        """Run callables in parallel, keyed by ID."""

    def run_until_idle(self, tasks: list[Task], refill: Refill) -> dict[str, Any]:
        """Run tasks and let completions enqueue follow-up tasks.

        The base implementation preserves compatibility with execute-only
        custom pools by running in barrier batches. Pools with native
        as-completed support should override this for true work-conserving
        refill.
        """
        results: dict[str, Any] = {}
        pending = list(tasks)
        while pending:
            batch_results = self.execute(pending)
            pending = []
            for task_id, result in batch_results.items():
                results[task_id] = result
                pending.extend(refill(task_id, result, set()))
        return results


class ThreadPool(Pool):
    """Run steps concurrently in a ThreadPoolExecutor."""

    def __init__(self, max_concurrency: int = 8) -> None:
        self.max_concurrency = max_concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        futures = {self.executor.submit(fn): task_id for task_id, fn in tasks}
        return {futures[f]: f.result() for f in as_completed(futures)}

    def run_until_idle(self, tasks: list[Task], refill: Refill) -> dict[str, Any]:
        futures = {self.executor.submit(fn): task_id for task_id, fn in tasks}
        results: dict[str, Any] = {}
        while futures:
            for future in as_completed(list(futures)):
                task_id = futures.pop(future)
                result = future.result()
                results[task_id] = result
                active_ids = set(futures.values())
                for new_task_id, fn in refill(task_id, result, active_ids):
                    futures[self.executor.submit(fn)] = new_task_id
                break
        return results

    def shutdown(self):
        self.executor.shutdown(wait=False)


class SequentialPool(Pool):
    """Runs everything one at a time — useful for testing and debugging."""

    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        return {task_id: fn() for task_id, fn in tasks}


class CallablePool(Pool):
    """Wrap a plain function as a pool with .execute()."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        return self.fn(tasks)


__all__ = ["CallablePool", "Pool", "Refill", "SequentialPool", "Task", "ThreadPool"]
