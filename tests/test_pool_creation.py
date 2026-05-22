import threading

from rlmflow.utils.pool import CallablePool, SequentialPool, ThreadPool
from rlmflow.rlm import RLMConfig, create_pool


def test_create_pool_defaults_to_threadpool_at_cpu_count():
    import os

    pool = create_pool(RLMConfig())

    try:
        assert isinstance(pool, ThreadPool)
        assert pool.max_concurrency == (os.cpu_count() or 1)
    finally:
        pool.shutdown()


def test_create_pool_explicit_none_max_concurrency_falls_back_to_sequential():
    pool = create_pool(RLMConfig(max_concurrency=None))

    assert isinstance(pool, SequentialPool)


def test_create_pool_max_concurrency_one_uses_sequential():
    """A pool of one isn't a pool — skip the threading overhead."""
    for n in (0, 1):
        pool = create_pool(RLMConfig(max_concurrency=n))
        assert isinstance(pool, SequentialPool), n


def test_create_pool_uses_threadpool_when_max_concurrency_is_set():
    pool = create_pool(RLMConfig(max_concurrency=3))

    try:
        assert isinstance(pool, ThreadPool)
        assert pool.max_concurrency == 3
    finally:
        pool.shutdown()


def test_create_pool_explicit_pool_wins_over_config():
    explicit = SequentialPool()

    pool = create_pool(RLMConfig(max_concurrency=3), explicit)

    assert pool is explicit


def test_create_pool_wraps_callable_pool():
    def run(tasks):
        return {task_id: fn() for task_id, fn in tasks}

    pool = create_pool(RLMConfig(max_concurrency=3), run)

    assert isinstance(pool, CallablePool)
    assert pool.execute([("x", lambda: 1)]) == {"x": 1}


def test_threadpool_run_until_idle_refills_before_slow_sibling_finishes():
    pool = ThreadPool(max_concurrency=2)
    events: list[str] = []
    fast_done = threading.Event()
    fast_second_started = threading.Event()

    def fast_first():
        events.append("fast_first_start")
        fast_done.set()
        return "fast1"

    def fast_second():
        events.append("fast_second_start")
        fast_second_started.set()
        return "fast2"

    def slow():
        events.append("slow_start")
        assert fast_done.wait(timeout=2)
        assert fast_second_started.wait(timeout=2)
        events.append("slow_done")
        return "slow"

    def refill(task_id, _result, active_ids):
        if task_id == "fast_first":
            assert "slow" in active_ids
            return [("fast_second", fast_second)]
        return []

    try:
        results = pool.run_until_idle(
            [("fast_first", fast_first), ("slow", slow)],
            refill,
        )
    finally:
        pool.shutdown()

    assert results == {
        "fast_first": "fast1",
        "fast_second": "fast2",
        "slow": "slow",
    }
    assert events.index("fast_second_start") < events.index("slow_done")


def test_callable_pool_run_until_idle_uses_execute_fallback():
    calls: list[list[str]] = []

    def run(tasks):
        calls.append([task_id for task_id, _ in tasks])
        return {task_id: fn() for task_id, fn in tasks}

    pool = CallablePool(run)

    def refill(task_id, _result, _active_ids):
        if task_id == "first":
            return [("second", lambda: 2)]
        return []

    assert pool.run_until_idle([("first", lambda: 1)], refill) == {
        "first": 1,
        "second": 2,
    }
    assert calls == [["first"], ["second"]]
