from rlmflow.pool import CallablePool, SequentialPool, ThreadPool
from rlmflow.rlm import RLMConfig, create_pool


def test_create_pool_defaults_to_sequential():
    pool = create_pool(RLMConfig())

    assert isinstance(pool, SequentialPool)


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
