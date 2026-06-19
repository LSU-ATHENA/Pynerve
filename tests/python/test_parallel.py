"""Tests for parallel execution modules."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pynerve._parallel_api import ForkServerPool, compute_persistence_parallel
from pynerve._parallel_patterns import ChunkedParallel, MapReducePH
from pynerve._parallel_pool import ParallelPH, _shared_memory_worker
from pynerve._shared_memory import SharedMemoryArray


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _sum_batch(x: np.ndarray) -> float:
    return float(x.sum())


def _add_two(a: float, b: float) -> float:
    return a + b


def _failing_fn(_x: np.ndarray) -> None:
    raise ValueError("worker error")


def _returns_list(x: np.ndarray) -> list[float]:
    return [float(x.sum()), float(x.mean())]


class TestForkServerPool:
    def test_init_defaults(self):
        pool = ForkServerPool()
        assert pool.n_workers >= 1
        r = repr(pool)
        assert "ForkServerPool" in r
        assert "active=False" in r

    def test_init_with_n_workers(self):
        pool = ForkServerPool(n_workers=2)
        assert pool.n_workers == 2

    def test_init_rejects_invalid_n_workers(self):
        with pytest.raises((TypeError, ValueError)):
            ForkServerPool(n_workers=-1)
        with pytest.raises((TypeError, ValueError)):
            ForkServerPool(n_workers=0)
        with pytest.raises((TypeError, ValueError)):
            ForkServerPool(n_workers="invalid")

    def test_context_manager_enter_and_exit(self):
        pool = ForkServerPool(n_workers=1)
        result = pool.__enter__()
        assert result is not None
        assert "active=True" in repr(pool)
        pool.__exit__(None, None, None)
        assert "active=False" in repr(pool)

    def test_context_manager_rejects_double_enter(self):
        pool = ForkServerPool(n_workers=1)
        pool.__enter__()
        try:
            with pytest.raises(RuntimeError, match="already active"):
                pool.__enter__()
        finally:
            pool.__exit__(None, None, None)

    def test_context_manager_terminate_on_exception(self):
        pool = ForkServerPool(n_workers=1)
        pool.__enter__()
        pool.__exit__(RuntimeError, RuntimeError("test"), None)
        assert "active=False" in repr(pool)

    def test_repr(self):
        pool = ForkServerPool(n_workers=4)
        r = repr(pool)
        assert "ForkServerPool" in r
        assert "4" in r


class TestComputePersistenceParallel:
    def test_rejects_non_callable_progress_callback(self):
        data = [np.array([[1.0, 2.0, 0.0]])]
        with pytest.raises(TypeError, match="progress_callback"):
            compute_persistence_parallel(data, progress_callback="not_callable")

    def test_rejects_non_iterable_batches(self):
        with pytest.raises(TypeError, match="data_batches"):
            compute_persistence_parallel("not_a_list")

    def test_empty_batches(self):
        result = compute_persistence_parallel([])
        assert result == []

    def test_calls_compute_persistence_batch(self):
        data = [np.array([[1.0, 2.0, 0.0]])]
        with patch("pynerve._parallel_api.ParallelPH", autospec=True) as mock_ph_class:
            mock_ph = mock_ph_class.return_value.__enter__.return_value
            mock_ph.map.return_value = ["result"]
            result = compute_persistence_parallel(data)
            mock_ph.map.assert_called_once()
            assert result == ["result"]


class TestParallelPHInit:
    def test_init_defaults(self):
        ph = ParallelPH()
        assert ph.n_workers >= 1
        assert ph.use_shared_memory is True
        assert ph.chunksize == 1
        r = repr(ph)
        assert "ParallelPH" in r
        assert "active=False" in r

    def test_init_custom_params(self):
        ph = ParallelPH(n_workers=2, use_shared_memory=False, chunksize=3)
        assert ph.n_workers == 2
        assert ph.use_shared_memory is False
        assert ph.chunksize == 3

    def test_init_rejects_invalid_n_workers(self):
        with pytest.raises((TypeError, ValueError)):
            ParallelPH(n_workers=-1)
        with pytest.raises((TypeError, ValueError)):
            ParallelPH(n_workers=0)

    def test_init_rejects_invalid_chunksize(self):
        with pytest.raises((TypeError, ValueError)):
            ParallelPH(chunksize=0)
        with pytest.raises((TypeError, ValueError)):
            ParallelPH(chunksize=-1)

    def test_init_rejects_invalid_use_shared_memory(self):
        with pytest.raises((TypeError, ValueError)):
            ParallelPH(use_shared_memory="yes")

    def test_repr(self):
        ph = ParallelPH(n_workers=4, use_shared_memory=False)
        r = repr(ph)
        assert "ParallelPH" in r
        assert "4" in r


class TestParallelPHContextManager:
    def test_enter_creates_pool(self):
        ph = ParallelPH(n_workers=1)
        result = ph.__enter__()
        assert result is ph
        assert "active=True" in repr(ph)
        ph.__exit__(None, None, None)

    def test_enter_rejects_double_enter(self):
        ph = ParallelPH(n_workers=1)
        ph.__enter__()
        try:
            with pytest.raises(RuntimeError, match="already active"):
                ph.__enter__()
        finally:
            ph.__exit__(None, None, None)

    def test_exit_close_on_success(self):
        ph = ParallelPH(n_workers=1)
        ph.__enter__()
        ph.__exit__(None, None, None)
        assert "active=False" in repr(ph)

    def test_exit_terminate_on_error(self):
        ph = ParallelPH(n_workers=1)
        ph.__enter__()
        ph.__exit__(RuntimeError, RuntimeError("x"), None)
        assert "active=False" in repr(ph)


class TestParallelPHMap:
    def test_map_outside_context(self):
        ph = ParallelPH(n_workers=1)
        with pytest.raises(RuntimeError, match="context manager"):
            ph.map(_identity, [np.array([1.0])])

    def test_rejects_non_callable_compute_fn(self):
        with ParallelPH(n_workers=1) as ph, pytest.raises(TypeError, match="compute_fn"):
            ph.map(42, [np.array([1.0])])

    def test_rejects_non_callable_callback(self):
        with ParallelPH(n_workers=1) as ph, pytest.raises(TypeError, match="callback"):
            ph.map(_sum_batch, [np.array([1.0])], callback="bad")

    def test_map_empty_batches(self):
        with ParallelPH(n_workers=1) as ph:
            result = ph.map(_identity, [])
            assert result == []

    def test_map_single_batch_standard(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with ParallelPH(n_workers=1, use_shared_memory=False) as ph:
            result = ph.map(_sum_batch, [data])
            assert len(result) == 1
            assert result[0] == 10.0

    def test_map_multiple_batches_standard(self):
        data = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        with ParallelPH(n_workers=1, use_shared_memory=False) as ph:
            result = ph.map(_sum_batch, data)
            assert len(result) == 2

    def test_map_callback(self):
        callbacks = []

        def cb(completed: int, total: int) -> None:
            callbacks.append((completed, total))

        data = [np.array([[1.0]]), np.array([[2.0]])]
        with ParallelPH(n_workers=1, use_shared_memory=False) as ph:
            ph.map(_sum_batch, data, callback=cb)
        assert len(callbacks) == 2
        assert callbacks[-1] == (2, 2)

    def test_map_shared_memory_mode(self):
        data = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        with ParallelPH(n_workers=1, use_shared_memory=True) as ph:
            result = ph.map(_sum_batch, data)
            assert len(result) == 2

    def test_map_shared_memory_single_batch_falls_back_to_standard(self):
        data = [np.array([[1.0, 2.0]])]
        with ParallelPH(n_workers=1, use_shared_memory=True) as ph:
            result = ph.map(_sum_batch, data)
            assert len(result) == 1
            assert result[0] == 3.0


class TestParallelPHMapUnordered:
    def test_map_unordered_outside_context(self):
        ph = ParallelPH(n_workers=1)
        with pytest.raises(RuntimeError, match="context manager"):
            list(ph.map_unordered(_identity, [np.array([1.0])]))

    def test_rejects_non_callable_compute_fn(self):
        with ParallelPH(n_workers=1) as ph, pytest.raises(TypeError, match="compute_fn"):
            list(ph.map_unordered("bad", [np.array([1.0])]))

    def test_map_unordered_empty_batches(self):
        with ParallelPH(n_workers=1) as ph:
            result = list(ph.map_unordered(_identity, []))
            assert result == []

    def test_map_unordered_returns_iterator(self):
        data = [np.array([[1.0, 2.0]])]
        with ParallelPH(n_workers=1, use_shared_memory=False) as ph:
            result = list(ph.map_unordered(_sum_batch, data))
            assert len(result) == 1


class TestParallelPHStarmap:
    def test_starmap_outside_context(self):
        ph = ParallelPH(n_workers=1)
        with pytest.raises(RuntimeError, match="context manager"):
            ph.starmap(_add_two, [(1.0, 2.0)])

    def test_rejects_non_callable_compute_fn(self):
        with ParallelPH(n_workers=1) as ph, pytest.raises(TypeError, match="compute_fn"):
            ph.starmap("bad", [(1.0,)])

    def test_rejects_none_args_list(self):
        with ParallelPH(n_workers=1) as ph, pytest.raises(TypeError, match="args_list"):
            ph.starmap(_add_two, None)

    def test_starmap_empty_args(self):
        with ParallelPH(n_workers=1) as ph:
            result = ph.starmap(_add_two, [])
            assert result == []

    def test_starmap_with_args(self):
        with ParallelPH(n_workers=1) as ph:
            result = ph.starmap(_add_two, [(1.0, 2.0), (3.0, 4.0)])
            assert result == [3.0, 7.0]


class TestSharedMemoryWorker:
    def test_worker_attaches_and_calls_fn(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        shm = SharedMemoryArray.from_array(data)
        try:
            args = (_sum_batch, shm.name, shm.shape, shm.dtype.str)
            result = _shared_memory_worker(args)
            assert result == 10.0
        finally:
            shm.close()

    def test_worker_closes_shared_memory_on_error(self):
        data = np.array([[1.0, 2.0]])
        shm = SharedMemoryArray.from_array(data)
        try:
            args = (_failing_fn, shm.name, shm.shape, shm.dtype.str)
            with pytest.raises(ValueError, match="worker error"):
                _shared_memory_worker(args)
        finally:
            shm.close()


class TestChunkedParallelInit:
    def test_init_defaults(self):
        cp = ChunkedParallel()
        assert cp.chunk_size == 1000
        assert cp.n_workers >= 1
        assert cp.aggregator is not None

    def test_init_custom_params(self):
        cp = ChunkedParallel(chunk_size=500, n_workers=2)
        assert cp.chunk_size == 500
        assert cp.n_workers == 2

    def test_init_rejects_invalid_chunk_size(self):
        with pytest.raises((TypeError, ValueError)):
            ChunkedParallel(chunk_size=-1)
        with pytest.raises((TypeError, ValueError)):
            ChunkedParallel(chunk_size=0)

    def test_init_rejects_non_callable_aggregator(self):
        with pytest.raises(TypeError, match="aggregator"):
            ChunkedParallel(aggregator="not_callable")

    def test_init_with_custom_aggregator(self):
        cp = ChunkedParallel(aggregator=sum)
        assert cp.aggregator is sum

    def test_repr(self):
        cp = ChunkedParallel(chunk_size=200, n_workers=3)
        r = repr(cp)
        assert "ChunkedParallel" in r
        assert "200" in r
        assert "3" in r


class TestChunkedParallelProcess:
    def test_process_rejects_non_callable_process_fn(self):
        cp = ChunkedParallel()
        data = np.array([[1.0, 2.0]])
        with pytest.raises(TypeError, match="process_fn"):
            cp.process(data, "not_callable")

    def test_process_rejects_non_callable_reduce_fn(self):
        cp = ChunkedParallel()
        data = np.array([[1.0, 2.0]])
        with pytest.raises(TypeError, match="reduce_fn"):
            cp.process(data, _sum_batch, reduce_fn="bad")

    def test_process_empty_data(self):
        cp = ChunkedParallel()
        data = np.zeros((0, 3))
        result = cp.process(data, _sum_batch)
        assert result == []

    def test_process_single_chunk(self):
        cp = ChunkedParallel(chunk_size=5, n_workers=1)
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = cp.process(data, _sum_batch)
        assert len(result) == 1
        assert result[0] == 10.0

    def test_process_multiple_chunks_default_aggregator(self):
        cp = ChunkedParallel(chunk_size=2, n_workers=1)
        data = np.array([[1.0], [2.0], [3.0], [4.0]])
        result = cp.process(data, _sum_batch)
        assert result == [3.0, 7.0]

    def test_process_with_custom_reduce_fn(self):
        cp = ChunkedParallel(chunk_size=2, n_workers=1)
        data = np.array([[1.0], [2.0], [3.0], [4.0]])
        result = cp.process(data, _sum_batch, reduce_fn=sum)
        assert result == 10.0

    def test_process_flattens_list_results(self):
        cp = ChunkedParallel(chunk_size=2, n_workers=1)
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = cp.process(data, _returns_list)
        assert result == [10.0, 2.5, 11.0, 5.5]


class TestChunkedParallelDefaultAggregator:
    def test_flattens_lists(self):
        cp = ChunkedParallel()
        results = [[1, 2], [3, 4]]
        assert cp._default_aggregator(results) == [1, 2, 3, 4]

    def test_passes_through_non_lists(self):
        cp = ChunkedParallel()
        results = [5, 10]
        assert cp._default_aggregator(results) == [5, 10]

    def test_mixed_list_and_non_list(self):
        cp = ChunkedParallel()
        results = [[1], 5]
        assert cp._default_aggregator(results) == [[1], 5]

    def test_empty_results(self):
        cp = ChunkedParallel()
        assert cp._default_aggregator([]) == []


class TestMapReducePH:
    def test_init_rejects_non_callable_map_fn(self):
        with pytest.raises(TypeError, match="callable"):
            MapReducePH("bad", _add_two)

    def test_init_rejects_non_callable_reduce_fn(self):
        with pytest.raises(TypeError, match="callable"):
            MapReducePH(_sum_batch, "bad")

    def test_init_defaults(self):
        mr = MapReducePH(_sum_batch, _add_two)
        assert mr.n_workers >= 1

    def test_init_custom_n_workers(self):
        mr = MapReducePH(_sum_batch, _add_two, n_workers=2)
        assert mr.n_workers == 2

    def test_init_rejects_invalid_n_workers(self):
        with pytest.raises((TypeError, ValueError)):
            MapReducePH(_sum_batch, _add_two, n_workers=-1)

    def test_repr(self):
        mr = MapReducePH(_sum_batch, _add_two, n_workers=2)
        r = repr(mr)
        assert "MapReducePH" in r

    def test_run_rejects_empty_partitions(self):
        mr = MapReducePH(_sum_batch, _add_two)
        with pytest.raises(ValueError, match="non-empty"):
            mr.run([])

    def test_run_single_partition(self):
        mr = MapReducePH(_sum_batch, _add_two, n_workers=1)
        data = np.array([[1.0, 2.0, 3.0]])
        result = mr.run([data])
        assert result == 6.0

    def test_run_multiple_partitions(self):
        mr = MapReducePH(_sum_batch, _add_two, n_workers=1)
        partitions = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        result = mr.run(partitions)
        assert result == 10.0

    def test_run_many_partitions(self):
        mr = MapReducePH(_sum_batch, _add_two, n_workers=1)
        partitions = [np.array([[float(i)]]) for i in range(1, 6)]
        result = mr.run(partitions)
        assert result == 15.0
