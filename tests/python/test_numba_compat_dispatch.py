"""Tests for _numba_compat.py, _numba_dispatch.py, _numba_distance.py."""

from __future__ import annotations

import numpy as np
import pytest

# No mock_gpu_deps — numba tests need real numba or the fallback shim, not a MagicMock


class TestNumbaCompat:
    """Covers _numba_compat.py."""

    def test_has_numba_flag(self):
        from pynerve._numba_compat import HAS_NUMBA
        assert isinstance(HAS_NUMBA, bool)

    def test_njit_decorator(self):
        from pynerve._numba_compat import njit

        @njit
        def my_func(x):
            return x + 1

        assert my_func(5) == 6

    def test_njit_decorator_with_kwargs(self):
        from pynerve._numba_compat import njit

        @njit(cache=True)
        def my_func(x):
            return x * 2

        assert my_func(10) == 20

    def test_prange(self):
        from pynerve._numba_compat import prange
        assert list(prange(5)) == [0, 1, 2, 3, 4]


class TestNumbaDistance:
    """Covers _numba_distance.py."""

    def test_pairwise_distances(self):
        from pynerve._numba_distance import numba_pairwise_distances
        points = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
        result = numba_pairwise_distances(points)
        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0
        assert abs(result[0, 1] - 5.0) < 1e-6
        assert result[0, 2] == 0.0

    def test_pairwise_distances_single_point(self):
        from pynerve._numba_distance import numba_pairwise_distances
        points = np.array([[1.0, 2.0, 3.0]])
        result = numba_pairwise_distances(points)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_nearest_neighbors(self):
        from pynerve._numba_distance import numba_nearest_neighbors
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        dists, indices = numba_nearest_neighbors(points, k=2)
        assert dists.shape == (3, 2)
        assert indices.shape == (3, 2)
        assert dists[0, 0] <= dists[0, 1]

    def test_nearest_neighbors_k_zero(self):
        from pynerve._numba_distance import numba_nearest_neighbors
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        dists, indices = numba_nearest_neighbors(points, k=0)
        assert dists.shape == (2, 0)
        assert indices.shape == (2, 0)


class TestNumbaDispatch:
    """Covers _numba_dispatch.py."""

    def test_compute_with_numba_pairwise(self):
        from pynerve._numba_dispatch import compute_with_numba
        points = np.random.rand(5, 3)
        result = compute_with_numba("pairwise_distances", points)
        assert result.shape == (5, 5)

    def test_compute_with_numba_unknown_op(self):
        from pynerve._numba_dispatch import compute_with_numba
        with pytest.raises(ValueError, match="Unknown operation"):
            compute_with_numba("nonexistent", 1, 2)

    def test_compute_with_numba_empty_op(self):
        from pynerve._numba_dispatch import compute_with_numba
        with pytest.raises(ValueError, match="non-empty"):
            compute_with_numba("", 1)

    def test_compute_with_numba_non_string_op(self):
        from pynerve._numba_dispatch import compute_with_numba
        with pytest.raises(ValueError, match="non-empty"):
            compute_with_numba(None, 1)

    def test_benchmark_numba_vs_numpy(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy

        def setup():
            data = np.random.rand(10, 3)
            return (data,)

        def func_numba(data):
            return np.sum(data)

        def func_numpy(data):
            return np.sum(data)

        result = benchmark_numba_vs_numpy(func_numba, func_numpy, setup, n_trials=2)
        assert "numba_mean" in result
        assert "numpy_mean" in result
        assert "speedup" in result
        assert "results_match" in result
        assert result["results_match"] is True

    def test_benchmark_invalid_func_numba(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy
        with pytest.raises(TypeError, match="func_numba"):
            benchmark_numba_vs_numpy("not callable", lambda x: x, lambda: (1,))

    def test_benchmark_invalid_func_numpy(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy
        with pytest.raises(TypeError, match="func_numpy"):
            benchmark_numba_vs_numpy(lambda x: x, "not callable", lambda: (1,))

    def test_benchmark_invalid_setup(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy
        with pytest.raises(TypeError, match="setup_fn"):
            benchmark_numba_vs_numpy(lambda x: x, lambda x: x, "not callable")

    def test_benchmark_setup_returns_non_tuple(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy
        with pytest.raises(TypeError, match="tuple"):
            benchmark_numba_vs_numpy(lambda x: x, lambda x: x, lambda: 42)

    def test_benchmark_invalid_n_trials(self):
        from pynerve._numba_dispatch import benchmark_numba_vs_numpy
        with pytest.raises(Exception, match="positive"):
            benchmark_numba_vs_numpy(
                lambda x: x, lambda x: x, lambda: (1,), n_trials=0
            )
