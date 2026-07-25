"""Tests for benchmark/_suite.py and benchmark/_compare_internal.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestJsonReady:
    """Covers _suite._json_ready."""

    def test_simple_types(self):
        from pynerve.benchmark._suite import _json_ready
        assert _json_ready(42) == 42
        assert _json_ready("hello") == "hello"
        assert _json_ready(3.14) == 3.14

    def test_inf_float(self):
        from pynerve.benchmark._suite import _json_ready
        assert _json_ready(float("inf")) is None
        assert _json_ready(float("nan")) is None

    def test_dict(self):
        from pynerve.benchmark._suite import _json_ready
        result = _json_ready({"a": 1, "b": float("inf")})
        assert result == {"a": 1, "b": None}

    def test_list(self):
        from pynerve.benchmark._suite import _json_ready
        result = _json_ready([1, 2, float("inf")])
        assert result == [1, 2, None]

    def test_object_with_dict(self):
        from pynerve.benchmark._suite import _json_ready

        class Obj:
            def __init__(self):
                self.x = 1
                self.y = 2

        result = _json_ready(Obj())
        assert result == {"x": 1, "y": 2}


class TestRunFullBenchmarkSuite:
    """Covers _suite.run_full_benchmark_suite."""

    def test_basic(self):
        from pynerve.benchmark._suite import run_full_benchmark_suite
        pytest.skip("run_full_benchmark_suite calls GUDHI persistence which is too slow for unit tests")

    def test_with_output_file(self, tmp_path):
        from pynerve.benchmark._suite import run_full_benchmark_suite
        pytest.skip("run_full_benchmark_suite calls GUDHI persistence which is too slow for unit tests")

    def test_empty_output_file_raises(self):
        from pynerve.benchmark._suite import run_full_benchmark_suite
        with pytest.raises(ValueError, match="non-empty"):
            run_full_benchmark_suite(output_file="")


class TestMakeComparisonFn:
    """Covers _suite._make_comparison_fn."""

    def test_ripser_fn(self):
        from pynerve.benchmark._suite import _make_comparison_fn
        fn = _make_comparison_fn("reference", "spheres", 100)
        assert callable(fn)

    def test_gudhi_fn(self):
        from pynerve.benchmark._suite import _make_comparison_fn
        fn = _make_comparison_fn("gudhi", "spheres", 100)
        assert callable(fn)

    def test_dionysus_fn(self):
        from pynerve.benchmark._suite import _make_comparison_fn
        fn = _make_comparison_fn("dionysus", "spheres", 100)
        assert callable(fn)

    def test_unknown_comparison(self):
        from pynerve.benchmark._suite import _make_comparison_fn
        with pytest.raises(ValueError, match="Unknown"):
            _make_comparison_fn("unknown", "spheres", 100)


class TestBenchmarkWithFallback:
    """Covers _suite._benchmark_with_fallback."""

    def test_success(self):
        from pynerve.benchmark._suite import _benchmark_with_fallback
        result = _benchmark_with_fallback(lambda: 42)
        assert result == 42

    def test_import_error(self):
        from pynerve.benchmark._suite import _benchmark_with_fallback

        def raise_import():
            raise ImportError("missing")

        result = _benchmark_with_fallback(raise_import)
        assert isinstance(result, str)
        assert "missing" in result


class TestResolveOutputFile:
    """Covers _suite._resolve_output_file."""

    def test_none(self):
        from pynerve.benchmark._suite import _resolve_output_file
        assert _resolve_output_file(None) is None

    def test_valid(self):
        from pynerve.benchmark._suite import _resolve_output_file
        result = _resolve_output_file("/tmp/test.json")
        assert isinstance(result, Path)

    def test_empty_raises(self):
        from pynerve.benchmark._suite import _resolve_output_file
        with pytest.raises(ValueError, match="non-empty"):
            _resolve_output_file("")


class TestBenchmarkGpuVsCpu:
    """Covers _compare_internal.benchmark_gpu_vs_cpu."""

    def test_basic(self):
        from pynerve.benchmark._compare_internal import benchmark_gpu_vs_cpu
        result = benchmark_gpu_vs_cpu(n_samples=50, max_dim=1)
        assert result.library1 == "Nerve CPU"
        assert result.library2 == "Nerve GPU"
        assert result.n_samples == 50

    def test_invalid_n_samples(self):
        from pynerve.benchmark._compare_internal import benchmark_gpu_vs_cpu
        with pytest.raises(ValueError, match="positive"):
            benchmark_gpu_vs_cpu(n_samples=0)

    def test_invalid_max_dim(self):
        from pynerve.benchmark._compare_internal import benchmark_gpu_vs_cpu
        with pytest.raises(ValueError, match="non-negative"):
            benchmark_gpu_vs_cpu(max_dim=-1)


class TestBenchmarkStreamingPersistence:
    """Covers _compare_internal.benchmark_streaming_persistence."""

    def test_basic(self):
        from pynerve.benchmark._compare_internal import benchmark_streaming_persistence
        result = benchmark_streaming_persistence(dataset="spheres", n_samples=100, chunk_size=50, n_runs=1)
        assert result.library1 == "Nerve (full)"
        assert result.library2 == "Nerve (streaming)"

    def test_invalid_dataset(self):
        from pynerve.benchmark._compare_internal import benchmark_streaming_persistence
        with pytest.raises(ValueError, match="non-empty"):
            benchmark_streaming_persistence(dataset="")

    def test_invalid_chunk_size(self):
        from pynerve.benchmark._compare_internal import benchmark_streaming_persistence
        with pytest.raises(ValueError, match="positive"):
            benchmark_streaming_persistence(chunk_size=0)


class TestBenchmarkPersistenceImage:
    """Covers _compare_internal.benchmark_persistence_image."""

    def test_basic(self):
        from pynerve.benchmark._compare_internal import benchmark_persistence_image
        result = benchmark_persistence_image(dataset="spheres", n_samples=50, n_runs=1)
        assert "mean_time" in result
        assert "std_time" in result
        assert "n_points" in result

    def test_invalid_n_samples(self):
        from pynerve.benchmark._compare_internal import benchmark_persistence_image
        with pytest.raises(ValueError, match="positive"):
            benchmark_persistence_image(n_samples=0)

    def test_invalid_n_runs(self):
        from pynerve.benchmark._compare_internal import benchmark_persistence_image
        with pytest.raises(ValueError, match="positive"):
            benchmark_persistence_image(n_runs=0)
