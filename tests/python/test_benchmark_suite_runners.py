"""Tests for benchmark/_suite.py — run_full_benchmark_suite, run_full_cross_comparison, and helpers.

All benchmark functions are mocked so tests verify orchestration logic, not real computations.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from pynerve.benchmark._comparison_types import BenchmarkComparison, GPUComparison
from pynerve.benchmark._scalability import ScalabilityResult
from pynerve.benchmark._suite import (
    _benchmark_with_fallback,
    _json_ready,
    _make_comparison_fn,
    _make_distance_fn,
    _make_streaming_fn,
    _resolve_output_file,
    _run_cross_benchmarks,
    _run_cross_comparisons,
    _run_scalability_and_streaming,
    run_full_benchmark_suite,
    run_full_cross_comparison,
)


def _mock_comparison(*args, **kwargs):
    return BenchmarkComparison(
        library1="nerve", library2="mock", dataset="test",
        n_samples=100, time1=0.01, time2=0.02, speedup=2.0,
    )


def _mock_scalability(*args, **kwargs):
    return ScalabilityResult(n_samples=[100, 200], times=[0.01, 0.02])


def _mock_gpu_comparison(*args, **kwargs):
    return GPUComparison(
        library1="cpu", library2="cuda", dataset="test",
        n_samples=100, time1=0.1, time2=0.01, speedup=10.0,
        gpu_available=True,
    )


def _mock_simple_dict(*args, **kwargs):
    return {"time": 0.01, "result": "ok"}


def _mock_persistence_image_benchmark(*args, **kwargs):
    return {"image_time": 0.05, "shape": [10, 10]}


@pytest.fixture(autouse=True)
def _mock_benchmark_functions():
    """Mock all slow benchmark functions so tests run instantly."""
    mock_map = {
        "benchmark_vs_ripser": _mock_comparison,
        "benchmark_vs_gudhi": _mock_comparison,
        "benchmark_vs_dionysus": _mock_comparison,
        "benchmark_scalability": _mock_scalability,
        "benchmark_complexity_analysis": _mock_scalability,
        "benchmark_gpu_vs_cpu": _mock_gpu_comparison,
        "benchmark_streaming_persistence": _mock_simple_dict,
        "benchmark_persistence_image": _mock_persistence_image_benchmark,
        "benchmark_distance_matrix": _mock_simple_dict,
        "benchmark_witness_complex": _mock_simple_dict,
    }
    patches = [patch(f"pynerve.benchmark._suite.{name}", side_effect=fn)
               for name, fn in mock_map.items()]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


class TestJsonReadyEdgeCases:
    def test_nested_list(self):
        assert _json_ready([{"a": [1, 2]}, {"b": [3]}]) == [{"a": [1, 2]}, {"b": [3]}]

    def test_object_with_no_dict_attr(self):
        class Simple:
            pass

        result = _json_ready(Simple())
        assert isinstance(result, dict)

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": [1, {"d": np.float64(3.0)}]}}}
        result = _json_ready(data)
        assert result == {"a": {"b": {"c": [1, {"d": 3.0}]}}}

    def test_float_negative_inf(self):
        assert _json_ready(float("-inf")) is None

    def test_numpy_scalar_to_python(self):
        assert _json_ready(np.float64(3.14)) == 3.14
        assert _json_ready(np.int64(42)) == 42


class TestBenchmarkWithFallback:
    def test_returns_json_ready(self):
        result = _benchmark_with_fallback(lambda: {"x": np.float64(1.0)})
        assert result == {"x": 1.0}

    def test_catches_import_error(self):
        def raise_import():
            raise ImportError("missing lib")

        result = _benchmark_with_fallback(raise_import)
        assert "missing lib" in result

    def test_catches_runtime_error(self):
        def raise_runtime():
            raise RuntimeError("boom")

        result = _benchmark_with_fallback(raise_runtime)
        assert "boom" in result

    def test_catches_value_error(self):
        def raise_value():
            raise ValueError("bad value")

        result = _benchmark_with_fallback(raise_value)
        assert "bad value" in result


class TestMakeComparisonFn:
    def test_reference_callable(self):
        fn = _make_comparison_fn("reference", "spheres", 100)
        assert callable(fn)

    def test_gudhi_callable(self):
        fn = _make_comparison_fn("gudhi", "torus", 200)
        assert callable(fn)

    def test_dionysus_callable(self):
        fn = _make_comparison_fn("dionysus", "swiss_roll", 300)
        assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown comparison"):
            _make_comparison_fn("invalid", "spheres", 100)


class TestResolveOutputFile:
    def test_none_returns_none(self):
        assert _resolve_output_file(None) is None

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _resolve_output_file("")

    def test_valid_path(self):
        p = _resolve_output_file("/tmp/test.json")
        assert str(p) == "/tmp/test.json"


class TestMakeDistanceFn:
    def test_euclidean_callable(self):
        fn = _make_distance_fn("euclidean")
        assert callable(fn)

    def test_manhattan_callable(self):
        fn = _make_distance_fn("manhattan")
        assert callable(fn)


class TestMakeStreamingFn:
    def test_callable(self):
        fn = _make_streaming_fn("spheres")
        assert callable(fn)

    def test_torus_callable(self):
        fn = _make_streaming_fn("torus")
        assert callable(fn)


class TestRunFullBenchmarkSuite:
    def test_returns_dict_with_comparisons_and_scalability(self):
        results = run_full_benchmark_suite()
        assert "comparisons" in results
        assert "scalability" in results

    def test_writes_json_output(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            run_full_benchmark_suite(output_file=tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert "comparisons" in loaded
            assert "scalability" in loaded
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_comparison_keys_present(self):
        results = run_full_benchmark_suite()
        comparisons = results["comparisons"]
        assert any("reference_" in k for k in comparisons)
        assert any("gudhi_" in k for k in comparisons)

    def test_scalability_keys_present(self):
        results = run_full_benchmark_suite()
        assert "spheres" in results["scalability"]
        assert "torus" in results["scalability"]


class TestRunFullCrossComparison:
    def test_returns_all_categories(self):
        results = run_full_cross_comparison()
        for cat in [
            "comparisons", "scalability", "streaming", "gpu",
            "distance", "witness", "persistence_image", "complexity",
        ]:
            assert cat in results, f"missing category: {cat}"

    def test_comparison_keys_include_dionysus(self):
        results = run_full_cross_comparison()
        comparisons = results["comparisons"]
        assert any("dionysus_" in k for k in comparisons)

    def test_writes_json_output(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            run_full_cross_comparison(output_file=tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert "comparisons" in loaded
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_distance_keys_present(self):
        results = run_full_cross_comparison()
        assert "euclidean" in results["distance"]
        assert "manhattan" in results["distance"]


class TestRunCrossComparisonsHelper:
    def test_populates_results_dict(self):
        results: dict = {"comparisons": {}}
        _run_cross_comparisons(results)
        comp = results["comparisons"]
        assert len(comp) > 0
        assert any("reference_spheres_500" in k for k in comp)
        assert any("gudhi_torus_1000" in k for k in comp)
        assert any("dionysus_swiss_roll_500" in k for k in comp)


class TestRunScalabilityAndStreamingHelper:
    def test_populates_results(self):
        results: dict = {"scalability": {}, "streaming": {}}
        _run_scalability_and_streaming(results)
        assert "spheres" in results["scalability"]
        assert "torus" in results["scalability"]
        assert "spheres" in results["streaming"]
        assert "torus" in results["streaming"]


class TestRunCrossBenchmarksHelper:
    def test_populates_results(self):
        results: dict = {
            "gpu": {}, "distance": {}, "witness": {},
            "persistence_image": {}, "complexity": {},
        }
        _run_cross_benchmarks(results)
        assert "euclidean" in results["distance"]
        assert "manhattan" in results["distance"]
        assert "spheres" in results["witness"]
        assert "spheres" in results["persistence_image"]
        assert "spheres" in results["complexity"]
        assert "gpu" in results
