from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _source_path(*parts: str) -> Path:
    """Resolve a source path, with a friendly message if the file is missing."""
    path = ROOT.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"Source file not found: {path}. Skipping test.")
    return path


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"nerve_{name}", ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"could not load tool module {name}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TOOL_CACHE: dict[str, Any] = {}


def _get_tool(name: str):
    if name not in _TOOL_CACHE:
        _TOOL_CACHE[name] = _load_tool(name)
    return _TOOL_CACHE[name]


def run_tests():
    return _get_tool("run_tests")


def backend_checks():
    return _get_tool("backend_checks")


def cuda_launch_audit():
    return _get_tool("cuda_launch_audit")


@pytest.mark.quality
def test_async_executor_benchmark_checks_cuda_status_and_finite_speedup() -> None:
    text = (_source_path("src", "gpu", "executor_async_ops.cpp")).read_text(encoding="utf-8")

    assert "async benchmark iterations must be positive" in text, "missing async iterations check"
    assert "cudaDeviceSynchronize() != cudaSuccess" in text, "missing cudaDeviceSynchronize check"
    assert "cudaStreamCreate(&streams[i]) != cudaSuccess" in text, "missing cudaStreamCreate check"
    assert "bench.speedup = bench.sync_time_ms / bench.async_time_ms" not in text, (
        "found unsafe async speedup"
    )
    assert ": 1.0;" in text, "missing ': 1.0;' fallback in async executor"


@pytest.mark.quality
def test_cuda_error_mapping_metrics_reject_non_finite_timing_stats() -> None:
    text = (_source_path("src", "persistence", "cuda", "error_mapping_cuda.cpp")).read_text(
        encoding="utf-8"
    )

    assert "const double elapsed_ms = profiled_ms.value()" in text, (
        "missing elapsed_ms profiled_ms in error mapping"
    )
    assert "!std::isfinite(elapsed_ms) || elapsed_ms < 0.0" in text, (
        "missing non-finite elapsed_ms check"
    )
    assert "cudaEventRecord(start_event) != cudaSuccess" in text, (
        "missing cudaEventRecord start check"
    )
    assert "cudaEventRecord(stop_event) != cudaSuccess" in text, (
        "missing cudaEventRecord stop check"
    )
    assert "stats.gpu_time_ms > 0.0 ? (stats.gpu_time_ms + stats.cpu_time_ms)" not in text, (
        "found unsafe stats ratio"
    )
    assert "const bool finite_stats =" in text, "missing finite_stats flag"
    assert ": 1.0;" in text, "missing ': 1.0;' fallback in error mapping"


@pytest.mark.quality
def test_regularization_gpu_benchmarks_reject_invalid_inputs_and_keep_finite_speedups() -> None:
    augmentation = (
        _source_path("src", "regularization", "augmentation_gpu_benchmark.inl")
    ).read_text(encoding="utf-8")
    loss = (_source_path("src", "regularization", "loss_kernels_benchmark.inl")).read_text(
        encoding="utf-8"
    )
    combined = "\n".join([augmentation, loss])

    assert "quiet_NaN" not in combined, "found quiet_NaN in regularization benchmarks"
    assert "numeric_limits<double>::infinity()" not in combined, (
        "found infinity() in regularization benchmarks"
    )
    assert "augmentation benchmark dimensions are invalid" in augmentation, (
        "missing dimensions check in augmentation"
    )
    assert "augmentation benchmark input size overflows" in augmentation, (
        "missing overflow check in augmentation"
    )
    assert "loss benchmark sizes must be non-negative" in loss, "missing non-negative check in loss"
    assert combined.count("return 1.0;") >= 2, (
        f"expected >=2 'return 1.0;', found {combined.count('return 1.0;')}"
    )


@pytest.mark.quality
def test_ml_diff_gradient_benchmark_uses_finite_speedup_default() -> None:
    text = (_source_path("src", "ml", "diff", "simplex_autodiff.cu")).read_text(encoding="utf-8")

    assert "quiet_NaN" not in text, "found quiet_NaN in simplex_autodiff"
    assert "double speedup = 1.0" in text, "missing double speedup = 1.0 in simplex_autodiff"
    assert "gradient benchmark produced non-finite output" in text, (
        "missing non-finite output check in simplex_autodiff"
    )


@pytest.mark.quality
def test_specialized_gpu_benchmarks_use_finite_speedup_fallbacks() -> None:
    zigzag = (_source_path("src", "specialized", "zigzag_persistence_benchmark.inl")).read_text(
        encoding="utf-8"
    )
    cup = (_source_path("src", "specialized", "cup_product_benchmark.inl")).read_text(
        encoding="utf-8"
    )
    combined = "\n".join([zigzag, cup])

    assert "std::numeric_limits<double>::infinity()" not in combined, (
        "found infinity() in specialized benchmarks"
    )
    assert "cohomology benchmark sizes are invalid" in cup, (
        "missing sizes check in cohomology benchmark"
    )
    assert combined.count("return 1.0;") >= 2, (
        f"expected >=2 'return 1.0;', found {combined.count('return 1.0;')}"
    )
    assert combined.count("bench.speedup = ratio") == 2, (
        f"expected exactly 2 'bench.speedup = ratio', found {combined.count('bench.speedup = ratio')}"
    )


@pytest.mark.quality
def test_validation_microbenchmarks_use_finite_condition_estimates() -> None:
    core = (_source_path("src", "validation", "ph5_ph6_microbenchmarks.cpp")).read_text(
        encoding="utf-8"
    )
    spectral = (
        _source_path("src", "validation", "ph5_ph6_microbenchmarks_spectral.cpp")
    ).read_text(encoding="utf-8")
    combined = "\n".join([core, spectral])

    assert "condition_estimate = std::numeric_limits<double>::infinity()" not in combined, (
        "found infinity() condition estimate"
    )
    assert "finiteConditionEstimate" in core, "expected finiteConditionEstimate in core"
    assert "sanitizeConditionEstimate" in spectral, "expected sanitizeConditionEstimate in spectral"
    assert combined.count("result.condition_estimate = 1.0") >= 2, (
        f"expected >=2 'condition_estimate = 1.0', found {combined.count('result.condition_estimate = 1.0')}"
    )


@pytest.mark.quality
def test_sheaf_learning_benchmark_uses_finite_sdp_backing_tensors() -> None:
    text = (_source_path("src", "sheaf", "sheaf_learning.cpp")).read_text(encoding="utf-8")

    assert "benchmark.sdp_time_ms = std::numeric_limits<double>::quiet_NaN()" not in text, (
        "found NaN sdp_time_ms"
    )
    assert "benchmark.speedup_factor = std::numeric_limits<double>::quiet_NaN()" not in text, (
        "found NaN speedup_factor"
    )
    assert "benchmark.speedup_factor = 1.0" in text or "speedup" not in text, (
        "missing speedup_factor = 1.0"
    )
    assert "benchmark.accuracy_ratio = 1.0" in text or "accuracy" not in text, (
        "missing accuracy_ratio = 1.0"
    )
