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
def test_calibration_gpu_benchmark_keeps_finite_timing_results() -> None:
    text = (_source_path("src", "runtime", "calibration_gpu_benchmark.inl")).read_text(
        encoding="utf-8"
    )

    assert "quiet_NaN" not in text, "found quiet_NaN in calibration_gpu_benchmark.inl"
    assert "numeric_limits<double>::infinity()" not in text, (
        "found infinity() in calibration_gpu_benchmark.inl"
    )
    assert "return 1.0;" in text, "missing 'return 1.0;' in calibration_gpu_benchmark.inl"


@pytest.mark.quality
def test_gpu_autoencoder_benchmarks_keep_finite_optional_results() -> None:
    texts = [
        (_source_path("src", "compression", "autoencoder_gpu_benchmark.inl")).read_text(
            encoding="utf-8"
        ),
        (_source_path("src", "compression", "encoder_cudnn.cpp")).read_text(encoding="utf-8"),
        (_source_path("src", "compression", "decoder_cudnn.cpp")).read_text(encoding="utf-8"),
    ]
    combined = "\n".join(texts)

    assert "quiet_NaN" not in combined, "found quiet_NaN in autoencoder benchmarks"
    assert "numeric_limits<double>::infinity()" not in combined, (
        "found infinity() in autoencoder benchmarks"
    )
    assert "speedup_fused = 1.0" in combined, (
        "missing 'speedup_fused = 1.0' in autoencoder benchmarks"
    )


@pytest.mark.quality
def test_warp_specialization_benchmark_uses_neutral_finite_defaults() -> None:
    source = (
        _source_path("src", "persistence", "cuda", "kernel_warp_specialized_cuda.cu")
    ).read_text(encoding="utf-8")
    header = (
        _source_path("src", "include", "nerve", "persistence", "cuda")
        / "cuda_warp_specialized_kernels.hpp"
    ).read_text(encoding="utf-8")

    assert "quiet_NaN" not in source, "found quiet_NaN in warp_specialized source"
    assert "may be NaN" not in header, "found 'may be NaN' in warp_specialized header"
    assert "bench.total_speedup = 1.0" in source, (
        "missing 'bench.total_speedup = 1.0' in warp_specialized source"
    )


@pytest.mark.quality
def test_blackwell_benchmark_uses_finite_empty_defaults() -> None:
    text = (_source_path("src", "persistence", "cuda", "cuda_blackwell_benchmark.cu")).read_text(
        encoding="utf-8"
    )

    assert "quiet_NaN" not in text, "found quiet_NaN in blackwell benchmark"
    assert "result.speedup = 1.0" in text, "missing 'result.speedup = 1.0' in blackwell benchmark"
    assert "std::isfinite(result.blackwell_time_ms)" in text, (
        "missing isfinite check in blackwell benchmark"
    )


@pytest.mark.quality
def test_tensor_core_benchmark_uses_finite_defaults_and_error_metrics() -> None:
    header = (
        _source_path("src", "include", "nerve", "persistence", "cuda") / "cuda_tensor_core.hpp"
    ).read_text(encoding="utf-8")
    source = (_source_path("src", "persistence", "cuda", "tensor_core_benchmark.cu")).read_text(
        encoding="utf-8"
    )

    assert "quiet_NaN" not in header, "found quiet_NaN in tensor_core header"
    assert "result.max_relative_error = std::numeric_limits<double>::infinity()" not in source, (
        "found infinity() in tensor_core source"
    )
    assert "result.mean_relative_error = std::numeric_limits<double>::infinity()" not in source, (
        "found infinity() in tensor_core source mean"
    )
    assert "double speedup = 1.0" in header, "missing 'double speedup = 1.0' in tensor_core header"
    assert "double max_relative_error = 0.0" in header, (
        "missing max_relative_error default in tensor_core header"
    )
    assert "result.max_relative_error = 1.0" in source, (
        "missing max_relative_error = 1.0 in tensor_core source"
    )
    assert "result.speedup = 1.0" in source, "missing 'result.speedup = 1.0' in tensor_core source"


@pytest.mark.quality
def test_adaptive_selector_benchmark_uses_neutral_finite_speedups() -> None:
    text = (
        _source_path("src", "persistence", "cuda", "detail", "adaptive_selector_benchmark.inl")
    ).read_text(encoding="utf-8")

    assert "quiet_NaN" not in text, "found quiet_NaN in adaptive_selector benchmark"
    assert "bench.speedup_feature = 1.0" in text, (
        "missing bench.speedup_feature = 1.0 in adaptive_selector"
    )
    assert "bench.speedup_predict = 1.0" in text, (
        "missing bench.speedup_predict = 1.0 in adaptive_selector"
    )


@pytest.mark.quality
def test_graph_gpu_benchmarks_use_neutral_finite_optional_timings() -> None:
    texts = [
        (_source_path("src", "graphs", "attention_gpu.cu")).read_text(encoding="utf-8"),
        (_source_path("src", "graphs", "message_passing_gpu.cu")).read_text(encoding="utf-8"),
    ]
    combined = "\n".join(texts)

    assert "quiet_NaN" not in combined, "found quiet_NaN in graph GPU benchmarks"
    assert "bench.tensor_core_time_ms = 0.0" in combined, (
        "missing tensor_core_time_ms = 0.0 in graph benchmarks"
    )
    assert "bench.gpu_fp16_time_ms = 0.0" in combined, (
        "missing gpu_fp16_time_ms = 0.0 in graph benchmarks"
    )
    assert combined.count("bench.speedup = 1.0") >= 2, (
        f"expected at least 2 'bench.speedup = 1.0', found {combined.count('bench.speedup = 1.0')}"
    )


@pytest.mark.quality
def test_gpu_benchmark_speedups_use_finite_neutral_ratios() -> None:
    paths = [
        _source_path("src", "probabilistic", "probabilistic_gpu.cu"),
        _source_path("src", "streaming", "gpu", "windowed_ph_cuda.cu"),
        _source_path("src", "filtration", "gpu", "vr_sparse_cuda.cu"),
        _source_path("src", "encoders", "encoder_gpu_kernels.cu"),
        _source_path("src", "encoders", "encoder_gpu_kernels_benchmark.inl"),
    ]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    combined = "\n".join(texts)

    assert all("finiteBenchmarkSpeedup" in text for text in texts), (
        "all GPU benchmark sources must contain finiteBenchmarkSpeedup"
    )
    assert (
        "bench.gpu_time_ms > 0.0 ? bench.cpu_time_ms / bench.gpu_time_ms : 0.0" not in combined
    ), "found unsafe ratio in GPU benchmarks"
    assert (
        "bench.fused_time_ms > 0.0 ? bench.cpu_time_ms / bench.fused_time_ms : 0.0" not in combined
    ), "found unsafe ratio in fused benchmarks"
    assert combined.count("return 1.0;") >= 4, (
        f"expected at least 4 'return 1.0;', found {combined.count('return 1.0;')}"
    )


@pytest.mark.quality
def test_cpu_distributed_benchmark_speedups_use_finite_neutral_ratios() -> None:
    paths = [
        _source_path("src", "metrics", "wasserstein", "distance_sinkhorn_ops.cpp"),
        _source_path("src", "filtration", "vr", "vr_ann_search_ops.cpp"),
        _source_path("src", "metrics", "matrix", "matrix_distance_avx512_ops.cpp"),
        _source_path("src", "distributed", "distributed_persistence.cpp"),
    ]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    combined = "\n".join(texts)

    assert all("finiteBenchmarkSpeedup" in text for text in texts), (
        "all CPU distributed sources must contain finiteBenchmarkSpeedup"
    )
    assert "std::numeric_limits<double>::infinity()" not in texts[1], (
        "found infinity() in vr_ann_search_ops.cpp"
    )
    assert (
        "bench.sinkhorn_time_ms > 0.0 ? bench.sliced_time_ms / bench.sinkhorn_time_ms : 0.0"
        not in combined
    ), "found unsafe sinkhorn ratio in combined sources"
    assert "bench.ann_time_ms > 0.0 ? bench.exact_time_ms / bench.ann_time_ms" not in combined, (
        "found unsafe ann ratio"
    )
    assert (
        "bench.speedup = bench.single_node_time_ms / bench.distributed_time_ms" not in combined
    ), "found unsafe distributed ratio"
    assert combined.count("return 1.0;") >= 4, (
        f"expected at least 4 'return 1.0;', found {combined.count('return 1.0;')}"
    )


@pytest.mark.quality
def test_simd_reduction_benchmark_helpers_keep_finite_ratios() -> None:
    simd = (_source_path("src", "include", "distance", "simd_distance.hpp")).read_text(
        encoding="utf-8"
    )
    avx512 = (_source_path("src", "persistence", "utils", "avx512_optimizer.cpp")).read_text(
        encoding="utf-8"
    )
    sparsity = (
        _source_path("src", "include", "nerve", "persistence", "reduction")
        / "reduction_sparsity_aware.hpp"
    ).read_text(encoding="utf-8")
    early_exit = (
        _source_path("src", "include", "nerve", "persistence", "utils") / "early_exit_optimizer.hpp"
    ).read_text(encoding="utf-8")
    combined = "\n".join([simd, avx512, sparsity, early_exit])

    assert "finiteSpeedup" in simd, "expected 'finiteSpeedup' in simd_distance.hpp"
    assert combined.count("finiteBenchmarkSpeedup") >= 3, (
        f"expected >=3 finiteBenchmarkSpeedup, found {combined.count('finiteBenchmarkSpeedup')}"
    )
    assert "distance benchmark iterations must be positive" in simd, (
        "missing distance benchmark iterations check in simd"
    )
    assert "sparsity benchmark iterations must be positive" in sparsity, (
        "missing sparsity benchmark iterations check"
    )
    assert "early exit benchmark iterations must be positive" in early_exit, (
        "missing early exit benchmark iterations check"
    )
    assert ".speedup = scalar_time / simd_time" not in simd, "found unsafe simd speedup ratio"
    assert "bench.speedup = bench.scalar_time_ms / bench.avx512_time_ms" not in avx512, (
        "found unsafe avx512 speedup ratio"
    )
    assert (
        "bench.speedup = bench.standard_time_ms / bench.sparsity_aware_time_ms" not in sparsity
    ), "found unsafe sparsity speedup ratio"
    assert "bench.speedup = bench.standard_time_ms / bench.early_exit_time_ms" not in early_exit, (
        "found unsafe early exit speedup ratio"
    )


@pytest.mark.quality
def test_encoder_spectral_benchmark_speedups_use_finite_neutral_ratios() -> None:
    paths = [
        _source_path("src", "encoders", "encoder_fusion.cpp"),
        _source_path("src", "encoders", "encoder_tensor_cores_benchmark.cpp"),
        _source_path("src", "spectral", "eigensolver_gpu.cu"),
        _source_path("src", "spectral", "eigensolver_gpu_benchmark.inl"),
        _source_path("src", "spectral", "dirac_operator_gpu.cu"),
    ]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    combined = "\n".join(texts)

    assert combined.count("finiteBenchmarkSpeedup") >= 5, (
        f"expected >=5, found {combined.count('finiteBenchmarkSpeedup')}"
    )
    assert "tensor core benchmark dimensions must be positive" in texts[1], (
        "missing dimensions check in tensor core benchmarks"
    )
    assert (
        "bench.fused_time_ms > 0.0 ? bench.unfused_time_ms / bench.fused_time_ms : 0.0"
        not in combined
    ), "found unsafe fused ratio in encoder/spectral"
    assert (
        "bench.fp16_time_ms > 0.0 ? bench.fp32_time_ms / bench.fp16_time_ms : 0.0" not in combined
    ), "found unsafe fp16 ratio in encoder/spectral"
    assert (
        "bench.gpu_time_ms > 0.0 ? bench.cpu_time_ms / bench.gpu_time_ms : 0.0" not in combined
    ), "found unsafe gpu ratio in encoder/spectral"
    assert "bench.speedup = bench.cpu_time_ms / bench.gpu_time_ms" not in combined, (
        "found unsafe speedup assignment in encoder/spectral"
    )


@pytest.mark.quality
def test_parallel_benchmark_speedups_use_finite_neutral_ratios() -> None:
    paths = [
        _source_path("src", "dmt", "discrete_gradient_parallel.cpp"),
        _source_path("src", "sheaf", "sheaf_construction_parallel.cpp"),
        _source_path("src", "sheaf", "sheaf_laplacian_gpu.cu"),
        _source_path("src", "sheaf", "sheaf_laplacian_gpu_multi.inl"),
        _source_path("src", "persistence", "utils", "cpp20_parallel_ph.cpp"),
    ]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    combined = "\n".join(texts)

    assert combined.count("finiteBenchmarkSpeedup") >= 5, (
        f"expected >=5, found {combined.count('finiteBenchmarkSpeedup')}"
    )
    assert "parallel Morse benchmark produced inconsistent pair count" in texts[0], (
        "missing inconsistent pair count check in parallel Morse"
    )
    assert (
        "bench.parallel_time_ms > 0.0 ? bench.sequential_time_ms / bench.parallel_time_ms"
        not in combined
    ), "found unsafe parallel ratio in parallel benchmarks"
    assert (
        "bench.simd_time_ms > 0.0 ? bench.sequential_time_ms / bench.simd_time_ms" not in combined
    ), "found unsafe simd ratio in parallel benchmarks"
    assert (
        "bench.gpu_time_ms > 0.0 ? bench.cpu_time_ms / bench.gpu_time_ms : 0.0" not in combined
    ), "found unsafe gpu ratio in parallel benchmarks"
    assert (
        "bench.parallel_time_ms > 0.0 ? bench.sequential_time_ms / bench.parallel_time_ms : 1.0"
        not in combined
    ), "found unsafe parallel fallback ratio"


@pytest.mark.quality
def test_approximate_distributed_benchmark_estimates_use_finite_ratios() -> None:
    paths = [
        _source_path("src", "persistence", "approximate", "approximate_perfect_hash_benchmark.cpp"),
        _source_path("src", "persistence", "approximate", "approximate_bloom_filter_ops.cpp"),
        _source_path("src", "persistence", "approximate", "approximate_nearest_neighbor_ops.cpp"),
        _source_path("src", "persistence", "distributed", "mpi_distributed_ph.cpp"),
        _source_path("src", "include", "nerve", "persistence", "approximate") / "perfect_hash.hpp",
        _source_path("src", "include", "nerve", "persistence", "approximate") / "bloom_filter.hpp",
        _source_path("src", "include", "nerve", "persistence", "distributed")
        / "mpi_distributed_ph.hpp",
    ]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    combined = "\n".join(texts)

    assert "finiteBenchmarkRatio" in texts[0], (
        "expected finiteBenchmarkRatio in perfect_hash benchmark"
    )
    assert combined.count("finiteBenchmarkSpeedup") >= 3, (
        f"expected >=3 finiteBenchmarkSpeedup, found {combined.count('finiteBenchmarkSpeedup')}"
    )
    assert "Bloom benchmark produced false negatives" in texts[1], (
        "missing false negatives check in bloom filter"
    )
    assert "double speedup_vs_std = 1.0" in texts[4], (
        "missing speedup_vs_std = 1.0 in perfect_hash.hpp"
    )
    assert "double speedup = 1.0" in texts[5], "missing double speedup = 1.0 in bloom_filter.hpp"
    assert "double estimated_speedup = 1.0" in texts[6], (
        "missing estimated_speedup = 1.0 in mpi header"
    )
    assert "return denominator > 0.0 ? numerator / denominator : 0.0" not in combined, (
        "found unsafe denominator check"
    )
    assert "bench.speedup = 0.0" not in texts[1], "found bench.speedup = 0.0 in bloom filter"
    assert "result.estimated_speedup = serial_estimate_ms / std::max" not in combined, (
        "found unsafe estimated_speedup"
    )


@pytest.mark.quality
def test_report_numa_and_default_benchmark_speedups_are_finite_neutral() -> None:
    report = (_source_path("src", "benchmarks", "performance_benchmark_report.cpp")).read_text(
        encoding="utf-8"
    )
    numa = (_source_path("src", "persistence", "memory", "memory_numa_optimizer.cpp")).read_text(
        encoding="utf-8"
    )
    defaults = [
        _source_path("src", "include", "nerve", "filtration", "vr_runtime.hpp"),
        _source_path("src", "include", "nerve", "metrics", "gpu_distances.hpp"),
        _source_path("src", "include", "nerve", "streaming", "gpu_streaming.hpp"),
        _source_path("src", "include", "nerve", "distributed", "mpi_persistence.hpp"),
    ]
    default_text = "\n".join(path.read_text(encoding="utf-8") for path in defaults)

    assert "finiteReportRatio" in report, (
        "expected finiteReportRatio in performance_benchmark_report"
    )
    assert "baseline.computation_time_ms / result.computation_time_ms" not in report, (
        "found unsafe report ratio"
    )
    assert "static_cast<double>(result.memory_usage_mb) / baseline.memory_usage_mb" not in report, (
        "found unsafe memory ratio"
    )
    assert "finiteBenchmarkSpeedup(bench.regular_time_ms, bench.numa_time_ms)" in numa, (
        "missing finiteBenchmarkSpeedup call in numa"
    )
    assert "clampPositive(bench.regular_time_ms)" not in numa, "found clampPositive in numa"
    assert default_text.count("double speedup = 1.0") >= 6, (
        f"expected >=6 'double speedup = 1.0', found {default_text.count('double speedup = 1.0')}"
    )


@pytest.mark.quality
def test_non_persistence_statistical_sentinels_are_finite() -> None:
    stats = (_source_path("src", "probabilistic", "probabilistic_statistics.cpp")).read_text(
        encoding="utf-8"
    )
    sampling = (_source_path("src", "probabilistic", "probabilistic_sampling.cpp")).read_text(
        encoding="utf-8"
    )
    differentiable = (
        _source_path("src", "persistence", "differentiable", "differentiable_ph5_ops.cpp")
    ).read_text(encoding="utf-8")
    tuner = (_source_path("src", "gpu", "tuner_nvidia_auto.cpp")).read_text(encoding="utf-8")

    assert "kDegenerateWaldStatistic" in stats, "expected kDegenerateWaldStatistic in statistics"
    assert "double best_loss = std::numeric_limits<double>::max()" in stats, (
        "missing best_loss sentinel in statistics"
    )
    assert "kLogPriorImpossible" in sampling, "expected kLogPriorImpossible in sampling"
    assert "kIllConditionedGradient" in differentiable, (
        "expected kIllConditionedGradient in differentiable"
    )
    assert "std::numeric_limits<double>::lowest()" in tuner, "expected lowest() sentinel in tuner"
    assert "std::numeric_limits<double>::infinity()" not in stats, "found infinity() in statistics"
    assert "-std::numeric_limits<double>::infinity()" not in sampling, (
        "found -infinity() in sampling"
    )
    assert (
        "gradient_condition_number =\n        min_norm > 0.0 ? (max_norm / min_norm) : std::numeric_limits<double>::infinity()"
        not in differentiable
    ), "found unsafe gradient_condition_number"
