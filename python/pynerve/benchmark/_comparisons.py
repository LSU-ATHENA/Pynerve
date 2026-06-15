"""Benchmark comparisons between Nerve and other persistence libraries."""

from __future__ import annotations

from ._compare_external import (
    benchmark_distance_matrix,
    benchmark_vs_dionysus,
    benchmark_vs_gudhi,
    benchmark_vs_ripser,
    benchmark_witness_complex,
)
from ._compare_internal import (
    benchmark_gpu_vs_cpu,
    benchmark_persistence_image,
    benchmark_streaming_persistence,
)
from ._comparison_types import BenchmarkComparison, GPUComparison

__all__ = [
    "BenchmarkComparison",
    "GPUComparison",
    "benchmark_vs_ripser",
    "benchmark_vs_gudhi",
    "benchmark_vs_dionysus",
    "benchmark_distance_matrix",
    "benchmark_witness_complex",
    "benchmark_gpu_vs_cpu",
    "benchmark_streaming_persistence",
    "benchmark_persistence_image",
]
