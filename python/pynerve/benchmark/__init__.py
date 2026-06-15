"""Benchmarking utilities for Nerve."""

from __future__ import annotations

from ._comparisons import (
    BenchmarkComparison,
    benchmark_distance_matrix,
    benchmark_gpu_vs_cpu,
    benchmark_persistence_image,
    benchmark_streaming_persistence,
    benchmark_vs_dionysus,
    benchmark_vs_gudhi,
    benchmark_vs_ripser,
    benchmark_witness_complex,
)
from ._scalability import ScalabilityResult, benchmark_complexity_analysis, benchmark_scalability
from ._suite import run_full_benchmark_suite, run_full_cross_comparison
from ._timer import Timer, TimerResult

__all__ = [
    "Timer",
    "TimerResult",
    "BenchmarkComparison",
    "ScalabilityResult",
    "benchmark_vs_ripser",
    "benchmark_vs_gudhi",
    "benchmark_vs_dionysus",
    "benchmark_scalability",
    "benchmark_streaming_persistence",
    "benchmark_gpu_vs_cpu",
    "benchmark_distance_matrix",
    "benchmark_witness_complex",
    "benchmark_persistence_image",
    "benchmark_complexity_analysis",
    "run_full_benchmark_suite",
    "run_full_cross_comparison",
]
