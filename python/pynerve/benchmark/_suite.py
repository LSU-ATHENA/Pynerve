"""Full benchmark suite runner for cross-library and scalability comparisons."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .._validation import validate_nonempty_string
from ._common import _BENCHMARK_RECOVERABLE_ERRORS
from ._comparisons import (
    benchmark_distance_matrix,
    benchmark_gpu_vs_cpu,
    benchmark_persistence_image,
    benchmark_streaming_persistence,
    benchmark_vs_dionysus,
    benchmark_vs_gudhi,
    benchmark_vs_ripser,
    benchmark_witness_complex,
)
from ._scalability import (
    benchmark_complexity_analysis,
    benchmark_scalability,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_ready(value.__dict__)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_full_benchmark_suite(output_file: str | None = None) -> dict[str, Any]:
    """Run a complete benchmark suite and optionally write results to JSON.

    Runs library comparisons (Reference, GUDHI) across datasets and sample sizes,
    plus scalability benchmarks.

    :param output_file: Optional file path for JSON output.
    :returns: A nested dictionary of benchmark results keyed by category
        (``"comparisons"``, ``"scalability"``).
    """
    output_path: Path | None = None
    if output_file is not None:
        output_path = Path(validate_nonempty_string(output_file, "output_file"))
    results: dict[str, Any] = {"comparisons": {}, "scalability": {}}

    for dataset in ["spheres", "torus", "swiss_roll"]:
        for n in [500, 1000]:
            key = f"{dataset}_{n}"
            try:
                results["comparisons"][f"reference_{key}"] = benchmark_vs_ripser(
                    dataset=dataset, n_samples=n
                )
            except _BENCHMARK_RECOVERABLE_ERRORS as e:
                results["comparisons"][f"reference_{key}"] = str(e)
            try:
                results["comparisons"][f"gudhi_{key}"] = benchmark_vs_gudhi(
                    dataset=dataset, n_samples=n
                )
            except _BENCHMARK_RECOVERABLE_ERRORS as e:
                results["comparisons"][f"gudhi_{key}"] = str(e)

    for dataset in ["spheres", "torus"]:
        results["scalability"][dataset] = benchmark_scalability(
            dataset=dataset, n_samples_range=[100, 200, 500, 1000]
        )

    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(_json_ready(results), f, indent=2)
    return results


def _benchmark_with_fallback(fn: Callable[[], Any]) -> Any:
    try:
        return _json_ready(fn())
    except _BENCHMARK_RECOVERABLE_ERRORS as e:
        return str(e)


def _make_comparison_fn(name: str, dataset: str, n: int) -> Callable[[], Any]:
    if name == "reference":
        return lambda: benchmark_vs_ripser(dataset=dataset, n_samples=n)
    if name == "gudhi":
        return lambda: benchmark_vs_gudhi(dataset=dataset, n_samples=n)
    if name == "dionysus":
        return lambda: benchmark_vs_dionysus(dataset=dataset, n_samples=n)
    raise ValueError(f"Unknown comparison: {name}")


def _run_cross_comparisons(results: dict[str, Any]) -> None:
    for dataset in ["spheres", "torus", "swiss_roll"]:
        for n in [500, 1000]:
            key = f"{dataset}_{n}"
            for name in ["reference", "gudhi", "dionysus"]:
                fn = _make_comparison_fn(name, dataset, n)
                results["comparisons"][f"{name}_{key}"] = _benchmark_with_fallback(fn)


def _make_streaming_fn(dataset: str) -> Callable[[], Any]:
    return lambda: benchmark_streaming_persistence(dataset=dataset, n_samples=1000, chunk_size=250)


def _run_scalability_and_streaming(results: dict[str, Any]) -> None:
    for dataset in ["spheres", "torus"]:
        results["scalability"][dataset] = _json_ready(
            benchmark_scalability(dataset=dataset, n_samples_range=[100, 200, 500, 1000])
        )
        results["streaming"][dataset] = _benchmark_with_fallback(_make_streaming_fn(dataset))


def _make_distance_fn(metric: str) -> Callable[[], Any]:
    return lambda: benchmark_distance_matrix(dataset="spheres", n_samples=500, metric=metric)


def _run_cross_benchmarks(results: dict[str, Any]) -> None:
    results["gpu"] = _benchmark_with_fallback(
        lambda: benchmark_gpu_vs_cpu(n_samples=500, max_dim=2)
    )
    for metric in ["euclidean", "manhattan"]:
        results["distance"][metric] = _benchmark_with_fallback(_make_distance_fn(metric))
    results["witness"]["spheres"] = _benchmark_with_fallback(
        lambda: benchmark_witness_complex(dataset="spheres", n_samples=500, n_landmarks=100)
    )
    results["persistence_image"]["spheres"] = _benchmark_with_fallback(
        lambda: benchmark_persistence_image(dataset="spheres", n_samples=500)
    )
    results["complexity"]["spheres"] = _benchmark_with_fallback(
        lambda: benchmark_complexity_analysis(dataset="spheres", max_dim=2)
    )


def _resolve_output_file(output_file: str | None) -> Path | None:
    if output_file is None:
        return None
    return Path(validate_nonempty_string(output_file, "output_file"))


def run_full_cross_comparison(output_file: str | None = None) -> dict[str, Any]:
    """Run an extended cross-comparison benchmark suite and optionally write results to JSON.

    Includes library comparisons, scalability, streaming, GPU, distance-matrix,
    witness-complex, persistence-image, and complexity-analysis benchmarks.

    :param output_file: Optional file path for JSON output.
    :returns: A nested dictionary of benchmark results keyed by category
        (``"comparisons"``, ``"scalability"``, ``"streaming"``, ``"gpu"``, ``"distance"``,
        ``"witness"``, ``"persistence_image"``, ``"complexity"``).
    """
    output_path = _resolve_output_file(output_file)

    results: dict[str, Any] = {
        "comparisons": {},
        "scalability": {},
        "streaming": {},
        "gpu": {},
        "distance": {},
        "witness": {},
        "persistence_image": {},
        "complexity": {},
    }

    _run_cross_comparisons(results)
    _run_scalability_and_streaming(results)
    _run_cross_benchmarks(results)

    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results
