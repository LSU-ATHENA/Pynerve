from __future__ import annotations

import asyncio
import warnings
from typing import Any, cast

import numpy as np

from .._compute_api import compute_persistence
from .._fallback_classes import PersistenceBackend
from .._types import PersistenceDiagramLike
from .._validation import (
    validate_nonempty_string,
    validate_nonnegative_int,
    validate_positive_int,
)
from ._common import (
    _benchmark_dataset,
    _compute_nerve_persistence,
    _time_runs,
)
from ._comparison_types import BenchmarkComparison, GPUComparison


def benchmark_gpu_vs_cpu(n_samples: int = 1000, max_dim: int = 2) -> GPUComparison:
    """Benchmark Nerve CPU persistence against its CUDA/Hybrid GPU backend.

    :param n_samples: Number of sample points (uses the ``"spheres"`` dataset).
    :param max_dim: Maximum homology dimension.
    :returns: A :class:`GPUComparison` with CPU-vs-GPU timings.
    :raises ValueError: If a parameter fails validation.
    """
    n_samples = validate_positive_int(n_samples, "n_samples")
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    data = _benchmark_dataset("spheres", n_samples)

    cpu_times = _time_runs(3, lambda: _compute_nerve_persistence(data, max_dim=max_dim))
    gpu_times = None
    try:

        def _run_gpu() -> Any:
            return compute_persistence(
                data,
                max_dim=max_dim,
                backend=PersistenceBackend.CUDA_HYBRID,
            )

        gpu_times = _time_runs(3, _run_gpu)
    except ImportError:
        warnings.warn("CuPy or GPU backend not available, skipping GPU benchmark", stacklevel=2)
    except RuntimeError as e:
        warnings.warn(f"GPU computation failed: {e}", stacklevel=2)

    mean_cpu = float(np.mean(cpu_times))
    result: dict[str, Any] = {
        "library1": "Nerve CPU",
        "library2": "Nerve GPU",
        "dataset": "spheres",
        "n_samples": n_samples,
        "time1": mean_cpu,
        "time2": None,
        "speedup": 1.0,
        "gpu_available": False,
    }

    if gpu_times is not None:
        mean_gpu = float(np.mean(gpu_times))
        result["time2"] = mean_gpu
        result["speedup"] = mean_cpu / mean_gpu if mean_gpu > 0 else 1.0
        result["gpu_available"] = True

    return GPUComparison(
        library1=result["library1"],
        library2=result["library2"],
        dataset=result["dataset"],
        n_samples=result["n_samples"],
        time1=float(result["time1"]),
        time2=float(result["time2"]) if result["time2"] is not None else None,
        speedup=float(result["speedup"]),
        gpu_available=bool(result["gpu_available"]),
    )


def benchmark_streaming_persistence(
    dataset: str = "spheres",
    n_samples: int = 2000,
    chunk_size: int = 500,
    n_runs: int = 3,
) -> BenchmarkComparison:
    """Benchmark full-batch persistence against streaming (incremental) persistence.

    :param dataset: Name of the synthetic dataset generator (``"spheres"``, ``"torus"``, or ``"swiss_roll"``).
    :param n_samples: Total number of sample points.
    :param chunk_size: Number of points per streaming chunk.
    :param n_runs: Number of timing runs to average over.
    :returns: A :class:`BenchmarkComparison` with full-batch vs streaming timings.
    :raises ValueError: If a parameter fails validation.
    """
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples = validate_positive_int(n_samples, "n_samples")
    chunk_size = validate_positive_int(chunk_size, "chunk_size")
    n_runs = validate_positive_int(n_runs, "n_runs")
    data = _benchmark_dataset(dataset, n_samples)
    baseline_times = _time_runs(n_runs, lambda: _compute_nerve_persistence(data, max_dim=1))

    streaming_times = []
    for _ in range(n_runs):
        chunks = [data[i : i + chunk_size] for i in range(0, data.shape[0], chunk_size)]

        async def _run_streaming() -> Any:
            from .._streaming_persistence import StreamingPersistence  # noqa: PLC0415

            sp = StreamingPersistence(chunk_size=chunk_size, use_gpu=False)
            from .._async_compute import AsyncIteratorABC  # noqa: PLC0415

            class _ChunkIterator(AsyncIteratorABC[Any]):
                def __init__(self, chunks_iter: Any) -> None:
                    self._chunks = iter(chunks_iter)

                def __aiter__(self) -> _ChunkIterator:
                    return self

                async def __anext__(self) -> Any:
                    try:
                        return next(self._chunks)
                    except StopIteration:
                        raise StopAsyncIteration from None

            results = []
            async for result in sp.stream_compute(_ChunkIterator(chunks), return_format="stats"):  # noqa: B023
                results.append(result)
            return results

        start = __import__("time").perf_counter()
        asyncio.run(_run_streaming())
        streaming_times.append(__import__("time").perf_counter() - start)

    mean_baseline = float(np.mean(baseline_times))
    mean_streaming = float(np.mean(streaming_times))
    return BenchmarkComparison(
        library1="Nerve (full)",
        library2="Nerve (streaming)",
        dataset=dataset,
        n_samples=n_samples,
        time1=mean_baseline,
        time2=mean_streaming,
        speedup=mean_streaming / mean_baseline if mean_baseline > 0 else 1.0,
    )


def benchmark_persistence_image(
    dataset: str = "spheres",
    n_samples: int = 500,
    n_runs: int = 3,
) -> dict[str, Any]:
    """Benchmark Nerve's persistence-image generation.

    :param dataset: Name of the synthetic dataset generator (``"spheres"``, ``"torus"``, or ``"swiss_roll"``).
    :param n_samples: Number of sample points.
    :param n_runs: Number of timing runs to average over.
    :returns: A dictionary with keys ``"mean_time"`` (seconds), ``"std_time"`` (seconds), and ``"n_points"``.
    :raises ValueError: If a parameter fails validation.
    """
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples = validate_positive_int(n_samples, "n_samples")
    n_runs = validate_positive_int(n_runs, "n_runs")
    data = _benchmark_dataset(dataset, n_samples)
    diagram_result = _compute_nerve_persistence(data, max_dim=1)

    if isinstance(diagram_result, dict) and "pairs" in diagram_result:
        diagram_array = np.asarray(diagram_result["pairs"], dtype=np.float64)
    elif isinstance(diagram_result, np.ndarray):
        diagram_array = diagram_result
    elif isinstance(diagram_result, dict):
        pairs_key = next((k for k in diagram_result if "pair" in k.lower()), None)
        diagram_array = (
            np.asarray(diagram_result[pairs_key], dtype=np.float64)
            if pairs_key
            else np.empty((0, 3), dtype=np.float64)
        )
    else:
        diagram_array = np.empty((0, 3), dtype=np.float64)

    nerve_times = _time_runs(n_runs, lambda: _run_nerve_persistence_image(diagram_array))
    return {
        "mean_time": float(np.mean(nerve_times)),
        "std_time": float(np.std(nerve_times)),
        "n_points": len(diagram_array),
    }


def _run_nerve_persistence_image(pairs: np.ndarray) -> np.ndarray:
    from .. import persistence_image  # noqa: PLC0415

    return persistence_image(cast(PersistenceDiagramLike, pairs), resolution=20, sigma=0.1)


def _run_gudhi_persistence_image(pairs: np.ndarray) -> Any:
    import gudhi  # noqa: PLC0415

    diag = [(int(row[2]), (row[0], row[1])) for row in pairs]
    img = gudhi.representations.PersistenceImage(resolution=[20, 20], bandwidth=0.1)  # pyright: ignore[reportAttributeAccessIssue]
    return img.fit_transform([diag])
