from __future__ import annotations

import numpy as np
import pytest
from pynerve.exceptions import InvalidArgumentError, ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_cupy_adapters_validate_inputs_before_optional_backend() -> None:
    from pynerve._cupy_api import batch_diagrams_cupy, compute_diagrams_cupy
    from pynerve._cupy_convert import from_cupy, to_cupy
    from pynerve._cupy_memory import CudaStream, GPUBuffer, UnifiedMemoryBuffer

    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValidationError, match="device_id"):
        compute_diagrams_cupy(points, device_id=1.5)
    with pytest.raises(ValidationError, match="max_radius"):
        compute_diagrams_cupy(points, max_radius=float("nan"))
    with pytest.raises(ValidationError, match="max_dim"):
        compute_diagrams_cupy(points, max_dim=1.5)
    with pytest.raises((ValueError, ValidationError), match="finite coordinates"):
        compute_diagrams_cupy(np.array([[float("nan"), 0.0]], dtype=np.float32))
    with pytest.raises((TypeError, ValidationError), match="point_clouds"):
        batch_diagrams_cupy(object())
    with pytest.raises(ValidationError, match="max_radius"):
        batch_diagrams_cupy([points], max_radius=float("inf"))

    with pytest.raises(ValidationError, match="device_id"):
        to_cupy(points, device_id=1.5)
    with pytest.raises(TypeError, match="object dtype"):
        to_cupy(np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="target type"):
        from_cupy(object(), target_type="invalid")

    with pytest.raises(ValidationError, match="size"):
        GPUBuffer(1.5)
    with pytest.raises(TypeError, match="object dtype"):
        GPUBuffer(1, dtype=object)
    with pytest.raises(ValidationError, match="device_id"):
        GPUBuffer(1, device_id=1.5)
    with pytest.raises(TypeError, match="non_blocking"):
        CudaStream(non_blocking=1)
    with pytest.raises(ValidationError, match="size"):
        UnifiedMemoryBuffer(1.5)
    with pytest.raises(TypeError, match="object dtype"):
        UnifiedMemoryBuffer(1, dtype=object)


def test_jit_and_numba_facades_validate_before_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pynerve._numba_dispatch import benchmark_numba_vs_numpy, compute_with_numba
    from pynerve.jit import (
        JITCache,
        batch_betti_curves,
        betti_curve,
        cached_jit,
        filter_pairs,
        pairwise_distances,
        persistence_image,
        vietoris_rips_edges,
    )

    assert persistence_image(np.empty((0, 2), dtype=np.float32), 4, 0.1).shape == (4, 4), (
        f"expected (4, 4), got {persistence_image(np.empty((0, 2), dtype=np.float32), 4, 0.1).shape}"
    )
    stats = benchmark_numba_vs_numpy(
        lambda values: values + 1,
        lambda values: values + 1,
        lambda: (np.array([1.0]),),
        n_trials=1,
    )
    assert stats["results_match"], "expected results to match"
    assert np.isfinite(stats["speedup"]), f"expected finite speedup, got {stats['speedup']}"

    monkeypatch.setattr("time.perf_counter", lambda: 1.0)
    zero_time_stats = benchmark_numba_vs_numpy(
        lambda values: values + 1,
        lambda values: values + 1,
        lambda: (np.array([1.0]),),
        n_trials=1,
    )
    assert zero_time_stats["speedup"] == 1.0, f"expected 1.0, got {zero_time_stats['speedup']}"

    with pytest.raises((ValueError, InvalidArgumentError), match="points"):
        pairwise_distances(np.array([[float("nan"), 0.0]], dtype=np.float32))
    with pytest.raises((ValueError, InvalidArgumentError), match="birth"):
        persistence_image(np.array([[float("nan"), 1.0]], dtype=np.float32))
    with pytest.raises(ValidationError, match="resolution"):
        persistence_image(np.empty((0, 2), dtype=np.float32), 0, 0.1)
    with pytest.raises(ValidationError, match="sigma"):
        persistence_image(np.empty((0, 2), dtype=np.float32), 4, float("nan"))
    with pytest.raises(ValidationError, match="threshold"):
        filter_pairs(np.array([[0.0, 1.0]], dtype=np.float32), float("nan"))
    with pytest.raises(ValidationError, match="max_dim"):
        betti_curve(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), 1.5, 4)
    with pytest.raises((ValueError, InvalidArgumentError), match="dimensions"):
        betti_curve(np.array([[0.0, 1.0, 0.5]], dtype=np.float32), 1, 4)
    with pytest.raises(ValidationError, match="max_dist"):
        vietoris_rips_edges(np.ones((2, 2), dtype=np.float32), float("nan"))
    with pytest.raises(ValueError, match="diagrams"):
        batch_betti_curves(np.empty((0, 1, 3), dtype=np.float32), 1, 4)
    with pytest.raises(TypeError, match="func"):
        JITCache().get_or_compile(object())
    with pytest.raises(TypeError, match="func"):
        cached_jit(object())
    with pytest.raises(ValueError, match="operation"):
        compute_with_numba("")
    with pytest.raises(ValueError, match="Unknown operation"):
        compute_with_numba("missing")
    with pytest.raises(ValidationError, match="n_trials"):
        benchmark_numba_vs_numpy(lambda x: x, lambda x: x, lambda: (np.array([1.0]),), 1.5)
    with pytest.raises(TypeError, match="setup_fn"):
        benchmark_numba_vs_numpy(lambda x: x, lambda x: x, lambda: np.array([1.0]), 1)
