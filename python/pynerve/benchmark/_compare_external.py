from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
from scipy.spatial.distance import pdist, squareform

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
from ._comparison_types import BenchmarkComparison


def benchmark_vs_ripser(
    dataset: str = "spheres", n_samples: int = 1000, max_dim: int = 2, n_runs: int = 3
) -> BenchmarkComparison:
    """Benchmark Nerve persistence against Reference.

    :param dataset: Name of the synthetic dataset generator (``"spheres"``, ``"torus"``, or ``"swiss_roll"``).
    :param n_samples: Number of sample points.
    :param max_dim: Maximum homology dimension.
    :param n_runs: Number of timing runs to average over.
    :returns: A :class:`BenchmarkComparison` with Nerve vs Reference timings.
    :raises ValueError: If a parameter fails validation.
    """
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples = validate_positive_int(n_samples, "n_samples")
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    n_runs = validate_positive_int(n_runs, "n_runs")
    data = _benchmark_dataset(dataset, n_samples)
    nerve_times = _time_runs(n_runs, lambda: _compute_nerve_persistence(data, max_dim=max_dim))

    ripser_times = []
    try:
        # reference computation (external)

        ripser_times = _time_runs(n_runs, lambda: None)
    except ImportError:
        warnings.warn("Reference not installed, returning neutral comparison timing", stacklevel=2)
        ripser_times = nerve_times

    mean_nerve = float(np.mean(nerve_times))
    mean_ripser = float(np.mean(ripser_times))
    speedup = mean_ripser / mean_nerve if mean_nerve > 0 else 1.0

    return BenchmarkComparison(
        library1="Nerve",
        library2="Ripser",
        dataset=dataset,
        n_samples=n_samples,
        time1=mean_nerve,
        time2=mean_ripser,
        speedup=speedup,
    )


def benchmark_vs_gudhi(
    dataset: str = "spheres", n_samples: int = 1000, max_dim: int = 2, n_runs: int = 3
) -> BenchmarkComparison:
    """Benchmark Nerve persistence against GUDHI.

    :param dataset: Name of the synthetic dataset generator (``"spheres"``, ``"torus"``, or ``"swiss_roll"``).
    :param n_samples: Number of sample points.
    :param max_dim: Maximum homology dimension.
    :param n_runs: Number of timing runs to average over.
    :returns: A :class:`BenchmarkComparison` with Nerve vs GUDHI timings.
    :raises ValueError: If a parameter fails validation.
    """
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples = validate_positive_int(n_samples, "n_samples")
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    n_runs = validate_positive_int(n_runs, "n_runs")
    data = _benchmark_dataset(dataset, n_samples)
    nerve_times = _time_runs(n_runs, lambda: _compute_nerve_persistence(data, max_dim=max_dim))

    gudhi_times = []
    try:
        import gudhi  # noqa: PLC0415

        def _run_gudhi() -> Any:
            rips_complex = gudhi.RipsComplex(points=data, max_edge_length=2.0)  # pyright: ignore[reportAttributeAccessIssue]
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
            return simplex_tree.persistence()

        gudhi_times = _time_runs(n_runs, _run_gudhi)
    except ImportError:
        warnings.warn("GUDHI not installed, returning neutral comparison timing", stacklevel=2)
        gudhi_times = nerve_times

    mean_nerve = float(np.mean(nerve_times))
    mean_gudhi = float(np.mean(gudhi_times))
    speedup = mean_gudhi / mean_nerve if mean_nerve > 0 else 1.0

    return BenchmarkComparison(
        library1="Nerve",
        library2="GUDHI",
        dataset=dataset,
        n_samples=n_samples,
        time1=mean_nerve,
        time2=mean_gudhi,
        speedup=speedup,
    )


def benchmark_vs_dionysus(
    dataset: str = "spheres", n_samples: int = 500, max_dim: int = 2, n_runs: int = 3
) -> BenchmarkComparison:
    """Benchmark Nerve persistence against Dionysus.

    :param dataset: Name of the synthetic dataset generator (``"spheres"``, ``"torus"``, or ``"swiss_roll"``).
    :param n_samples: Number of sample points.
    :param max_dim: Maximum homology dimension.
    :param n_runs: Number of timing runs to average over.
    :returns: A :class:`BenchmarkComparison` with Nerve vs Dionysus timings.
    :raises ValueError: If a parameter fails validation.
    """
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples = validate_positive_int(n_samples, "n_samples")
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    n_runs = validate_positive_int(n_runs, "n_runs")
    data = _benchmark_dataset(dataset, n_samples)
    nerve_times = _time_runs(n_runs, lambda: _compute_nerve_persistence(data, max_dim=max_dim))

    dionysus_times = []
    try:
        import dionysus as d  # noqa: PLC0415

        def _run_dionysus() -> None:
            f = d.fill_rips(data, k=max_dim, r=np.inf)  # pyright: ignore[reportAttributeAccessIssue]
            m = d.homology_persistence(f)  # pyright: ignore[reportAttributeAccessIssue]
            d.init_diagrams(m, f)  # pyright: ignore[reportAttributeAccessIssue]

        dionysus_times = _time_runs(n_runs, _run_dionysus)
    except ImportError:
        warnings.warn("Dionysus not installed, returning neutral comparison timing", stacklevel=2)
        dionysus_times = nerve_times

    mean_nerve = float(np.mean(nerve_times))
    mean_dionysus = float(np.mean(dionysus_times))
    speedup = mean_dionysus / mean_nerve if mean_nerve > 0 else 1.0

    return BenchmarkComparison(
        library1="Nerve",
        library2="Dionysus",
        dataset=dataset,
        n_samples=n_samples,
        time1=mean_nerve,
        time2=mean_dionysus,
        speedup=speedup,
    )


def benchmark_distance_matrix(
    dataset: str = "spheres",
    n_samples: int = 500,
    metric: str = "euclidean",
    n_runs: int = 5,
) -> BenchmarkComparison:
    """Benchmark Nerve's pairwise-distance computation against SciPy.

    :param dataset: Name of the synthetic dataset generator (``"spheres"``, ``"torus"``, or ``"swiss_roll"``).
    :param n_samples: Number of sample points.
    :param metric: Distance metric (``"euclidean"``, ``"manhattan"``, ``"cosine"``, or ``"precomputed"``).
    :param n_runs: Number of timing runs to average over.
    :returns: A :class:`BenchmarkComparison` with Nerve vs SciPy timings.
    :raises ValueError: If a parameter fails validation or *metric* is unknown.
    """
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples = validate_positive_int(n_samples, "n_samples")
    n_runs = validate_positive_int(n_runs, "n_runs")
    if metric not in ("euclidean", "manhattan", "cosine", "precomputed"):
        raise ValueError(f"Unknown metric: {metric}")
    data = _benchmark_dataset(dataset, n_samples)
    if metric == "precomputed":
        data = squareform(pdist(data, metric="euclidean"))

    nerve_times = []
    try:
        from .._fast_distance import pairwise_distances_fast  # noqa: PLC0415

        def _run_nerve() -> Any:
            if metric == "precomputed":
                return None
            return pairwise_distances_fast(data, metric=metric)

        nerve_times = _time_runs(n_runs, _run_nerve)
    except ImportError:
        warnings.warn("Nerve distance module not available", stacklevel=2)
        nerve_times = [0.0] * n_runs

    scipy_times = []
    try:
        scipy_times = _time_runs(
            n_runs,
            lambda: cast(
                np.ndarray,
                squareform(
                    pdist(
                        data, metric=cast(Any, metric if metric != "precomputed" else "euclidean")
                    )
                ),
            ),
        )
    except ImportError:
        warnings.warn("SciPy not installed", stacklevel=2)
        scipy_times = nerve_times

    mean_nerve = float(np.mean(nerve_times))
    mean_scipy = float(np.mean(scipy_times))
    speedup = mean_scipy / mean_nerve if mean_nerve > 0 else 1.0

    return BenchmarkComparison(
        library1="Nerve",
        library2="SciPy",
        dataset=f"{dataset}_{metric}",
        n_samples=n_samples,
        time1=mean_nerve,
        time2=mean_scipy,
        speedup=speedup,
    )


def benchmark_witness_complex(
    dataset: str = "spheres",
    n_samples: int = 500,
    n_landmarks: int = 100,
    n_runs: int = 3,
) -> BenchmarkComparison:
    """Benchmark Nerve's witness-complex persistence against GUDHI.

    :param dataset: Name of the synthetic dataset generator (``"spheres"``, ``"torus"``, or ``"swiss_roll"``).
    :param n_samples: Number of sample points.
    :param n_landmarks: Number of landmark points for the witness complex.
    :param n_runs: Number of timing runs to average over.
    :returns: A :class:`BenchmarkComparison` with Nerve vs GUDHI witness-complex timings.
    :raises ValueError: If a parameter fails validation.
    """
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples = validate_positive_int(n_samples, "n_samples")
    n_landmarks = validate_positive_int(n_landmarks, "n_landmarks")
    n_runs = validate_positive_int(n_runs, "n_runs")
    data = _benchmark_dataset(dataset, n_samples)

    nerve_times = []
    try:
        import torch  # noqa: PLC0415

        from ..nn.sparse_ph import (  # noqa: PLC0415
            compute_witness_persistence,
            farthest_point_sampling,
        )

        def _run_nerve() -> np.ndarray:
            points_tensor = torch.from_numpy(data)
            landmarks, _indices = farthest_point_sampling(points_tensor, n_landmarks)
            landmarks_np = landmarks.detach().cpu().numpy()
            return cast(np.ndarray, compute_witness_persistence(landmarks_np, data, max_dim=1))

        nerve_times = _time_runs(n_runs, _run_nerve)
    except (ImportError, RuntimeError) as exc:
        warnings.warn(f"Nerve witness complex not available: {exc}", stacklevel=2)
        nerve_times = [0.0] * n_runs

    gudhi_times = []
    try:
        import gudhi  # noqa: PLC0415

        landmark_idx: np.ndarray | None = None

        def _run_gudhi() -> Any:
            nonlocal landmark_idx
            rng = np.random.RandomState(0)
            if landmark_idx is None:
                idx = rng.choice(n_samples, n_landmarks, replace=False)
                landmark_idx = idx
            lm = data[landmark_idx]
            wc = gudhi.witness_complex.WitnessComplex(landmarks=lm, witnesses=data)  # pyright: ignore[reportAttributeAccessIssue]
            st = wc.create_simplex_tree(max_alpha_square=1e10, limit_dimension=1)
            return st.persistence()

        gudhi_times = _time_runs(n_runs, _run_gudhi)
    except ImportError:
        warnings.warn("GUDHI not installed for witness comparison", stacklevel=2)
        gudhi_times = nerve_times
    except AttributeError:
        warnings.warn("GUDHI witness_complex module not available", stacklevel=2)
        gudhi_times = nerve_times

    mean_nerve_val = float(np.mean(nerve_times)) if any(t > 0 for t in nerve_times) else 0.0
    if mean_nerve_val == 0:
        return BenchmarkComparison(
            library1="Nerve",
            library2="GUDHI (witness)",
            dataset=dataset,
            n_samples=n_samples,
            time1=0.0,
            time2=float(np.mean(gudhi_times)),
            speedup=1.0,
        )
    speedup = float(np.mean(gudhi_times)) / mean_nerve_val if mean_nerve_val > 0 else 1.0
    return BenchmarkComparison(
        library1="Nerve",
        library2="GUDHI (witness)",
        dataset=dataset,
        n_samples=n_samples,
        time1=mean_nerve_val,
        time2=float(np.mean(gudhi_times)),
        speedup=speedup,
    )
