from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._validation import validate_nonempty_string, validate_nonnegative_int, validate_positive_int
from .._validation import validate_positive_finite as _validate_positive_finite
from ._common import _benchmark_dataset, _compute_nerve_persistence, _time_runs


@dataclass
class ScalabilityResult:
    """Result of a scalability benchmark measuring runtime vs input size.

    :param n_samples: List of sample sizes tested.
    :param times: Mean runtimes corresponding to each sample size in seconds.
    :param fit_result: Optional ``(intercept, exponent)`` tuple from a log-log linear fit.
    """

    n_samples: list[int]
    times: list[float]
    fit_result: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate all fields after dataclass initialisation."""
        self.n_samples = [validate_positive_int(n, "n_samples") for n in self.n_samples]
        self.times = [_validate_positive_finite(t, "times") for t in self.times]
        if self.fit_result is not None:
            if len(self.fit_result) != 2:
                raise ValueError("fit_result must contain exactly two values")
            intercept, exponent = (float(self.fit_result[0]), float(self.fit_result[1]))
            if not math.isfinite(intercept) or not math.isfinite(exponent):
                raise ValueError("fit_result values must be finite")
            self.fit_result = (intercept, exponent)

    def estimate_complexity(self) -> str:
        """Estimate asymptotic time complexity from a log-log fit.

        :returns: A human-readable complexity string, e.g. ``"O(n^{1.85}) ~ O(n^2)"``.
        :raises ValueError: If *n_samples* and *times* have mismatched lengths or contain non-positive values.
        """
        if len(self.n_samples) != len(self.times):
            raise ValueError("n_samples and times must have the same length")
        if any(n <= 0 for n in self.n_samples) or any(t <= 0 for t in self.times):
            raise ValueError("n_samples and times must be positive")
        if not np.isfinite(self.n_samples).all() or not np.isfinite(self.times).all():
            raise ValueError("n_samples and times must be finite")
        if len(self.n_samples) < 2:
            return "Insufficient data"

        log_n = np.log(self.n_samples)
        log_t = np.log(self.times)
        coeffs = np.polyfit(log_n, log_t, 1)
        exponent = coeffs[0]
        self.fit_result = (coeffs[1], exponent)

        if exponent < 1.2:
            return f"O(n^{exponent:.2f}) ~ O(n)"
        elif exponent < 2.2:
            return f"O(n^{exponent:.2f}) ~ O(n^2)"
        elif exponent < 3.2:
            return f"O(n^{exponent:.2f}) ~ O(n^3)"
        else:
            return f"O(n^{exponent:.2f})"


def benchmark_scalability(
    dataset: str = "spheres",
    n_samples_range: list[int] | None = None,
    max_dim: int = 2,
    n_runs: int = 3,
) -> ScalabilityResult:
    """Measure Nerve persistence runtime across multiple sample sizes.

    :param dataset: Name of the synthetic dataset generator (``"spheres"``, ``"torus"``, or ``"swiss_roll"``).
    :param n_samples_range: List of sample sizes to test. Defaults to ``[100, 200, 500, 1000, 2000]``.
    :param max_dim: Maximum homology dimension.
    :param n_runs: Number of timing runs to average over per sample size.
    :returns: A :class:`ScalabilityResult` containing the timing data and complexity estimate.
    :raises ValueError: If a parameter fails validation.
    """
    if n_samples_range is None:
        n_samples_range = [100, 200, 500, 1000, 2000]
    dataset = validate_nonempty_string(dataset, "dataset")
    n_samples_range = [validate_positive_int(n, "n_samples_range") for n in n_samples_range]
    max_dim = validate_nonnegative_int(max_dim, "max_dim")
    n_runs = validate_positive_int(n_runs, "n_runs")

    n_samples_list = []
    times = []
    for n in n_samples_range:
        data = _benchmark_dataset(dataset, n)
        run_times = _time_runs(
            n_runs,
            (lambda data=data: _compute_nerve_persistence(data, max_dim=max_dim)),  # type: ignore[misc]
        )
        n_samples_list.append(n)
        times.append(float(np.mean(run_times)))

    result = ScalabilityResult(n_samples=n_samples_list, times=times)
    result.estimate_complexity()
    return result


def benchmark_complexity_analysis(
    dataset: str = "spheres",
    max_dim: int = 2,
    n_runs: int = 3,
) -> ScalabilityResult:
    """Convenience wrapper that runs :func:`benchmark_scalability` with a fixed size range.

    :param dataset: Name of the synthetic dataset generator.
    :param max_dim: Maximum homology dimension.
    :param n_runs: Number of timing runs to average over per sample size.
    :returns: A :class:`ScalabilityResult` containing the timing data and complexity estimate.
    :raises ValueError: If a parameter fails validation.
    """
    return benchmark_scalability(
        dataset=dataset,
        n_samples_range=[100, 200, 500, 1000, 2000],
        max_dim=max_dim,
        n_runs=n_runs,
    )
