from __future__ import annotations

from dataclasses import dataclass

from .._validation import (
    validate_nonempty_string,
    validate_positive_int,
)
from .._validation import (
    validate_nonnegative_finite as _validate_nonnegative_number,
)


@dataclass
class BenchmarkComparison:
    """Result of a library-to-library benchmark comparison.

    :param library1: Name of the first persistence library.
    :param library2: Name of the second persistence library.
    :param dataset: Identifier of the benchmark dataset.
    :param n_samples: Number of sample points in the dataset.
    :param time1: Mean wall-clock time for the first library in seconds.
    :param time2: Mean wall-clock time for the second library in seconds.
    :param speedup: Speedup factor computed as ``time2 / time1``.
    :param memory1: Optional peak memory usage for the first library in MiB.
    :param memory2: Optional peak memory usage for the second library in MiB.
    """

    library1: str
    library2: str
    dataset: str
    n_samples: int
    time1: float
    time2: float
    speedup: float
    memory1: float | None = None
    memory2: float | None = None

    def __post_init__(self) -> None:
        """Validate all fields after dataclass initialisation."""
        self.library1 = validate_nonempty_string(self.library1, "library1")
        self.library2 = validate_nonempty_string(self.library2, "library2")
        self.dataset = validate_nonempty_string(self.dataset, "dataset")
        self.n_samples = validate_positive_int(self.n_samples, "n_samples")
        self.time1 = _validate_nonnegative_number(self.time1, "time1")
        self.time2 = _validate_nonnegative_number(self.time2, "time2")
        self.speedup = _validate_nonnegative_number(self.speedup, "speedup")
        if self.memory1 is not None:
            self.memory1 = _validate_nonnegative_number(self.memory1, "memory1")
        if self.memory2 is not None:
            self.memory2 = _validate_nonnegative_number(self.memory2, "memory2")

    def __repr__(self) -> str:
        """Return a human-readable speedup summary."""
        return f"{self.library1} vs {self.library2}: {self.speedup:.2f}x speedup"


@dataclass
class GPUComparison:
    """Result of a CPU-vs-GPU benchmark comparison.

    :param library1: Name of the first (CPU) library variant.
    :param library2: Name of the second (GPU) library variant.
    :param dataset: Identifier of the benchmark dataset.
    :param n_samples: Number of sample points in the dataset.
    :param time1: Mean wall-clock time for the CPU library in seconds.
    :param time2: Mean wall-clock time for the GPU library in seconds, or ``None`` if GPU was unavailable.
    :param speedup: Speedup factor computed as ``time1 / time2``.
    :param gpu_available: Whether a GPU backend was successfully used.
    """

    library1: str
    library2: str
    dataset: str
    n_samples: int
    time1: float
    time2: float | None
    speedup: float
    gpu_available: bool
