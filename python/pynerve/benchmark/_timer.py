from __future__ import annotations

import timeit
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._constants import EPS_1e_9
from .._validation import (
    validate_nonempty_string,
    validate_positive_int,
)
from .._validation import (
    validate_nonnegative_finite as _validate_nonnegative_number,
)
from .._validation import (
    validate_positive_finite as _validate_positive_finite,
)


@dataclass
class TimerResult:
    """Aggregated timing statistics from a :class:`Timer` run.

    :param mean_time: Mean per-iteration time in seconds.
    :param std_time: Standard deviation of per-iteration times in seconds.
    :param min_time: Minimum per-iteration time in seconds.
    :param max_time: Maximum per-iteration time in seconds.
    :param n_runs: Total number of iterations executed.
    """

    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    n_runs: int

    def __post_init__(self) -> None:
        """Validate all fields after dataclass initialisation."""
        self.mean_time = _validate_nonnegative_number(self.mean_time, "mean_time")
        self.std_time = _validate_nonnegative_number(self.std_time, "std_time")
        self.min_time = _validate_nonnegative_number(self.min_time, "min_time")
        self.max_time = _validate_nonnegative_number(self.max_time, "max_time")
        self.n_runs = validate_positive_int(self.n_runs, "n_runs")
        if self.min_time > self.max_time:
            raise ValueError("min_time must not exceed max_time")

    def __repr__(self) -> str:
        """Return a human-readable summary with millisecond precision."""
        return f"TimerResult(mean={self.mean_time * 1000:.2f}ms, std={self.std_time * 1000:.2f}ms)"


class Timer:
    """Precision timer built on top of :mod:`timeit`.

    :param stmt: Code statement to time, passed to :func:`timeit.timeit`.
    :param setup: Setup code executed once before timing, passed to :func:`timeit.timeit`.
    :param globals: Optional namespace dictionary for the timed code.
    :param label: Optional human-readable label. Defaults to *stmt*.
    :raises TypeError: If *globals* is supplied but is not a dictionary.
    :raises ValueError: If *stmt* or *setup* is empty.
    """

    def __init__(
        self,
        stmt: str = "pass",
        setup: str = "pass",
        *,
        globals: dict[str, Any] | None = None,
        label: str | None = None,
    ):
        """Initialise the timer with statement, setup, and optional globals namespace."""
        stmt = validate_nonempty_string(stmt, "stmt")
        setup = validate_nonempty_string(setup, "setup")
        if globals is not None and not isinstance(globals, dict):
            raise TypeError("globals must be a dictionary")
        if label is not None:
            label = validate_nonempty_string(label, "label")
        self.stmt = stmt
        self.setup = setup
        self.globals = globals
        self.label = label or stmt

    def timeit(self, number: int = 100000) -> float:
        """Execute the timed statement a fixed number of times and return the total duration.

        :param number: Number of loop iterations.
        :returns: Total elapsed time in seconds.
        :raises ValueError: If *number* is not a positive integer.
        """
        number = validate_positive_int(number, "number")
        return timeit.timeit(self.stmt, setup=self.setup, globals=self.globals, number=number)

    def blocked_autorange(self, min_run_time: float = 0.1, max_runs: int = 1000) -> TimerResult:
        """Auto-calibrate the iteration count and return aggregated timing statistics.

        Uses :func:`timeit.Timer.autorange` to determine a suitable iteration count,
        then repeats the measurement several times.

        :param min_run_time: Minimum total time for the autorange calibration in seconds.
        :param max_runs: Maximum number of auto-range steps.
        :returns: A :class:`TimerResult` with mean, std, min, max, and total iterations.
        :raises ValueError: If *min_run_time* or *max_runs* fail validation.
        """
        min_run_time = _validate_positive_finite(min_run_time, "min_run_time")
        max_runs = validate_positive_int(max_runs, "max_runs")

        t = timeit.Timer(self.stmt, setup=self.setup, globals=self.globals)
        number, total_time = t.autorange(lambda _number, _time_taken: None)

        times = []
        for _ in range(min(10, max(1, int(min_run_time / (total_time / number + EPS_1e_9))))):
            dt = t.timeit(number)
            times.append(dt / number)

        times_arr = np.array(times)
        return TimerResult(
            mean_time=float(np.mean(times_arr)),
            std_time=float(np.std(times_arr)),
            min_time=float(np.min(times_arr)),
            max_time=float(np.max(times_arr)),
            n_runs=len(times) * number,
        )
