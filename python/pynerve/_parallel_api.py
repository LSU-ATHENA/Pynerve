"""High-level parallel persistence API and ForkServerPool."""

from __future__ import annotations

import os
from collections.abc import Callable
from multiprocessing import get_context
from typing import Any

import numpy as np

from ._compute_api import compute_persistence
from ._parallel_pool import ParallelPH
from ._shared_memory import _validate_batches
from ._validation import (
    validate_optional_positive_int,
)

__all__ = ["compute_persistence_parallel", "ForkServerPool"]


def _compute_persistence_batch(batch: np.ndarray) -> Any:
    return compute_persistence(batch)


def compute_persistence_parallel(
    data_batches: list[np.ndarray],
    n_workers: int | None = None,
    use_shared_memory: bool = True,
    progress_callback: Callable[..., Any] | None = None,
) -> list[Any]:
    """Compute persistence diagrams for multiple data batches in parallel.

    :param data_batches: Iterable of point-cloud arrays.
    :param n_workers: Number of worker processes (defaults to CPU count).
    :param use_shared_memory: If *True*, use :class:`SharedMemoryArray` for
        data transfer.
    :param progress_callback: Optional ``f(completed, total)`` invoked after
        each batch.
    :returns: List of persistence results, one per batch.
    :raises TypeError: If *progress_callback* is not callable.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    data_batches = _validate_batches(data_batches, "data_batches")
    with ParallelPH(n_workers=n_workers, use_shared_memory=use_shared_memory) as pool:
        return pool.map(_compute_persistence_batch, data_batches, progress_callback)


class ForkServerPool:
    """Context manager for a ``forkserver``-based process pool.

    Uses the ``forkserver`` start method (safer than fork on many platforms).
    The managed object is the raw ``multiprocessing.Pool``.

    :param n_workers: Number of worker processes (defaults to CPU count).
    :raises ValueError: If *n_workers* is out of range.
    """

    def __init__(self, n_workers: int | None = None) -> None:
        n_workers = validate_optional_positive_int(n_workers, "n_workers")
        self.n_workers: int = n_workers or os.cpu_count() or 1
        self._ctx = get_context("forkserver")
        self._pool: Any = None

    def __repr__(self) -> str:
        active = self._pool is not None
        return f"ForkServerPool(n_workers={self.n_workers}, active={active})"

    def __enter__(self) -> Any:
        """Create and start the forkserver pool.

        :returns: The underlying ``multiprocessing.Pool``.
        :raises RuntimeError: If the context is already active.
        """
        if self._pool is not None:
            raise RuntimeError("ForkServerPool context is already active")
        self._pool = self._ctx.Pool(processes=self.n_workers)
        return self._pool

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Shut down the forkserver pool.

        Closes on success, terminates on exception.
        """
        if self._pool:
            if exc_type is None:
                self._pool.close()
            else:
                self._pool.terminate()
            self._pool.join()
            self._pool = None
