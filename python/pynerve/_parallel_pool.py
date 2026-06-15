"""ParallelPH  --  multiprocessing pool wrapper for persistence computations."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from multiprocessing import get_context
from typing import Any

import numpy as np

from ._shared_memory import SharedMemoryArray, _validate_batches
from ._validation import (
    validate_bool,
    validate_optional_positive_int,
    validate_positive_int,
)

__all__ = [
    "_shared_memory_worker",
]


def _shared_memory_worker(args: tuple[Callable[..., Any], str, tuple[int, ...], str]) -> Any:
    compute_fn, name, shape, dtype_str = args
    shared = SharedMemoryArray.attach(name, shape, np.dtype(dtype_str))
    try:
        return compute_fn(shared.array)
    finally:
        shared.close()


class ParallelPH:
    """Context manager wrapping a ``multiprocessing.Pool`` for parallel persistence.

    Supports both standard pickle-based and shared-memory data transfer.
    Use as a context manager::

        with ParallelPH(n_workers=4) as pool:
            results = pool.map(compute_fn, batches)

    :param n_workers: Number of worker processes (defaults to CPU count).
    :param use_shared_memory: When *True* and more than one batch is given,
        data is transferred via :class:`SharedMemoryArray` instead of pickle.
    :param chunksize: Chunk size passed to ``pool.imap``.
    :raises TypeError: If any parameter has an invalid type.
    :raises ValueError: If numeric parameters are out of range.
    """

    def __init__(
        self,
        n_workers: int | None = None,
        use_shared_memory: bool = True,
        chunksize: int = 1,
    ) -> None:
        n_workers = validate_optional_positive_int(n_workers, "n_workers")
        chunksize = validate_positive_int(chunksize, "chunksize")
        use_shared_memory = validate_bool(use_shared_memory, "use_shared_memory")
        self.n_workers: int = n_workers or os.cpu_count() or 1
        self.use_shared_memory: bool = use_shared_memory
        self.chunksize: int = chunksize
        self._pool: Any = None

    def __repr__(self) -> str:
        active = self._pool is not None
        return (
            f"ParallelPH(n_workers={self.n_workers}, "
            f"use_shared_memory={self.use_shared_memory}, active={active})"
        )

    def __enter__(self) -> ParallelPH:
        """Create and start the worker pool.

        :returns: *self*.
        :raises RuntimeError: If the context is already active.
        """
        if self._pool is not None:
            raise RuntimeError("ParallelPH context is already active")
        ctx = get_context("spawn")
        self._pool = ctx.Pool(processes=self.n_workers)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Shut down the worker pool.

        Closes on success, terminates on exception.
        """
        if self._pool:
            if exc_type is None:
                self._pool.close()
            else:
                self._pool.terminate()
            self._pool.join()
            self._pool = None

    def map(
        self,
        compute_fn: Callable[..., Any],
        data_batches: list[np.ndarray],
        callback: Callable[..., Any] | None = None,
    ) -> list[Any]:
        """Apply *compute_fn* to each batch and return ordered results.

        When :attr:`use_shared_memory` is *True* and more than one batch is
        given, data is transferred to workers via :class:`SharedMemoryArray`.

        :param compute_fn: Function to apply to each batch.
        :param data_batches: Iterable of numpy arrays, one per task.
        :param callback: Optional ``f(completed, total)`` called after each
            batch completes.
        :returns: Ordered list of results, one per input batch.
        :raises TypeError: If *compute_fn* or *callback* is not callable.
        :raises RuntimeError: If the context manager is not active.
        """
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        batches = _validate_batches(data_batches, "data_batches")
        if not self._pool:
            raise RuntimeError("Use ParallelPH as context manager")

        if not batches:
            return []

        if self.use_shared_memory and len(batches) > 1:
            return self._map_shared_memory(compute_fn, batches, callback)
        return self._map_standard(compute_fn, batches, callback)

    def _map_standard(
        self,
        compute_fn: Callable[..., Any],
        data_batches: list[np.ndarray],
        callback: Callable[..., Any] | None,
    ) -> list[Any]:
        results: list[Any] = []
        pool: Any = self._pool
        for i, result in enumerate(pool.imap(compute_fn, data_batches, chunksize=self.chunksize)):
            results.append(result)
            if callback:
                callback(i + 1, len(data_batches))

        return results

    def _map_shared_memory(
        self,
        compute_fn: Callable[..., Any],
        data_batches: list[np.ndarray],
        callback: Callable[..., Any] | None,
    ) -> list[Any]:
        shm_arrays: list[SharedMemoryArray] = []
        for batch in data_batches:
            shm = SharedMemoryArray.from_array(batch)
            shm_arrays.append(shm)
        shm_batches = [(compute_fn, shm.name, shm.shape, shm.dtype.str) for shm in shm_arrays]

        results: list[Any] = []
        pool: Any = self._pool
        try:
            for i, result in enumerate(
                pool.imap(_shared_memory_worker, shm_batches, chunksize=self.chunksize)
            ):
                results.append(result)
                if callback:
                    callback(i + 1, len(data_batches))
        finally:
            for shm in shm_arrays:
                shm.close()

        return results

    def map_unordered(
        self, compute_fn: Callable[..., Any], data_batches: list[np.ndarray]
    ) -> Iterator[Any]:
        """Apply *compute_fn* to each batch, yielding results as they arrive.

        :param compute_fn: Function to apply to each batch.
        :param data_batches: Iterable of numpy arrays, one per task.
        :returns: Iterator yielding results in arbitrary (completion) order.
        :raises TypeError: If *compute_fn* is not callable.
        :raises RuntimeError: If the context manager is not active.
        """
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")
        batches = _validate_batches(data_batches, "data_batches")
        if not self._pool:
            raise RuntimeError("Use ParallelPH as context manager")

        if not batches:
            return iter(())

        return self._pool.imap_unordered(compute_fn, batches, chunksize=self.chunksize)  # type: ignore[no-any-return]

    def starmap(
        self, compute_fn: Callable[..., Any], args_list: list[tuple[Any, ...]]
    ) -> list[Any]:
        """Apply *compute_fn* to each tuple of arguments.

        :param compute_fn: Function to apply (called as ``fn(*args)``).
        :param args_list: List of argument tuples.
        :returns: Ordered list of results.
        :raises TypeError: If *compute_fn* is not callable or *args_list* is
            *None*.
        :raises RuntimeError: If the context manager is not active.
        """
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")
        if args_list is None:
            raise TypeError("args_list must be a list of argument tuples")
        if not self._pool:
            raise RuntimeError("Use ParallelPH as context manager")

        if not args_list:
            return []

        return self._pool.starmap(compute_fn, args_list, chunksize=self.chunksize)  # type: ignore[no-any-return]
