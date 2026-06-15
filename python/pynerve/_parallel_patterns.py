"""Higher-order parallel patterns: ChunkedParallel, MapReducePH."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from ._parallel_pool import ParallelPH
from ._validation import (
    validate_optional_positive_int,
    validate_positive_int,
)


class ChunkedParallel:
    """Split data into chunks, process each in parallel, then aggregate.

    Useful for large arrays that cannot fit in a single worker's memory.

    :param chunk_size: Maximum number of rows per chunk.
    :param n_workers: Number of worker processes (defaults to CPU count).
    :param aggregator: Callable that combines chunk results. Defaults to
        flattening if every result is a list.
    :raises TypeError: If *aggregator* is not callable.
    :raises ValueError: If numeric parameters are out of range.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        n_workers: int | None = None,
        aggregator: Callable[..., Any] | None = None,
    ) -> None:
        chunk_size = validate_positive_int(chunk_size, "chunk_size")
        n_workers = validate_optional_positive_int(n_workers, "n_workers")
        if aggregator is not None and not callable(aggregator):
            raise TypeError("aggregator must be callable")
        self.chunk_size: int = chunk_size
        self.n_workers: int = n_workers or os.cpu_count() or 1
        self.aggregator: Callable[..., Any] = aggregator or self._default_aggregator

    def __repr__(self) -> str:
        return f"ChunkedParallel(chunk_size={self.chunk_size}, n_workers={self.n_workers})"

    def process(
        self,
        data: np.ndarray,
        process_fn: Callable[..., Any],
        reduce_fn: Callable[..., Any] | None = None,
    ) -> Any:
        """Split *data*, process chunks in parallel, then reduce.

        :param data: Input array (first dimension is chunked).
        :param process_fn: Per-chunk processing function.
        :param reduce_fn: Optional reduction ``f(chunk_results)`` called on the
            combined results from all chunks. Defaults to
            :attr:`aggregator`.
        :returns: Aggregated result.
        :raises TypeError: If *process_fn* or *reduce_fn* is not callable.
        """
        if not callable(process_fn):
            raise TypeError("process_fn must be callable")
        if reduce_fn is not None and not callable(reduce_fn):
            raise TypeError("reduce_fn must be callable")
        data = np.asarray(data)
        if len(data) == 0:
            return reduce_fn([]) if reduce_fn else self.aggregator([])

        n_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
        chunks = [data[i * self.chunk_size : (i + 1) * self.chunk_size] for i in range(n_chunks)]

        with ParallelPH(n_workers=self.n_workers) as pool:
            chunk_results = pool.map(process_fn, chunks)

        if reduce_fn:
            return reduce_fn(chunk_results)
        return self.aggregator(chunk_results)

    def _default_aggregator(self, results: list[Any]) -> list[Any]:
        if all(isinstance(r, list) for r in results):
            return [item for sublist in results for item in sublist]
        return results


class MapReducePH:
    """Parallel map-reduce for persistence workflows.

    Splits data across workers (map phase), then folds results pairwise
    (reduce phase).

    :param map_fn: Per-partition map function.
    :param reduce_fn: Binary reduction ``f(a, b)`` for combining mapped
        results.
    :param n_workers: Number of worker processes (defaults to CPU count).
    :raises TypeError: If *map_fn* or *reduce_fn* is not callable.
    :raises ValueError: If *n_workers* is out of range.
    """

    def __init__(
        self,
        map_fn: Callable[..., Any],
        reduce_fn: Callable[..., Any],
        n_workers: int | None = None,
    ) -> None:
        if not callable(map_fn) or not callable(reduce_fn):
            raise TypeError("map_fn and reduce_fn must be callable")
        n_workers = validate_optional_positive_int(n_workers, "n_workers")
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn
        self.n_workers: int = n_workers or os.cpu_count() or 1

    def __repr__(self) -> str:
        return (
            f"MapReducePH(map_fn={self.map_fn.__name__!r}, "
            f"reduce_fn={self.reduce_fn.__name__!r}, n_workers={self.n_workers})"
        )

    def run(self, data_partitions: list[np.ndarray]) -> Any:
        """Execute the map-reduce pipeline.

        :param data_partitions: List of data partitions, one per map task.
        :returns: Final reduced result.
        :raises ValueError: If *data_partitions* is empty.
        """
        if not data_partitions:
            raise ValueError("data_partitions must be non-empty")
        with ParallelPH(n_workers=self.n_workers) as pool:
            mapped = pool.map(self.map_fn, data_partitions)

        result = mapped[0]
        for m in mapped[1:]:
            result = self.reduce_fn(result, m)

        return result
