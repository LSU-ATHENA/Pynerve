"""Smart tiered cache with memory/disk routing."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Literal

import numpy as np

from .._validation import validate_positive_int
from ._engine import (
    _CACHE_RECOVERABLE_ERRORS,
    _CACHE_SIZE_LIMIT,
    _MISSING,
    _ONE_MIB,
    HAS_DISKCACHE,
    DiagramCache,
    PersistentDiagramCache,
    _validate_cache_key,
)

logger = logging.getLogger("pynerve.cache")


def _validate_ignore_args(ignore_args: Sequence[str] | None) -> set[str]:
    if ignore_args is None:
        return set()
    if isinstance(ignore_args, (str, bytes)) or not isinstance(ignore_args, Sequence):
        raise TypeError("ignore_args must be a sequence of argument names")
    result = set(ignore_args)
    if not all(isinstance(name, str) and name for name in result):
        raise ValueError("ignore_args must contain non-empty strings")
    return result


class SmartCache:
    """Tiered cache that routes small results to memory and large results to disk.

    Results whose serialised size is below *small_threshold* are stored in
    an in-memory :class:`DiagramCache` (fast).  Larger results are stored
    in a :class:`PersistentDiagramCache` disk tier, avoiding the overhead
    of filling memory with bulky objects.

    :param memory_maxsize: Maximum in-memory entries
    :param disk_size_limit: Maximum disk space in bytes; defaults to the
        ``NERVE_CACHE_SIZE`` environment variable or 10 GiB
    :param small_threshold: Serialised size (bytes) below which results stay
        in memory
    """

    def __init__(
        self,
        memory_maxsize: int = 128,
        disk_size_limit: int | None = None,
        small_threshold: int = _ONE_MIB,
    ) -> None:
        memory_maxsize = validate_positive_int(memory_maxsize, "memory_maxsize")
        disk_size_limit = validate_positive_int(
            disk_size_limit or _CACHE_SIZE_LIMIT, "disk_size_limit"
        )
        small_threshold = validate_positive_int(small_threshold, "small_threshold")
        self.memory_cache = DiagramCache(memory_maxsize=memory_maxsize)
        self.disk_cache: PersistentDiagramCache | None = None
        if HAS_DISKCACHE:
            self.disk_cache = PersistentDiagramCache(size_limit=disk_size_limit)
        else:
            logger.warning("diskcache not installed; SmartCache disk tier unavailable")
        self.small_threshold = small_threshold

    def __repr__(self) -> str:
        return (
            f"SmartCache(memory_maxsize={self.memory_cache.memory_maxsize}, "
            f"disk={'yes' if self.disk_cache else 'no'}, "
            f"small_threshold={self.small_threshold})"
        )

    def __enter__(self) -> SmartCache:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        self.memory_cache.close()
        return False

    def get(self, key: str, data: np.ndarray | None = None) -> Any | None:
        """Retrieve a cached result by key.

        Checks memory first, then disk.  If the result came from disk and
        its size is below *small_threshold*, it is promoted to memory.

        :param key: Cache key string
        :param data: Optional numpy array used for diagnostic warnings on disk miss
        :returns: Cached result or ``None``
        :raises ValueError: If *key* is empty or not a string
        """
        key = _validate_cache_key(key)
        result = self.memory_cache.get_by_key(key, _MISSING)
        if result is not _MISSING:
            return result

        if self.disk_cache is not None:
            cached = self.disk_cache.get_by_key(key, _MISSING)
            if cached is not _MISSING:
                result = cached
                result_size = len(pickle.dumps(result))
                if result_size < self.small_threshold:
                    self.memory_cache.set_by_key(key, result)
                return result

        if data is not None and self.disk_cache is not None:
            logger.warning(
                "SmartCache.get() called with data but disk cache miss for key %s", key[:16]
            )

        return None

    def set(self, key: str, result: Any, data: np.ndarray | None = None) -> None:
        """Cache a result, routing it to the appropriate tier.

        Small results (serialised size < *small_threshold*) go to memory.
        Large results go to the disk tier, which requires *data* to
        construct the storage key.

        :param key: Cache key string
        :param result: Value to cache
        :param data: Numpy array required by the disk tier for large results
        :raises ValueError: If *key* is empty or not a string
        """
        key = _validate_cache_key(key)
        result_size = len(pickle.dumps(result))

        if result_size < self.small_threshold:
            self.memory_cache.set_by_key(key, result)
        elif self.disk_cache is not None and data is not None:
            self.disk_cache.set(data, result, key=key)
        else:
            logger.debug(
                "Result for key %s not cached: size=%d exceeds small_threshold=%d "
                "and no disk cache available",
                key[:16],
                result_size,
                self.small_threshold,
            )


def get_cache_stats(cache: DiagramCache) -> dict[str, int]:
    """Return usage statistics for a :class:`DiagramCache` instance.

    :param cache: The cache instance to inspect
    :returns: Dictionary with keys ``memory_entries``, ``memory_maxsize``,
        and (if disk is enabled) ``disk_entries`` and ``disk_size``
    :raises TypeError: If *cache* is not a :class:`DiagramCache`
    """
    if not isinstance(cache, DiagramCache):
        raise TypeError("cache must be a DiagramCache")
    with cache._lock:
        stats = {
            "memory_entries": len(cache._cache),
            "memory_maxsize": cache.memory_maxsize,
        }

    if cache._disk_cache is not None:
        with suppress(*_CACHE_RECOVERABLE_ERRORS):
            stats["disk_entries"] = len(cache._disk_cache)
            stats["disk_size"] = cache._disk_cache.volume()

    return stats
