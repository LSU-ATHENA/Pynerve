"""Two-tier and persistent diagram caches with decorator support."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import pickle
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._validation import (
    validate_bool,
    validate_nonempty_string,
    validate_optional_positive_int,
    validate_positive_int,
)

logger = logging.getLogger("pynerve.cache")


def _validate_cache_key(key: str) -> str:
    return validate_nonempty_string(key, "cache_key")


try:
    import diskcache as dc

    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False

_ONE_MIB = 1024 * 1024
_TEN_GIB = 10 * 1024 * 1024 * 1024
_SEVEN_DAYS = 86400 * 7
_THIRTY_DAYS = 86400 * 30

_CACHE_DIR = os.environ.get("NERVE_CACHE_DIR", ".nerve_cache")
_CACHE_SIZE_LIMIT = int(os.environ.get("NERVE_CACHE_SIZE", str(_TEN_GIB)))
_CACHE_TTL = int(os.environ.get("NERVE_CACHE_TTL", str(_THIRTY_DAYS)))
_MISSING = object()
_CACHE_RECOVERABLE_ERRORS = (
    EOFError,
    OSError,
    pickle.PickleError,
)


class DiagramCache:
    """Two-tier in-memory and optional disk-backed cache for diagram persistence computations.

    Uses an LRU eviction policy for the in-memory tier via
    :class:`collections.OrderedDict`.  The optional disk tier (backed by
    ``diskcache``) persists results beyond process lifetime.

    :param memory_maxsize: Maximum number of entries in the in-memory LRU cache
    :param ttl: Time-to-live in seconds for cache entries; ``None`` means no expiry
    :param use_disk: Whether to enable the disk-cache tier
    :param disk_path: Directory path for the disk cache; defaults to the
        ``NERVE_CACHE_DIR`` environment variable or ``.nerve_cache``
    :raises ImportError: If ``use_disk=True`` but the ``diskcache`` package is
        not installed
    """

    def __init__(
        self,
        memory_maxsize: int = 1024,
        ttl: int | None = None,
        use_disk: bool = False,
        disk_path: str | None = None,
    ) -> None:
        memory_maxsize = validate_positive_int(memory_maxsize, "memory_maxsize")
        ttl = validate_optional_positive_int(ttl, "ttl")
        use_disk = validate_bool(use_disk, "use_disk")
        self.maxsize: int = memory_maxsize
        self.memory_maxsize: int = memory_maxsize
        self.ttl: int | None = ttl
        self.use_disk: bool = use_disk
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._write_times: dict[Any, float] = {}
        self._lock: threading.Lock = threading.Lock()
        self._disk_cache: Any = None

        if use_disk and HAS_DISKCACHE:
            disk_path = disk_path or _CACHE_DIR
            logger.info("Initializing disk cache at %s (ttl=%s)", disk_path, ttl)
            self._disk_cache = dc.Cache(disk_path)  # pyright: ignore[reportPossiblyUnboundVariable]
        elif use_disk and not HAS_DISKCACHE:
            raise ImportError(
                "diskcache is required when use_disk=True. Install it with: pip install diskcache"
            )

    def __repr__(self) -> str:
        mem_size = len(self._cache)
        disk = "enabled" if self._disk_cache else "disabled"
        return f"DiagramCache(memory_maxsize={self.memory_maxsize}, ttl={self.ttl}, entries={mem_size}, disk={disk})"

    def __enter__(self) -> DiagramCache:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> Literal[False]:
        self.close()
        return False

    def _make_key(self, data: np.ndarray, **params: Any) -> str:
        """Generate cache key using SHA-256.

        Uses SHA-256 for collision resistance appropriate for production caching.
        Call :meth:`_make_key_fast` for the faster Adler32 variant.
        """
        array = np.ascontiguousarray(data)
        digest = hashlib.sha256()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.view(np.uint8).tobytes())
        digest.update(pickle.dumps(sorted(params.items())))
        return digest.hexdigest()

    def _make_key_fast(self, data: np.ndarray, **params: Any) -> str:
        """Generate cache key using Adler32 (fast, non-cryptographic).

        Suitable for high-throughput caching where collision resistance is
        not a concern. Use :meth:`_make_key` for the default SHA-256 variant.
        """
        array = np.ascontiguousarray(data)
        h = zlib.adler32(str(array.shape).encode())
        h = zlib.adler32(str(array.dtype).encode(), h)
        h = zlib.adler32(array.view(np.uint8).tobytes(), h)
        h = zlib.adler32(pickle.dumps(sorted(params.items())), h)
        return format(h, "x")

    def get(self, data: np.ndarray, **params: Any) -> Any | None:
        """Retrieve a cached result by input data and parameters.

        :param data: Input numpy array used as part of the cache key
        :param params: Additional keyword parameters incorporated into the cache key
        :returns: Cached result if found, or ``None``
        """
        key = self._make_key(data, **params)
        return self.get_by_key(key)

    def get_by_key(self, key: Any, default: Any = None) -> Any:
        """Retrieve a cached result by its explicit cache key.

        Checks the in-memory cache first (with expiry validation and LRU
        promotion), then falls back to the disk cache if enabled.

        :param key: Cache key string
        :param default: Value returned when the key is not found
        :returns: Cached result or *default*
        """
        with self._lock:
            if key in self._cache:
                if self._is_expired(key):
                    logger.debug("Cache entry expired: %s", key[:16])
                    self._remove_key(key)
                    return default
                self._cache.move_to_end(key)
                return self._cache[key]

        disk_cache = self._disk_cache
        if disk_cache is not None:
            with suppress(*_CACHE_RECOVERABLE_ERRORS):
                result = disk_cache.get(key, _MISSING)
                if result is not _MISSING:
                    logger.debug("Disk cache hit: %s", key[:16])
                    self._add_to_memory(key, result)
                    return result

        return default

    def set(self, data: np.ndarray, result: Any, **params: Any) -> None:
        """Store a result in the cache keyed by input data and parameters.

        :param data: Input numpy array used as part of the cache key
        :param result: Value to cache
        :param params: Additional keyword parameters incorporated into the cache key
        """
        key = self._make_key(data, **params)
        self.set_by_key(key, result)

    def set_by_key(self, key: Any, result: Any) -> None:
        """Store a result in the cache under an explicit key.

        Writes to both the in-memory LRU cache and, if enabled, the disk cache.

        :param key: Cache key string
        :param result: Value to cache
        """
        self._add_to_memory(key, result)

        if self._disk_cache is not None:
            with suppress(*_CACHE_RECOVERABLE_ERRORS):
                self._disk_cache.set(key, result, expire=self.ttl)

    def _add_to_memory(self, key: str, value: Any) -> None:
        with self._lock:
            if key not in self._cache and len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)

            self._cache[key] = value
            self._write_times[key] = time.time()

    def _is_expired(self, key: str) -> bool:
        return self.ttl is not None and time.time() - self._write_times[key] > self.ttl

    def _remove_key(self, key: str) -> None:
        self._cache.pop(key, None)
        self._write_times.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from both the in-memory and disk caches."""
        with self._lock:
            self._cache.clear()
            self._write_times.clear()

        if self._disk_cache is not None:
            with suppress(*_CACHE_RECOVERABLE_ERRORS):
                self._disk_cache.clear()
        logger.debug("Cache cleared")

    def close(self) -> None:
        """Close the disk cache, flushing any pending writes.

        Safe to call even when the disk cache is not enabled.
        """
        if self._disk_cache is not None:
            self._disk_cache.close()
            logger.debug("Disk cache closed")


class PersistentDiagramCache(DiagramCache):
    """Persistent disk-backed diagram cache with configurable size and TTL.

    Extends :class:`DiagramCache` with a mandatory disk tier, automatic
    directory creation, and a configurable size limit.

    :param cache_dir: Directory path for the cache; expands ``~``.
        Defaults to the ``NERVE_CACHE_DIR`` environment variable or
        ``.nerve_cache``
    :param size_limit: Maximum disk space in bytes; defaults to the
        ``NERVE_CACHE_SIZE`` environment variable or 10 GiB
    :param ttl: Time-to-live in seconds for entries; defaults to the
        ``NERVE_CACHE_TTL`` environment variable or 30 days
    :param memory_maxsize: Maximum number of in-memory LRU entries
    :raises RuntimeError: If ``diskcache`` is not installed
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        size_limit: int | None = None,
        ttl: int | None = None,
        memory_maxsize: int = 1000,
    ) -> None:
        resolved_cache_dir: str = cache_dir or _CACHE_DIR
        size_limit = size_limit if size_limit is not None else _CACHE_SIZE_LIMIT
        ttl = ttl if ttl is not None else _CACHE_TTL
        size_limit = validate_positive_int(size_limit, "size_limit")
        ttl = validate_positive_int(ttl, "ttl")
        memory_maxsize = validate_positive_int(memory_maxsize, "memory_maxsize")
        if not HAS_DISKCACHE:
            raise RuntimeError("diskcache required for persistent caching")

        cache_path = Path(resolved_cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)

        super().__init__(
            memory_maxsize=memory_maxsize, ttl=ttl, use_disk=True, disk_path=str(cache_path)
        )

        self._disk_cache.size_limit = size_limit

    def __repr__(self) -> str:
        return (
            f"PersistentDiagramCache(cache_dir={self._disk_cache.directory!r}, "
            f"size_limit={self._disk_cache.size_limit}, "
            f"ttl={self.ttl}, memory_maxsize={self.maxsize})"
        )


def cached_persistence(
    memory_maxsize: int = 128,
    ttl: int | None = None,
    use_disk: bool = False,
    key_fn: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that caches function results in a :class:`DiagramCache`.

    Each wrapped function receives its own cache instance.  The cache key
    is derived from the ``data`` array and keyword arguments, or from an
    optional custom key function.

    :param memory_maxsize: Maximum in-memory entries for the underlying
        :class:`DiagramCache`
    :param ttl: Entry time-to-live in seconds; ``None`` for no expiry
    :param use_disk: Whether to enable disk-backed persistence
    :param key_fn: Optional callable ``(data, **kwargs) -> key`` for custom
        cache-key logic
    :returns: A decorator that wraps the target function with caching
    :raises TypeError: If *key_fn* is provided but is not callable
    """
    if key_fn is not None and not callable(key_fn):
        raise TypeError("key_fn must be callable")
    cache = DiagramCache(memory_maxsize=memory_maxsize, ttl=ttl, use_disk=use_disk)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(data: np.ndarray, **kwargs: Any) -> Any:
            if key_fn is not None:
                cache_key = key_fn(data, **kwargs)
            else:
                cache_key = cache._make_key(data, **kwargs)

            result = cache.get_by_key(cache_key, _MISSING)
            if result is not _MISSING:
                return result

            result = func(data, **kwargs)
            cache.set_by_key(cache_key, result)

            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator
