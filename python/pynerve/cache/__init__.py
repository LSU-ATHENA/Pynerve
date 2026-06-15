"""Caching and memoization for expensive topology computations.

Environment variables:
    NERVE_CACHE_DIR: Directory for the disk cache (default: ``.nerve_cache``).
    NERVE_CACHE_SIZE: Maximum disk cache size in bytes (default: 10 GiB).
    NERVE_CACHE_TTL: Cache entry TTL in seconds (default: 30 days).
"""

from ._engine import DiagramCache, PersistentDiagramCache, _validate_cache_key, cached_persistence
from ._memoize import MemoizePersistent, memoize_persistent
from ._smart import SmartCache, get_cache_stats

__all__ = [
    "DiagramCache",
    "PersistentDiagramCache",
    "cached_persistence",
    "MemoizePersistent",
    "memoize_persistent",
    "SmartCache",
    "_validate_cache_key",
    "get_cache_stats",
]
