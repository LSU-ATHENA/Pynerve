"""Shared-memory multiprocessing helpers (re-export facade)."""

from ._parallel_api import ForkServerPool, compute_persistence_parallel
from ._parallel_patterns import ChunkedParallel, MapReducePH
from ._parallel_pool import ParallelPH
from ._shared_memory import SharedMemoryArray

__all__ = [
    "ChunkedParallel",
    "ForkServerPool",
    "MapReducePH",
    "ParallelPH",
    "SharedMemoryArray",
    "compute_persistence_parallel",
]
