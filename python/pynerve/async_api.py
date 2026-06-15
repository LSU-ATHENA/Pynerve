"""Asynchronous persistence API for Nerve.

Usage::

    import pynerve.async_api as nerve_async

    # Single computation
    result = await nerve_async.compute_persistence_async(points, max_dim=2)

    # Streaming mode for large datasets
    async for chunk in nerve_async.stream_persistence(data, chunk_size=500):
        process(chunk)

GPU and diagram loader components::

    from pynerve.async_api import AsyncGPUTransfer, AsyncDiagramLoader

    loader = AsyncDiagramLoader(max_concurrent=8)
    diagrams = await loader.load_batch(["file1.npy", "file2.npy"])
"""

from __future__ import annotations

from ._async_facade import (
    compute_persistence_async,
    load_diagrams_async,
    stream_persistence,
)
from ._async_gpu import AsyncGPUTransfer
from ._async_loader import AsyncDiagramLoader

__all__ = [
    "AsyncDiagramLoader",
    "AsyncGPUTransfer",
    "compute_persistence_async",
    "load_diagrams_async",
    "stream_persistence",
]
