"""Async persistence API.

Usage::

    import pynerve.async_api as nerve_async

    # Single computation
    result = await nerve_async.compute_persistence_async(points, max_dim=2)

    # Streaming mode for large datasets
    async for chunk in nerve_async.stream_persistence(data, chunk_size=500):
        process(chunk)

Note:
    The async API requires the ``pynerve_internal`` C++ extension to be installed
    (``pip install pynerve`` from source). If the extension is missing, calls
    will raise :class:`pynerve.BackendRequiredError`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from . import _core_import_error
from ._compute_api import compute_persistence
from ._compute_core import PersistenceResult
from ._validation import validate_points


def _require_async_deps() -> tuple[Any, ...]:
    """Lazy-import async submodules with user-friendly error messages."""
    if _core_import_error is not None:
        raise type(_core_import_error)(
            "The async API requires the pynerve_internal C++ extension. "
            "To fix this, install pynerve from source:\n"
            "  pip install -e ./python\n"
            "Or install a pre-built wheel from PyPI:\n"
            "  pip install pynerve\n"
            "Without the C++ extension, only type definitions, utilities, "
            "and sub-packages (nn, torch, etc.) are available.\n"
            f"Original import error: {_core_import_error}"
        )
    try:
        from ._async_compute import (  # noqa: PLC0415
            AsyncPersistenceComputer as _AsyncPersistenceComputer,
        )
        from ._async_loader import AsyncDiagramLoader as _AsyncDiagramLoader  # noqa: PLC0415
        from ._streaming_persistence import (  # noqa: PLC0415
            StreamingPersistence as _StreamingPersistence,
        )
    except ImportError as exc:
        raise ImportError(
            "Failed to import async persistence components. "
            "This may indicate a broken installation. "
            "Try reinstalling with 'pip install --force-reinstall pynerve'."
        ) from exc
    return _AsyncPersistenceComputer, _AsyncDiagramLoader, _StreamingPersistence


def _validate_filepaths(filepaths: list[str]) -> list[str]:
    if isinstance(filepaths, (str, bytes)) or not isinstance(filepaths, Iterable):
        raise TypeError("filepaths must be an iterable of paths")
    paths = list(filepaths)
    if not paths:
        raise ValueError("filepaths must be non-empty")
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"file not found: {p}")
    return paths


async def compute_persistence_async(
    data: np.ndarray, max_workers: int = 1, **kwargs: Any
) -> PersistenceResult:
    """Compute a persistence diagram asynchronously.

    :param data: Point cloud of shape ``(n_points, n_features)``.
    :param max_workers: Maximum number of concurrent workers.
    :param kwargs: Additional keyword arguments forwarded to
        :func:`compute_persistence`.
    :returns: The computed :class:`PersistenceResult`.
    :raises BackendRequiredError: If the C++ extension is not installed.
    """
    data = validate_points(data)

    async def single_batch() -> AsyncIterator[np.ndarray]:
        yield data

    def compute_batch(batch: np.ndarray) -> Any:
        return compute_persistence(batch, **kwargs)

    async_persistence_computer, _, _ = _require_async_deps()
    async with async_persistence_computer(max_workers=max_workers) as computer:
        async for result in computer.compute_batch_async(single_batch(), compute_batch):
            if isinstance(result, dict):
                return PersistenceResult.from_dict(result)
            return result  # type: ignore[no-any-return]
    raise RuntimeError("single-batch persistence produced no result")


async def load_diagrams_async(filepaths: list[str], max_concurrent: int = 8) -> Any:
    """Load multiple persistence diagram files concurrently.

    Supports ``.npy``, ``.pkl``, and ``.bin`` formats.

    :param filepaths: List of file paths to load.
    :param max_concurrent: Maximum number of concurrent file reads.
    :returns: List of diagram arrays (one per file).
    :raises FileNotFoundError: If any path does not exist.
    :raises BackendRequiredError: If the C++ extension is not installed.
    """
    filepaths = _validate_filepaths(filepaths)
    _, async_loader_cls, _ = _require_async_deps()
    loader = async_loader_cls(max_concurrent=max_concurrent)
    return await loader.load_batch(filepaths)


async def stream_persistence(
    data_source: str,
    chunk_size: int = 1000,
    return_format: str = "diagrams",
    max_buffered_chunks: int = 3,
    use_gpu: bool = False,
    **persistence_kwargs: Any,
) -> AsyncIterator[PersistenceResult | dict[Any, Any]]:
    """Stream persistence computation over a large dataset in chunks.

    :param data_source: Path to the data file.
    :param chunk_size: Number of points per chunk.
    :param return_format: Either ``"diagrams"`` or ``"raw"``.
    :param max_buffered_chunks: Maximum number of chunks buffered ahead.
    :param use_gpu: If *True*, attempt GPU-accelerated computation.
    :param persistence_kwargs: Additional keyword arguments forwarded to the
        persistence backend.
    :returns: Async iterator yielding results as they are computed.
    :raises TypeError: If *use_gpu* is not a boolean.
    :raises BackendRequiredError: If the C++ extension is not installed.
    """
    if not isinstance(use_gpu, bool):
        raise TypeError("use_gpu must be a boolean")
    _, _, streaming_cls = _require_async_deps()
    streamer = streaming_cls(
        chunk_size=chunk_size,
        max_buffered_chunks=max_buffered_chunks,
        use_gpu=use_gpu,
        **persistence_kwargs,
    )
    async for result in streamer.stream_compute(data_source, return_format=return_format):
        yield result


__all__ = [
    "compute_persistence_async",
    "load_diagrams_async",
    "stream_persistence",
]
