"""Async persistence-diagram file loading."""

from __future__ import annotations

import asyncio
import pickle
import struct
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import numpy as np

from ._validation import validate_diagram_array, validate_positive_int


def _validate_filepath(filepath: str | Path) -> Path:
    if not isinstance(filepath, (str, Path)) or not str(filepath):
        raise ValueError("filepath must be a non-empty path")
    return Path(filepath)


def _validate_filepaths(filepaths: list[str]) -> list[str | Path]:
    if isinstance(filepaths, (str, bytes)) or not isinstance(filepaths, Iterable):
        raise TypeError("filepaths must be an iterable of paths")
    paths: list[str | Path] = list(filepaths)
    if not paths:
        raise ValueError("filepaths must be non-empty")
    for filepath in paths:
        _validate_filepath(filepath)
    return paths


def _validate_diagram_array(array: np.ndarray, source: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return validate_diagram_array(array, name=str(source))


class AsyncDiagramLoader:
    """Async loader for persistence diagram files.

    Supports ``.npy``, ``.pkl``, and ``.bin`` formats. Concurrency is
    throttled by a semaphore.

    :param max_concurrent: Maximum number of concurrent file reads.
    :raises ValueError: If *max_concurrent* is not a positive integer.
    """

    def __init__(self, max_concurrent: int = 8):
        max_concurrent = validate_positive_int(max_concurrent, "max_concurrent")
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def load_file(self, filepath: str) -> np.ndarray:
        """Load one diagram file.

        The format is inferred from the file suffix (``.npy``, ``.pkl``,
        or ``.bin``).

        :param filepath: Path to the diagram file.
        :returns: Diagram array of shape ``(n_pairs, 3)``.
        :raises ValueError: If the file format is unsupported.
        """
        path = _validate_filepath(filepath)
        async with self._semaphore:
            loaders = {
                ".npy": self._load_npy,
                ".pkl": self._load_pickle,
                ".bin": self._load_binary,
            }
            try:
                return await loaders[path.suffix.lower()](path)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown file format: {filepath}. Supported: .npy, .pkl, .bin"
                ) from exc

    async def load_batch(self, filepaths: list[str]) -> list[np.ndarray]:
        """Load multiple diagram files concurrently.

        Results are returned in the same order as *filepaths*.

        :param filepaths: List of file paths to load.
        :returns: List of diagram arrays, one per file.
        """
        validated: list[str | Path] = _validate_filepaths(filepaths)
        tasks = [self.load_file(str(fp)) for fp in validated]
        return await asyncio.gather(*tasks)

    async def stream_directory(
        self, directory: str | Path, pattern: str = "*.npy", batch_size: int = 32
    ) -> AsyncIterator[list[np.ndarray]]:
        """Yield diagram files from a directory in fixed-size batches.

        Files matching *pattern* are sorted alphabetically and loaded in
        groups of *batch_size*.

        :param directory: Directory to scan.
        :param pattern: Glob pattern for file selection.
        :param batch_size: Number of files per yielded batch.
        :returns: Async iterator of diagram batches.
        :raises ValueError: If *pattern* is empty or *batch_size* is invalid.
        """
        batch_size = validate_positive_int(batch_size, "batch_size")
        dir_path = _validate_filepath(directory)
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a non-empty string")
        files = sorted(str(p) for p in dir_path.glob(pattern))

        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            diagrams = await self.load_batch(batch_files)
            yield diagrams

    async def _load_npy(self, filepath: Path) -> np.ndarray:
        await asyncio.sleep(0)
        return _validate_diagram_array(np.load(filepath), filepath)

    async def _load_pickle(self, filepath: Path) -> np.ndarray:
        await asyncio.sleep(0)
        data = filepath.read_bytes()
        return _validate_diagram_array(pickle.loads(data), filepath)

    async def _load_binary(self, filepath: Path) -> np.ndarray:
        await asyncio.sleep(0)
        with filepath.open("rb") as f:
            header = f.read(8)
            n_pairs = self._decode_pair_count(header, filepath)
            data = f.read(n_pairs * 3 * 4)
        return _validate_diagram_array(
            self._decode_binary_payload(data, n_pairs, filepath), filepath
        )

    @staticmethod
    def _decode_pair_count(header: bytes, filepath: Path) -> int:
        if len(header) != 8:
            raise ValueError(f"{filepath} has an incomplete binary diagram header")
        return int(struct.unpack("Q", header)[0])

    @staticmethod
    def _decode_binary_payload(data: bytes, n_pairs: int, filepath: Path) -> np.ndarray:
        expected_bytes = n_pairs * 3 * np.dtype(np.float32).itemsize
        if len(data) != expected_bytes:
            raise ValueError(f"{filepath} has an incomplete binary diagram payload")
        return np.frombuffer(data, dtype=np.float32).reshape(n_pairs, 3)


__all__ = [
    "AsyncDiagramLoader",
]
