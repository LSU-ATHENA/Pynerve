"""Streaming persistence computation for large datasets."""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import AsyncIterator as AsyncIteratorABC
from pathlib import Path
from typing import Any

import numpy as np

from ._async_compute import AsyncPersistenceComputer
from ._compute_api import compute_persistence
from ._compute_core import PersistenceResult
from ._fallback_classes import PersistenceBackend
from ._validation import validate_bool, validate_diagram_array, validate_positive_int
from .exceptions import InvalidArgumentError


def _validate_streaming_array(array: np.ndarray, source: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise InvalidArgumentError(f"{source} data must be a 2D array", parameter=source)
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise InvalidArgumentError(f"{source} data must be non-empty", parameter=source)
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidArgumentError(f"{source} data must have a numeric dtype", parameter=source)
    if not np.isfinite(array).all():
        raise InvalidArgumentError(
            f"{source} data must contain only finite coordinates", parameter=source
        )
    return array


class StreamingPersistence:
    """Chunked persistence computation for file or async-iterator inputs."""

    def __init__(
        self,
        chunk_size: int = 1000,
        max_buffered_chunks: int = 3,
        use_gpu: bool = True,
        **persistence_kwargs: Any,
    ):
        chunk_size = validate_positive_int(chunk_size, "chunk_size")
        max_buffered_chunks = validate_positive_int(max_buffered_chunks, "max_buffered_chunks")
        use_gpu = validate_bool(use_gpu, "use_gpu")
        self.chunk_size = chunk_size
        self.max_buffered = max_buffered_chunks
        self.use_gpu = use_gpu
        self.persistence_kwargs = dict(persistence_kwargs)

    async def stream_compute(
        self,
        data_source: str | AsyncIterator[Any],
        return_format: str = "diagrams",
        **persistence_kwargs: Any,
    ) -> AsyncIterator[PersistenceResult | dict[str, Any]]:
        """Yield persistence results for each chunk."""
        if return_format not in {"diagrams", "betti", "stats"}:
            raise InvalidArgumentError(
                "return_format must be 'diagrams', 'betti', or 'stats'", parameter="return_format"
            )

        compute_kwargs = self._compute_kwargs(persistence_kwargs)
        if isinstance(data_source, (str, Path)):
            chunks = self._stream_from_file(str(data_source))
        else:
            if not isinstance(data_source, AsyncIteratorABC):
                raise InvalidArgumentError(
                    "data_source must be a path or async iterator", parameter="data_source"
                )
            chunks = data_source

        async with AsyncPersistenceComputer(
            max_workers=1, buffer_size=self.max_buffered
        ) as computer:
            async for result in computer.compute_batch_async(
                chunks, lambda chunk: self._compute_chunk(chunk, compute_kwargs)
            ):
                yield self._format_result(result, return_format)

    def _compute_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        compute_kwargs = {**self.persistence_kwargs, **overrides}
        if not self.use_gpu and "backend" not in compute_kwargs:
            compute_kwargs["backend"] = PersistenceBackend.CPU_EXACT
        return compute_kwargs

    async def _stream_from_file(self, filepath: str) -> AsyncIterator[np.ndarray]:
        """Yield chunks from HDF5, NPZ, or NPY files."""
        path = Path(filepath)
        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            async for chunk in self._stream_from_hdf5(path):
                yield chunk
            return

        if suffix == ".npy":
            data = _validate_streaming_array(np.load(path), str(path))
        elif suffix == ".npz":
            with np.load(path) as archive:
                if not archive.files:
                    raise InvalidArgumentError(
                        f"{filepath} does not contain any arrays", parameter="filepath"
                    )
                key = "data" if "data" in archive.files else archive.files[0]
                data = _validate_streaming_array(np.asarray(archive[key]), str(path))
        else:
            raise InvalidArgumentError(
                f"Unsupported streaming file format: {filepath}", parameter="filepath"
            )

        for start in range(0, data.shape[0], self.chunk_size):
            yield np.asarray(data[start : start + self.chunk_size])

    async def _stream_from_hdf5(self, path: Path) -> AsyncIterator[np.ndarray]:
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("h5py is required to stream HDF5 files") from exc

        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise InvalidArgumentError(
                    f"{path} does not contain a 'data' dataset", parameter="path"
                )
            dataset = handle["data"]
            for start in range(0, dataset.shape[0], self.chunk_size):
                yield _validate_streaming_array(
                    np.asarray(dataset[start : start + self.chunk_size]), str(path)
                )

    def _compute_chunk(
        self, chunk: np.ndarray, persistence_kwargs: dict[str, Any]
    ) -> PersistenceResult:
        chunk = _validate_streaming_array(chunk, "stream chunk")
        return compute_persistence(chunk, **persistence_kwargs)

    def _format_result(self, result: dict[str, Any], return_format: str) -> dict[str, Any]:
        if return_format == "betti":
            return self._diagrams_to_betti(result)
        if return_format == "stats":
            return self._diagrams_to_stats(result)
        return result

    def _pair_array(self, result: Any) -> np.ndarray:

        if isinstance(result, dict):
            if "pairs" not in result:
                raise InvalidArgumentError(
                    "persistence result must include a 'pairs' entry", parameter="result"
                )
            pairs = result["pairs"]
        elif isinstance(result, PersistenceResult):
            pairs = result.pairs
        else:
            pairs = result

        pairs = np.asarray(pairs, dtype=float)
        if pairs.size == 0:
            return np.empty((0, 3), dtype=float)
        return validate_diagram_array(pairs, name="result")

    def _diagrams_to_betti(self, result: Any) -> dict[str, int]:
        if isinstance(result, dict) and "betti_numbers" in result:
            return {f"betti_{dim}": int(value) for dim, value in enumerate(result["betti_numbers"])}

        from ._compute_core import PersistenceResult as _PersistenceResult  # noqa: PLC0415

        if isinstance(result, _PersistenceResult) and result.betti_numbers:
            return {f"betti_{dim}": int(value) for dim, value in enumerate(result.betti_numbers)}

        betti: dict[str, int] = {}
        pairs = self._pair_array(result)
        if pairs.shape[1] < 3:
            raise ValueError("betti conversion requires pair dimensions")
        for _birth, dth, dim in pairs[:, :3]:
            key = f"betti_{int(dim)}"
            betti.setdefault(key, 0)
            if np.isinf(dth) or dth > 1e9:
                betti[key] += 1

        max_dim = max((int(dim) for dim in pairs[:, 2]), default=0)
        for dim in range(max_dim + 1):
            betti.setdefault(f"betti_{dim}", 0)
        return betti

    def _diagrams_to_stats(self, result: Any) -> dict[str, float]:
        pairs = self._pair_array(result)
        finite = np.isfinite(pairs[:, 1])
        persistence = pairs[finite, 1] - pairs[finite, 0]
        return {
            "num_features": int(len(pairs)),
            "avg_persistence": float(persistence.mean()) if persistence.size else 0.0,
            "max_persistence": float(persistence.max()) if persistence.size else 0.0,
        }


__all__ = [
    "StreamingPersistence",
]
