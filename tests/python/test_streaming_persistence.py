"""Tests for _streaming_persistence.py -- StreamingPersistence class."""

from __future__ import annotations

import asyncio
import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestStreamingPersistenceInit:
    def test_default_init(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        assert sp.chunk_size == 1000
        assert sp.max_buffered == 3
        assert sp.use_gpu is True

    def test_custom_init(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=500, max_buffered_chunks=5, use_gpu=False)
        assert sp.chunk_size == 500
        assert sp.max_buffered == 5
        assert sp.use_gpu is False

    def test_invalid_chunk_size(self):
        from pynerve._streaming_persistence import StreamingPersistence

        with pytest.raises(Exception, match="positive"):
            StreamingPersistence(chunk_size=0)

    def test_invalid_max_buffered(self):
        from pynerve._streaming_persistence import StreamingPersistence

        with pytest.raises(Exception, match="positive"):
            StreamingPersistence(max_buffered_chunks=0)

    def test_extra_kwargs_stored(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(max_dim=2, backend="cpu")
        assert sp.persistence_kwargs.get("max_dim") == 2


class TestValidateStreamingArray:
    def test_valid_array(self):
        from pynerve._streaming_persistence import _validate_streaming_array

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_streaming_array(arr, "test")
        assert result.shape == (2, 2)

    def test_1d_array_rejected(self):
        from pynerve._streaming_persistence import _validate_streaming_array
        from pynerve.exceptions import InvalidArgumentError

        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InvalidArgumentError, match="2D"):
            _validate_streaming_array(arr, "test")

    def test_empty_array_rejected(self):
        from pynerve._streaming_persistence import _validate_streaming_array
        from pynerve.exceptions import InvalidArgumentError

        arr = np.empty((0, 2))
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            _validate_streaming_array(arr, "test")

    def test_non_numeric_rejected(self):
        from pynerve._streaming_persistence import _validate_streaming_array
        from pynerve.exceptions import InvalidArgumentError

        arr = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(InvalidArgumentError, match="numeric"):
            _validate_streaming_array(arr, "test")

    def test_non_finite_rejected(self):
        from pynerve._streaming_persistence import _validate_streaming_array
        from pynerve.exceptions import InvalidArgumentError

        arr = np.array([[1.0, float("inf")], [3.0, 4.0]])
        with pytest.raises(InvalidArgumentError, match="finite"):
            _validate_streaming_array(arr, "test")

    def test_nan_rejected(self):
        from pynerve._streaming_persistence import _validate_streaming_array
        from pynerve.exceptions import InvalidArgumentError

        arr = np.array([[1.0, float("nan")], [3.0, 4.0]])
        with pytest.raises(InvalidArgumentError, match="finite"):
            _validate_streaming_array(arr, "test")


class TestStreamingCompute:
    def test_invalid_return_format(self):
        from pynerve._streaming_persistence import StreamingPersistence
        from pynerve.exceptions import InvalidArgumentError

        sp = StreamingPersistence()

        async def _run():
            async for _ in sp.stream_compute("dummy.npy", return_format="bad"):
                pass

        with pytest.raises(InvalidArgumentError, match="return_format"):
            asyncio.run(_run())

    def test_unsupported_file_format(self):
        from pynerve._streaming_persistence import StreamingPersistence
        from pynerve.exceptions import InvalidArgumentError

        sp = StreamingPersistence()

        async def _run():
            async for _ in sp.stream_compute("dummy.xyz"):
                pass

        with pytest.raises(InvalidArgumentError, match="Unsupported"):
            asyncio.run(_run())

    def test_stream_from_npy(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=5)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            data = np.random.RandomState(0).randn(12, 3).astype(np.float64)
            np.save(f.name, data)
            path = f.name

        try:
            async def _run():
                chunks = []
                async for chunk in sp._stream_from_file(path):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())
            assert len(chunks) == 3
            assert chunks[0].shape == (5, 3)
            assert chunks[1].shape == (5, 3)
            assert chunks[2].shape == (2, 3)
        finally:
            os.unlink(path)

    def test_stream_from_npz(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=10)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            data = np.random.RandomState(0).randn(15, 2).astype(np.float64)
            np.savez(f.name, data=data)
            path = f.name

        try:
            async def _run():
                chunks = []
                async for chunk in sp._stream_from_file(path):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())
            assert len(chunks) == 2
            assert chunks[0].shape == (10, 2)
            assert chunks[1].shape == (5, 2)
        finally:
            os.unlink(path)

    def test_async_iterator_source(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()

        async def async_chunks():
            yield np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
            yield np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float64)

        async def _run():
            results = []
            async for r in sp.stream_compute(async_chunks(), return_format="diagrams"):
                results.append(r)
            return results

        results = asyncio.run(_run())
        assert len(results) == 2

    def test_non_async_iterator_source(self):
        from pynerve._streaming_persistence import StreamingPersistence
        from pynerve.exceptions import InvalidArgumentError

        sp = StreamingPersistence()

        async def _run():
            async for _ in sp.stream_compute([1, 2, 3]):
                pass

        with pytest.raises(InvalidArgumentError, match="path or async"):
            asyncio.run(_run())


class TestStreamingFormatResults:
    def test_format_diagrams(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        result = {"pairs": [[0.0, 1.0, 0.0]]}
        formatted = sp._format_result(result, "diagrams")
        assert formatted == result

    def test_format_betti(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        # Use float('inf') so the infinite-death branch (dth > 1e9) is hit
        result = {"pairs": np.array([[0.0, float("inf"), 0.0], [0.0, 0.5, 1.0]])}
        betti = sp._format_result(result, "betti")
        assert "betti_0" in betti
        assert "betti_1" in betti
        assert betti["betti_0"] == 1  # inf death counts as infinite feature

    def test_format_stats(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        result = {"pairs": np.array([[0.0, 1.0, 0.0], [0.0, 3.0, 1.0]])}
        stats = sp._format_result(result, "stats")
        assert "num_features" in stats
        assert "avg_persistence" in stats
        assert "max_persistence" in stats
        assert stats["num_features"] == 2
        assert stats["max_persistence"] == pytest.approx(3.0)

    def test_format_betti_with_existing_betti(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        result = {"betti_numbers": [2, 1, 0]}
        betti = sp._format_result(result, "betti")
        assert betti["betti_0"] == 2
        assert betti["betti_1"] == 1

    def test_format_stats_empty(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        result = {"pairs": np.empty((0, 3))}
        stats = sp._format_result(result, "stats")
        assert stats["num_features"] == 0
        assert stats["avg_persistence"] == 0.0


class TestComputeKwargs:
    def test_gpu_disabled_adds_backend(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(use_gpu=False)
        kwargs = sp._compute_kwargs({})
        assert "backend" in kwargs

    def test_gpu_enabled_no_backend(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(use_gpu=True)
        kwargs = sp._compute_kwargs({})
        assert "backend" not in kwargs

    def test_overrides_merged(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(max_dim=1)
        kwargs = sp._compute_kwargs({"max_dim": 2})
        assert kwargs["max_dim"] == 2
