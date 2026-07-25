"""Tests for _async_compute.py and _async_loader.py — async persistence and diagram loading."""

from __future__ import annotations

import asyncio
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestAsyncPersistenceComputer:
    """Covers _async_compute.py — AsyncPersistenceComputer."""

    def test_construct(self):
        from pynerve._async_compute import AsyncPersistenceComputer
        computer = AsyncPersistenceComputer(max_workers=2, buffer_size=2)
        assert computer.max_workers == 2
        assert computer.buffer_size == 2
        assert computer._closed is False

    def test_construct_invalid_workers(self):
        from pynerve._async_compute import AsyncPersistenceComputer
        with pytest.raises(Exception, match="positive"):
            AsyncPersistenceComputer(max_workers=0)

    def test_construct_invalid_buffer(self):
        from pynerve._async_compute import AsyncPersistenceComputer
        with pytest.raises(Exception, match="positive"):
            AsyncPersistenceComputer(buffer_size=0)

    def test_compute_batch_async(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            async def data_source():
                yield np.random.rand(10, 3)

            def compute_fn(batch):
                return {"pairs": [(0.0, 1.0, 0)], "betti": [1]}

            computer = AsyncPersistenceComputer(max_workers=2, buffer_size=2)
            results = []
            async for result in computer.compute_batch_async(data_source(), compute_fn):
                results.append(result)
            await computer.close()
            return results

        results = asyncio.run(_run())
        assert len(results) == 1
        assert "pairs" in results[0]

    def test_compute_batch_multiple(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            async def data_source():
                for _ in range(3):
                    yield np.random.rand(5, 3)

            def compute_fn(batch):
                return {"n": len(batch)}

            computer = AsyncPersistenceComputer(max_workers=2, buffer_size=1)
            results = []
            async for result in computer.compute_batch_async(data_source(), compute_fn):
                results.append(result)
            await computer.close()
            return results

        results = asyncio.run(_run())
        assert len(results) == 3

    def test_compute_batch_not_async_iterator(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            computer = AsyncPersistenceComputer(max_workers=2)
            with pytest.raises(TypeError, match="async iterator"):
                async for _ in computer.compute_batch_async([1, 2, 3]):
                    pass
            await computer.close()

        asyncio.run(_run())

    def test_compute_batch_not_callable(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            async def data_source():
                yield np.random.rand(5, 3)

            computer = AsyncPersistenceComputer(max_workers=2)
            with pytest.raises(TypeError, match="callable"):
                async for _ in computer.compute_batch_async(data_source(), "not callable"):
                    pass
            await computer.close()

        asyncio.run(_run())

    def test_close_twice(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            computer = AsyncPersistenceComputer(max_workers=2)
            await computer.close()
            await computer.close()
            return computer._closed

        result = asyncio.run(_run())
        assert result is True

    def test_context_manager(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            async with AsyncPersistenceComputer(max_workers=2) as computer:
                assert computer._closed is False
            return computer._closed

        result = asyncio.run(_run())
        assert result is True

    def test_compute_after_close(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            async def data_source():
                yield np.random.rand(5, 3)

            computer = AsyncPersistenceComputer(max_workers=2)
            await computer.close()
            with pytest.raises(RuntimeError, match="closed"):
                async for _ in computer.compute_batch_async(data_source()):
                    pass

        asyncio.run(_run())

    def test_default_compute(self):
        from pynerve._async_compute import AsyncPersistenceComputer
        computer = AsyncPersistenceComputer(max_workers=2)
        result = computer._default_compute(np.random.rand(10, 3))
        assert result is not None

        async def _close():
            await computer.close()

        asyncio.run(_close())


class TestAsyncDiagramLoader:
    """Covers _async_loader.py — AsyncDiagramLoader."""

    def test_construct(self):
        from pynerve._async_loader import AsyncDiagramLoader
        loader = AsyncDiagramLoader(max_concurrent=4)
        assert loader.max_concurrent == 4

    def test_construct_invalid(self):
        from pynerve._async_loader import AsyncDiagramLoader
        with pytest.raises(Exception, match="positive"):
            AsyncDiagramLoader(max_concurrent=0)

    def test_load_npy(self, tmp_path):
        from pynerve._async_loader import AsyncDiagramLoader
        diag = np.array([[0.0, 1.0, 0], [0.5, 2.0, 1]], dtype=np.float32)
        path = tmp_path / "test.npy"
        np.save(path, diag)
        loader = AsyncDiagramLoader()
        result = asyncio.run(loader.load_file(str(path)))
        assert result.shape == (2, 3)

    def test_load_pickle(self, tmp_path):
        from pynerve._async_loader import AsyncDiagramLoader
        diag = np.array([[0.0, 1.0, 0], [0.5, 2.0, 1]], dtype=np.float32)
        path = tmp_path / "test.pkl"
        path.write_bytes(pickle.dumps(diag))
        loader = AsyncDiagramLoader()
        result = asyncio.run(loader.load_file(str(path)))
        assert result.shape == (2, 3)

    def test_load_binary(self, tmp_path):
        from pynerve._async_loader import AsyncDiagramLoader
        diag = np.array([[0.0, 1.0, 0], [0.5, 2.0, 1]], dtype=np.float32)
        path = tmp_path / "test.bin"
        header = struct.pack("Q", diag.shape[0])
        payload = diag.tobytes()
        path.write_bytes(header + payload)
        loader = AsyncDiagramLoader()
        result = asyncio.run(loader.load_file(str(path)))
        assert result.shape == (2, 3)

    def test_load_unknown_format(self, tmp_path):
        from pynerve._async_loader import AsyncDiagramLoader
        path = tmp_path / "test.xyz"
        path.write_bytes(b"data")
        loader = AsyncDiagramLoader()
        with pytest.raises(ValueError, match="Unknown file format"):
            asyncio.run(loader.load_file(str(path)))

    def test_load_nonexistent_file(self):
        from pynerve._async_loader import AsyncDiagramLoader
        loader = AsyncDiagramLoader()
        with pytest.raises(FileNotFoundError):
            asyncio.run(loader.load_file("/nonexistent/file.npy"))

    def test_load_batch(self, tmp_path):
        from pynerve._async_loader import AsyncDiagramLoader
        diag = np.array([[0.0, 1.0, 0]], dtype=np.float32)
        paths = []
        for i in range(3):
            p = tmp_path / f"test_{i}.npy"
            np.save(p, diag)
            paths.append(str(p))
        loader = AsyncDiagramLoader()
        results = asyncio.run(loader.load_batch(paths))
        assert len(results) == 3
        for r in results:
            assert r.shape == (1, 3)

    def test_load_batch_empty(self):
        from pynerve._async_loader import AsyncDiagramLoader
        loader = AsyncDiagramLoader()
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(loader.load_batch([]))

    def test_load_batch_string_not_list(self):
        from pynerve._async_loader import AsyncDiagramLoader
        loader = AsyncDiagramLoader()
        with pytest.raises(TypeError, match="iterable"):
            asyncio.run(loader.load_batch("single_path"))

    def test_stream_directory(self, tmp_path):
        from pynerve._async_loader import AsyncDiagramLoader
        diag = np.array([[0.0, 1.0, 0]], dtype=np.float32)
        for i in range(5):
            np.save(tmp_path / f"test_{i}.npy", diag)

        async def _run():
            loader = AsyncDiagramLoader()
            batches = []
            async for batch in loader.stream_directory(str(tmp_path), pattern="*.npy", batch_size=2):
                batches.append(batch)
            return batches

        batches = asyncio.run(_run())
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[2]) == 1

    def test_stream_directory_empty_pattern(self, tmp_path):
        from pynerve._async_loader import AsyncDiagramLoader

        async def _run():
            loader = AsyncDiagramLoader()
            with pytest.raises(ValueError, match="pattern"):
                async for _ in loader.stream_directory(str(tmp_path), pattern=""):
                    pass

        asyncio.run(_run())

    def test_stream_directory_invalid_batch_size(self, tmp_path):
        from pynerve._async_loader import AsyncDiagramLoader

        async def _run():
            loader = AsyncDiagramLoader()
            with pytest.raises(Exception, match="positive"):
                async for _ in loader.stream_directory(str(tmp_path), batch_size=0):
                    pass

        asyncio.run(_run())

    def test_validate_filepath_valid(self):
        from pynerve._async_loader import _validate_filepath
        result = _validate_filepath("/tmp/test.npy")
        assert isinstance(result, Path)

    def test_validate_filepath_empty(self):
        from pynerve._async_loader import _validate_filepath
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepath("")

    def test_validate_filepath_invalid_type(self):
        from pynerve._async_loader import _validate_filepath
        with pytest.raises(ValueError, match="path"):
            _validate_filepath(42)

    def test_validate_filepaths_valid(self):
        from pynerve._async_loader import _validate_filepaths
        result = _validate_filepaths(["/tmp/a.npy", "/tmp/b.npy"])
        assert len(result) == 2

    def test_validate_filepaths_empty(self):
        from pynerve._async_loader import _validate_filepaths
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepaths([])

    def test_validate_filepaths_string(self):
        from pynerve._async_loader import _validate_filepaths
        with pytest.raises(TypeError, match="iterable"):
            _validate_filepaths("single")

    def test_decode_pair_count_valid(self):
        from pynerve._async_loader import AsyncDiagramLoader
        header = struct.pack("Q", 42)
        result = AsyncDiagramLoader._decode_pair_count(header, Path("test.bin"))
        assert result == 42

    def test_decode_pair_count_short_header(self):
        from pynerve._async_loader import AsyncDiagramLoader
        with pytest.raises(ValueError, match="incomplete"):
            AsyncDiagramLoader._decode_pair_count(b"short", Path("test.bin"))

    def test_decode_binary_payload_valid(self):
        from pynerve._async_loader import AsyncDiagramLoader
        diag = np.array([[0.0, 1.0, 0]], dtype=np.float32)
        result = AsyncDiagramLoader._decode_binary_payload(diag.tobytes(), 1, Path("test.bin"))
        assert result.shape == (1, 3)

    def test_decode_binary_payload_wrong_size(self):
        from pynerve._async_loader import AsyncDiagramLoader
        with pytest.raises(ValueError, match="incomplete"):
            AsyncDiagramLoader._decode_binary_payload(b"short", 10, Path("test.bin"))

    def test_validate_diagram_array_empty(self):
        from pynerve._async_loader import _validate_diagram_array
        result = _validate_diagram_array(np.empty((0, 0)), Path("test.npy"))
        assert result.shape == (0, 3)
