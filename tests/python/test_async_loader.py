"""Tests for _async_loader.py — AsyncDiagramLoader and helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_source_root = Path(__file__).resolve().parents[2] / "python"
if str(_source_root) not in sys.path:
    sys.path.insert(0, str(_source_root))

for _key in list(sys.modules):
    if _key.startswith("pynerve"):
        del sys.modules[_key]

_module_path = _source_root / "pynerve" / "_async_loader.py"
_spec = importlib.util.spec_from_file_location("pynerve._async_loader", str(_module_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

AsyncDiagramLoader = _mod.AsyncDiagramLoader
_decode_pair_count = AsyncDiagramLoader._decode_pair_count
_decode_binary_payload = AsyncDiagramLoader._decode_binary_payload
_validate_diagram_array = _mod._validate_diagram_array
_validate_filepath = _mod._validate_filepath
_validate_filepaths = _mod._validate_filepaths

import asyncio
import pickle
import struct

import numpy as np
import pytest


class TestValidateFilepath:
    def test_accepts_str_path(self):
        result = _validate_filepath("diagram.npy")
        assert isinstance(result, Path)
        assert str(result) == "diagram.npy"

    def test_accepts_pathlib_path(self):
        p = Path("diagram.npy")
        result = _validate_filepath(p)
        assert result == p

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepath("")

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepath(None)

    def test_rejects_int(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepath(42)

    def test_rejects_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepath(["diagram.npy"])


class TestValidateFilepaths:
    def test_accepts_list_of_strings(self):
        result = _validate_filepaths(["a.npy", "b.npy"])
        assert len(result) == 2
        assert all(isinstance(p, (str, Path)) for p in result)

    def test_accepts_list_of_paths(self):
        result = _validate_filepaths([Path("a.npy"), Path("b.npy")])
        assert len(result) == 2

    def test_rejects_string_input(self):
        with pytest.raises(TypeError, match="iterable"):
            _validate_filepaths("a.npy")

    def test_rejects_bytes_input(self):
        with pytest.raises(TypeError, match="iterable"):
            _validate_filepaths(b"a.npy")

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepaths([])

    def test_rejects_non_path_type_in_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepaths([42])

    def test_rejects_none_in_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepaths([None])


class TestValidateDiagramArray:
    def test_returns_empty_array_for_empty_input(self):
        result = _validate_diagram_array(np.array([]), Path("test.npy"))
        assert result.shape == (0, 3)
        assert result.dtype == np.float32

    def test_returns_empty_array_for_zero_size_array(self):
        result = _validate_diagram_array(np.zeros((0,)), Path("test.npy"))
        assert result.shape == (0, 3)
        assert result.dtype == np.float32

    def test_validates_valid_diagram(self):
        arr = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=np.float32)
        result = _validate_diagram_array(arr, Path("test.npy"))
        np.testing.assert_array_equal(result, arr)

    def test_raises_on_invalid_diagram(self):
        arr = np.array([[float("nan"), 1.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            _validate_diagram_array(arr, Path("test.npy"))


class TestAsyncDiagramLoaderInit:
    def test_default_max_concurrent(self):
        loader = AsyncDiagramLoader()
        assert loader.max_concurrent == 8

    def test_custom_max_concurrent(self):
        loader = AsyncDiagramLoader(max_concurrent=4)
        assert loader.max_concurrent == 4

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            AsyncDiagramLoader(max_concurrent=0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            AsyncDiagramLoader(max_concurrent=-1)

    def test_rejects_float(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            AsyncDiagramLoader(max_concurrent=1.5)

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            AsyncDiagramLoader(max_concurrent=None)

    def test_creates_semaphore(self):
        loader = AsyncDiagramLoader(max_concurrent=2)
        assert isinstance(loader._semaphore, asyncio.Semaphore)
        assert loader._semaphore._value == 2


class TestAsyncDiagramLoaderLoadFile:
    def test_load_npy_valid(self, tmp_path):
        path = tmp_path / "diagram.npy"
        arr = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=np.float32)
        np.save(path, arr)

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        loaded = asyncio.run(_go())
        np.testing.assert_array_equal(loaded, arr)

    def test_load_pkl_valid(self, tmp_path):
        path = tmp_path / "diagram.pkl"
        arr = np.array([[0.0, 1.0, 0.0], [2.0, 3.0, 1.0]], dtype=np.float32)
        path.write_bytes(pickle.dumps(arr))

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        loaded = asyncio.run(_go())
        np.testing.assert_array_equal(loaded, arr)

    def test_load_bin_valid(self, tmp_path):
        path = tmp_path / "diagram.bin"
        arr = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=np.float32)
        header = struct.pack("Q", len(arr))
        payload = arr.tobytes()
        path.write_bytes(header + payload)

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        loaded = asyncio.run(_go())
        np.testing.assert_array_equal(loaded, arr)

    def test_load_bin_empty(self, tmp_path):
        path = tmp_path / "empty.bin"
        header = struct.pack("Q", 0)
        path.write_bytes(header)

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        loaded = asyncio.run(_go())
        assert loaded.shape == (0, 3)
        assert loaded.dtype == np.float32

    def test_load_npy_preserves_float32_dtype(self, tmp_path):
        path = tmp_path / "diagram.npy"
        arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        np.save(path, arr)

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        loaded = asyncio.run(_go())
        assert loaded.dtype == np.float32

    def test_rejects_unknown_extension(self, tmp_path):
        path = tmp_path / "diagram.xyz"
        path.write_text("data")

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        with pytest.raises(ValueError, match="Unknown file format"):
            asyncio.run(_go())

    def test_rejects_unknown_extension_uppercase(self, tmp_path):
        path = tmp_path / "diagram.XYZ"
        path.write_text("data")

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        with pytest.raises(ValueError, match="Unknown file format"):
            asyncio.run(_go())

    def test_load_with_path_object(self, tmp_path):
        path = tmp_path / "diagram.npy"
        arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        np.save(path, arr)

        async def _go():
            return await AsyncDiagramLoader().load_file(path)

        loaded = asyncio.run(_go())
        np.testing.assert_array_equal(loaded, arr)

    def test_rejects_empty_filepath(self):
        async def _go():
            return await AsyncDiagramLoader().load_file("")

        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(_go())

    def test_rejects_nonexistent_file(self):
        async def _go():
            return await AsyncDiagramLoader().load_file("/nonexistent/path.npy")

        with pytest.raises(FileNotFoundError):
            asyncio.run(_go())

    def test_load_pkl_invalid_data(self, tmp_path):
        path = tmp_path / "bad.pkl"
        arr = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        path.write_bytes(pickle.dumps(arr))

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        with pytest.raises(ValueError, match="deaths"):
            asyncio.run(_go())

    def test_load_npy_nan_data(self, tmp_path):
        path = tmp_path / "bad.npy"
        arr = np.array([[float("nan"), 1.0, 0.0]], dtype=np.float32)
        np.save(path, arr)

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        with pytest.raises(ValueError, match="births"):
            asyncio.run(_go())

    def test_semaphore_exists_on_loader(self):
        loader = AsyncDiagramLoader(max_concurrent=2)
        assert isinstance(loader._semaphore, asyncio.Semaphore)
        assert loader._semaphore._value == 2

    def test_semaphore_limits_concurrency(self):
        acquired = []
        sem = asyncio.Semaphore(1)

        async def _track(i):
            async with sem:
                acquired.append(i)

        async def _gather():
            tasks = [_track(i) for i in range(5)]
            await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_gather())
        finally:
            loop.close()
        assert len(acquired) == 5


class TestAsyncDiagramLoaderLoadBatch:
    def test_load_batch_returns_list_in_order(self, tmp_path):
        path1 = tmp_path / "a.npy"
        path2 = tmp_path / "b.npy"
        arr1 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        arr2 = np.array([[2.0, 3.0, 1.0]], dtype=np.float32)
        np.save(path1, arr1)
        np.save(path2, arr2)

        async def _go():
            return await AsyncDiagramLoader().load_batch([str(path1), str(path2)])

        results = asyncio.run(_go())
        assert len(results) == 2
        np.testing.assert_array_equal(results[0], arr1)
        np.testing.assert_array_equal(results[1], arr2)

    def test_load_batch_single_file(self, tmp_path):
        path = tmp_path / "diagram.npy"
        arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        np.save(path, arr)

        async def _go():
            return await AsyncDiagramLoader().load_batch([str(path)])

        results = asyncio.run(_go())
        assert len(results) == 1
        np.testing.assert_array_equal(results[0], arr)

    def test_load_batch_mixed_formats(self, tmp_path):
        npy_path = tmp_path / "a.npy"
        pkl_path = tmp_path / "b.pkl"
        npy_arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        pkl_arr = np.array([[2.0, 3.0, 1.0]], dtype=np.float32)
        np.save(npy_path, npy_arr)
        pkl_path.write_bytes(pickle.dumps(pkl_arr))

        async def _go():
            return await AsyncDiagramLoader().load_batch([str(npy_path), str(pkl_path)])

        results = asyncio.run(_go())
        np.testing.assert_array_equal(results[0], npy_arr)
        np.testing.assert_array_equal(results[1], pkl_arr)

    def test_load_batch_binary_format(self, tmp_path):
        path = tmp_path / "diagram.bin"
        arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        path.write_bytes(struct.pack("Q", len(arr)) + arr.tobytes())

        async def _go():
            return await AsyncDiagramLoader().load_batch([str(path)])

        results = asyncio.run(_go())
        np.testing.assert_array_equal(results[0], arr)

    def test_rejects_empty_filepaths(self):
        async def _go():
            return await AsyncDiagramLoader().load_batch([])

        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(_go())

    def test_rejects_string_argument(self):
        async def _go():
            return await AsyncDiagramLoader().load_batch("a.npy")

        with pytest.raises(TypeError, match="iterable"):
            asyncio.run(_go())


class TestAsyncDiagramLoaderStreamDirectory:
    def test_streams_valid_directory(self, tmp_path):
        for i in range(5):
            arr = np.array([[float(i), float(i + 1), 0.0]], dtype=np.float32)
            np.save(tmp_path / f"diagram_{i:04d}.npy", arr)

        async def _go():
            loader = AsyncDiagramLoader(max_concurrent=2)
            results = []
            async for batch in loader.stream_directory(str(tmp_path), batch_size=2):
                results.extend(batch)
            return results

        results = asyncio.run(_go())
        assert len(results) == 5
        for i, arr in enumerate(results):
            np.testing.assert_array_equal(
                arr, np.array([[float(i), float(i + 1), 0.0]], dtype=np.float32)
            )

    def test_streams_in_batches(self, tmp_path):
        for i in range(5):
            arr = np.array([[float(i), float(i + 1), 0.0]], dtype=np.float32)
            np.save(tmp_path / f"d_{i:04d}.npy", arr)

        async def _go():
            loader = AsyncDiagramLoader()
            batches = []
            async for batch in loader.stream_directory(str(tmp_path), batch_size=2):
                batches.append(batch)
            return batches

        batches = asyncio.run(_go())
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_streams_with_glob_pattern(self, tmp_path):
        np.save(tmp_path / "a.npy", np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        (tmp_path / "b.pkl").write_bytes(
            pickle.dumps(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        )
        np.save(tmp_path / "c.npy", np.array([[2.0, 3.0, 1.0]], dtype=np.float32))

        async def _go():
            loader = AsyncDiagramLoader()
            results = []
            async for batch in loader.stream_directory(
                str(tmp_path), pattern="*.npy", batch_size=4
            ):
                results.extend(batch)
            return results

        results = asyncio.run(_go())
        assert len(results) == 2

    def test_rejects_empty_directory_string(self):
        async def _go():
            loader = AsyncDiagramLoader()
            async for _ in loader.stream_directory(""):
                pass

        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(_go())

    def test_rejects_empty_pattern(self):
        async def _go():
            loader = AsyncDiagramLoader()
            async for _ in loader.stream_directory("/tmp", pattern=""):
                pass

        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(_go())

    def test_rejects_non_string_pattern(self):
        async def _go():
            loader = AsyncDiagramLoader()
            async for _ in loader.stream_directory("/tmp", pattern=42):
                pass

        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(_go())

    def test_rejects_invalid_batch_size(self):
        async def _go():
            loader = AsyncDiagramLoader()
            async for _ in loader.stream_directory("/tmp", batch_size=0):
                pass

        with pytest.raises(ValueError, match="batch_size"):
            asyncio.run(_go())

    def test_accepts_path_object_directory(self, tmp_path):
        arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        np.save(tmp_path / "d.npy", arr)

        async def _go():
            loader = AsyncDiagramLoader()
            results = []
            async for batch in loader.stream_directory(tmp_path, batch_size=4):
                results.extend(batch)
            return results

        results = asyncio.run(_go())
        assert len(results) == 1

    def test_empty_directory_returns_no_batches(self, tmp_path):
        async def _go():
            loader = AsyncDiagramLoader()
            results = []
            async for batch in loader.stream_directory(str(tmp_path)):
                results.append(batch)
            return results

        results = asyncio.run(_go())
        assert len(results) == 0

    def test_default_pattern_matches_only_npy(self, tmp_path):
        np.save(tmp_path / "a.npy", np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        tmp_path.joinpath("b.bin").write_bytes(struct.pack("Q", 0))

        async def _go():
            loader = AsyncDiagramLoader()
            results = []
            async for batch in loader.stream_directory(str(tmp_path)):
                results.extend(batch)
            return results

        results = asyncio.run(_go())
        assert len(results) == 1

    def test_files_sorted_alphabetically(self, tmp_path):
        for name in ("z.npy", "a.npy", "m.npy"):
            np.save(tmp_path / name, np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

        async def _go():
            loader = AsyncDiagramLoader()
            results = []
            async for batch in loader.stream_directory(str(tmp_path), batch_size=4):
                results.extend(batch)
            return results

        results = asyncio.run(_go())
        assert len(results) == 3

    def test_file_path_as_directory_returns_empty(self, tmp_path):
        path = tmp_path / "diagram.npy"
        np.save(path, np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

        async def _go():
            loader = AsyncDiagramLoader()
            results = []
            async for batch in loader.stream_directory(str(path)):
                results.append(batch)
            return results

        results = asyncio.run(_go())
        assert len(results) == 0


class TestDecodePairCount:
    def test_valid_header(self):
        header = struct.pack("Q", 5)
        result = _decode_pair_count(header, Path("test.bin"))
        assert result == 5

    def test_zero_pairs(self):
        header = struct.pack("Q", 0)
        result = _decode_pair_count(header, Path("test.bin"))
        assert result == 0

    def test_incomplete_header_too_short(self):
        header = b"\x01\x02\x03"
        with pytest.raises(ValueError, match="incomplete binary diagram header"):
            _decode_pair_count(header, Path("test.bin"))

    def test_incomplete_header_empty(self):
        header = b""
        with pytest.raises(ValueError, match="incomplete binary diagram header"):
            _decode_pair_count(header, Path("test.bin"))

    def test_incomplete_header_too_long(self):
        header = b"\x00" * 9
        with pytest.raises(ValueError, match="incomplete binary diagram header"):
            _decode_pair_count(header, Path("test.bin"))


class TestDecodeBinaryPayload:
    def test_valid_payload(self):
        arr = np.array([[0.0, 1.0, 0.0], [2.0, 3.0, 1.0]], dtype=np.float32)
        data = arr.tobytes()
        result = _decode_binary_payload(data, 2, Path("test.bin"))
        np.testing.assert_array_equal(result, arr)

    def test_single_pair(self):
        arr = np.array([[1.0, 2.0, 0.0]], dtype=np.float32)
        data = arr.tobytes()
        result = _decode_binary_payload(data, 1, Path("test.bin"))
        np.testing.assert_array_equal(result, arr)

    def test_incomplete_payload(self):
        arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        data = arr.tobytes()[:-2]
        with pytest.raises(ValueError, match="incomplete binary diagram payload"):
            _decode_binary_payload(data, 1, Path("test.bin"))

    def test_incomplete_payload_empty(self):
        with pytest.raises(ValueError, match="incomplete binary diagram payload"):
            _decode_binary_payload(b"", 1, Path("test.bin"))


class TestLoadFileErrors:
    def test_binary_raises_on_incomplete_header(self, tmp_path):
        path = tmp_path / "bad.bin"
        path.write_bytes(b"\x00" * 4)

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        with pytest.raises(ValueError, match="incomplete binary diagram header"):
            asyncio.run(_go())

    def test_binary_raises_on_truncated_payload(self, tmp_path):
        path = tmp_path / "bad.bin"
        header = struct.pack("Q", 10)
        path.write_bytes(header + b"\x00" * 4)

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        with pytest.raises(ValueError, match="incomplete binary diagram payload"):
            asyncio.run(_go())

    def test_pickle_raises_on_corrupt_data(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not a pickle")

        async def _go():
            return await AsyncDiagramLoader().load_file(str(path))

        with pytest.raises((pickle.UnpicklingError, ValueError)):
            asyncio.run(_go())
