from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pynerve._compute_core import PersistenceResult
from pynerve._fallback_classes import PersistenceBackend
from pynerve._streaming_persistence import (
    StreamingPersistence,
    _validate_streaming_array,
)
from pynerve.exceptions import InvalidArgumentError, ValidationError


class TestValidateStreamingArray:
    def test_accepts_valid_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_streaming_array(arr, "test")
        assert result.shape == (2, 2)
        assert result.dtype == arr.dtype

    def test_converts_list_to_array(self):
        result = _validate_streaming_array([[1.0, 2.0], [3.0, 4.0]], "test")
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_rejects_1d_array(self):
        with pytest.raises(InvalidArgumentError, match="2D array"):
            _validate_streaming_array(np.array([1.0, 2.0, 3.0]), "test")

    def test_rejects_empty_array_rows(self):
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            _validate_streaming_array(np.empty((0, 2)), "test")

    def test_rejects_empty_array_cols(self):
        with pytest.raises(InvalidArgumentError, match="non-empty"):
            _validate_streaming_array(np.empty((2, 0)), "test")

    def test_rejects_non_numeric_dtype(self):
        with pytest.raises(InvalidArgumentError, match="numeric dtype"):
            _validate_streaming_array(np.array([["a", "b"], ["c", "d"]]), "test")

    def test_rejects_nan(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(InvalidArgumentError, match="finite coordinates"):
            _validate_streaming_array(arr, "test")

    def test_rejects_inf(self):
        arr = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(InvalidArgumentError, match="finite coordinates"):
            _validate_streaming_array(arr, "test")

    def test_rejects_neg_inf(self):
        arr = np.array([[1.0, 2.0], [-np.inf, 4.0]])
        with pytest.raises(InvalidArgumentError, match="finite coordinates"):
            _validate_streaming_array(arr, "test")

    def test_includes_source_in_error(self):
        with pytest.raises(InvalidArgumentError, match="my_source data must"):
            _validate_streaming_array(np.array([1.0, 2.0]), "my_source")


class TestStreamingPersistenceInit:
    def test_defaults(self):
        sp = StreamingPersistence()
        assert sp.chunk_size == 1000
        assert sp.max_buffered == 3
        assert sp.use_gpu is True
        assert sp.persistence_kwargs == {}

    def test_custom_params(self):
        sp = StreamingPersistence(
            chunk_size=500,
            max_buffered_chunks=5,
            use_gpu=False,
            max_dim=1,
            max_radius=2.0,
        )
        assert sp.chunk_size == 500
        assert sp.max_buffered == 5
        assert sp.use_gpu is False
        assert sp.persistence_kwargs == {"max_dim": 1, "max_radius": 2.0}

    def test_rejects_zero_chunk_size(self):
        with pytest.raises(ValidationError, match="chunk_size"):
            StreamingPersistence(chunk_size=0)

    def test_rejects_negative_chunk_size(self):
        with pytest.raises(ValidationError, match="chunk_size"):
            StreamingPersistence(chunk_size=-1)

    def test_rejects_non_int_chunk_size(self):
        with pytest.raises(ValidationError, match="chunk_size"):
            StreamingPersistence(chunk_size=1.5)

    def test_rejects_bool_chunk_size(self):
        with pytest.raises(ValidationError, match="chunk_size"):
            StreamingPersistence(chunk_size=True)

    def test_rejects_zero_max_buffered(self):
        with pytest.raises(ValidationError, match="max_buffered_chunks"):
            StreamingPersistence(max_buffered_chunks=0)

    def test_rejects_negative_max_buffered(self):
        with pytest.raises(ValidationError, match="max_buffered_chunks"):
            StreamingPersistence(max_buffered_chunks=-1)

    def test_rejects_non_bool_use_gpu(self):
        with pytest.raises(ValidationError, match="use_gpu"):
            StreamingPersistence(use_gpu="yes")


class TestStreamingPersistenceComputeKwargs:
    def test_passthrough_gpu_enabled(self):
        sp = StreamingPersistence(use_gpu=True, max_dim=2)
        result = sp._compute_kwargs({"max_radius": 1.0})
        assert result == {"max_dim": 2, "max_radius": 1.0}
        assert "backend" not in result

    def test_cpu_default_backend(self):
        sp = StreamingPersistence(use_gpu=False, max_dim=2)
        result = sp._compute_kwargs({})
        assert result == {"max_dim": 2, "backend": PersistenceBackend.CPU_EXACT}

    def test_explicit_backend_overrides_cpu_default(self):
        sp = StreamingPersistence(use_gpu=False, max_dim=2)
        result = sp._compute_kwargs({"backend": PersistenceBackend.CUDA_HYBRID})
        assert result == {
            "max_dim": 2,
            "backend": PersistenceBackend.CUDA_HYBRID,
        }

    def test_overrides_merge_correctly(self):
        sp = StreamingPersistence(use_gpu=True, max_dim=2, max_radius=3.0)
        result = sp._compute_kwargs({"max_dim": 4, "seed": 42})
        assert result == {"max_dim": 4, "max_radius": 3.0, "seed": 42}

    def test_no_kwargs_cpu_backend(self):
        sp = StreamingPersistence(use_gpu=False)
        result = sp._compute_kwargs({})
        assert result == {"backend": PersistenceBackend.CPU_EXACT}

    def test_no_kwargs_gpu_enabled(self):
        sp = StreamingPersistence(use_gpu=True)
        result = sp._compute_kwargs({})
        assert result == {}


class TestStreamingPersistenceStreamComputeValidation:
    def test_rejects_invalid_return_format(self):
        sp = StreamingPersistence()
        import asyncio

        async def run():
            async for _ in sp.stream_compute("data", return_format="xml"):
                pass

        with pytest.raises(InvalidArgumentError, match="return_format"):
            asyncio.run(run())

    def test_rejects_non_async_iterator_data_source(self):
        sp = StreamingPersistence()
        import asyncio

        async def run():
            async for _ in sp.stream_compute(42, return_format="diagrams"):
                pass

        with pytest.raises(InvalidArgumentError, match="data_source"):
            asyncio.run(run())

    def test_rejects_none_data_source(self):
        sp = StreamingPersistence()
        import asyncio

        async def run():
            async for _ in sp.stream_compute(None, return_format="diagrams"):
                pass

        with pytest.raises(InvalidArgumentError, match="data_source"):
            asyncio.run(run())


class TestStreamingPersistenceFormatResult:
    def test_diagrams_passthrough(self):
        sp = StreamingPersistence()
        result = {"pairs": [[0.0, 1.0, 0]]}
        assert sp._format_result(result, "diagrams") is result

    def test_betti_delegates(self):
        sp = StreamingPersistence()
        result = {"pairs": [[0.0, float("inf"), 0]]}
        formatted = sp._format_result(result, "betti")
        assert "betti_0" in formatted

    def test_stats_delegates(self):
        sp = StreamingPersistence()
        result = {"pairs": [[0.0, 1.0, 0], [0.5, 2.0, 0]]}
        formatted = sp._format_result(result, "stats")
        assert "num_features" in formatted
        assert "avg_persistence" in formatted
        assert "max_persistence" in formatted


class TestStreamingPersistencePairArray:
    def test_dict_with_pairs_key(self):
        sp = StreamingPersistence()
        result = {"pairs": [[0.0, 1.0, 0]]}
        arr = sp._pair_array(result)
        assert arr.shape == (1, 3)
        assert arr[0, 0] == 0.0
        assert arr[0, 1] == 1.0
        assert arr[0, 2] == 0.0

    def test_dict_without_pairs_key(self):
        sp = StreamingPersistence()
        result = {"not_pairs": []}
        with pytest.raises(InvalidArgumentError, match="pairs"):
            sp._pair_array(result)

    def test_persistence_result_instance(self):
        sp = StreamingPersistence()
        pr = PersistenceResult(pairs=[(0.0, 1.0, 0), (0.5, 2.0, 1)])
        arr = sp._pair_array(pr)
        assert arr.shape == (2, 3)
        assert arr[0, 0] == 0.0
        assert arr[1, 2] == 1.0

    def test_persistence_result_empty_pairs(self):
        sp = StreamingPersistence()
        pr = PersistenceResult(pairs=[])
        arr = sp._pair_array(pr)
        assert arr.shape == (0, 3)

    def test_raw_list_of_tuples(self):
        sp = StreamingPersistence()
        result = [(0.0, 1.0, 0), (1.0, float("inf"), 1)]
        arr = sp._pair_array(result)
        assert arr.shape == (2, 3)

    def test_empty_list(self):
        sp = StreamingPersistence()
        result = []
        arr = sp._pair_array(result)
        assert arr.shape == (0, 3)

    def test_raw_array(self):
        sp = StreamingPersistence()
        result = np.array([[0.0, 1.0, 0], [1.0, 2.0, 1]])
        arr = sp._pair_array(result)
        assert arr.shape == (2, 3)


class TestStreamingPersistenceDiagramsToBetti:
    def test_dict_with_precomputed_betti(self):
        sp = StreamingPersistence()
        result = {"pairs": [], "betti_numbers": [3, 2, 0]}
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 3, "betti_1": 2, "betti_2": 0}

    def test_persistence_result_with_betti(self):
        sp = StreamingPersistence()
        pr = PersistenceResult(pairs=[], betti_numbers=[5, 1])
        betti = sp._diagrams_to_betti(pr)
        assert betti == {"betti_0": 5, "betti_1": 1}

    def test_raw_pairs_with_infinite_death(self):
        sp = StreamingPersistence()
        result = [[0.0, float("inf"), 0], [0.5, float("inf"), 1]]
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 1, "betti_1": 1}

    def test_raw_pairs_all_finite_deaths(self):
        sp = StreamingPersistence()
        result = [[0.0, 1.0, 0], [0.5, 2.0, 0]]
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 0}

    def test_raw_pairs_large_death(self):
        sp = StreamingPersistence()
        result = [[0.0, 1e10, 0]]
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 1}

    def test_raw_pairs_mixed_deaths(self):
        sp = StreamingPersistence()
        result = [[0.0, float("inf"), 0], [0.5, 1.0, 0], [1.0, 2.0, 1]]
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 1, "betti_1": 0}

    def test_empty_pairs(self):
        sp = StreamingPersistence()
        result = np.empty((0, 3))
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 0}

    def test_pairs_only_two_columns_raises(self):
        sp = StreamingPersistence()
        result = [[0.0, 1.0]]
        with pytest.raises(ValueError, match="betti conversion"):
            sp._diagrams_to_betti(result)

    def test_gaps_in_dimensions(self):
        sp = StreamingPersistence()
        result = [[0.0, float("inf"), 2]]
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 0, "betti_1": 0, "betti_2": 1}

    def test_single_dim_zero(self):
        sp = StreamingPersistence()
        result = [[0.0, float("inf"), 0], [0.5, float("inf"), 0], [1.0, float("inf"), 0]]
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 3}

    def test_dict_betti_overrides_pairs(self):
        sp = StreamingPersistence()
        result = {"pairs": [[0.0, float("inf"), 1]], "betti_numbers": [2, 1]}
        betti = sp._diagrams_to_betti(result)
        assert betti == {"betti_0": 2, "betti_1": 1}


class TestStreamingPersistenceDiagramsToStats:
    def test_standard_pairs(self):
        sp = StreamingPersistence()
        result = [[0.0, 1.0, 0], [0.5, 2.0, 0], [10.0, 12.0, 1]]
        stats = sp._diagrams_to_stats(result)
        assert stats["num_features"] == 3
        assert stats["avg_persistence"] == pytest.approx((1.0 + 1.5 + 2.0) / 3)
        assert stats["max_persistence"] == pytest.approx(2.0)

    def test_all_infinite_deaths(self):
        sp = StreamingPersistence()
        result = [[0.0, float("inf"), 0], [1.0, float("inf"), 1]]
        stats = sp._diagrams_to_stats(result)
        assert stats["num_features"] == 2
        assert stats["avg_persistence"] == 0.0
        assert stats["max_persistence"] == 0.0

    def test_mixed_finite_and_infinite(self):
        sp = StreamingPersistence()
        result = [[0.0, 1.0, 0], [0.5, float("inf"), 1]]
        stats = sp._diagrams_to_stats(result)
        assert stats["num_features"] == 2
        assert stats["avg_persistence"] == pytest.approx(1.0)
        assert stats["max_persistence"] == pytest.approx(1.0)

    def test_empty_pairs(self):
        sp = StreamingPersistence()
        result = np.empty((0, 3))
        stats = sp._diagrams_to_stats(result)
        assert stats["num_features"] == 0
        assert stats["avg_persistence"] == 0.0
        assert stats["max_persistence"] == 0.0

    def test_single_pair_finite(self):
        sp = StreamingPersistence()
        result = [[5.0, 8.0, 0]]
        stats = sp._diagrams_to_stats(result)
        assert stats["num_features"] == 1
        assert stats["avg_persistence"] == pytest.approx(3.0)
        assert stats["max_persistence"] == pytest.approx(3.0)

    def test_zero_persistence(self):
        sp = StreamingPersistence()
        result = [[1.0, 1.0, 0]]
        stats = sp._diagrams_to_stats(result)
        assert stats["num_features"] == 1
        assert stats["avg_persistence"] == pytest.approx(0.0)
        assert stats["max_persistence"] == pytest.approx(0.0)

    def test_negative_birth_finite_deaths(self):
        sp = StreamingPersistence()
        result = [[-2.0, 5.0, 0], [-1.0, 3.0, 1]]
        stats = sp._diagrams_to_stats(result)
        assert stats["num_features"] == 2
        assert stats["avg_persistence"] == pytest.approx((7.0 + 4.0) / 2)
        assert stats["max_persistence"] == pytest.approx(7.0)

    def test_persistence_result_input(self):
        sp = StreamingPersistence()
        pr = PersistenceResult(pairs=[(0.0, 1.0, 0), (1.0, 3.0, 0)])
        stats = sp._diagrams_to_stats(pr)
        assert stats["num_features"] == 2
        assert stats["avg_persistence"] == pytest.approx(1.5)
        assert stats["max_persistence"] == pytest.approx(2.0)


class TestStreamingPersistenceComputeChunk:
    def test_rejects_invalid_chunk(self):
        sp = StreamingPersistence()
        with pytest.raises(InvalidArgumentError, match="2D array"):
            sp._compute_chunk(np.array([1.0, 2.0]), {})

    def test_rejects_chunk_with_nan(self):
        sp = StreamingPersistence()
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(InvalidArgumentError, match="stream chunk data"):
            sp._compute_chunk(arr, {})

    def test_valid_chunk(self, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(chunk_size=100)
        arr = np.random.default_rng(42).random((10, 2))
        result = sp._compute_chunk(arr, {"max_dim": 0, "max_radius": 1.0})
        assert isinstance(result, PersistenceResult)
        assert result.max_dim == 0


class TestStreamingPersistenceStreamFromFile:
    def test_rejects_unsupported_format(self, tmp_path: Path):
        sp = StreamingPersistence()
        filepath = tmp_path / "data.csv"
        filepath.write_text("1,2,3\n4,5,6\n")
        import asyncio

        async def run():
            async for _ in sp._stream_from_file(str(filepath)):
                pass

        with pytest.raises(InvalidArgumentError, match="Unsupported streaming file format"):
            asyncio.run(run())

    def test_rejects_nonexistent_file(self):
        sp = StreamingPersistence()
        import asyncio

        async def run():
            async for _ in sp._stream_from_file("/nonexistent/path/data.npy"):
                pass

        with pytest.raises((FileNotFoundError, OSError)):
            asyncio.run(run())

    def test_yields_chunks_from_npy(self, tmp_path: Path):
        sp = StreamingPersistence(chunk_size=3)
        data = np.random.default_rng(42).random((7, 4))
        filepath = tmp_path / "data.npy"
        np.save(str(filepath), data)
        import asyncio

        async def run():
            chunks = []
            async for chunk in sp._stream_from_file(str(filepath)):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 3
        assert chunks[0].shape == (3, 4)
        assert chunks[1].shape == (3, 4)
        assert chunks[2].shape == (1, 4)
        assert np.allclose(np.concatenate(chunks), data)

    def test_yields_single_chunk_smaller_than_chunk_size(self, tmp_path: Path):
        sp = StreamingPersistence(chunk_size=100)
        data = np.random.default_rng(42).random((5, 2))
        filepath = tmp_path / "data.npy"
        np.save(str(filepath), data)
        import asyncio

        async def run():
            chunks = []
            async for chunk in sp._stream_from_file(str(filepath)):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        assert chunks[0].shape == (5, 2)

    def test_yields_chunks_from_npz_with_data_key(self, tmp_path: Path):
        sp = StreamingPersistence(chunk_size=3)
        data = np.random.default_rng(42).random((5, 3))
        filepath = tmp_path / "data.npz"
        np.savez(str(filepath), data=data)
        import asyncio

        async def run():
            chunks = []
            async for chunk in sp._stream_from_file(str(filepath)):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 2
        combined = np.concatenate(chunks)
        assert np.allclose(combined, data)

    def test_yields_chunks_from_npz_without_data_key(self, tmp_path: Path):
        sp = StreamingPersistence(chunk_size=3)
        data = np.random.default_rng(42).random((5, 2))
        filepath = tmp_path / "data.npz"
        np.savez(str(filepath), my_array=data)
        import asyncio

        async def run():
            chunks = []
            async for chunk in sp._stream_from_file(str(filepath)):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 2
        combined = np.concatenate(chunks)
        assert np.allclose(combined, data)

    def test_rejects_empty_npz(self, tmp_path: Path):
        sp = StreamingPersistence()
        filepath = tmp_path / "empty.npz"
        np.savez(str(filepath))
        import asyncio

        async def run():
            async for _ in sp._stream_from_file(str(filepath)):
                pass

        with pytest.raises(InvalidArgumentError, match="does not contain any arrays"):
            asyncio.run(run())


class TestStreamingPersistenceIntegration:
    def test_full_pipeline_with_async_iterator(self, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        rng = np.random.default_rng(42)
        data = rng.random((23, 2))
        import asyncio

        async def chunk_generator():
            for start in range(0, len(data), 10):
                yield data[start : start + 10]

        async def run():
            results = []
            async for result in sp.stream_compute(chunk_generator(), return_format="diagrams"):
                results.append(result)
            return results

        results = asyncio.run(run())
        assert len(results) == 3

    def test_full_pipeline_with_betti_format(self, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        rng = np.random.default_rng(42)
        data = rng.random((15, 2))
        import asyncio

        async def chunk_generator():
            for start in range(0, len(data), 10):
                yield data[start : start + 10]

        async def run():
            results = []
            async for result in sp.stream_compute(chunk_generator(), return_format="betti"):
                results.append(result)
            return results

        results = asyncio.run(run())
        assert len(results) == 2
        for r in results:
            assert any(k.startswith("betti_") for k in r)

    def test_full_pipeline_with_stats_format(self, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        rng = np.random.default_rng(42)
        data = rng.random((12, 3))
        import asyncio

        async def chunk_generator():
            for start in range(0, len(data), 10):
                yield data[start : start + 10]

        async def run():
            results = []
            async for result in sp.stream_compute(chunk_generator(), return_format="stats"):
                results.append(result)
            return results

        results = asyncio.run(run())
        assert len(results) == 2
        for r in results:
            assert "num_features" in r
            assert "avg_persistence" in r
            assert "max_persistence" in r

    def test_stream_from_npy_file(self, tmp_path: Path, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(chunk_size=5, use_gpu=False)
        rng = np.random.default_rng(42)
        data = rng.random((12, 2))
        filepath = tmp_path / "data.npy"
        np.save(str(filepath), data)
        import asyncio

        async def run():
            results = []
            async for result in sp.stream_compute(str(filepath), return_format="diagrams"):
                results.append(result)
            return results

        results = asyncio.run(run())
        assert len(results) == 3

    def test_stream_from_npz_file(self, tmp_path: Path, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(chunk_size=6, use_gpu=False)
        rng = np.random.default_rng(42)
        data = rng.random((14, 2))
        filepath = tmp_path / "data.npz"
        np.savez(str(filepath), data=data)
        import asyncio

        async def run():
            results = []
            async for result in sp.stream_compute(str(filepath), return_format="diagrams"):
                results.append(result)
            return results

        results = asyncio.run(run())
        assert len(results) == 3

    def test_persistence_kwargs_forwarded(self, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(
            chunk_size=10,
            use_gpu=False,
            max_dim=0,
            max_radius=1.0,
            backend=PersistenceBackend.CPU_EXACT,
        )
        rng = np.random.default_rng(42)
        data = rng.random((8, 2))
        import asyncio

        async def chunk_generator():
            yield data

        async def run():
            results = []
            async for result in sp.stream_compute(chunk_generator(), return_format="diagrams"):
                results.append(result)
            return results

        results = asyncio.run(run())
        assert len(results) == 1
        assert isinstance(results[0], PersistenceResult)

    def test_uses_cpu_exact_when_gpu_disabled(self, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(chunk_size=10, use_gpu=False, max_dim=0)
        kwargs = sp._compute_kwargs({})
        assert kwargs.get("backend") == PersistenceBackend.CPU_EXACT

    def test_all_return_formats_stream(self, nerve_core: None):  # noqa: ARG002
        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        rng = np.random.default_rng(42)
        data = rng.random((8, 2))
        import asyncio

        async def chunk_generator():
            yield data

        async def run(fmt):
            results = []
            async for result in sp.stream_compute(chunk_generator(), return_format=fmt):
                results.append(result)
            return results

        for fmt in ("diagrams", "betti", "stats"):
            results = asyncio.run(run(fmt))
            assert len(results) == 1


class TestStreamingPersistenceHdf5:
    def test_hdf5_missing_h5py_raises_runtime_error(self, tmp_path: Path):
        sp = StreamingPersistence()
        import sys

        if "h5py" in sys.modules:
            pytest.skip("h5py is already imported")

        with pytest.raises((RuntimeError, ImportError)):
            import asyncio

            async def run():
                async for _ in sp._stream_from_hdf5(tmp_path / "test.h5"):
                    pass

            asyncio.run(run())
