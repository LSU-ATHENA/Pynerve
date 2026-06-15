"""Comprehensive correctness tests for _torch_diagrams and _streaming_persistence."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pynerve._torch_diagrams import (  # noqa: E402
    birth_death,
    encode_diagram_embedding,
    encode_diagram_rows,
    encoder_output_dim,
    persistence,
    validate_diagram,
)
from pynerve.exceptions import ValidationError  # noqa: E402


class _LinearEncoder(torch.nn.Module):
    """Encoder that projects (batch, n_pairs, cols) -> (batch, n_pairs, output_dim)."""

    def __init__(self, input_dim: int = 2, output_dim: int = 8):
        super().__init__()
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _GraphEncoder(torch.nn.Module):
    """Encoder that returns a graph-level 2D output (1, d) via mean pooling + linear."""

    def __init__(self, input_dim: int = 2, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1, n_pairs, cols) -> (1, output_dim)
        pooled = x.mean(dim=1)  # (1, cols)
        return self.linear(pooled)  # (1, output_dim)


class _VecEncoder(torch.nn.Module):
    """Encoder that returns a 1D vector (d,) without batch dimension."""

    output_dim: int = 4

    def __init__(self, output_dim: int = 4):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.output_dim, device=x.device)


class _NoAttrEncoder(torch.nn.Module):
    """Encoder without output_dim attribute."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# torch diagrams


class TestBirthDeath:
    """birth_death() and persistence() correctness."""

    def test_extracts_first_two_columns(self):
        d = torch.tensor([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=torch.float64)
        bd = birth_death(d)
        assert bd.shape == (2, 2)
        assert bd.dtype == d.dtype
        torch.testing.assert_close(bd, d[:, :2])

    def test_works_with_extra_columns(self):
        d = torch.tensor([[0.1, 0.9, 0.0, 42.0], [0.3, 0.7, 1.0, 99.0]], dtype=torch.float32)
        bd = birth_death(d)
        torch.testing.assert_close(bd, d[:, :2])

    def test_single_pair(self):
        d = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
        bd = birth_death(d)
        assert bd.shape == (1, 2)

    def test_integer_tensor(self):
        d = torch.tensor([[0, 5], [2, 3]], dtype=torch.int64)
        bd = birth_death(d)
        assert bd.shape == (2, 2)
        assert bd.dtype == torch.int64

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="shape"):
            birth_death(torch.tensor([0.0, 1.0]))

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="shape"):
            birth_death(torch.zeros(1, 2, 3))

    def test_rejects_single_column(self):
        with pytest.raises(ValueError, match="shape"):
            birth_death(torch.tensor([[0.0]]))


class TestPersistence:
    """persistence() computes death - birth."""

    def test_persistence_basic(self):
        d = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        pers = persistence(d)
        assert pers.shape == (2,)
        torch.testing.assert_close(pers, torch.tensor([1.0, 1.5], dtype=torch.float64))

    def test_persistence_zero_persistence(self):
        d = torch.tensor([[0.5, 0.5], [3.0, 3.0]], dtype=torch.float32)
        pers = persistence(d)
        torch.testing.assert_close(pers, torch.zeros(2, dtype=torch.float32))

    def test_persistence_single_pair(self):
        d = torch.tensor([[2.0, 7.0]], dtype=torch.float64)
        pers = persistence(d)
        torch.testing.assert_close(pers, torch.tensor([5.0], dtype=torch.float64))

    def test_persistence_with_infinity(self):
        d = torch.tensor([[0.0, float("inf")], [1.0, 3.0]], dtype=torch.float64)
        pers = persistence(d)
        assert pers[0] == float("inf")
        torch.testing.assert_close(pers[1], torch.tensor(2.0, dtype=torch.float64))

    def test_persistence_preserves_dtype(self):
        for dtype in [torch.float32, torch.float64]:
            d = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=dtype)
            assert persistence(d).dtype == dtype

    def test_persistence_with_extra_columns(self):
        d = torch.tensor([[0.0, 3.0, 0.0], [1.0, 4.0, 1.0]], dtype=torch.float32)
        pers = persistence(d)
        torch.testing.assert_close(pers, torch.tensor([3.0, 3.0], dtype=torch.float32))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="shape"):
            persistence(torch.tensor([0.0, 1.0]))


class TestValidateDiagram:
    """validate_diagram() input validation."""

    def test_accepts_valid_2d(self):
        d = torch.randn(10, 3)
        validate_diagram(d)  # no exception

    def test_accepts_exactly_min_cols(self):
        validate_diagram(torch.zeros(5, 2), min_cols=2)
        validate_diagram(torch.zeros(5, 4), min_cols=4)

    def test_rejects_1d_tensor(self):
        with pytest.raises(ValueError, match="shape"):
            validate_diagram(torch.randn(10))

    def test_rejects_3d_tensor(self):
        with pytest.raises(ValueError, match="shape"):
            validate_diagram(torch.randn(2, 10, 3))

    def test_rejects_0d_scalar(self):
        with pytest.raises(ValueError, match="shape"):
            validate_diagram(torch.tensor(1.0))

    def test_rejects_too_few_cols(self):
        with pytest.raises(ValueError, match="shape"):
            validate_diagram(torch.zeros(5, 1), min_cols=2)

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="tensor"):
            validate_diagram([[0.0, 1.0]])

    def test_rejects_numpy_array(self):
        with pytest.raises(TypeError, match="tensor"):
            validate_diagram(np.array([[0.0, 1.0]]))

    def test_custom_name_in_error(self):
        with pytest.raises(ValueError, match="my_diagram"):
            validate_diagram(torch.tensor([0.0]), name="my_diagram")

    def test_custom_min_cols(self):
        d = torch.zeros(5, 3)
        validate_diagram(d, min_cols=3)
        with pytest.raises(ValueError, match="shape"):
            validate_diagram(d, min_cols=4)

    def test_empty_pairs_ok(self):
        validate_diagram(torch.empty(0, 2))

    def test_with_birth_gt_death_values(self):
        """validate_diagram does not check birth/death ordering."""
        d = torch.tensor([[5.0, 1.0], [3.0, 0.0]])
        validate_diagram(d)  # no exception -- only shape is checked


class TestEncoderOutputDim:
    """encoder_output_dim() correctness."""

    def test_returns_output_dim_attr(self):
        enc = _LinearEncoder(input_dim=2, output_dim=32)
        assert encoder_output_dim(enc) == 32

    def test_returns_default_when_no_attr(self):
        enc = _NoAttrEncoder()
        assert encoder_output_dim(enc) == 64

    def test_returns_default_with_custom_default(self):
        enc = _NoAttrEncoder()
        assert encoder_output_dim(enc, default=128) == 128

    def test_returns_int(self):
        enc = _LinearEncoder(input_dim=2, output_dim=64)
        assert isinstance(encoder_output_dim(enc), int)


class TestEncodeDiagramRows:
    """encode_diagram_rows() encodes diagram row-by-row."""

    def test_3d_encoder_output_preserved(self):
        enc = _LinearEncoder(input_dim=2, output_dim=4)
        d = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
        result = encode_diagram_rows(enc, d)
        assert result.shape == (2, 4)

    def test_3d_encoder_output_with_extra_cols(self):
        enc = _LinearEncoder(input_dim=3, output_dim=4)
        d = torch.tensor([[0.0, 1.0, 0.0], [0.5, 2.0, 1.0]], dtype=torch.float32)
        result = encode_diagram_rows(enc, d, min_cols=3)
        assert result.shape == (2, 4)

    def test_2d_graph_output_expanded(self):
        enc = _GraphEncoder(input_dim=2, output_dim=8)
        d = torch.tensor([[0.0, 1.0], [0.5, 2.0], [0.2, 0.8]], dtype=torch.float32)
        result = encode_diagram_rows(enc, d)
        assert result.shape == (3, 8)
        # all rows should be identical (graph-level expanded)
        torch.testing.assert_close(result[0], result[1])
        torch.testing.assert_close(result[1], result[2])

    def test_1d_output_expanded(self):
        enc = _VecEncoder(output_dim=4)
        d = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
        result = encode_diagram_rows(enc, d)
        assert result.shape == (2, 4)
        torch.testing.assert_close(result[0], result[1])

    def test_2d_single_row_output_expanded(self):
        """2D output with shape (1, d) and 1 input pair."""
        enc = _GraphEncoder(input_dim=2, output_dim=8)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        result = encode_diagram_rows(enc, d)
        assert result.shape == (1, 8)

    def test_empty_diagram_linearencoder(self):
        enc = _LinearEncoder(input_dim=2, output_dim=4)
        d = torch.empty((0, 2), dtype=torch.float32)
        result = encode_diagram_rows(enc, d)
        assert result.shape == (0, 4)

    def test_empty_diagram_graph_encoder(self):
        enc = _GraphEncoder(input_dim=2, output_dim=8)
        d = torch.empty((0, 2), dtype=torch.float32)
        result = encode_diagram_rows(enc, d)
        assert result.ndim == 2
        assert result.shape[1] == 8

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="tensor"):
            encode_diagram_rows(torch.nn.Identity(), np.array([[0.0, 1.0]]))

    def test_rejects_3d_diagram(self):
        with pytest.raises(ValueError, match="shape"):
            encode_diagram_rows(torch.nn.Identity(), torch.zeros(1, 2, 3))

    def test_preserves_dtype(self):
        enc = _LinearEncoder(input_dim=2, output_dim=4).to(torch.float64)
        d = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        result = encode_diagram_rows(enc, d)
        assert result.dtype == torch.float64


class TestEncodeDiagramEmbedding:
    """encode_diagram_embedding() produces a mean-pooled vector."""

    def test_mean_pooling_basic(self):
        enc = _LinearEncoder(input_dim=2, output_dim=4)
        d = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
        embedding = encode_diagram_embedding(enc, d)
        assert embedding.shape == (4,)

    def test_empty_diagram_returns_zeros(self):
        enc = _LinearEncoder(input_dim=2, output_dim=6)
        d = torch.empty((0, 2), dtype=torch.float32)
        embedding = encode_diagram_embedding(enc, d)
        assert embedding.shape == (6,)
        torch.testing.assert_close(embedding, torch.zeros(6))

    def test_single_pair_equals_row(self):
        enc = _LinearEncoder(input_dim=2, output_dim=4)
        d = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        embedding = encode_diagram_embedding(enc, d)
        rows = encode_diagram_rows(enc, d)
        torch.testing.assert_close(embedding, rows.squeeze(0))

    def test_preserves_dtype(self):
        enc = _LinearEncoder(input_dim=2, output_dim=4).to(torch.float64)
        d = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        embedding = encode_diagram_embedding(enc, d)
        assert embedding.dtype == d.dtype

    def test_rejects_invalid_diagram(self):
        enc = _LinearEncoder(input_dim=2, output_dim=4)
        with pytest.raises(ValueError, match="shape"):
            encode_diagram_embedding(enc, torch.tensor([0.0, 1.0]))


def _pynerve_streaming_available() -> bool:
    """Check that pynerve can be imported and _core is present."""
    try:
        import pynerve

        return getattr(pynerve, "_core", None) is not None
    except ImportError:
        return False


async def _async_chunks(chunks: list[np.ndarray]) -> AsyncIterator[np.ndarray]:
    """Yield numpy chunks asynchronously."""
    for chunk in chunks:
        yield chunk


def _make_point_cloud(n_points: int, dim: int = 2, seed: int = 42) -> np.ndarray:
    """Generate a reproducible point cloud."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_points, dim)).astype(np.float64)


@pytest.mark.skipif(
    not _pynerve_streaming_available(), reason="nerve_internal C++ extension required"
)
# streaming persistence


class TestStreamingPersistenceBasic:
    """Basic StreamingPersistence construction and validation."""

    def test_constructs_with_defaults(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        assert sp.chunk_size == 1000
        assert sp.max_buffered == 3
        assert sp.use_gpu is True

    def test_constructs_with_custom_args(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=500, max_buffered_chunks=5, use_gpu=False)
        assert sp.chunk_size == 500
        assert sp.max_buffered == 5
        assert sp.use_gpu is False

    def test_constructs_with_persistence_kwargs(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(max_dim=2, max_radius=1.5)
        assert sp.persistence_kwargs == {"max_dim": 2, "max_radius": 1.5}

    def test_rejects_zero_chunk_size(self):
        from pynerve._streaming_persistence import StreamingPersistence

        with pytest.raises((TypeError, ValueError, ValidationError)):
            StreamingPersistence(chunk_size=0)

    def test_rejects_negative_chunk_size(self):
        from pynerve._streaming_persistence import StreamingPersistence

        with pytest.raises((TypeError, ValueError, ValidationError)):
            StreamingPersistence(chunk_size=-1)

    def test_rejects_zero_max_buffered(self):
        from pynerve._streaming_persistence import StreamingPersistence

        with pytest.raises((TypeError, ValueError, ValidationError)):
            StreamingPersistence(max_buffered_chunks=0)


@pytest.mark.skipif(
    not _pynerve_streaming_available(), reason="nerve_internal C++ extension required"
)
class TestStreamingComputeChunk:
    """_compute_chunk() processes individual chunks correctly."""

    def test_process_chunk_returns_result(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        data = _make_point_cloud(10)
        result = sp._compute_chunk(data, {})
        assert result is not None
        assert hasattr(result, "pairs")
        assert hasattr(result, "betti_numbers")

    def test_process_chunk_single_point(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=1, use_gpu=False)
        data = _make_point_cloud(1)
        result = sp._compute_chunk(data, {})
        assert result is not None

    def test_process_chunk_passes_kwargs(self):
        from pynerve._compute_core import PersistenceResult
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=100, use_gpu=False)
        data = _make_point_cloud(50)
        result = sp._compute_chunk(data, {"max_dim": 1, "max_radius": 1.0})
        assert isinstance(result, PersistenceResult)
        assert len(result.pairs) > 0

    def test_process_chunk_rejects_empty(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        with pytest.raises(Exception):  # noqa: B017
            sp._compute_chunk(np.empty((0, 2)), {})

    def test_process_chunk_rejects_1d(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        with pytest.raises(Exception):  # noqa: B017
            sp._compute_chunk(np.array([1.0, 2.0]), {})

    def test_process_chunk_rejects_non_finite(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        with pytest.raises(Exception):  # noqa: B017
            sp._compute_chunk(np.array([[0.0, float("nan")]]), {})

    def test_process_chunk_rejects_non_numeric_dtype(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=10, use_gpu=False)
        with pytest.raises(Exception):  # noqa: B017
            sp._compute_chunk(np.array([["a", "b"]]), {})


@pytest.mark.skipif(
    not _pynerve_streaming_available(), reason="nerve_internal C++ extension required"
)
# async generator helpers


class TestStreamComputeAsync:
    """stream_compute() yields results for async chunked datasets."""

    @pytest.mark.asyncio
    async def test_stream_compute_basic(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=50, max_buffered_chunks=2, use_gpu=False)
        data = _make_point_cloud(150)
        chunks = [data[i : i + 50] for i in range(0, 150, 50)]
        results = []
        async for result in sp.stream_compute(_async_chunks(chunks), return_format="diagrams"):
            results.append(result)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, dict) or hasattr(r, "pairs")

    @pytest.mark.asyncio
    async def test_stream_compute_single_chunk(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=100, max_buffered_chunks=2, use_gpu=False)
        data = _make_point_cloud(30)
        results = []
        async for result in sp.stream_compute(_async_chunks([data]), return_format="diagrams"):
            results.append(result)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_stream_compute_return_format_betti(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=50, use_gpu=False)
        data = _make_point_cloud(50)
        results = []
        async for result in sp.stream_compute(_async_chunks([data]), return_format="betti"):
            results.append(result)
        assert len(results) == 1
        # betti format returns dict with betti_0, betti_1, ...
        assert isinstance(results[0], dict)
        assert any(k.startswith("betti_") for k in results[0])

    @pytest.mark.asyncio
    async def test_stream_compute_return_format_stats(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=50, use_gpu=False)
        data = _make_point_cloud(50)
        results = []
        async for result in sp.stream_compute(_async_chunks([data]), return_format="stats"):
            results.append(result)
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert "num_features" in results[0]
        assert "avg_persistence" in results[0]
        assert "max_persistence" in results[0]

    @pytest.mark.asyncio
    async def test_stream_compute_rejects_invalid_return_format(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=50, use_gpu=False)
        data = _make_point_cloud(50)
        with pytest.raises(Exception):  # noqa: B017
            async for _ in sp.stream_compute(_async_chunks([data]), return_format="invalid"):
                pass

    @pytest.mark.asyncio
    async def test_total_result_collects_all_chunks(self):
        """Aggregated pairs from chunks should produce a non-empty set."""
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(
            chunk_size=25,
            max_buffered_chunks=2,
            use_gpu=False,
            max_dim=1,
            max_radius=float("inf"),
        )
        data = _make_point_cloud(75, seed=1)
        chunks = [data[i : i + 25] for i in range(0, 75, 25)]
        all_pairs: list[tuple[float, float, int]] = []
        async for result in sp.stream_compute(_async_chunks(chunks), return_format="diagrams"):
            if isinstance(result, dict):
                pairs = result.get("pairs", [])
            else:
                pairs = getattr(result, "pairs", [])
            all_pairs.extend(pairs)
        assert len(all_pairs) > 0

    @pytest.mark.asyncio
    async def test_single_element_chunks(self):
        """Extreme case: chunk_size=1, each chunk contains one point."""
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=1, max_buffered_chunks=1, use_gpu=False)
        data = _make_point_cloud(5, seed=3)
        chunks = [data[i : i + 1] for i in range(5)]
        results: list = []
        async for result in sp.stream_compute(_async_chunks(chunks), return_format="diagrams"):
            results.append(result)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_streaming_matches_non_streaming(self):
        """Full persistence on all data should produce at least the same betti
        counts as summing individual chunk betti numbers, since each chunk is
        computed independently."""
        from pynerve import compute_persistence
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(
            chunk_size=30,
            max_buffered_chunks=2,
            use_gpu=False,
            max_dim=1,
            max_radius=float("inf"),
        )
        data = _make_point_cloud(90, seed=7)
        chunks = [data[i : i + 30] for i in range(0, 90, 30)]

        # Collect betti from streaming
        streamed_betti: dict[int, int] = {}
        async for result in sp.stream_compute(_async_chunks(chunks), return_format="betti"):
            for k, v in result.items():
                dim = int(k.split("_")[1])
                streamed_betti[dim] = streamed_betti.get(dim, 0) + v

        # Full computation for reference
        full_result = compute_persistence(data, max_dim=1, max_radius=float("inf"))
        full_betti = full_result.betti_numbers
        full_count = sum(full_betti)

        # Total streamed features should be >= full (chunk boundaries can create
        # extra features at splits)
        streamed_count = sum(streamed_betti.values())
        assert streamed_count >= full_count

    @pytest.mark.asyncio
    async def test_streaming_with_empty_chunks_raises(self):
        """Empty chunks are rejected by validation."""
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence(chunk_size=50, use_gpu=False)
        empty = np.empty((0, 2), dtype=np.float64)
        with pytest.raises(Exception):  # noqa: B017
            async for _ in sp.stream_compute(_async_chunks([empty]), return_format="diagrams"):
                pass


@pytest.mark.skipif(
    not _pynerve_streaming_available(), reason="nerve_internal C++ extension required"
)
class TestAsyncPersistenceComputerWorkerCounts:
    """AsyncPersistenceComputer with different worker counts."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("workers", [1, 2, 4])
    async def test_different_worker_counts(self, workers: int):
        from pynerve._async_compute import AsyncPersistenceComputer
        from pynerve._compute_api import compute_persistence

        data = _make_point_cloud(40, seed=11)
        chunks = [data[i : i + 20] for i in range(0, 40, 20)]

        async with AsyncPersistenceComputer(max_workers=workers, buffer_size=2) as computer:
            results: list = []
            async for result in computer.compute_batch_async(
                _async_chunks(chunks),
                lambda chunk: compute_persistence(chunk, max_dim=1, max_radius=float("inf")),
            ):
                results.append(result)
            assert len(results) == 2
            for r in results:
                assert hasattr(r, "pairs")

    def test_async_computer_rejects_zero_workers(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        with pytest.raises((TypeError, ValueError, ValidationError)):
            AsyncPersistenceComputer(max_workers=0)

    def test_async_computer_rejects_non_async_iterator(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            async with AsyncPersistenceComputer(max_workers=1) as computer:
                with pytest.raises(TypeError, match="async iterator"):
                    async for _ in computer.compute_batch_async([np.array([[1.0, 2.0]])]):
                        pass

        asyncio.run(_run())

    @pytest.mark.asyncio
    async def test_async_computer_context_manager(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async with AsyncPersistenceComputer(max_workers=1, buffer_size=1) as computer:
            assert not computer._closed
        assert computer._closed

    @pytest.mark.asyncio
    async def test_async_computer_rejects_use_after_close(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        computer = AsyncPersistenceComputer(max_workers=1)
        await computer.close()
        assert computer._closed
        with pytest.raises(RuntimeError, match="closed"):
            async for _ in computer.compute_batch_async(_async_chunks([])):
                pass


@pytest.mark.skipif(
    not _pynerve_streaming_available(), reason="nerve_internal C++ extension required"
)
class TestStreamingFormatResults:
    """_format_result() and helper formatting."""

    def test_format_diagrams_passthrough(self):
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        result_in = {"pairs": [(0.0, 1.0, 0)], "betti_numbers": [1]}
        out = sp._format_result(result_in, "diagrams")
        assert out is result_in

    def test_format_betti_from_dict(self):
        from pynerve._compute_core import PersistenceResult
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        result = PersistenceResult(
            pairs=[(0.0, float("inf"), 0), (0.0, 1.0, 1)],
            betti_numbers=[1, 1],
        )
        out = sp._format_result(result, "betti")
        assert out == {"betti_0": 1, "betti_1": 1}

    def test_format_stats_from_result(self):
        from pynerve._compute_core import PersistenceResult
        from pynerve._streaming_persistence import StreamingPersistence

        sp = StreamingPersistence()
        result = PersistenceResult(
            pairs=[(0.0, 1.0, 0), (0.0, 2.0, 0), (0.0, float("inf"), 1)],
            betti_numbers=[1, 1],
        )
        out = sp._format_result(result, "stats")
        assert out["num_features"] == 3
        assert out["avg_persistence"] == pytest.approx(1.5)
        assert out["max_persistence"] == pytest.approx(2.0)
