from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pynerve.exceptions import InvalidArgumentError, ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_prng_key_rejects_invalid_public_inputs() -> None:
    from pynerve.random import PRNGKey, manual_seed, reproducible, split

    key = PRNGKey(123)
    assert key.normal(shape=(2, 3)).shape == (2, 3), (
        f"expected (2, 3), got {key.normal(shape=(2, 3)).shape}"
    )
    assert key.uniform(0.0, 1.0, shape=2).shape == (2,), (
        f"expected (2,), got {key.uniform(0.0, 1.0, shape=2).shape}"
    )
    assert key.randint(1, 4, size=(2,)).shape == (2,), (
        f"expected (2,), got {key.randint(1, 4, size=(2,)).shape}"
    )
    assert key.choice(np.array([1, 2, 3]), size=2).shape == (2,), (
        f"expected (2,), got {key.choice(np.array([1, 2, 3]), size=2).shape}"
    )
    assert key.permutation(3).shape == (3,), f"expected (3,), got {key.permutation(3).shape}"
    assert len(split(2)) == 2, f"expected 2, got {len(split(2))}"
    with reproducible(5) as seeded_key:
        assert isinstance(seeded_key, PRNGKey), f"expected PRNGKey, got {type(seeded_key).__name__}"

    with pytest.raises((TypeError, InvalidArgumentError), match="seed"):
        PRNGKey(1.5)
    with pytest.raises((ValueError, InvalidArgumentError), match="counter"):
        PRNGKey(1, counter=-1)
    with pytest.raises(ValidationError, match="n"):
        key.split(1.5)
    with pytest.raises(ValidationError, match="n"):
        key.split(0)
    with pytest.raises(ValidationError, match="shape"):
        key.normal(shape=(1.5,))
    with pytest.raises(ValidationError, match="shape"):
        key.normal(shape=(-1,))
    with pytest.raises(ValidationError, match="high"):
        key.uniform(0.0, float("nan"))
    with pytest.raises((TypeError, InvalidArgumentError), match="low"):
        key.randint(1.5)
    with pytest.raises((ValueError, InvalidArgumentError), match="high"):
        key.randint(4, 4)
    with pytest.raises((TypeError, InvalidArgumentError), match="replace"):
        key.choice([1, 2], replace=1)
    with pytest.raises((ValueError, InvalidArgumentError), match="p"):
        key.choice([1, 2], p=np.array([float("nan"), 1.0]))
    with pytest.raises((TypeError, InvalidArgumentError), match="seed"):
        manual_seed(1.2)


def test_pipeline_helpers_reject_invalid_diagrams_and_parameters() -> None:
    from pynerve.pipeline import Pipeline, analysis_pipeline, vr_pipeline

    def compute_valid(_):
        return {"pairs": [[0.0, 1.0, 0], [0.5, float("inf"), 0]]}

    result = analysis_pipeline(compute_valid, ["diagram", "vector"])(None)
    assert result["vector"] == [1.0, float("inf")], f"expected [1.0, inf], got {result['vector']}"

    with pytest.raises((TypeError, ValidationError, InvalidArgumentError), match="index"):
        Pipeline().insert_step(1.5, "noop", lambda value: value)
    with pytest.raises(ValidationError, match="max_dim"):
        vr_pipeline(max_dim=1.5)
    with pytest.raises(ValidationError, match="min_persistence"):
        vr_pipeline(min_persistence=float("nan"))
    with pytest.raises(ValidationError, match="max_radius"):
        vr_pipeline(max_radius=float("inf"))
    with pytest.raises((TypeError, ValidationError), match="representations"):
        analysis_pipeline(compute_valid, "vector")
    with pytest.raises((ValueError, InvalidArgumentError), match="representations"):
        analysis_pipeline(compute_valid, ["vector", "vector"])

    with pytest.raises((ValueError, ValidationError), match="births"):
        analysis_pipeline(lambda _: [[float("nan"), 1.0]], ["vector"])(None)
    with pytest.raises((ValueError, ValidationError), match="deaths"):
        analysis_pipeline(lambda _: [[1.0, 0.0]], ["vector"])(None)
    with pytest.raises((ValueError, ValidationError), match="deaths"):
        analysis_pipeline(lambda _: [[0.0, float("nan")]], ["vector"])(None)


def test_format_helpers_reject_invalid_diagrams_and_points(tmp_path: Path) -> None:
    from pynerve import formats

    diagram = [(0.0, 1.0, 0), (0.0, float("inf"), 1)]
    json_path = tmp_path / "diagram.json"
    formats.save_json(diagram, json_path)
    loaded = formats.load_json(json_path)
    assert loaded["diagram"][1][1] == float("inf"), f"expected inf, got {loaded['diagram'][1][1]}"
    assert '"death": null' in formats.to_external(diagram), (
        f"expected 'death': null in external output, got {formats.to_external(diagram)}"
    )

    with pytest.raises((ValueError, ValidationError), match="births"):
        formats.save_json([(float("nan"), 1.0, 0)], tmp_path / "bad_birth.json")
    with pytest.raises((ValueError, ValidationError), match="deaths"):
        formats.to_external([(1.0, 0.0, 0)])
    with pytest.raises((ValueError, ValidationError), match="births"):
        formats.from_external({"dgms": [np.array([[float("nan"), 1.0]])]})
    with pytest.raises((ValueError, ValidationError), match="deaths"):
        formats.from_gudhi([(0, (1.0, 0.0))])
    with pytest.raises((ValueError, ValidationError), match="dimensions"):
        formats.from_giotto(np.array([[0.0, 1.0, 1.5]]))

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("birth,death,dimension\nnan,1.0,0\n")
    with pytest.raises((ValueError, ValidationError), match="births"):
        formats.load_csv(csv_path)
    malformed_csv_path = tmp_path / "malformed.csv"
    malformed_csv_path.write_text("birth,death,dimension\n0.0,not-a-number,0\n")
    with pytest.raises(ValueError, match="CSV row 2"):
        formats.load_csv(malformed_csv_path)
    short_csv_path = tmp_path / "short.csv"
    short_csv_path.write_text("0.0,1.0\n")
    with pytest.raises(ValueError, match="CSV row 1"):
        formats.load_csv(short_csv_path)

    off_path = tmp_path / "bad.off"
    off_path.write_text("OFF\n1 0 0\nnan 0 0\n")
    with pytest.raises((ValueError, InvalidArgumentError), match="OFF vertex coordinates"):
        formats.load_off(off_path)
    with pytest.raises((ValueError, InvalidArgumentError), match="finite coordinates"):
        formats.save_off(np.array([[float("nan"), 0.0, 0.0]]), tmp_path / "bad_save.off")
    with pytest.raises((ValueError, InvalidArgumentError), match="face indices"):
        formats.save_off(np.zeros((1, 3)), tmp_path / "bad_face.off", faces=[[1, 0, 0]])

    ply_path = tmp_path / "bad.ply"
    ply_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "0 nan 0\n"
    )
    with pytest.raises((ValueError, InvalidArgumentError), match="PLY vertex coordinates"):
        formats.load_ply(ply_path)


def test_fast_numpy_helpers_reject_invalid_numeric_inputs() -> None:
    pytest.importorskip("scipy")
    from pynerve import fast_ops

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert fast_ops.pairwise_distances(points).shape == (3, 3), (
        f"expected (3, 3), got {fast_ops.pairwise_distances(points).shape}"
    )
    assert fast_ops.pairwise_distances_broadcast(points).shape == (3, 3), (
        f"expected (3, 3), got {fast_ops.pairwise_distances_broadcast(points).shape}"
    )
    assert fast_ops.vr_edges(points, 1.1).shape[1] == 2, (
        f"expected 2 columns, got {fast_ops.vr_edges(points, 1.1).shape[1]}"
    )
    assert fast_ops.connected_components(np.array([[0, 1]]), 3).shape == (3,), (
        f"expected (3,), got {fast_ops.connected_components(np.array([[0, 1]]), 3).shape}"
    )
    assert fast_ops.minimum_spanning_tree(points).shape[1] == 2, (
        f"expected 2 columns, got {fast_ops.minimum_spanning_tree(points).shape[1]}"
    )
    assert fast_ops.sort_filtration(np.array([[0], [1]]), np.array([0.0, 1.0]))[0].shape == (2,), (
        f"expected (2,), got {fast_ops.sort_filtration(np.array([[0], [1]]), np.array([0.0, 1.0]))[0].shape}"
    )

    with pytest.raises(ValidationError, match="finite coordinates"):
        fast_ops.pairwise_distances(np.array([[float("nan"), 0.0]]))
    with pytest.raises(ValueError, match="metric"):
        fast_ops.pairwise_distances(points, metric="")
    with pytest.raises(ValidationError, match="k"):
        fast_ops.nearest_neighbors(points, k=1.5)
    with pytest.raises(ValidationError, match="max_dist"):
        fast_ops.sparse_distance_matrix(points, max_dist=float("nan"))
    with pytest.raises(ValueError, match="output_type"):
        fast_ops.sparse_distance_matrix(points, max_dist=1.0, output_type="bad")
    with pytest.raises(ValidationError, match="n_vertices"):
        fast_ops.connected_components(np.array([[0, 1]]), 2.5)
    with pytest.raises(TypeError, match="edges"):
        fast_ops.connected_components(np.array([[0.0, 1.0]]), 2)
    with pytest.raises(ValidationError, match="max_dist"):
        fast_ops.vr_edges(points, float("nan"))
    with pytest.raises(ValidationError, match="max_dim"):
        fast_ops.enumerate_simplices(points, 1.0, max_dim=1.5)
    with pytest.raises(ValueError, match="unique"):
        fast_ops.simplex_boundary(np.array([0, 0]))
    with pytest.raises(ValueError, match="matching lengths"):
        fast_ops.sort_filtration(np.array([[0], [1]]), np.array([0.0]))
    with pytest.raises(ValueError, match="filtration_values"):
        fast_ops.sort_filtration(np.array([[0]]), np.array([float("nan")]))


def test_cache_utilities_reject_invalid_public_inputs(tmp_path: Path) -> None:
    from pynerve.cache import (
        DiagramCache,
        MemoizePersistent,
        PersistentDiagramCache,
        SmartCache,
        cached_persistence,
        get_cache_stats,
        memoize_persistent,
    )

    cache = DiagramCache(memory_maxsize=2, ttl=1)
    data = np.array([[0.0, 1.0]])
    cache.set(data, "value", max_dim=1)
    assert cache.get(data, max_dim=1) == "value", (
        f"expected 'value', got {cache.get(data, max_dim=1)}"
    )
    assert get_cache_stats(cache)["memory_entries"] == 1, (
        f"expected 1, got {get_cache_stats(cache)['memory_entries']}"
    )

    with pytest.raises(ValidationError, match="memory_maxsize"):
        DiagramCache(memory_maxsize=1.5)
    with pytest.raises(ValidationError, match="memory_maxsize"):
        DiagramCache(memory_maxsize=0)
    with pytest.raises(ValidationError, match="ttl"):
        DiagramCache(ttl=1.5)
    with pytest.raises(ValidationError, match="use_disk"):
        DiagramCache(use_disk=1)
    with pytest.raises(TypeError, match="key_fn"):
        cached_persistence(key_fn=object())
    with pytest.raises(TypeError, match="func"):
        MemoizePersistent(object(), cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="ignore_args"):
        MemoizePersistent(lambda value: value, cache_dir=str(tmp_path), ignore_args=[""])
    with pytest.raises(ValidationError, match="ttl"):
        memoize_persistent(cache_dir=str(tmp_path), ttl=1.5)
    with pytest.raises(ValidationError, match="size_limit"):
        PersistentDiagramCache(cache_dir=str(tmp_path), size_limit=0)
    with pytest.raises(ValidationError, match="memory_maxsize"):
        SmartCache(memory_maxsize=1.5)
    with pytest.raises(ValidationError, match="disk_size_limit"):
        SmartCache(disk_size_limit=1.5)
    with pytest.raises(ValidationError, match="cache_key"):
        SmartCache().set("", "value")
    with pytest.raises(TypeError, match="cache"):
        get_cache_stats(object())


def test_multiprocessing_shared_helpers_validate_public_inputs() -> None:
    from pynerve.mp_shared import (
        ChunkedParallel,
        ForkServerPool,
        MapReducePH,
        ParallelPH,
        SharedMemoryArray,
        compute_persistence_parallel,
    )

    shared = SharedMemoryArray((2, 2), np.float64)
    try:
        shared.array[...] = np.ones((2, 2))
        assert np.allclose(np.asarray(shared), 1.0), (
            f"expected all close to 1.0, got {np.asarray(shared)}"
        )
    finally:
        shared.close()

    with pytest.raises((TypeError, Exception), match="dimensions must be integers"):
        SharedMemoryArray((1.5,), np.float64)
    with pytest.raises((ValueError, Exception), match="must contain at least one dimension"):
        SharedMemoryArray((), np.float64)
    with pytest.raises(TypeError, match="object dtype"):
        SharedMemoryArray((1,), object)
    with pytest.raises(ValueError, match="array"):
        SharedMemoryArray.from_array(np.array(1.0))
    with pytest.raises(ValidationError, match="n_workers"):
        ParallelPH(n_workers=1.5)
    with pytest.raises(ValidationError, match="use_shared_memory"):
        ParallelPH(use_shared_memory=1)
    with pytest.raises(ValidationError, match="chunksize"):
        ParallelPH(chunksize=0)

    with ParallelPH(n_workers=1) as pool:
        with pytest.raises(TypeError, match="data_batches"):
            pool.map(lambda value: value, None)
        with pytest.raises(TypeError, match="args_list"):
            pool.starmap(lambda value: value, None)

    with pytest.raises(TypeError, match="aggregator"):
        ChunkedParallel(aggregator=object())
    with pytest.raises(ValidationError, match="chunk_size"):
        ChunkedParallel(chunk_size=1.5)
    with pytest.raises(TypeError, match="map_fn"):
        MapReducePH(object(), lambda _left, right: right)
    with pytest.raises(ValidationError, match="n_workers"):
        ForkServerPool(n_workers=1.5)
    with pytest.raises(TypeError, match="progress_callback"):
        compute_persistence_parallel([], progress_callback=object())


def test_diagnostics_helpers_validate_public_inputs() -> None:
    import io

    from pynerve.diagnostics import (
        DebugMode,
        DiagnosticInfo,
        DiagnosticsCollector,
        check_data_quality,
        diagnose_failure,
        verbose,
    )

    info = DiagnosticInfo("compute", 0.0, n_points=2, backend="cpu")
    assert "compute" in repr(info), f"expected 'compute' in {repr(info)}"
    collector = DiagnosticsCollector()
    with collector.track("stage", n_points=1):
        pass
    assert collector.summary()["n_operations"] == 1, (
        f"expected 1, got {collector.summary()['n_operations']}"
    )
    assert "Diagnostic Report" in collector.report(), (
        f"expected 'Diagnostic Report' in {collector.report()}"
    )
    assert "Invalid input" in diagnose_failure("stage", ValueError("bad"), context={"x": 1}), (
        f"expected 'Invalid input' in {diagnose_failure('stage', ValueError('bad'), context={'x': 1})}"
    )
    assert check_data_quality(np.ones((2, 2)))["valid"], "expected valid data quality"
    stream = io.StringIO()
    with DebugMode(stream=stream):
        pass
    assert "Debug mode enabled" in stream.getvalue(), (
        f"expected 'Debug mode enabled' in {stream.getvalue()}"
    )

    with pytest.raises(ValidationError, match="operation"):
        DiagnosticInfo("", 0.0)
    with pytest.raises(ValidationError, match="duration"):
        DiagnosticInfo("stage", float("nan"))
    with pytest.raises(ValidationError, match="n_points"):
        DiagnosticInfo("stage", 0.0, n_points=-1)
    with pytest.raises(ValidationError, match="n_simplices"):
        DiagnosticInfo("stage", 0.0, n_simplices=1.5)
    with pytest.raises(TypeError, match="enabled"), verbose(enabled=1):
        pass
    with pytest.raises(TypeError, match="exception"):
        diagnose_failure("stage", object())
    with pytest.raises(TypeError, match="context"):
        diagnose_failure("stage", ValueError("bad"), context=object())
    with pytest.raises((TypeError, ValidationError), match="operation|unexpected keyword"):
        check_data_quality(np.ones((1, 1)), operation="")
    with pytest.raises(TypeError, match="print_intermediate"):
        DebugMode(print_intermediate=1)
    with pytest.raises(TypeError, match="stream"):
        DebugMode(stream=object())


def test_benchmark_helpers_validate_public_inputs() -> None:
    from pynerve.benchmark import (
        BenchmarkComparison,
        ScalabilityResult,
        Timer,
        TimerResult,
        benchmark_scalability,
        benchmark_vs_ripser,
        run_full_benchmark_suite,
    )

    timer = Timer("x = 1", globals={})
    assert timer.timeit(1) >= 0.0, f"expected >= 0.0, got {timer.timeit(1)}"
    assert timer.blocked_autorange(min_run_time=0.0001, max_runs=1).n_runs >= 1, (
        f"expected >= 1, got {timer.blocked_autorange(min_run_time=0.0001, max_runs=1).n_runs}"
    )
    assert "TimerResult" in repr(TimerResult(0.1, 0.0, 0.1, 0.1, 1)), (
        f"expected 'TimerResult' in {repr(TimerResult(0.1, 0.0, 0.1, 0.1, 1))}"
    )
    assert "speedup" in repr(BenchmarkComparison("Nerve", "Other", "spheres", 1, 0.1, 0.1, 1.0)), (
        f"expected 'speedup' in {repr(BenchmarkComparison('Nerve', 'Other', 'spheres', 1, 0.1, 0.1, 1.0))}"
    )
    assert ScalabilityResult([10, 20], [0.1, 0.4]).estimate_complexity(), (
        "expected estimate_complexity to return truthy value"
    )

    with pytest.raises(ValidationError, match="stmt"):
        Timer("", globals={})
    with pytest.raises(ValidationError, match="setup"):
        Timer("pass", setup="", globals={})
    with pytest.raises(TypeError, match="globals"):
        Timer("pass", globals=object())
    with pytest.raises(ValidationError, match="number"):
        timer.timeit(1.5)
    with pytest.raises(ValidationError, match="min_run_time"):
        timer.blocked_autorange(min_run_time=float("nan"))
    with pytest.raises(ValidationError, match="max_runs"):
        timer.blocked_autorange(max_runs=1.5)
    with pytest.raises(ValidationError, match="n_runs"):
        TimerResult(0.1, 0.0, 0.1, 0.1, 0)
    with pytest.raises(ValueError, match="min_time"):
        TimerResult(0.1, 0.0, 0.2, 0.1, 1)
    with pytest.raises(ValidationError, match="time1"):
        BenchmarkComparison("A", "B", "spheres", 1, float("nan"), 0.1, 1.0)
    with pytest.raises(ValidationError, match="time2"):
        BenchmarkComparison("A", "B", "spheres", 1, 0.1, float("inf"), 1.0)
    with pytest.raises(ValidationError, match="speedup"):
        BenchmarkComparison("A", "B", "spheres", 1, 0.1, 0.1, float("inf"))
    with pytest.raises(ValidationError, match="n_samples"):
        ScalabilityResult([1.5], [0.1])
    with pytest.raises(ValidationError, match="times"):
        ScalabilityResult([1], [float("nan")])
    with pytest.raises(ValueError, match="fit_result"):
        ScalabilityResult([1], [0.1], fit_result=(0.0, float("nan")))
    with pytest.raises(ValidationError, match="n_runs"):
        benchmark_vs_ripser(n_runs=1.5)
    with pytest.raises(ValidationError, match="n_samples_range"):
        benchmark_scalability(n_samples_range=[1.5])
    with pytest.raises(ValidationError, match="output_file"):
        run_full_benchmark_suite(output_file="")


def test_async_helpers_validate_public_inputs(tmp_path: Path, nerve_core: None) -> None:  # noqa: ARG001, PLR0915
    import asyncio
    import pickle

    from pynerve._async_compute import AsyncPersistenceComputer
    from pynerve._async_gpu import AsyncGPUTransfer
    from pynerve._async_loader import AsyncDiagramLoader
    from pynerve._streaming_persistence import StreamingPersistence
    from pynerve.async_api import compute_persistence_async, load_diagrams_async

    async def _valid_batches():
        yield np.ones((2, 2), dtype=np.float32)

    async def _collect_compute():
        async with AsyncPersistenceComputer(max_workers=1, buffer_size=1) as computer:
            return [
                item
                async for item in computer.compute_batch_async(
                    _valid_batches(), lambda batch: {"shape": batch.shape}
                )
            ]

    assert asyncio.run(_collect_compute())[0]["shape"] == (2, 2), (
        f"expected (2, 2), got {asyncio.run(_collect_compute())[0]['shape']}"
    )

    diagram_path = tmp_path / "diagram.npy"
    np.save(diagram_path, np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

    async def _load_valid():
        loader = AsyncDiagramLoader(max_concurrent=1)
        return await loader.load_batch([diagram_path])

    assert asyncio.run(_load_valid())[0].shape == (1, 3), (
        f"expected (1, 3), got {asyncio.run(_load_valid())[0].shape}"
    )
    assert asyncio.run(load_diagrams_async([diagram_path], max_concurrent=1))[0].shape == (1, 3), (
        f"expected (1, 3), got {asyncio.run(load_diagrams_async([diagram_path], max_concurrent=1))[0].shape}"
    )

    with pytest.raises(ValidationError, match="max_workers"):
        AsyncPersistenceComputer(max_workers=1.5)
    with pytest.raises(ValidationError, match="buffer_size"):
        AsyncPersistenceComputer(buffer_size=0)

    async def _bad_batches_type():
        async with AsyncPersistenceComputer(max_workers=1) as computer:
            return [item async for item in computer.compute_batch_async([np.ones((1, 1))])]

    with pytest.raises(TypeError, match="data_batches"):
        asyncio.run(_bad_batches_type())

    with pytest.raises(ValidationError, match="max_concurrent"):
        AsyncDiagramLoader(max_concurrent=1.5)
    with pytest.raises(TypeError, match="filepaths"):
        asyncio.run(load_diagrams_async("diagram.npy"))

    bad_diagram = tmp_path / "bad.npy"
    np.save(bad_diagram, np.array([[float("nan"), 1.0, 0.0]], dtype=np.float32))

    async def _load_bad():
        return await AsyncDiagramLoader(max_concurrent=1).load_file(bad_diagram)

    with pytest.raises(ValueError, match="births"):
        asyncio.run(_load_bad())

    bad_pickle = tmp_path / "bad.pkl"
    bad_pickle.write_bytes(pickle.dumps(np.array([[1.0, 0.0, 0.0]], dtype=np.float32)))

    async def _load_bad_pickle():
        return await AsyncDiagramLoader(max_concurrent=1).load_file(bad_pickle)

    with pytest.raises(ValueError, match="deaths"):
        asyncio.run(_load_bad_pickle())

    with pytest.raises(TypeError, match="device_id"):
        AsyncGPUTransfer(device_id=1.5)
    with pytest.raises(ValueError, match="device_id"):
        AsyncGPUTransfer(device_id=-1)
    with pytest.raises(ValidationError, match="chunk_size"):
        StreamingPersistence(chunk_size=1.5)
    with pytest.raises(ValidationError, match="use_gpu"):
        StreamingPersistence(use_gpu=1)

    streamer = StreamingPersistence(chunk_size=2, max_buffered_chunks=1, use_gpu=False)
    with pytest.raises(ValueError, match="births"):
        streamer._diagrams_to_stats({"pairs": [[float("nan"), 1.0, 0.0]]})

    async def _bad_source_type():
        async for _ in streamer.stream_compute([np.ones((1, 1))]):
            pass

    with pytest.raises((TypeError, ValueError), match="data_source"):
        asyncio.run(_bad_source_type())

    with pytest.raises((ValueError, ValidationError), match="points|data"):
        asyncio.run(compute_persistence_async(np.array([[float("nan"), 0.0]])))
