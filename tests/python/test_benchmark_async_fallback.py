from __future__ import annotations

import contextlib
import dataclasses
import os
import tempfile

import numpy as np
import pytest
from pynerve._constants import EPS_1e_9
from pynerve._fallback_classes import (
    EventType,
    PersistenceBackend,
    PersistenceEngine,
    PersistenceMode,
    PersistenceOptions,
    PH5PH6Config,
    PH5PH6Engine,
    PH5PH6Metrics,
)
from pynerve.benchmark._common import (
    _BENCHMARK_RECOVERABLE_ERRORS,
    _benchmark_dataset,
)
from pynerve.benchmark._comparison_types import BenchmarkComparison, GPUComparison
from pynerve.benchmark._scalability import ScalabilityResult
from pynerve.benchmark._timer import Timer, TimerResult


class TestBenchmarkCommon:
    def test_recoverable_errors_are_exception_classes(self):
        assert isinstance(_BENCHMARK_RECOVERABLE_ERRORS, tuple)
        for exc in _BENCHMARK_RECOVERABLE_ERRORS:
            assert issubclass(exc, Exception)

    def test_benchmark_dataset_valid_datasets_return_ndarray(self):
        for dataset in ("spheres", "torus", "swiss_roll"):
            result = _benchmark_dataset(dataset, 10)
            assert isinstance(result, np.ndarray)
            assert len(result) == 10
            assert result.dtype == np.float64

    def test_benchmark_dataset_unknown_dataset_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            _benchmark_dataset("nonexistent", 10)

    def test_benchmark_dataset_empty_string_raises(self):
        with pytest.raises(ValueError):
            _benchmark_dataset("", 10)

    def test_benchmark_dataset_zero_samples_raises(self):
        with pytest.raises(ValueError):
            _benchmark_dataset("spheres", 0)

    def test_benchmark_dataset_negative_samples_raises(self):
        with pytest.raises(ValueError):
            _benchmark_dataset("spheres", -1)

    def test_benchmark_dataset_non_integer_samples_raises(self):
        with pytest.raises((TypeError, ValueError)):
            _benchmark_dataset("spheres", "abc")

    def test_benchmark_dataset_reproducible_with_seed_zero(self):
        a = _benchmark_dataset("spheres", 100)
        b = _benchmark_dataset("spheres", 100)
        assert np.array_equal(a, b)

    def test_benchmark_dataset_different_sizes(self):
        small = _benchmark_dataset("torus", 5)
        large = _benchmark_dataset("torus", 50)
        assert len(small) == 5
        assert len(large) == 50


class TestBenchmarkComparisonTypes:
    def test_benchmark_comparison_valid_construction(self):
        bc = BenchmarkComparison(
            library1="nerve",
            library2="ripser",
            dataset="spheres",
            n_samples=500,
            time1=0.1,
            time2=0.2,
            speedup=2.0,
        )
        assert bc.library1 == "nerve"
        assert bc.speedup == 2.0
        assert bc.memory1 is None

    def test_benchmark_comparison_with_memory(self):
        bc = BenchmarkComparison(
            library1="a",
            library2="b",
            dataset="s",
            n_samples=1,
            time1=0.0,
            time2=0.0,
            speedup=0.0,
            memory1=100.0,
            memory2=200.0,
        )
        assert bc.memory1 == 100.0
        assert bc.memory2 == 200.0

    def test_benchmark_comparison_repr(self):
        bc = BenchmarkComparison(
            library1="A", library2="B", dataset="s", n_samples=1, time1=1.0, time2=2.0, speedup=2.0
        )
        assert "A vs B" in repr(bc)
        assert "2.00x" in repr(bc)

    def test_benchmark_comparison_negative_time_raises(self):
        with pytest.raises(ValueError):
            BenchmarkComparison(
                library1="a",
                library2="b",
                dataset="s",
                n_samples=1,
                time1=-1.0,
                time2=0.1,
                speedup=1.0,
            )

    def test_benchmark_comparison_negative_speedup_raises(self):
        with pytest.raises(ValueError):
            BenchmarkComparison(
                library1="a",
                library2="b",
                dataset="s",
                n_samples=1,
                time1=0.1,
                time2=0.1,
                speedup=-0.1,
            )

    def test_benchmark_comparison_nan_time_raises(self):
        with pytest.raises(ValueError):
            BenchmarkComparison(
                library1="a",
                library2="b",
                dataset="s",
                n_samples=1,
                time1=float("nan"),
                time2=0.1,
                speedup=1.0,
            )

    def test_benchmark_comparison_inf_speedup_raises(self):
        with pytest.raises(ValueError):
            BenchmarkComparison(
                library1="a",
                library2="b",
                dataset="s",
                n_samples=1,
                time1=0.1,
                time2=0.1,
                speedup=float("inf"),
            )

    def test_benchmark_comparison_ninf_speedup_raises(self):
        with pytest.raises(ValueError):
            BenchmarkComparison(
                library1="a",
                library2="b",
                dataset="s",
                n_samples=1,
                time1=0.1,
                time2=0.1,
                speedup=float("-inf"),
            )

    def test_benchmark_comparison_empty_library_raises(self):
        with pytest.raises(ValueError):
            BenchmarkComparison(
                library1="",
                library2="b",
                dataset="s",
                n_samples=1,
                time1=0.0,
                time2=0.0,
                speedup=1.0,
            )

    def test_benchmark_comparison_zero_samples_raises(self):
        with pytest.raises(ValueError):
            BenchmarkComparison(
                library1="a",
                library2="b",
                dataset="s",
                n_samples=0,
                time1=0.0,
                time2=0.0,
                speedup=1.0,
            )

    def test_benchmark_comparison_negative_memory_raises(self):
        with pytest.raises(ValueError):
            BenchmarkComparison(
                library1="a",
                library2="b",
                dataset="s",
                n_samples=1,
                time1=0.0,
                time2=0.0,
                speedup=1.0,
                memory1=-1.0,
            )

    def test_gpu_comparison_valid_construction(self):
        gc = GPUComparison(
            library1="cpu",
            library2="cuda",
            dataset="spheres",
            n_samples=500,
            time1=0.5,
            time2=0.05,
            speedup=10.0,
            gpu_available=True,
        )
        assert gc.library1 == "cpu"
        assert gc.time2 == 0.05
        assert gc.gpu_available is True

    def test_gpu_comparison_time2_none_allowed(self):
        gc = GPUComparison(
            library1="cpu",
            library2="cuda",
            dataset="spheres",
            n_samples=500,
            time1=0.5,
            time2=None,
            speedup=0.0,
            gpu_available=False,
        )
        assert gc.time2 is None

    def test_gpu_comparison_is_dataclass(self):
        assert dataclasses.is_dataclass(GPUComparison)


class TestScalabilityResult:
    def test_valid_construction(self):
        sr = ScalabilityResult(n_samples=[100, 200], times=[0.1, 0.2])
        assert sr.n_samples == [100, 200]
        assert sr.fit_result is None

    def test_with_fit_result(self):
        sr = ScalabilityResult(
            n_samples=[100, 200],
            times=[0.1, 0.2],
            fit_result=(0.5, 1.8),
        )
        assert sr.fit_result == (0.5, 1.8)

    def test_fit_result_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly two"):
            ScalabilityResult(n_samples=[100], times=[0.1], fit_result=(0.5,))

    def test_fit_result_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            ScalabilityResult(n_samples=[100], times=[0.1], fit_result=(float("nan"), 1.0))

    def test_zero_sample_raises(self):
        with pytest.raises(ValueError):
            ScalabilityResult(n_samples=[0], times=[0.1])

    def test_negative_time_raises(self):
        with pytest.raises(ValueError):
            ScalabilityResult(n_samples=[100], times=[-0.1])

    def test_estimate_complexity_insufficient_data(self):
        sr = ScalabilityResult(n_samples=[100], times=[0.1])
        result = sr.estimate_complexity()
        assert result == "Insufficient data"

    def test_estimate_complexity_mismatched_length_raises(self):
        sr = ScalabilityResult(n_samples=[100, 200], times=[0.1])
        with pytest.raises(ValueError, match="same length"):
            sr.estimate_complexity()

    def test_estimate_complexity_zero_times_rejected_by_post_init(self):
        with pytest.raises(ValueError):
            ScalabilityResult(n_samples=[100, 200], times=[0.1, 0.0])

    def test_estimate_complexity_linear(self):
        n = [100, 200, 500, 1000, 2000]
        t = [float(x) for x in n]
        sr = ScalabilityResult(n_samples=n, times=t)
        result = sr.estimate_complexity()
        assert "~ O(n)" in result

    def test_estimate_complexity_quadratic(self):
        n = [100, 200, 500, 1000]
        t = [float(x * x) for x in n]
        sr = ScalabilityResult(n_samples=n, times=t)
        result = sr.estimate_complexity()
        assert "~ O(n^2)" in result

    def test_estimate_complexity_cubic(self):
        n = [100, 200, 500]
        t = [float(x * x * x) for x in n]
        sr = ScalabilityResult(n_samples=n, times=t)
        result = sr.estimate_complexity()
        assert "~ O(n^3)" in result

    def test_estimate_complexity_super_cubic(self):
        n = [100, 200, 500]
        t = [float(x**3.5) for x in n]
        sr = ScalabilityResult(n_samples=n, times=t)
        result = sr.estimate_complexity()
        assert result.startswith("O(n^")
        assert "~" not in result

    def test_estimate_complexity_sets_fit_result(self):
        n = [100, 200, 500, 1000]
        t = [float(x * x) for x in n]
        sr = ScalabilityResult(n_samples=n, times=t)
        sr.estimate_complexity()
        assert sr.fit_result is not None
        assert len(sr.fit_result) == 2

    def test_estimate_complexity_positive_guard_after_construction(self):
        sr = ScalabilityResult(n_samples=[100, 200], times=[0.1, 0.2])
        object.__setattr__(sr, "times", [0.1, 0.0])
        with pytest.raises(ValueError, match="positive"):
            sr.estimate_complexity()

    def test_estimate_complexity_nonfinite_rejected_by_post_init(self):
        with pytest.raises(ValueError):
            ScalabilityResult(n_samples=[100, 200], times=[float("inf"), 0.1])


class TestTimerResult:
    def test_valid_construction(self):
        tr = TimerResult(mean_time=0.1, std_time=0.01, min_time=0.09, max_time=0.12, n_runs=100)
        assert tr.mean_time == 0.1
        assert tr.n_runs == 100

    def test_min_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="min_time must not exceed"):
            TimerResult(mean_time=0.1, std_time=0.01, min_time=0.15, max_time=0.10, n_runs=100)

    def test_negative_mean_time_raises(self):
        with pytest.raises(ValueError):
            TimerResult(mean_time=-0.1, std_time=0.01, min_time=0.0, max_time=0.0, n_runs=1)

    def test_zero_runs_raises(self):
        with pytest.raises(ValueError):
            TimerResult(mean_time=0.0, std_time=0.0, min_time=0.0, max_time=0.0, n_runs=0)

    def test_repr_shows_ms(self):
        tr = TimerResult(
            mean_time=0.00123, std_time=0.00045, min_time=0.001, max_time=0.002, n_runs=100
        )
        r = repr(tr)
        assert "ms" in r

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            TimerResult(mean_time=float("nan"), std_time=0.0, min_time=0.0, max_time=0.0, n_runs=1)


class TestTimer:
    def test_default_construction(self):
        t = Timer()
        assert t.stmt == "pass"
        assert t.setup == "pass"
        assert t.globals is None
        assert t.label == "pass"

    def test_with_label(self):
        t = Timer(stmt="x = 1", label="my-timer")
        assert t.label == "my-timer"

    def test_empty_stmt_raises(self):
        with pytest.raises(ValueError):
            Timer(stmt="")

    def test_empty_setup_raises(self):
        with pytest.raises(ValueError):
            Timer(setup="")

    def test_globals_not_dict_raises(self):
        with pytest.raises(TypeError, match="dictionary"):
            Timer(globals="not-a-dict")

    def test_label_not_empty(self):
        with pytest.raises(ValueError):
            Timer(label="")

    def test_timeit_positive(self):
        t = Timer(stmt="x = 1 + 2")
        dt = t.timeit(number=1000)
        assert dt >= 0.0

    def test_timeit_zero_number_raises(self):
        t = Timer()
        with pytest.raises(ValueError):
            t.timeit(number=0)

    def test_timeit_negative_number_raises(self):
        t = Timer()
        with pytest.raises(ValueError):
            t.timeit(number=-10)

    def test_timeit_with_globals(self):
        t = Timer(stmt="y = a + b", globals={"a": 1, "b": 2})
        dt = t.timeit(number=1000)
        assert dt >= 0.0

    def test_blocked_autorange_returns_timer_result(self):
        t = Timer(stmt="x = sum(range(10))")
        result = t.blocked_autorange(min_run_time=0.05, max_runs=10)
        assert isinstance(result, TimerResult)
        assert result.n_runs > 0
        assert result.mean_time >= 0.0
        assert result.min_time <= result.max_time

    def test_blocked_autorange_zero_min_run_time_raises(self):
        t = Timer()
        with pytest.raises(ValueError):
            t.blocked_autorange(min_run_time=0.0)

    def test_blocked_autorange_negative_min_run_time_raises(self):
        t = Timer()
        with pytest.raises(ValueError):
            t.blocked_autorange(min_run_time=-1.0)

    def test_blocked_autorange_zero_max_runs_raises(self):
        t = Timer()
        with pytest.raises(ValueError):
            t.blocked_autorange(max_runs=0)

    def test_blocked_autorange_nan_min_run_time_raises(self):
        t = Timer()
        with pytest.raises(ValueError):
            t.blocked_autorange(min_run_time=float("nan"))


class TestBenchmarkSuiteHelpers:
    def test_json_ready_primitives(self):
        from pynerve.benchmark._suite import _json_ready

        assert _json_ready(42) == 42
        assert _json_ready("hello") == "hello"
        assert _json_ready(True) is True
        assert _json_ready(None) is None

    def test_json_ready_list(self):
        from pynerve.benchmark._suite import _json_ready

        assert _json_ready([1, 2, 3]) == [1, 2, 3]

    def test_json_ready_nested_dict(self):
        from pynerve.benchmark._suite import _json_ready

        result = _json_ready({"a": {"b": 1}})
        assert result == {"a": {"b": 1}}

    def test_json_ready_nan_becomes_none(self):
        from pynerve.benchmark._suite import _json_ready

        assert _json_ready(float("nan")) is None

    def test_json_ready_inf_becomes_none(self):
        from pynerve.benchmark._suite import _json_ready

        assert _json_ready(float("inf")) is None
        assert _json_ready(float("-inf")) is None

    def test_json_ready_numpy_scalar_to_python(self):
        from pynerve.benchmark._suite import _json_ready

        assert _json_ready(np.float64(3.14)) == 3.14
        assert _json_ready(np.int64(42)) == 42

    def test_json_ready_object_with_dict(self):
        from pynerve.benchmark._suite import _json_ready

        class Obj:
            def __init__(self):
                self.x = 1
                self.y = 2

        assert _json_ready(Obj()) == {"x": 1, "y": 2}

    def test_benchmark_with_fallback_catches_errors(self):
        from pynerve.benchmark._suite import _benchmark_with_fallback

        result = _benchmark_with_fallback(lambda: (_ for _ in ()).throw(ValueError("test")))
        assert "test" in result

    def test_benchmark_with_fallback_returns_json(self):
        from pynerve.benchmark._suite import _benchmark_with_fallback

        result = _benchmark_with_fallback(lambda: 42)
        assert result == 42

    def test_make_comparison_fn_reference(self):
        from pynerve.benchmark._suite import _make_comparison_fn

        fn = _make_comparison_fn("reference", "spheres", 500)
        assert callable(fn)

    def test_make_comparison_fn_gudhi(self):
        from pynerve.benchmark._suite import _make_comparison_fn

        fn = _make_comparison_fn("gudhi", "torus", 1000)
        assert callable(fn)

    def test_make_comparison_fn_dionysus(self):
        from pynerve.benchmark._suite import _make_comparison_fn

        fn = _make_comparison_fn("dionysus", "swiss_roll", 500)
        assert callable(fn)

    def test_make_comparison_fn_unknown_raises(self):
        from pynerve.benchmark._suite import _make_comparison_fn

        with pytest.raises(ValueError, match="Unknown comparison"):
            _make_comparison_fn("nonexistent", "spheres", 500)

    def test_resolve_output_file_none(self):
        from pynerve.benchmark._suite import _resolve_output_file

        assert _resolve_output_file(None) is None

    def test_resolve_output_file_empty_raises(self):
        from pynerve.benchmark._suite import _resolve_output_file

        with pytest.raises(ValueError):
            _resolve_output_file("")

    def test_resolve_output_file_valid(self):
        from pynerve.benchmark._suite import _resolve_output_file

        p = _resolve_output_file("/tmp/test.json")
        assert isinstance(p, os.PathLike) or hasattr(p, "open")

    def test_make_distance_fn(self):
        from pynerve.benchmark._suite import _make_distance_fn

        fn = _make_distance_fn("euclidean")
        assert callable(fn)

    def test_make_streaming_fn(self):
        from pynerve.benchmark._suite import _make_streaming_fn

        fn = _make_streaming_fn("spheres")
        assert callable(fn)


class TestAsyncFacadeValidation:
    def test_validate_filepaths_empty_list_raises(self):
        from pynerve._async_facade import _validate_filepaths

        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepaths([])

    def test_validate_filepaths_non_iterable_raises(self):
        from pynerve._async_facade import _validate_filepaths

        with pytest.raises(TypeError, match="iterable"):
            _validate_filepaths(42)

    def test_validate_filepaths_single_string_raises(self):
        from pynerve._async_facade import _validate_filepaths

        with pytest.raises(TypeError):
            _validate_filepaths("a_string")

    def test_validate_filepaths_nonexistent_raises(self):
        from pynerve._async_facade import _validate_filepaths

        with pytest.raises(FileNotFoundError, match="file not found"):
            _validate_filepaths(["/nonexistent/path/file.bin"])

    def test_validate_filepaths_valid(self):
        from pynerve._async_facade import _validate_filepaths

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as f:
            result = _validate_filepaths([f.name])
            assert result == [f.name]

    def test_require_async_deps_without_core_extension(self):
        import pynerve
        from pynerve._async_facade import _require_async_deps

        if pynerve._core_import_error is not None:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                _require_async_deps()
        else:
            with contextlib.suppress(ImportError):
                _require_async_deps()

    def test_stream_persistence_non_bool_gpu_raises(self):
        from pynerve._async_facade import stream_persistence

        with pytest.raises(TypeError, match="boolean"):
            gen = stream_persistence("data", use_gpu="yes")
            import asyncio

            asyncio.run(gen.__anext__())


class TestAsyncCompute:
    def test_async_computer_construction(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        apc = AsyncPersistenceComputer(max_workers=2, buffer_size=3)
        assert apc.max_workers == 2
        assert apc.buffer_size == 3
        if not apc._closed:
            import asyncio

            asyncio.run(apc.close())

    def test_async_computer_defaults(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        apc = AsyncPersistenceComputer()
        assert apc.max_workers == 4
        assert apc.buffer_size == 3
        if not apc._closed:
            import asyncio

            asyncio.run(apc.close())

    def test_async_computer_zero_max_workers_raises(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        with pytest.raises(ValueError):
            AsyncPersistenceComputer(max_workers=0)

    def test_async_computer_zero_buffer_size_raises(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        with pytest.raises(ValueError):
            AsyncPersistenceComputer(buffer_size=0)

    def test_async_computer_compute_batch_rejects_non_async_iterator(self):
        import asyncio

        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            apc = AsyncPersistenceComputer(max_workers=1)
            try:
                with pytest.raises(TypeError, match="async iterator"):
                    async for _ in apc.compute_batch_async([1, 2, 3]):
                        pass
            finally:
                await apc.close()

        asyncio.run(_run())

    def test_async_computer_compute_batch_rejects_closed(self):
        import asyncio

        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            apc = AsyncPersistenceComputer(max_workers=1)
            await apc.close()
            with pytest.raises(RuntimeError, match="closed"):

                async def _gen():
                    if False:
                        yield np.array([])

                async for _ in apc.compute_batch_async(_gen()):
                    pass

        asyncio.run(_run())

    def test_async_computer_close_idempotent(self):
        import asyncio

        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            apc = AsyncPersistenceComputer(max_workers=1)
            await apc.close()
            await apc.close()

        asyncio.run(_run())

    def test_async_computer_context_manager(self):
        import asyncio

        from pynerve._async_compute import AsyncPersistenceComputer

        async def _run():
            async with AsyncPersistenceComputer(max_workers=1) as apc:
                assert not apc._closed
            assert apc._closed

        asyncio.run(_run())

    def test_async_computer_compute_batch_with_process_results(self):
        import asyncio

        import pynerve
        from pynerve._async_compute import AsyncPersistenceComputer

        if pynerve._core_import_error is not None:
            pytest.skip("C++ extension not available")

        async def _data():
            yield np.array([[1.0, 2.0, 3.0]])

        async def _run():
            async with AsyncPersistenceComputer(max_workers=2, buffer_size=1) as apc:
                results = []
                async for r in apc.compute_batch_async(_data()):
                    results.append(r)
                assert len(results) == 1

        asyncio.run(_run())


class TestFallbackEnums:
    def test_persistence_mode_values(self):
        assert PersistenceMode.EXACT.value == "EXACT"
        assert PersistenceMode.APPROX.value == "APPROX"

    def test_persistence_backend_values(self):
        assert PersistenceBackend.CPU_EXACT.value == "CPU_EXACT"
        assert PersistenceBackend.CPU_ADAPTIVE_ACCELERATION.value == "CPU_ADAPTIVE_ACCELERATION"
        assert PersistenceBackend.CUDA_HYBRID.value == "CUDA_HYBRID"

    def test_persistence_engine_values(self):
        assert PersistenceEngine.AUTO.value == "auto"
        assert PersistenceEngine.PH0.value == "ph0"
        assert PersistenceEngine.PH3.value == "ph3"
        assert PersistenceEngine.PH4.value == "ph4"
        assert PersistenceEngine.PH5.value == "ph5"
        assert PersistenceEngine.PH6.value == "ph6"

    def test_event_type_values(self):
        assert EventType.ADD.value == "add"
        assert EventType.REMOVE.value == "remove"


class TestPersistenceOptions:
    def test_default_construction(self):
        opts = PersistenceOptions()
        assert opts.mode == PersistenceMode.EXACT
        assert opts.backend == PersistenceBackend.CPU_ADAPTIVE_ACCELERATION
        assert opts.max_dim == 2
        assert opts.max_radius is None
        assert opts.threads == 0
        assert opts.error_tolerance == 0.0

    def test_is_frozen(self):
        opts = PersistenceOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            opts.max_dim = 5

    def test_replace_method(self):
        opts = PersistenceOptions(max_dim=3)
        new_opts = opts.replace(max_dim=5)
        assert opts.max_dim == 3
        assert new_opts.max_dim == 5

    def test_replace_method_preserves_other_fields(self):
        opts = PersistenceOptions(max_dim=3, max_radius=1.5)
        new_opts = opts.replace(threads=4)
        assert new_opts.max_dim == 3
        assert new_opts.max_radius == 1.5
        assert new_opts.threads == 4

    def test_max_dim_bool_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="max_dim"):
            PersistenceOptions(max_dim=True)

    def test_max_dim_negative_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="max_dim"):
            PersistenceOptions(max_dim=-1)

    def test_max_dim_float_truncated(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="max_dim"):
            PersistenceOptions(max_dim=2.5)

    def test_max_radius_non_number_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="max_radius"):
            PersistenceOptions(max_radius="abc")

    def test_max_radius_nan_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="max_radius"):
            PersistenceOptions(max_radius=float("nan"))

    def test_max_radius_negative_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="max_radius"):
            PersistenceOptions(max_radius=-0.5)

    def test_max_radius_inf_allowed(self):
        opts = PersistenceOptions(max_radius=float("inf"))
        assert opts.max_radius == float("inf")

    def test_max_radius_none_allowed(self):
        opts = PersistenceOptions(max_radius=None)
        assert opts.max_radius is None

    def test_threads_bool_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="threads"):
            PersistenceOptions(threads=False)

    def test_threads_negative_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="threads"):
            PersistenceOptions(threads=-2)

    def test_threads_float_truncated_to_int(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="threads"):
            PersistenceOptions(threads=3.5)

    def test_error_tolerance_string_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="error_tolerance"):
            PersistenceOptions(error_tolerance="low")

    def test_error_tolerance_negative_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="error_tolerance"):
            PersistenceOptions(error_tolerance=-0.1)

    def test_error_tolerance_nan_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="error_tolerance"):
            PersistenceOptions(error_tolerance=float("nan"))

    def test_error_tolerance_inf_raises(self):
        from pynerve.exceptions._validation import ValidationError

        with pytest.raises(ValidationError, match="error_tolerance"):
            PersistenceOptions(error_tolerance=float("inf"))


class TestPH5PH6Config:
    def test_default_values(self):
        cfg = PH5PH6Config()
        assert cfg.numerical_tolerance == EPS_1e_9
        assert cfg.max_iterations == 1000
        assert cfg.enable_stability_checks is True
        assert cfg.validate_results is True
        assert cfg.require_bitwise_reproducibility is False
        assert cfg.enable_checksum_validation is True
        assert cfg.computation_id == ""

    def test_is_frozen(self):
        cfg = PH5PH6Config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_iterations = 500

    def test_repr_non_empty(self):
        cfg = PH5PH6Config()
        r = repr(cfg)
        assert "PH5PH6Config" in r

    def test_repr_empty_fields_omitted(self):
        cfg = PH5PH6Config(max_iterations=0, computation_id="")
        r = repr(cfg)
        assert r.count("=") > 0


class TestPH5PH6Metrics:
    def test_default_values(self):
        m = PH5PH6Metrics()
        assert m.computation_time_ms == 0.0
        assert m.peak_memory_bytes == 0
        assert m.original_simplices == 0
        assert m.final_simplices == 0
        assert m.compression_ratio == 1.0
        assert m.quality_score == 0.0
        assert m.passed_stability_checks is False
        assert m.numerical_errors == 0
        assert m.checksum_validation_passed is False

    def test_repr(self):
        m = PH5PH6Metrics(computation_time_ms=1.5, peak_memory_bytes=1024)
        r = repr(m)
        assert "PH5PH6Metrics" in r
        assert "1.5" in r
        assert "1024" in r


class TestPH5PH6Engine:
    def test_default_construction(self):
        engine = PH5PH6Engine()
        assert isinstance(engine.config, PH5PH6Config)

    def test_with_config(self):
        cfg = PH5PH6Config(max_iterations=500)
        engine = PH5PH6Engine(config=cfg)
        assert engine.config.max_iterations == 500

    def test_repr(self):
        engine = PH5PH6Engine()
        r = repr(engine)
        assert "PH5PH6Engine" in r
