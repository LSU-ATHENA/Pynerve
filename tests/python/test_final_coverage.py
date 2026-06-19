from __future__ import annotations

import asyncio
import math
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class TestValidationReexport:
    def test_all_names_are_importable(self):
        from pynerve._validation import __all__ as all_names

        for name in all_names:
            __import__("pynerve._validation", fromlist=[name])


class TestCupyOps:
    def test_compute_diagrams_delegates(self):
        with patch("pynerve.cupy_ops.compute_diagrams_cupy") as mock:
            mock.return_value = {"pairs": [], "betti_numbers": [0]}
            from pynerve.cupy_ops import compute_diagrams

            result = compute_diagrams(np.ones((10, 3)), max_radius=2.0, max_dim=1, device_id=0)
            mock.assert_called_once()
            assert result == {"pairs": [], "betti_numbers": [0]}

    def test_compute_diagrams_defaults(self):
        with patch("pynerve.cupy_ops.compute_diagrams_cupy") as mock:
            mock.return_value = {"pairs": []}
            from pynerve.cupy_ops import compute_diagrams

            result = compute_diagrams(np.ones((5, 2)))
            mock.assert_called_once_with(ANY, None, 2, 0)
            assert result == {"pairs": []}

    def test_batch_diagrams_delegates_kwargs(self):
        with patch("pynerve.cupy_ops.batch_diagrams_cupy") as mock:
            mock.return_value = [{"pairs": []}]
            from pynerve.cupy_ops import batch_diagrams

            clouds = [np.ones((5, 2)), np.ones((3, 2))]
            result = batch_diagrams(clouds, max_radius=1.0, max_dim=1)
            mock.assert_called_once_with(clouds, max_radius=1.0, max_dim=1)
            assert result == [{"pairs": []}]


class TestFallbackClasses:
    def test_persistence_options_replace_mode(self):
        from pynerve._fallback_classes import (
            PersistenceMode,
            PersistenceOptions,
        )

        opts = PersistenceOptions(max_dim=2)
        new_opts = opts.replace(mode=PersistenceMode.APPROX)
        assert new_opts.mode == PersistenceMode.APPROX
        assert new_opts is not opts

    def test_persistence_options_replace_backend(self):
        from pynerve._fallback_classes import PersistenceBackend, PersistenceOptions

        opts = PersistenceOptions()
        new_opts = opts.replace(backend=PersistenceBackend.CUDA_HYBRID)
        assert new_opts.backend == PersistenceBackend.CUDA_HYBRID

    def test_persistence_options_replace_max_radius(self):
        from pynerve._fallback_classes import PersistenceOptions

        opts = PersistenceOptions(max_radius=None)
        new_opts = opts.replace(max_radius=2.5)
        assert new_opts.max_radius == 2.5

    def test_persistence_options_replace_threads(self):
        from pynerve._fallback_classes import PersistenceOptions

        opts = PersistenceOptions()
        new_opts = opts.replace(threads=8)
        assert new_opts.threads == 8

    def test_persistence_options_replace_error_tolerance(self):
        from pynerve._fallback_classes import PersistenceOptions

        opts = PersistenceOptions()
        new_opts = opts.replace(error_tolerance=0.001)
        assert new_opts.error_tolerance == 0.001

    def test_persistence_options_accepts_inf_max_radius(self):
        from pynerve._fallback_classes import PersistenceOptions

        opts = PersistenceOptions(max_radius=float("inf"))
        assert math.isinf(opts.max_radius)

    def test_persistence_options_validates_error_tolerance_inf(self):
        from pynerve._fallback_classes import PersistenceOptions
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError):
            PersistenceOptions(error_tolerance=float("inf"))

    def test_ph5ph6_config_repr_full(self):
        from pynerve._fallback_classes import PH5PH6Config

        cfg = PH5PH6Config(
            numerical_tolerance=1e-12,
            max_iterations=500,
            computation_id="test-id",
        )
        r = repr(cfg)
        assert "PH5PH6Config" in r

    def test_ph5ph6_metrics_repr_full(self):
        from pynerve._fallback_classes import PH5PH6Metrics

        m = PH5PH6Metrics(
            computation_time_ms=10.5,
            peak_memory_bytes=2048,
            original_simplices=100,
            final_simplices=50,
            compression_ratio=2.0,
            quality_score=0.95,
            passed_stability_checks=True,
            numerical_errors=0,
            checksum_validation_passed=True,
        )
        r = repr(m)
        assert "PH5PH6Metrics" in r

    def test_ph5ph6_engine_repr(self):
        from pynerve._fallback_classes import PH5PH6Config, PH5PH6Engine

        cfg = PH5PH6Config(max_iterations=500)
        engine = PH5PH6Engine(config=cfg)
        r = repr(engine)
        assert "PH5PH6Engine" in r
        assert "max_iterations=500" in r


class TestSharedMemory:
    def test_from_array_zero_dimensional_raises(self):
        from pynerve._shared_memory import SharedMemoryArray

        with pytest.raises(ValueError, match="at least one dimension"):
            SharedMemoryArray.from_array(np.array(3.14))

    def test_from_array_preserves_data(self):
        from pynerve._shared_memory import SharedMemoryArray

        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        shm = SharedMemoryArray.from_array(data)
        try:
            np.testing.assert_array_equal(shm.array, data)
        finally:
            shm.close()

    def test_attach_creates_correct_instance(self):
        from pynerve._shared_memory import SharedMemoryArray

        original = SharedMemoryArray((3,), np.float64)
        try:
            attached = SharedMemoryArray.attach(original.name, original.shape, original.dtype)
            try:
                assert attached.name == original.name
                assert attached.shape == original.shape
                assert not attached._owner
            finally:
                attached.close()
        finally:
            original.close()

    def test_getitem_when_closed_raises(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        shm.close()
        with pytest.raises(RuntimeError, match="is closed"):
            _ = shm[0]

    def test_setitem_when_closed_raises(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        shm.close()
        with pytest.raises(RuntimeError, match="is closed"):
            shm[0] = 1.0

    def test_array_when_closed_raises(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        shm.close()
        with pytest.raises(RuntimeError, match="is closed"):
            _ = shm.array

    def test_close_idempotent(self):
        from pynerve._shared_memory import SharedMemoryArray

        shm = SharedMemoryArray((3,), np.float64)
        shm.close()
        shm.close()

    def test_context_manager_enter_exit(self):
        from pynerve._shared_memory import SharedMemoryArray

        with SharedMemoryArray((4,), np.float64) as shm:
            assert not shm._closed
            assert isinstance(shm.array, np.ndarray)

    def test_name_rejects_empty_string(self):
        from pynerve._shared_memory import SharedMemoryArray

        with pytest.raises(ValueError, match="name must be a non-empty string"):
            SharedMemoryArray((3,), np.float64, name="")

    def test_name_rejects_non_string(self):
        from pynerve._shared_memory import SharedMemoryArray

        with pytest.raises(ValueError, match="name must be a non-empty string"):
            SharedMemoryArray((3,), np.float64, name=42)

    def test_object_dtype_raises(self):
        from pynerve._shared_memory import SharedMemoryArray

        with pytest.raises(TypeError, match="object dtype"):
            SharedMemoryArray((3,), np.dtype(object))


class TestAsyncCompute:
    @pytest.mark.asyncio
    async def test_init_validates_max_workers(self):
        from pynerve._async_compute import AsyncPersistenceComputer
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError):
            AsyncPersistenceComputer(max_workers=0)

    @pytest.mark.asyncio
    async def test_init_validates_buffer_size(self):
        from pynerve._async_compute import AsyncPersistenceComputer
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError):
            AsyncPersistenceComputer(buffer_size=0)

    @pytest.mark.asyncio
    async def test_compute_batch_async_rejects_non_async_iter(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        apc = AsyncPersistenceComputer(max_workers=1)
        try:
            async for _ in apc.compute_batch_async([np.ones((2, 3))]):
                pass
        except (TypeError, asyncio.CancelledError):
            pass
        finally:
            await apc.close()

    @pytest.mark.asyncio
    async def test_compute_batch_async_rejects_closed(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        apc = AsyncPersistenceComputer(max_workers=1)
        await apc.close()
        with pytest.raises(RuntimeError, match="is closed"):

            async def _src():
                yield np.ones((2, 3))

            async for _ in apc.compute_batch_async(_src()):
                pass

    @pytest.mark.asyncio
    async def test_compute_batch_async_rejects_non_callable_fn(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        apc = AsyncPersistenceComputer(max_workers=1)
        try:
            with pytest.raises(TypeError, match="callable"):

                async def _src():
                    yield np.ones((2, 3))

                async for _ in apc.compute_batch_async(_src(), compute_fn=42):
                    pass
        finally:
            await apc.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        apc = AsyncPersistenceComputer(max_workers=1)
        await apc.close()
        await apc.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        async with AsyncPersistenceComputer(max_workers=1) as apc:
            assert not apc._closed

    @pytest.mark.asyncio
    async def test_default_compute_calls_persistence(self):
        with patch("pynerve._async_compute.compute_persistence") as mock:
            mock.return_value = None
            from pynerve._async_compute import AsyncPersistenceComputer

            apc = AsyncPersistenceComputer(max_workers=1)
            try:
                apc._default_compute(np.ones((5, 2)))
                mock.assert_called_once()
            finally:
                await apc.close()

    @pytest.mark.asyncio
    async def test_compute_batch_async_with_custom_fn(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        apc = AsyncPersistenceComputer(max_workers=1, buffer_size=1)
        try:

            async def _src():
                yield np.ones((2, 3))

            results = []
            async for r in apc.compute_batch_async(
                _src(), compute_fn=lambda x, **kw: {"result": x.shape}
            ):
                results.append(r)
            assert len(results) == 1
            assert results[0]["result"] == (2, 3)
        finally:
            await apc.close()

    @pytest.mark.asyncio
    async def test_compute_batch_async_with_empty_source(self):
        from pynerve._async_compute import AsyncPersistenceComputer

        apc = AsyncPersistenceComputer(max_workers=1, buffer_size=1)
        try:

            async def _src():
                if False:
                    yield

            results = []
            async for r in apc.compute_batch_async(_src()):
                results.append(r)
            assert len(results) == 0
        finally:
            await apc.close()


class TestAsyncFacade:
    def test_validate_filepaths_empty_list(self):
        from pynerve._async_facade import _validate_filepaths

        with pytest.raises(ValueError, match="non-empty"):
            _validate_filepaths([])

    def test_validate_filepaths_nonexistent(self):
        from pynerve._async_facade import _validate_filepaths

        with pytest.raises(FileNotFoundError, match="file not found"):
            _validate_filepaths(["/nonexistent/path/12345.xyz"])

    def test_validate_filepaths_non_iterable(self):
        from pynerve._async_facade import _validate_filepaths

        with pytest.raises(TypeError, match="iterable"):
            _validate_filepaths(42)

    def test_validate_filepaths_string_instead_of_list(self):
        from pynerve._async_facade import _validate_filepaths

        with pytest.raises(TypeError, match="iterable"):
            _validate_filepaths("some_string")

    def test_require_async_deps_without_extension(self):
        with patch("pynerve._async_facade._core_import_error", ImportError("mock error")):
            from pynerve._async_facade import _require_async_deps

            with pytest.raises(ImportError):
                _require_async_deps()

    @pytest.mark.asyncio
    async def test_compute_persistence_async_with_mock(self):
        from pynerve._compute_core import PersistenceResult

        class _FakeComputer:
            def __init__(self, max_workers=1):
                self.max_workers = max_workers

            async def compute_batch_async(self, data_batches, compute_fn, **kwargs):
                async for batch in data_batches:
                    result = compute_fn(batch, **kwargs)
                    yield result

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with (
            patch(
                "pynerve._async_facade._require_async_deps",
                return_value=(_FakeComputer, MagicMock(), MagicMock()),
            ),
            patch("pynerve._async_facade.compute_persistence") as mock_cp,
        ):
            mock_cp.return_value = {
                "pairs": [],
                "betti_numbers": [0],
                "max_dim": 1,
                "max_radius": 0.0,
                "diagnostics": {},
            }
            from pynerve._async_facade import compute_persistence_async

            points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            result = await compute_persistence_async(points, max_workers=1, max_dim=1)
            assert isinstance(result, PersistenceResult)

    @pytest.mark.asyncio
    async def test_stream_persistence_use_gpu_type_error(self):
        from pynerve._async_facade import stream_persistence

        with pytest.raises(TypeError, match="use_gpu must be a boolean"):
            async for _ in stream_persistence("data.npy", use_gpu="yes"):
                pass


class TestAsyncGPU:
    def test_init_rejects_bool_device_id(self):
        from pynerve._async_gpu import AsyncGPUTransfer

        with pytest.raises(TypeError, match="device_id must be an integer"):
            AsyncGPUTransfer(device_id=True)

    def test_init_rejects_negative_device_id(self):
        from pynerve._async_gpu import AsyncGPUTransfer

        with pytest.raises(ValueError, match="non-negative"):
            AsyncGPUTransfer(device_id=-1)

    def test_init_rejects_float_device_id(self):
        from pynerve._async_gpu import AsyncGPUTransfer

        with pytest.raises(TypeError, match="device_id must be an integer"):
            AsyncGPUTransfer(device_id=1.5)

    def test_init_accepts_int_device_id(self):
        from pynerve._async_gpu import AsyncGPUTransfer

        transfer = AsyncGPUTransfer(device_id=0)
        assert transfer.device_id == 0

    def test_init_accepts_numpy_int_device_id(self):
        from pynerve._async_gpu import AsyncGPUTransfer

        transfer = AsyncGPUTransfer(device_id=np.int64(1))
        assert transfer.device_id == 1

    def test_require_cupy_transfer_raises(self):
        with patch("pynerve._async_gpu.HAS_CUPY", False):
            from pynerve._async_gpu import _require_cupy_transfer

            with pytest.raises(RuntimeError, match="CuPy required"):
                _require_cupy_transfer()


class TestDiagnostics:
    def test_verbose_debug_level(self):
        from pynerve.diagnostics import verbose

        with verbose(enabled=True, level="debug"):
            pass

    def test_verbose_trace_level(self):
        from pynerve.diagnostics import verbose

        with verbose(enabled=True, level="trace"):
            pass

    def test_verbose_rejects_invalid_level(self):
        from pynerve.diagnostics import verbose

        with pytest.raises(ValueError, match="level must be"), verbose(level="invalid"):
            pass

    def test_verbose_rejects_non_bool_enabled(self):
        from pynerve.diagnostics import verbose

        with pytest.raises(TypeError, match="boolean"), verbose(enabled="yes"):
            pass

    def test_diagnostics_collector_empty_summary(self):
        from pynerve.diagnostics import DiagnosticsCollector

        dc = DiagnosticsCollector()
        assert dc.summary() == {}

    def test_diagnostics_collector_empty_report(self):
        from pynerve.diagnostics import DiagnosticsCollector

        dc = DiagnosticsCollector()
        r = dc.report()
        assert "Diagnostic Report" in r

    def test_diagnostics_collector_error_capture(self):
        from pynerve.diagnostics import DiagnosticsCollector

        dc = DiagnosticsCollector()
        try:
            with dc.track("test_op", n_points=5):
                raise ValueError("test error")
        except ValueError:
            pass
        assert len(dc.diagnostics) == 1
        assert dc.diagnostics[0].error == "test error"

    def test_diagnostics_collector_unexpected_kwargs_warn(self):
        from pynerve.diagnostics import DiagnosticsCollector

        dc = DiagnosticsCollector()
        with (
            pytest.warns(UserWarning, match="Unexpected keyword arguments"),
            dc.track("test_op", unexpected_kwarg=42),
        ):
            pass

    def test_diagnostic_info_repr_error(self):
        from pynerve.diagnostics import DiagnosticInfo

        info = DiagnosticInfo(operation="failed_op", duration=0.5, error="something went wrong")
        r = repr(info)
        assert "ERROR" in r

    def test_diagnostic_info_repr_ok(self):
        from pynerve.diagnostics import DiagnosticInfo

        info = DiagnosticInfo(operation="ok_op", duration=0.1)
        r = repr(info)
        assert "OK" in r

    def test_diagnostics_collector_repr(self):
        from pynerve.diagnostics import DiagnosticsCollector

        dc = DiagnosticsCollector()
        r = repr(dc)
        assert "operations=0" in r

    def test_diagnostics_collector_report_with_error(self):
        from pynerve.diagnostics import DiagnosticsCollector

        dc = DiagnosticsCollector()
        try:
            with dc.track("bad_op"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        r = dc.report()
        assert "ERROR" in r
        assert "boom" in r

    def test_diagnostics_collector_summary_with_data(self):
        from pynerve.diagnostics import DiagnosticsCollector

        dc = DiagnosticsCollector()
        with dc.track("op1", n_points=10):
            pass
        with dc.track("op2", n_points=20):
            pass
        s = dc.summary()
        assert s["n_operations"] == 2
        assert s["n_errors"] == 0
        assert s["error_rate"] == 0.0


class TestFormatsInterop:
    def test_from_gudhi_unknown_format_raises(self):
        from pynerve._formats_interop import from_gudhi
        from pynerve.exceptions import InvalidArgumentError

        with pytest.raises(InvalidArgumentError, match="Unknown format_type"):
            from_gudhi([], format_type="invalid")

    def test_to_external_with_filepath(self):
        from pynerve._formats_interop import to_external

        diagram = [(0.0, 1.0, 0), (0.5, 2.0, 1)]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            result = to_external(diagram, filepath=path)
            assert Path(path).exists()
            assert result == str(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_external_with_empty_diagram(self):
        from pynerve._formats_interop import to_external

        result = to_external([])
        import json

        data = json.loads(result)
        assert data["diagrams"] == []

    def test_resolve_simplex_iter_from_dict(self):
        from pynerve._formats_interop import _resolve_simplex_iter

        source = {"simplices": [([0, 1], 1.0), ([2], 2.0)]}
        result = _resolve_simplex_iter(source)
        assert result == [([0, 1], 1.0), ([2], 2.0)]

    def test_resolve_simplex_iter_from_list(self):
        from pynerve._formats_interop import _resolve_simplex_iter

        source = [([0, 1], 0.5), ([1, 2], 1.5)]
        result = _resolve_simplex_iter(source)
        assert result is source

    def test_resolve_simplex_iter_unsupported(self):
        from pynerve._formats_interop import _resolve_simplex_iter
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unsupported simplex_tree source"):
            _resolve_simplex_iter(42)

    def test_diagram_entry_nan_birth(self):
        from pynerve._formats_interop import _validate_diagram_entry
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="births must be finite"):
            _validate_diagram_entry(float("nan"), 1.0, 0)

    def test_diagram_entry_inf_birth(self):
        from pynerve._formats_interop import _validate_diagram_entry
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="births must be finite"):
            _validate_diagram_entry(float("inf"), 1.0, 0)

    def test_diagram_entry_nan_death(self):
        from pynerve._formats_interop import _validate_diagram_entry
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="deaths must be finite or \\+inf"):
            _validate_diagram_entry(0.0, float("nan"), 0)

    def test_diagram_entry_neginf_death(self):
        from pynerve._formats_interop import _validate_diagram_entry
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="deaths must be finite or \\+inf"):
            _validate_diagram_entry(0.0, float("-inf"), 0)

    def test_diagram_entry_death_less_than_birth(self):
        from pynerve._formats_interop import _validate_diagram_entry
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="deaths must be greater than or equal"):
            _validate_diagram_entry(1.0, 0.5, 0)

    def test_diagram_entry_valid_with_inf_death(self):
        from pynerve._formats_interop import _validate_diagram_entry

        birth, death, dim = _validate_diagram_entry(0.0, float("inf"), 1)
        assert birth == 0.0
        assert math.isinf(death)
        assert dim == 1


class TestPHLayer:
    def test_effective_max_radius_explicit_finite(self):
        from pynerve.diff.ph_layer import _effective_max_radius

        points = torch.randn(5, 3)
        result = _effective_max_radius(points, 2.0)
        assert result == 2.0

    def test_effective_max_radius_single_point(self):
        from pynerve.diff.ph_layer import _effective_max_radius

        points = torch.randn(1, 3)
        result = _effective_max_radius(points, float("inf"))
        assert result == 0.0

    def test_validate_persistence_inputs_wrong_filtration(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 3)
        with pytest.raises(ValueError, match="only rips"):
            _validate_persistence_inputs(points, 1, "alpha", {})

    def test_validate_persistence_inputs_wrong_dim(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(5, 3)
        with pytest.raises(ValueError, match="shape \\(batch, n_points, dim\\)"):
            _validate_persistence_inputs(points, 1, "rips", {})

    def test_validate_persistence_inputs_negative_max_dim(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 3)
        with pytest.raises(ValueError, match="non-negative"):
            _validate_persistence_inputs(points, -1, "rips", {})

    def test_validate_persistence_inputs_empty_batch(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(0, 5, 3)
        with pytest.raises(ValueError, match="non-empty"):
            _validate_persistence_inputs(points, 1, "rips", {})

    def test_validate_persistence_inputs_empty_points(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 0, 3)
        with pytest.raises(ValueError, match="non-empty"):
            _validate_persistence_inputs(points, 1, "rips", {})

    def test_validate_persistence_inputs_empty_features(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 0)
        with pytest.raises(ValueError, match="non-empty"):
            _validate_persistence_inputs(points, 1, "rips", {})

    def test_validate_persistence_inputs_non_finite(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 3)
        points[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            _validate_persistence_inputs(points, 1, "rips", {})

    def test_validate_persistence_inputs_max_radius_zero(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 3)
        with pytest.raises(ValueError, match="positive"):
            _validate_persistence_inputs(points, 1, "rips", {"max_radius": 0.0})

    def test_validate_persistence_inputs_max_radius_nan(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 3)
        with pytest.raises(ValueError, match="positive"):
            _validate_persistence_inputs(points, 1, "rips", {"max_radius": float("nan")})

    def test_validate_persistence_inputs_max_radius_none(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 3)
        max_radius, metric, reduction = _validate_persistence_inputs(
            points, 1, "rips", {"max_radius": None}
        )
        assert math.isinf(max_radius)

    def test_validate_persistence_inputs_default_max_radius(self):
        from pynerve.diff.ph_layer import _validate_persistence_inputs

        points = torch.randn(2, 5, 3)
        max_radius, metric, reduction = _validate_persistence_inputs(points, 1, "rips", {})
        assert math.isinf(max_radius)
        assert metric == "euclidean"
        assert reduction == "clearing"

    def test_setup_persistence_options_without_metric_attr(self):
        import sys
        import types

        from pynerve.diff.ph_layer import _setup_persistence_options

        class MockOptions:
            max_dim = 0
            max_radius = 0.0

        mock_core = types.ModuleType("mock_core")
        mock_core.PersistenceOptions = MockOptions
        sys.modules["mock_core"] = mock_core
        try:
            opts = _setup_persistence_options(mock_core, 1, float("inf"), "euclidean", "clearing")
            assert opts.max_dim == 1
            assert opts.max_radius == 0.0
        finally:
            del sys.modules["mock_core"]

    def test_setup_persistence_options_without_metric_attr_non_euclidean(self):
        import sys
        import types

        from pynerve.diff.ph_layer import _setup_persistence_options

        class MockOptions:
            max_dim = 0
            max_radius = 0.0

        mock_core = types.ModuleType("mock_core")
        mock_core.PersistenceOptions = MockOptions
        sys.modules["mock_core"] = mock_core
        try:
            with pytest.raises(ValueError, match="only supports euclidean"):
                _setup_persistence_options(mock_core, 1, float("inf"), "manhattan", "clearing")
        finally:
            del sys.modules["mock_core"]

    def test_setup_persistence_options_without_reduction_attr_non_clearing(self):
        import sys
        import types

        from pynerve.diff.ph_layer import _setup_persistence_options

        class MockOptions:
            max_dim = 0
            max_radius = 0.0
            metric = "euclidean"

        mock_core = types.ModuleType("mock_core")
        mock_core.PersistenceOptions = MockOptions
        sys.modules["mock_core"] = mock_core
        try:
            with pytest.raises(ValueError, match="only supports clearing"):
                _setup_persistence_options(mock_core, 1, float("inf"), "euclidean", "exhaustive")
        finally:
            del sys.modules["mock_core"]

    def test_setup_persistence_options_with_both_attrs(self):
        class MockOptions:
            max_dim = 0
            max_radius = 0.0
            metric = "euclidean"
            reduction = "clearing"

        import sys
        import types

        mock_core = types.ModuleType("pynerve_internal")
        mock_core.PersistenceOptions = MockOptions
        sys.modules["pynerve_internal"] = mock_core
        try:
            from pynerve.diff.ph_layer import _setup_persistence_options

            opts = _setup_persistence_options(mock_core, 1, 2.0, "euclidean", "clearing")
            assert opts.max_dim == 1
            assert opts.max_radius == 2.0
            assert opts.metric == "euclidean"
            assert opts.reduction == "clearing"
        finally:
            del sys.modules["pynerve_internal"]

    def test_differentiable_vietoris_rips_init_negative_dim(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        with pytest.raises(ValueError, match="non-negative"):
            DifferentiableVietorisRips(max_dim=-1)

    def test_differentiable_vietoris_rips_init_with_radius(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        layer = DifferentiableVietorisRips(max_dim=1, max_radius=2.0)
        assert layer.max_dim == 1
        assert layer.max_radius == 2.0

    def test_differentiable_vietoris_rips_forward_wrong_dim(self):
        from pynerve.diff.ph_layer import DifferentiableVietorisRips

        layer = DifferentiableVietorisRips(max_dim=1)
        points = torch.randn(5, 3)
        with pytest.raises(ValueError, match="shape \\(batch, n_points, dim\\)"):
            layer.forward(points)

    def test_differentiable_alpha_complex_forward_raises(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex

        layer = DifferentiableAlphaComplex(max_dim=2)
        with pytest.raises(RuntimeError, match="not exposed"):
            layer.forward(torch.randn(2, 5, 3))

    def test_differentiable_alpha_complex_init_negative_dim(self):
        from pynerve.diff.ph_layer import DifferentiableAlphaComplex

        with pytest.raises(ValueError, match="non-negative"):
            DifferentiableAlphaComplex(max_dim=-1)

    def test_differentiable_cubical_forward_raises(self):
        from pynerve.diff.ph_layer import DifferentiableCubical

        layer = DifferentiableCubical(max_dim=2)
        with pytest.raises(RuntimeError, match="not exposed"):
            layer.forward(torch.randn(2, 5, 5))

    def test_differentiable_cubical_init_negative_dim(self):
        from pynerve.diff.ph_layer import DifferentiableCubical

        with pytest.raises(ValueError, match="non-negative"):
            DifferentiableCubical(max_dim=-1)

    def test_filtration_learning_layer_init_negative_input_dim(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        with pytest.raises(ValueError, match="input_dim must be positive"):
            FiltrationLearningLayer(input_dim=0)

    def test_filtration_learning_layer_init_negative_hidden_dim(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        with pytest.raises(ValueError, match="hidden dimensions must be positive"):
            FiltrationLearningLayer(input_dim=2, hidden_dims=[64, -1])

    def test_filtration_learning_layer_forward_wrong_dim(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3)
        points = torch.randn(5, 3)
        with pytest.raises(ValueError, match="shape \\(batch, n_points, dim\\)"):
            layer.forward(points)

    def test_filtration_learning_layer_forward_shape(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=3, hidden_dims=[16])
        points = torch.randn(2, 5, 3)
        output = layer.forward(points)
        assert output.shape == (2, 5)

    def test_filtration_learning_layer_default_hidden_dims(self):
        from pynerve.diff.ph_layer import FiltrationLearningLayer

        layer = FiltrationLearningLayer(input_dim=2)
        points = torch.randn(1, 3, 2)
        output = layer.forward(points)
        assert output.shape == (1, 3)

    def test_learnable_filtration_persistence_forward(self):
        from pynerve.diff.ph_layer import LearnableFiltrationPersistence

        layer = LearnableFiltrationPersistence(input_dim=2, max_dim=0, hidden_dims=[16])
        points = torch.randn(1, 3, 2)
        filt_values = layer.filtration(points)
        assert filt_values.shape == (1, 3)

    def test_pad_diagrams_across_batches_empty(self):
        from pynerve.diff.ph_layer import _pad_diagrams_across_batches

        result = _pad_diagrams_across_batches([], 0, 2, torch.float32, torch.device("cpu"))
        assert len(result) == 1
        assert result[0].shape == (2, 0, 2)

    def test_pad_diagrams_across_batches_with_data(self):
        from pynerve.diff.ph_layer import _pad_diagrams_across_batches

        d1 = [torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)]
        d2 = [torch.tensor([[0.0, 0.3]], dtype=torch.float32)]
        result = _pad_diagrams_across_batches([d1, d2], 0, 2, torch.float32, torch.device("cpu"))
        assert result[0].shape == (2, 2, 2)
        assert torch.allclose(result[0][0], d1[0])
        assert result[0][1, 1, 0] == 0.0

    def test_accumulate_merge_gradients_basic(self):
        from pynerve.diff.ph_layer import _accumulate_merge_gradients

        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        dists = torch.cdist(pts, pts)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_grad = diff / (dists.unsqueeze(-1) + 1e-8)
        grad_target = torch.zeros(1, 3, 2, dtype=torch.float64)
        grad_diag = torch.tensor([[0.1, 0.1], [0.2, 0.2]], dtype=torch.float64)

        _accumulate_merge_gradients(pts, grad_diag, dists, dist_grad, grad_target, 0)
        assert torch.isfinite(grad_target).all()

    def test_compute_persistence_backward_basic(self):
        from pynerve.diff.ph_layer import _compute_persistence_backward

        points = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=False)
        grad_diags = (
            torch.zeros(2, 5, 2, dtype=torch.float64),
            torch.zeros(2, 5, 2, dtype=torch.float64),
        )
        result = _compute_persistence_backward(
            points, grad_diags, 1, "rips", {"max_radius": float("inf")}
        )
        assert result.shape == points.shape

    def test_compute_persistence_backward_non_rips(self):
        from pynerve.diff.ph_layer import _compute_persistence_backward

        points = torch.randn(2, 5, 3, dtype=torch.float64)
        grad_diags = (torch.zeros(2, 5, 2, dtype=torch.float64),)
        result = _compute_persistence_backward(points, grad_diags, 1, "alpha", {})
        assert result.shape == points.shape
        assert torch.allclose(result, torch.zeros_like(points))

    def test_compute_persistence_backward_h0_only(self):
        from pynerve.diff.ph_layer import _compute_persistence_backward

        points = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=False)
        grad_diags_h0 = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        grad_diags = (grad_diags_h0, torch.zeros(0, 2, dtype=torch.float64))
        result = _compute_persistence_backward(points, grad_diags, 1, "rips", {})
        assert result.shape == points.shape
        assert torch.isfinite(result).all()

    def test_compute_persistence_forward_no_core(self):
        import builtins

        original_import = builtins.__import__

        def _block_import(name, *args, **kwargs):
            if name == "pynerve_internal":
                raise ImportError("mock import error")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_import):
            from pynerve.diff.ph_layer import _compute_persistence_forward

            with pytest.raises(ImportError):
                _compute_persistence_forward(torch.randn(2, 5, 3), 1, "rips")


class TestLossModules:
    def test_betti_number_loss_soft_step(self):
        from pynerve.diff._loss_modules import BettiNumberLoss

        loss = BettiNumberLoss(threshold=0.1, temperature=0.1)
        x = torch.tensor([0.0, 0.1, 0.2, 1.0])
        result = loss.soft_step(x)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_diagram_complexity_loss_total_persistence(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="total_persistence")
        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        result = loss.forward(diagram)
        assert result.item() == pytest.approx(3.0)

    def test_diagram_complexity_loss_persistence_entropy(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="persistence_entropy")
        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        result = loss.forward(diagram)
        assert torch.isfinite(result).all()

    def test_diagram_complexity_loss_persistence_entropy_zero(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="persistence_entropy")
        diagram = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        result = loss.forward(diagram)
        assert result.item() == 0.0

    def test_diagram_complexity_loss_num_features(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="num_features")
        diagram = torch.tensor([[0.0, 1.0], [0.0, 0.5], [0.0, 0.05]], dtype=torch.float32)
        result = loss.forward(diagram)
        assert result.item() == 2.0

    def test_diagram_complexity_loss_max_persistence(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="max_persistence")
        diagram = torch.tensor([[0.0, 1.0], [0.0, 0.5]], dtype=torch.float32)
        result = loss.forward(diagram)
        assert result.item() == 1.0

    def test_diagram_complexity_loss_empty(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="total_persistence")
        diagram = torch.empty((0, 2), dtype=torch.float32)
        result = loss.forward(diagram)
        assert result.item() == 0.0

    def test_diagram_complexity_loss_unknown_measure(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        with pytest.raises(ValueError, match="unknown complexity measure"):
            DiagramComplexityLoss(measure="unknown")

    def test_diagram_complexity_loss_three_col_input(self):
        from pynerve.diff._loss_modules import DiagramComplexityLoss

        loss = DiagramComplexityLoss(measure="total_persistence")
        diagram = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float32)
        result = loss.forward(diagram)
        assert result.item() == pytest.approx(3.0)

    def test_stability_loss_non_callable(self):
        from pynerve.diff._loss_modules import StabilityLoss

        loss = StabilityLoss(epsilon=0.01, num_samples=2)
        with pytest.raises(TypeError, match="must be callable"):
            loss.forward(torch.ones(2, 2), 42)

    def test_stability_loss_empty_diagrams(self):
        from pynerve.diff._loss_modules import StabilityLoss

        loss = StabilityLoss(epsilon=0.01, num_samples=3)
        with pytest.raises(ValueError, match="orig_diagrams must be non-empty"):
            loss.forward(torch.ones(2, 2), lambda _pts: [])

    def test_multiscale_loss_mismatched_lengths(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        loss = MultiScaleTopologyLoss(scales=(0.1, 0.5, 1.0))
        diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="target_diagrams length must match scales"):
            loss.forward(diagram, [diagram])

    def test_multiscale_loss_empty_scales(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        with pytest.raises(ValueError, match="non-empty"):
            MultiScaleTopologyLoss(scales=())

    def test_multiscale_loss_negative_scale(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        with pytest.raises(ValueError, match="must be positive"):
            MultiScaleTopologyLoss(scales=(0.0,))

    def test_multiscale_loss_all_scales(self):
        from pynerve.diff._loss_modules import MultiScaleTopologyLoss

        loss = MultiScaleTopologyLoss(scales=(0.01, 0.1))
        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        target1 = torch.tensor([[0.0, 0.5]], dtype=torch.float32)
        result = loss.forward(diagram, [target1, target1])
        assert torch.isfinite(result).all()

    def test_landscape_loss_landscape_method(self):
        from pynerve.diff._loss_modules import LandscapeLoss

        loss = LandscapeLoss(n_layers=2, resolution=10)
        diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        landscape = loss.landscape(diagram)
        assert landscape is not None
        assert torch.isfinite(landscape).all()

    def test_landscape_loss_forward(self):
        from pynerve.diff._loss_modules import LandscapeLoss

        loss = LandscapeLoss(n_layers=3, resolution=50)
        d1 = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        d2 = torch.tensor([[0.0, 1.0], [0.0, 2.5]], dtype=torch.float32)
        result = loss.forward(d1, d2)
        assert torch.isfinite(result).all()

    def test_landscape_loss_init_validation(self):
        from pynerve.diff._loss_modules import LandscapeLoss
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError):
            LandscapeLoss(n_layers=0)

        with pytest.raises(ValidationError):
            LandscapeLoss(resolution=0)
