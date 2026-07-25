"""Tests for _diagnostics_data.py, _diagnostics_failure.py, _diagnostics_system.py."""

from __future__ import annotations

import io

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestDataQuality:
    """Covers _diagnostics_data.py — check_data_quality and helpers."""

    def test_valid_data(self):
        from pynerve._diagnostics_data import check_data_quality
        data = np.random.rand(50, 3).astype(np.float64)
        report = check_data_quality(data)
        assert report["valid"] is True
        assert report["errors"] == []

    def test_nan_values(self):
        from pynerve._diagnostics_data import check_data_quality
        data = np.array([[0.0, 1.0, 2.0], [np.nan, 1.0, 2.0], [3.0, 4.0, 5.0]])
        report = check_data_quality(data)
        assert report["valid"] is False
        assert any("NaN" in e for e in report["errors"])

    def test_inf_values(self):
        from pynerve._diagnostics_data import check_data_quality
        data = np.array([[0.0, 1.0], [np.inf, 2.0], [3.0, 4.0]])
        report = check_data_quality(data)
        assert report["valid"] is True
        assert any("Inf" in w for w in report["warnings"])

    def test_duplicates(self):
        from pynerve._diagnostics_data import check_data_quality
        data = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]])
        report = check_data_quality(data)
        assert any("duplicate" in w for w in report["warnings"])

    def test_zero_variance(self):
        from pynerve._diagnostics_data import check_data_quality
        data = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        report = check_data_quality(data)
        assert any("zero variance" in w for w in report["warnings"])

    def test_large_range(self, monkeypatch):
        from pynerve._diagnostics_data import check_data_quality
        monkeypatch.setenv("NERVE_COORD_RANGE_WARN", "1.0")
        data = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
        report = check_data_quality(data)
        assert any("range" in w for w in report["warnings"])

    def test_large_dataset(self, monkeypatch):
        from pynerve._diagnostics_data import check_data_quality
        monkeypatch.setenv("NERVE_NPOINTS_WARN", "5")
        data = np.random.rand(10, 3)
        report = check_data_quality(data)
        assert any("Large dataset" in w for w in report["warnings"])

    def test_wrong_ndim(self):
        from pynerve._diagnostics_data import check_data_quality
        data = np.random.rand(5, 3, 2)
        report = check_data_quality(data)
        assert report["valid"] is False
        assert any("2D" in e for e in report["errors"])

    def test_empty_data(self):
        from pynerve._diagnostics_data import check_data_quality
        data = np.empty((0, 3))
        report = check_data_quality(data)
        assert report["valid"] is False
        assert any("non-empty" in e for e in report["errors"])

    def test_non_numeric_dtype(self):
        from pynerve._diagnostics_data import check_data_quality
        data = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=object)
        report = check_data_quality(data)
        assert report["valid"] is False
        assert any("numeric" in e for e in report["errors"])

    def test_non_array_input(self):
        from pynerve._diagnostics_data import check_data_quality
        report = check_data_quality("not an array")
        assert report["valid"] is False
        assert any("numpy" in e.lower() for e in report["errors"])

    def test_torch_tensor_input(self):
        import torch
        from pynerve._diagnostics_data import check_data_quality
        data = torch.rand(10, 3)
        report = check_data_quality(data)
        assert report["valid"] is True

    def test_empty_torch_tensor(self):
        import torch
        from pynerve._diagnostics_data import check_data_quality
        data = torch.empty(0, 3)
        report = check_data_quality(data)
        assert report["valid"] is False

    def test_extract_array_none_for_unknown(self):
        from pynerve._diagnostics_data import _extract_array
        assert _extract_array("hello") is None
        assert _extract_array(42) is None

    def test_extract_array_numpy(self):
        from pynerve._diagnostics_data import _extract_array
        arr = np.array([1.0, 2.0])
        result = _extract_array(arr)
        assert result is arr

    def test_validate_shape_non_2d(self):
        from pynerve._diagnostics_data import _validate_shape_and_dtype, DataQualityReport
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        arr = np.zeros((2, 2, 2))
        result = _validate_shape_and_dtype(arr, report)
        assert result is False
        assert len(report["errors"]) > 0

    def test_validate_shape_empty(self):
        from pynerve._diagnostics_data import _validate_shape_and_dtype, DataQualityReport
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        arr = np.zeros((0, 3))
        result = _validate_shape_and_dtype(arr, report)
        assert result is False


class TestFailureDiagnosis:
    """Covers _diagnostics_failure.py — diagnose_failure and FailureDiagnosis."""

    def test_diagnose_value_error(self):
        from pynerve._diagnostics_failure import diagnose_failure
        diag = diagnose_failure("test_op", ValueError("bad input"))
        assert "test_op" in str(diag)
        assert diag.cause_category == "invalid_input"
        assert len(diag.suggestions) > 0

    def test_diagnose_memory_error(self):
        from pynerve._diagnostics_failure import diagnose_failure
        diag = diagnose_failure("big_op", MemoryError())
        assert diag.cause_category == "out_of_memory"

    def test_diagnose_runtime_error(self):
        from pynerve._diagnostics_failure import diagnose_failure
        diag = diagnose_failure("compute", RuntimeError("GPU failed"))
        assert diag.cause_category == "runtime_failure"

    def test_diagnose_unknown_exception(self):
        from pynerve._diagnostics_failure import diagnose_failure

        class CustomError(Exception):
            pass

        diag = diagnose_failure("custom", CustomError("weird"))
        assert diag.cause_category == "unknown"

    def test_diagnose_with_data(self):
        from pynerve._diagnostics_failure import diagnose_failure
        data = np.random.rand(100, 3)
        diag = diagnose_failure("op", ValueError("bad"), data=data)
        assert diag.data_info is not None
        assert "shape" in diag.data_info
        assert diag.data_info["shape"] == (100, 3)

    def test_diagnose_with_context(self):
        from pynerve._diagnostics_failure import diagnose_failure
        diag = diagnose_failure("op", ValueError("bad"), context={"engine": "ph5"})
        assert diag.context == {"engine": "ph5"}

    def test_diagnose_empty_operation(self):
        from pynerve._diagnostics_failure import diagnose_failure
        with pytest.raises(ValueError, match="non-empty"):
            diagnose_failure("", ValueError("bad"))

    def test_diagnose_non_exception(self):
        from pynerve._diagnostics_failure import diagnose_failure
        with pytest.raises(TypeError, match="Exception"):
            diagnose_failure("op", "not an exception")

    def test_diagnose_context_not_mapping(self):
        from pynerve._diagnostics_failure import diagnose_failure
        with pytest.raises(TypeError, match="mapping"):
            diagnose_failure("op", ValueError("bad"), context="not a dict")

    def test_failure_diagnosis_is_str(self):
        from pynerve._diagnostics_failure import FailureDiagnosis
        fd = FailureDiagnosis("hello", cause_category="test")
        assert isinstance(fd, str)
        assert fd == "hello"
        assert fd.cause_category == "test"

    def test_failure_diagnosis_default_suggestions(self):
        from pynerve._diagnostics_failure import FailureDiagnosis
        fd = FailureDiagnosis("msg")
        assert fd.suggestions == []

    def test_classify_shape_error(self):
        from pynerve._diagnostics_failure import _classify
        from pynerve.exceptions import ShapeError
        cat, sugg = _classify(ShapeError("mismatch"))
        assert cat == "shape_mismatch"

    def test_classify_convergence_error(self):
        from pynerve._diagnostics_failure import _classify
        from pynerve.exceptions import ConvergenceError
        cat, sugg = _classify(ConvergenceError("no converge"))
        assert cat == "no_convergence"

    def test_classify_device_error(self):
        from pynerve._diagnostics_failure import _classify
        from pynerve.exceptions import DeviceError
        cat, sugg = _classify(DeviceError("bad device"))
        assert cat == "device_error"

    def test_classify_numerical_error(self):
        from pynerve._diagnostics_failure import _classify
        from pynerve.exceptions import NumericalError
        cat, sugg = _classify(NumericalError("nan"))
        assert cat == "numerical_error"

    def test_classify_oom_error(self):
        from pynerve._diagnostics_failure import _classify
        from pynerve.exceptions import OutOfMemoryError
        cat, sugg = _classify(OutOfMemoryError("oom"))
        assert cat == "nerve_out_of_memory"

    def test_classify_gpu_memory_error(self):
        from pynerve._diagnostics_failure import _classify
        from pynerve.exceptions import GPUMemoryError
        cat, sugg = _classify(GPUMemoryError("gpu oom"))
        assert cat == "gpu_out_of_memory"

    def test_classify_backend_required(self):
        from pynerve._diagnostics_failure import _classify
        from pynerve.exceptions import BackendRequiredError
        cat, sugg = _classify(BackendRequiredError("no core", backend="test"))
        assert cat == "backend_missing"

    def test_classify_io_error(self):
        from pynerve._diagnostics_failure import _classify
        from pynerve.exceptions import NerveIOError
        cat, sugg = _classify(NerveIOError("io fail"))
        assert cat == "io_failure"

    def test_build_message(self):
        from pynerve._diagnostics_failure import _build_message
        msg = _build_message("op", ValueError("bad"), "test_cat", ["fix1"], None, None)
        assert "op" in msg
        assert "bad" in msg
        assert "fix1" in msg

    def test_build_message_with_data_info(self):
        from pynerve._diagnostics_failure import _build_message
        msg = _build_message("op", ValueError("bad"), "cat", ["s1"], {"shape": (10, 3)}, None)
        assert "shape" in msg
        assert "(10, 3)" in msg

    def test_build_message_with_context(self):
        from pynerve._diagnostics_failure import _build_message
        msg = _build_message("op", ValueError("bad"), "cat", ["s1"], None, {"engine": "ph5"})
        assert "engine" in msg
        assert "ph5" in msg


class TestSystemDiagnostics:
    """Covers _diagnostics_system.py — profile_memory, DebugMode, check_gpu_availability, system_info."""

    def test_profile_memory(self):
        from pynerve._diagnostics_system import profile_memory

        def my_func(x):
            return x * 2

        result, stats = profile_memory(my_func, 5)
        assert result == 10
        assert "memory_before_mb" in stats
        assert "memory_after_mb" in stats
        assert "memory_delta_mb" in stats

    def test_profile_memory_not_callable(self):
        from pynerve._diagnostics_system import profile_memory
        with pytest.raises(TypeError, match="callable"):
            profile_memory("not callable")

    def test_debug_mode_construct(self):
        from pynerve._diagnostics_system import DebugMode
        dm = DebugMode()
        assert dm.print_intermediate is False
        assert dm.stream is None

    def test_debug_mode_construct_with_stream(self):
        from pynerve._diagnostics_system import DebugMode
        stream = io.StringIO()
        dm = DebugMode(print_intermediate=True, stream=stream)
        assert dm.print_intermediate is True
        assert dm.stream is stream

    def test_debug_mode_invalid_print_intermediate(self):
        from pynerve._diagnostics_system import DebugMode
        with pytest.raises(TypeError, match="boolean"):
            DebugMode(print_intermediate="yes")

    def test_debug_mode_invalid_stream(self):
        from pynerve._diagnostics_system import DebugMode
        with pytest.raises(TypeError, match="write"):
            DebugMode(stream="not a stream")

    def test_debug_mode_context_manager(self):
        from pynerve._diagnostics_system import DebugMode
        stream = io.StringIO()
        with DebugMode(stream=stream) as dm:
            assert dm is not None
        assert "Debug mode enabled" in stream.getvalue()
        assert "Debug mode disabled" in stream.getvalue()

    def test_debug_mode_exception_handling(self):
        from pynerve._diagnostics_system import DebugMode
        stream = io.StringIO()
        dm = DebugMode(stream=stream)
        with pytest.raises(ValueError):
            with dm:
                raise ValueError("test error")
        assert "ValueError" in stream.getvalue()

    def test_debug_mode_repr(self):
        from pynerve._diagnostics_system import DebugMode
        dm = DebugMode(print_intermediate=True)
        assert "DebugMode" in repr(dm)
        assert "True" in repr(dm)

    def test_check_gpu_availability_no_gpu(self):
        from pynerve._diagnostics_system import check_gpu_availability
        info = check_gpu_availability()
        assert "cuda_available" in info
        assert "device_count" in info
        assert "devices" in info
        assert isinstance(info["cuda_available"], bool)

    def test_system_info(self):
        from pynerve._diagnostics_system import system_info
        info = system_info()
        assert "python_version" in info
        assert "platform" in info
        assert "cpu_count" in info
        assert "gpu_info" in info
        assert "pynerve_version" in info
