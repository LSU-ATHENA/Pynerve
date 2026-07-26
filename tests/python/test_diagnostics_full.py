"""Tests for diagnostics modules -- data quality, failure diagnosis, system profiling."""

from __future__ import annotations

import io
import os
import warnings

import numpy as np
import pytest

from pynerve._diagnostics_data import (
    DataQualityReport,
    _check_duplicates,
    _check_nan_inf,
    _check_size,
    _check_variance_and_range,
    _extract_array,
    _validate_shape_and_dtype,
    check_data_quality,
)
from pynerve._diagnostics_failure import (
    FailureDiagnosis,
    _build_message,
    _classify,
    diagnose_failure,
)
from pynerve._diagnostics_system import (
    DebugMode,
    check_gpu_availability,
    profile_memory,
    system_info,
)
from pynerve.exceptions import (
    BackendRequiredError,
    ConvergenceError,
    DeviceError,
    GPUMemoryError,
    NerveIOError,
    NumericalError,
    OutOfMemoryError,
    ShapeError,
)


# _diagnostics_data


class TestExtractArray:
    def test_numpy_returns_same(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert _extract_array(arr) is arr

    def test_torch_tensor(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _extract_array(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_empty_torch_returns_none(self):
        torch = pytest.importorskip("torch")
        t = torch.empty(0)
        assert _extract_array(t) is None

    def test_non_array_returns_none(self):
        assert _extract_array("not_an_array") is None
        assert _extract_array(42) is None
        assert _extract_array(None) is None

    def test_torch_cuda_detected(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([[1.0, 2.0]])
        result = _extract_array(t)
        assert isinstance(result, np.ndarray)


class TestValidateShapeAndDtype:
    def test_valid(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        assert _validate_shape_and_dtype(arr, report) is True
        assert report["valid"] is True

    def test_1d_array_fails(self):
        arr = np.array([1.0, 2.0, 3.0])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        assert _validate_shape_and_dtype(arr, report) is False
        assert report["valid"] is False
        assert len(report["errors"]) >= 1

    def test_empty_array_fails(self):
        arr = np.empty((0, 3))
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        assert _validate_shape_and_dtype(arr, report) is False
        assert report["valid"] is False

    def test_non_numeric_dtype_fails(self):
        arr = np.array([["a", "b"]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        assert _validate_shape_and_dtype(arr, report) is False
        assert report["valid"] is False


class TestCheckNanInf:
    def test_no_nan_inf(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_nan_inf(arr, report)
        assert report["valid"] is True
        assert len(report["errors"]) == 0

    def test_nan_detected(self):
        arr = np.array([[1.0, float("nan")], [3.0, 4.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_nan_inf(arr, report)
        assert report["valid"] is False
        assert any("NaN" in e for e in report["errors"])

    def test_inf_detected(self):
        arr = np.array([[1.0, float("inf")], [3.0, 4.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_nan_inf(arr, report)
        assert any("Inf" in w for w in report["warnings"])


class TestCheckDuplicates:
    def test_no_duplicates(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_duplicates(arr, report)
        assert len(report["warnings"]) == 0

    def test_duplicates_found(self):
        arr = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_duplicates(arr, report)
        assert any("duplicate" in w for w in report["warnings"])


class TestCheckVarianceAndRange:
    def test_no_variance_issues(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_variance_and_range(arr, report)
        assert len(report["warnings"]) == 0

    def test_zero_variance(self):
        arr = np.array([[1.0, 2.0], [1.0, 2.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_variance_and_range(arr, report)
        assert any("zero variance" in w for w in report["warnings"])

    def test_large_range(self):
        arr = np.array([[0.0, 0.0], [1e7, 0.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_variance_and_range(arr, report)
        assert any("Large coordinate" in w for w in report["warnings"])


class TestCheckSize:
    def test_small_dataset_no_warning(self):
        arr = np.array([[1.0, 2.0]])
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_size(arr, report)
        assert len(report["warnings"]) == 0

    def test_large_dataset_warning(self, monkeypatch):
        monkeypatch.setenv("NERVE_NPOINTS_WARN", "5")
        arr = np.random.randn(10, 3)
        report = DataQualityReport(valid=True, warnings=[], errors=[])
        _check_size(arr, report)
        assert any("Large dataset" in w for w in report["warnings"])


class TestCheckDataQuality:
    def test_valid_data(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        report = check_data_quality(arr)
        assert report["valid"] is True

    def test_nan_data(self):
        arr = np.array([[1.0, float("nan")]])
        report = check_data_quality(arr)
        assert report["valid"] is False

    def test_non_array_input(self):
        report = check_data_quality("not_an_array")
        assert report["valid"] is False

    def test_empty_torch_tensor(self):
        torch = pytest.importorskip("torch")
        t = torch.empty(0)
        report = check_data_quality(t)
        assert report["valid"] is False
        assert any("non-empty" in e for e in report["errors"])

    def test_torch_tensor(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        report = check_data_quality(t)
        assert report["valid"] is True

    def test_1d_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        report = check_data_quality(arr)
        assert report["valid"] is False

    def test_empty_numpy(self):
        arr = np.empty((0, 3))
        report = check_data_quality(arr)
        assert report["valid"] is False


# _diagnostics_failure


class TestFailureDiagnosis:
    def test_construction(self):
        diag = FailureDiagnosis("test message")
        assert str(diag) == "test message"
        assert diag.cause_category == "unknown"
        assert diag.suggestions == []
        assert diag.data_info is None

    def test_with_all_fields(self):
        diag = FailureDiagnosis(
            "error",
            cause_category="test",
            suggestions=["s1", "s2"],
            data_info={"shape": (10, 3)},
            context={"env": "test"},
        )
        assert diag.cause_category == "test"
        assert diag.suggestions == ["s1", "s2"]
        assert diag.data_info == {"shape": (10, 3)}
        assert diag.context == {"env": "test"}

    def test_string_operations(self):
        diag = FailureDiagnosis("hello world")
        assert "hello" in diag
        assert diag.startswith("hello")
        assert len(diag) == 11


class TestClassify:
    def test_memory_error(self):
        cat, suggestions = _classify(MemoryError("oom"))
        assert cat == "out_of_memory"
        assert len(suggestions) > 0

    def test_shape_error(self):
        cat, suggestions = _classify(ShapeError("bad"))
        assert cat == "shape_mismatch"

    def test_gpu_memory_error(self):
        cat, _ = _classify(GPUMemoryError("oom"))
        assert cat == "gpu_out_of_memory"

    def test_unknown_exception(self):
        cat, suggestions = _classify(Exception("unknown"))
        assert cat == "unknown"
        assert len(suggestions) == 3

    def test_value_error(self):
        cat, _ = _classify(ValueError("bad value"))
        assert cat == "invalid_input"


class TestBuildMessage:
    def test_basic_message(self):
        msg = _build_message("compute", ValueError("bad"), "invalid", ["fix1"], None, None)
        assert "compute" in msg
        assert "bad" in msg
        assert "fix1" in msg

    def test_with_data_info(self):
        msg = _build_message("op", Exception("e"), "cat", ["s1"], {"shape": (5,)}, None)
        assert "shape" in msg
        assert "5" in msg

    def test_with_context(self):
        msg = _build_message("op", Exception("e"), "cat", ["s1"], None, {"key": "val"})
        assert "key" in msg
        assert "val" in msg


class TestDiagnoseFailure:
    def test_basic_diagnosis(self):
        diag = diagnose_failure("compute", ValueError("bad value"))
        assert isinstance(diag, FailureDiagnosis)
        assert diag.cause_category == "invalid_input"

    def test_with_data(self):
        data = np.array([[1.0, 2.0]])
        diag = diagnose_failure("compute", ShapeError("bad"), data=data)
        assert diag.data_info is not None
        assert "shape" in diag.data_info

    def test_with_context(self):
        diag = diagnose_failure("compute", RuntimeError("fail"), context={"gpu": True})
        assert diag.context is not None
        assert diag.context["gpu"] is True

    def test_non_exception_raises(self):
        with pytest.raises(TypeError, match="exception"):
            diagnose_failure("op", "not_an_exception")  # type: ignore[arg-type]

    def test_invalid_context_raises(self):
        with pytest.raises(TypeError, match="mapping"):
            diagnose_failure("op", Exception("e"), context=42)  # type: ignore[arg-type]

    def test_empty_operation_raises(self):
        with pytest.raises(Exception):
            diagnose_failure("", Exception("e"))

    def test_backend_required(self):
        diag = diagnose_failure("compute", BackendRequiredError("missing"))
        assert diag.cause_category == "backend_missing"

    def test_convergence(self):
        diag = diagnose_failure("compute", ConvergenceError("no conv"))
        assert diag.cause_category == "no_convergence"

    def test_device_error(self):
        diag = diagnose_failure("compute", DeviceError("device"))
        assert diag.cause_category == "device_error"

    def test_io_error(self):
        diag = diagnose_failure("compute", NerveIOError("io"))
        assert diag.cause_category == "io_failure"

    def test_numerical_error(self):
        diag = diagnose_failure("compute", NumericalError("num"))
        assert diag.cause_category == "numerical_error"

    def test_out_of_memory(self):
        diag = diagnose_failure("compute", OutOfMemoryError("oom"))
        assert diag.cause_category == "nerve_out_of_memory"


# _diagnostics_system


class TestProfileMemory:
    def test_basic(self):
        def simple_func():
            return 42

        result, stats = profile_memory(simple_func)
        assert result == 42
        assert "memory_before_mb" in stats
        assert "memory_after_mb" in stats
        assert "memory_delta_mb" in stats
        assert isinstance(stats["memory_delta_mb"], float)

    def test_with_args(self):
        def add(a, b):
            return a + b

        result, stats = profile_memory(add, 1, b=2)
        assert result == 3

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            profile_memory("not_callable")  # type: ignore[arg-type]


class TestDebugMode:
    def test_context_manager(self):
        with DebugMode() as dm:
            assert dm.print_intermediate is False
            assert dm.stream is None

    def test_with_stream(self):
        stream = io.StringIO()
        with DebugMode(print_intermediate=True, stream=stream) as dm:
            assert dm.print_intermediate is True
        output = stream.getvalue()
        assert "Debug mode enabled" in output
        assert "Debug mode disabled" in output

    def test_normal_exit(self):
        stream = io.StringIO()
        with DebugMode(stream=stream):
            pass
        output = stream.getvalue()
        assert "Debug mode disabled" in output

    def test_exception_traceback(self):
        stream = io.StringIO()
        try:
            with DebugMode(stream=stream):
                raise ValueError("test error")
        except ValueError:
            pass
        output = stream.getvalue()
        assert "Exception caught: ValueError" in output
        assert "Debug mode disabled" in output

    def test_repr(self):
        dm = DebugMode(print_intermediate=True)
        assert "DebugMode" in repr(dm)

    def test_invalid_print_intermediate(self):
        with pytest.raises(TypeError, match="boolean"):
            DebugMode(print_intermediate="yes")  # type: ignore[arg-type]

    def test_invalid_stream(self):
        with pytest.raises(TypeError, match="write"):
            DebugMode(stream=42)  # type: ignore[arg-type]

    def test_file_stream_writes(self):
        f = io.StringIO()
        dm = DebugMode(stream=f)
        with dm:
            pass
        assert "Debug mode enabled" in f.getvalue()


class TestCheckGpuAvailability:
    def test_returns_dict(self):
        result = check_gpu_availability()
        assert isinstance(result, dict)
        assert "cuda_available" in result
        assert "cuda_version" in result
        assert "device_count" in result
        assert "devices" in result

    def test_no_cupy_returns_false(self):
        result = check_gpu_availability()
        assert isinstance(result["cuda_available"], bool)


class TestSystemInfo:
    def test_returns_dict(self):
        info = system_info()
        assert "python_version" in info
        assert "platform" in info
        assert "processor" in info
        assert "cpu_count" in info
        assert "gpu_info" in info

    def test_has_pynerve_version(self):
        info = system_info()
        assert "pynerve_version" in info
