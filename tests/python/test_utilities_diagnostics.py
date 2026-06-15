"""Tests for diagnostics utility module."""

from __future__ import annotations

import time

import numpy as np
import pytest
from pynerve.diagnostics import (
    DebugMode,
    DiagnosticInfo,
    DiagnosticsCollector,
    FailureDiagnosis,
    check_data_quality,
    check_gpu_availability,
    diagnose_failure,
    profile_memory,
    system_info,
    verbose,
)
from pynerve.exceptions import BackendRequiredError

# diagnostics.py


class TestDiagnosticInfo:
    def test_creation(self):
        info = DiagnosticInfo(operation="test", duration=1.5)
        assert info.operation == "test"
        assert info.duration == 1.5
        assert info.memory_delta is None
        assert info.error is None

    def test_full_fields(self):
        info = DiagnosticInfo(
            operation="compute",
            duration=2.0,
            memory_delta=10.5,
            gpu_memory_delta=5.0,
            n_points=1000,
            n_simplices=500,
            backend="cpu",
        )
        assert info.n_points == 1000
        assert info.n_simplices == 500
        assert info.backend == "cpu"

    def test_error_field(self):
        info = DiagnosticInfo(operation="fail", duration=0.0, error="something went wrong")
        assert info.error == "something went wrong"

    def test_repr(self):
        info = DiagnosticInfo(operation="test", duration=0.1)
        r = repr(info)
        assert "OK" in r
        assert "test" in r

    def test_repr_with_error(self):
        info = DiagnosticInfo(operation="fail", duration=0.0, error="boom")
        r = repr(info)
        assert "ERROR" in r

    def test_negative_duration_raises(self):
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="duration"):
            DiagnosticInfo(operation="x", duration=-1.0)

    def test_empty_operation_raises(self):
        from pynerve.exceptions import ValidationError

        with pytest.raises(ValidationError, match="operation"):
            DiagnosticInfo(operation="", duration=0.0)


class TestDiagnosticsCollector:
    def test_empty_initial_state(self):
        dc = DiagnosticsCollector()
        assert len(dc.diagnostics) == 0

    def test_track_context_manager(self):
        dc = DiagnosticsCollector()
        with dc.track("sleep", n_points=100):
            time.sleep(0.01)
        assert len(dc.diagnostics) == 1
        info = dc.diagnostics[0]
        assert info.operation == "sleep"
        assert info.duration > 0
        assert info.n_points == 100

    def test_track_multiple_operations(self):
        dc = DiagnosticsCollector()
        with dc.track("op1"):
            time.sleep(0.01)
        with dc.track("op2"):
            time.sleep(0.01)
        assert len(dc.diagnostics) == 2

    def test_track_captures_error(self):
        dc = DiagnosticsCollector()
        with pytest.raises(ValueError, match="test_error"), dc.track("will_fail"):
            raise ValueError("test_error")
        assert len(dc.diagnostics) == 1
        assert dc.diagnostics[0].error == "test_error"
        assert dc.diagnostics[0].duration > 0

    def test_report_contains_operations(self):
        dc = DiagnosticsCollector()
        with dc.track("alpha"):
            time.sleep(0.005)
        with dc.track("beta", n_points=50):
            time.sleep(0.005)
        report = dc.report()
        assert "alpha" in report
        assert "beta" in report
        assert "Total time" in report

    def test_summary_structure(self):
        dc = DiagnosticsCollector()
        with dc.track("a"):
            time.sleep(0.005)
        with dc.track("b"):
            time.sleep(0.01)
        s = dc.summary()
        assert s["n_operations"] == 2
        assert s["n_errors"] == 0
        assert s["total_time"] > 0
        assert s["mean_time"] > 0
        assert s["max_time"] > 0
        assert s["error_rate"] == 0.0

    def test_summary_empty(self):
        dc = DiagnosticsCollector()
        assert dc.summary() == {}

    def test_summary_with_errors(self):
        dc = DiagnosticsCollector()
        with dc.track("ok"):
            time.sleep(0.005)
        with pytest.raises(RuntimeError, match="fail"), dc.track("fail"):
            raise RuntimeError("fail")
        s = dc.summary()
        assert s["n_errors"] == 1
        assert s["error_rate"] == 0.5

    def test_diagnostics_property_returns_copy(self):
        dc = DiagnosticsCollector()
        with dc.track("op"):
            time.sleep(0.005)
        diags = dc.diagnostics
        diags.clear()
        assert len(dc.diagnostics) == 1

    def test_repr(self):
        dc = DiagnosticsCollector()
        r = repr(dc)
        assert "DiagnosticsCollector" in r

    def test_track_yields_diagnostic_info(self):
        dc = DiagnosticsCollector()
        with dc.track("test") as info:
            assert isinstance(info, DiagnosticInfo)
            assert info.operation == "test"
            assert info.duration == 0.0  # not yet completed

    def test_unexpected_kwargs_warns(self):
        dc = DiagnosticsCollector()
        with pytest.warns(UserWarning, match="Unexpected"), dc.track("test", foo="bar"):
            pass


class TestVerbose:
    def test_context_manager_enables(self):
        with verbose(True, level="info"):
            pass

    def test_context_manager_disables(self):
        with verbose(False):
            pass

    def test_trace_level(self):
        with verbose(True, level="trace"):
            pass

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level"), verbose(True, level="invalid"):
            pass

    def test_non_bool_enabled_raises(self):
        with pytest.raises(TypeError, match="boolean"), verbose("yes"):  # type: ignore[arg-type]
            pass


class TestFailureDiagnosis:
    def test_is_string_subclass(self):
        fd = FailureDiagnosis("test message")
        assert isinstance(fd, str)
        assert fd == "test message"

    def test_has_structured_attributes(self):
        fd = FailureDiagnosis(
            "msg",
            cause_category="out_of_memory",
            suggestions=["reduce points"],
            data_info={"shape": (100, 3)},
            context={"step": 1},
        )
        assert fd.cause_category == "out_of_memory"
        assert fd.suggestions == ["reduce points"]
        assert fd.data_info == {"shape": (100, 3)}
        assert fd.context == {"step": 1}

    def test_default_values(self):
        fd = FailureDiagnosis("test")
        assert fd.cause_category == "unknown"
        assert fd.suggestions == []
        assert fd.data_info is None
        assert fd.context is None


class TestDiagnoseFailure:
    def test_categorizes_memory_error(self):
        result = diagnose_failure("alloc", MemoryError("oom"))
        assert result.cause_category == "out_of_memory"
        assert "alloc" in result

    def test_categorizes_value_error(self):
        result = diagnose_failure("validate", ValueError("bad input"))
        assert result.cause_category == "invalid_input"

    def test_categorizes_backend_error(self):
        result = diagnose_failure("setup", BackendRequiredError("no core"))
        assert result.cause_category == "backend_missing"

    def test_categorizes_unknown_error(self):
        result = diagnose_failure("op", Exception("generic"))
        assert result.cause_category == "unknown"

    def test_data_info_from_numpy_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = diagnose_failure("op", ValueError("bad"), data=arr)
        assert result.data_info is not None
        assert result.data_info["shape"] == (2, 2)
        assert "int" in result.data_info["dtype"]

    def test_non_exception_raises(self):
        with pytest.raises(TypeError, match="Exception"):
            diagnose_failure("op", "not_an_exception")  # type: ignore[arg-type]

    def test_context_not_mapping_raises(self):
        with pytest.raises(TypeError, match="mapping"):
            diagnose_failure("op", ValueError("x"), context=42)  # type: ignore[arg-type]


class TestCheckDataQuality:
    def test_valid_array_passes(self):
        data = np.random.randn(100, 3).astype(np.float32)
        result = check_data_quality(data)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_1d_array_fails(self):
        data = np.array([1, 2, 3])
        result = check_data_quality(data)
        assert result["valid"] is False
        assert any("2D" in e for e in result["errors"])

    def test_empty_array_fails(self):
        data = np.empty((0, 3))
        result = check_data_quality(data)
        assert result["valid"] is False

    def test_non_numeric_fails(self):
        data = np.array([["a", "b"], ["c", "d"]])
        result = check_data_quality(data)
        assert result["valid"] is False

    def test_nan_detected(self):
        data = np.array([[1.0, 2.0], [np.nan, 4.0]])
        result = check_data_quality(data)
        assert result["valid"] is False
        assert any("NaN" in e for e in result["errors"])

    def test_inf_warning(self):
        data = np.array([[1.0, 2.0], [np.inf, 4.0]])
        result = check_data_quality(data)
        assert any("Inf" in w for w in result["warnings"])

    def test_duplicate_points_warning(self):
        data = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        result = check_data_quality(data)
        assert any("duplicate" in w for w in result["warnings"])

    def test_zero_variance_warning(self):
        data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = check_data_quality(data)
        assert any("zero variance" in w for w in result["warnings"])

    def test_non_array_input(self):
        result = check_data_quality([1, 2, 3])
        assert result["valid"] is False
        assert any("numpy" in e for e in result["errors"])


class TestProfileMemory:
    def test_requires_psutil(self):
        psutil = pytest.importorskip("psutil")
        assert psutil is not None

    def test_profiles_function(self):
        pytest.importorskip("psutil")
        result, stats = profile_memory(lambda x: x * 2, 21)
        assert result == 42
        assert "memory_before_mb" in stats
        assert "memory_after_mb" in stats
        assert "memory_delta_mb" in stats
        assert isinstance(stats["memory_before_mb"], float)

    def test_non_callable_raises(self):
        pytest.importorskip("psutil")
        with pytest.raises(TypeError, match="callable"):
            profile_memory(42)  # type: ignore[arg-type]


class TestDebugMode:
    def test_context_manager_enter_exit(self):
        with DebugMode() as dm:
            assert isinstance(dm, DebugMode)

    def test_print_intermediate_option(self):
        with DebugMode(print_intermediate=True):
            pass

    def test_custom_stream(self):
        import io

        stream = io.StringIO()
        with DebugMode(stream=stream) as dm:
            dm._write("hello")
        assert "hello" in stream.getvalue()

    def test_stream_captures_exception(self):
        import io

        stream = io.StringIO()
        with pytest.raises(ValueError), DebugMode(stream=stream):
            raise ValueError("test")
        output = stream.getvalue()
        assert "ValueError" in output

    def test_non_bool_print_intermediate_raises(self):
        with pytest.raises(TypeError, match="boolean"):
            DebugMode(print_intermediate=42)  # type: ignore[arg-type]

    def test_invalid_stream_raises(self):
        with pytest.raises(TypeError, match="stream"):
            DebugMode(stream=42)  # type: ignore[arg-type]

    def test_repr(self):
        dm = DebugMode(print_intermediate=True)
        r = repr(dm)
        assert "DebugMode" in r


class TestCheckGPUAvailability:
    def test_returns_dict(self):
        result = check_gpu_availability()
        assert isinstance(result, dict)
        assert "cuda_available" in result
        assert "device_count" in result
        assert "devices" in result

    def test_device_count_is_int(self):
        result = check_gpu_availability()
        assert isinstance(result["device_count"], int)


class TestSystemInfo:
    def test_returns_dict(self):
        result = system_info()
        assert isinstance(result, dict)
        assert "python_version" in result
        assert "platform" in result
        assert "cpu_count" in result
        assert "gpu_info" in result

    def test_cpu_count_is_int(self):
        result = system_info()
        assert isinstance(result["cpu_count"], int)
