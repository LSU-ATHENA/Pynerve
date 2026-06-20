from __future__ import annotations

import os
from unittest.mock import patch

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
from pynerve._diagnostics_system import (
    DebugMode,
    check_gpu_availability,
    profile_memory,
    system_info,
)
from pynerve._error_codes import (
    E00_IO_TIMEOUT,
    E30_DET_MISMATCH,
    E50_PH_ABORT,
    E53_PH4_BUDGET_EXCEEDED,
    E54_PH4_INVALID_INPUT,
    E60_NUMA_BIND_FAIL,
    E85_MATRIX_STRUCTURE,
    E87_INVALID_BETTI_NUMBERS,
    E88_INVALID_SIMPLICES,
    E91_SHAPE_ERROR,
    ErrorCategory,
)
from pynerve.exceptions._base import NerveError
from pynerve.exceptions._cpp import (
    BettiError,
    BudgetExceededError,
    DeterminismError,
    DimensionError,
    InvalidArgumentError,
    InvalidSimplexError,
    MatrixStructureError,
    NerveIOError,
    NUMAError,
    PersistenceError,
    ShapeMismatchError,
    TypeMismatchError,
)
from pynerve.exceptions._validation import (
    BackendRequiredError,
    DeviceError,
    DtypeError,
    ShapeError,
    ValidationError,
)


class TestExtractArray:
    def test_numpy_array_passthrough(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _extract_array(arr)
        assert result is arr

    def test_non_array_returns_none(self):
        assert _extract_array([1, 2, 3]) is None
        assert _extract_array("hello") is None
        assert _extract_array(42) is None

    def test_torch_tensor_returns_numpy(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _extract_array(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_torch_tensor_empty_numel_returns_none(self):
        torch = pytest.importorskip("torch")
        t = torch.empty(0, 3)
        result = _extract_array(t)
        assert result is None

    def test_torch_cuda_tensor_returns_cpu_numpy(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([[1.0, 2.0]])
        with patch.object(type(t), "is_cuda", True), patch.object(type(t), "cpu", return_value=t):
            result = _extract_array(t)
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))

    def test_torch_tensor_numpy_not_ndarray_returns_none(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([[1.0, 2.0]])
        with patch.object(type(t), "numpy", return_value=[1, 2]):
            result = _extract_array(t)
            assert result is None


class TestValidateShapeAndDtype:
    def test_valid_array_passes(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        assert _validate_shape_and_dtype(arr, report) is True
        assert report["valid"] is True

    def test_1d_array_fails(self):
        arr = np.array([1.0, 2.0])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        assert _validate_shape_and_dtype(arr, report) is False
        assert report["valid"] is False
        assert any("2D" in e for e in report["errors"])

    def test_3d_array_fails(self):
        arr = np.zeros((2, 2, 2))
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        assert _validate_shape_and_dtype(arr, report) is False
        assert report["valid"] is False

    def test_empty_array_fails(self):
        arr = np.empty((0, 3))
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        assert _validate_shape_and_dtype(arr, report) is False
        assert report["valid"] is False
        assert any("non-empty" in e for e in report["errors"])

    def test_non_numeric_dtype_fails(self):
        arr = np.array([["a", "b"], ["c", "d"]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        assert _validate_shape_and_dtype(arr, report) is False
        assert report["valid"] is False
        assert any("numeric" in e for e in report["errors"])


class TestCheckNanInf:
    def test_nan_detected(self):
        arr = np.array([[1.0, 2.0], [np.nan, 4.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_nan_inf(arr, report)
        assert report["valid"] is False
        assert any("NaN" in e for e in report["errors"])

    def test_inf_warns(self):
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_nan_inf(arr, report)
        assert report["valid"] is True
        assert any("Inf" in w for w in report["warnings"])

    def test_nan_and_inf_both(self):
        arr = np.array([[np.nan, np.inf]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_nan_inf(arr, report)
        assert report["valid"] is False
        assert any("NaN" in e for e in report["errors"])
        assert any("Inf" in w for w in report["warnings"])

    def test_clean_array_no_issues(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_nan_inf(arr, report)
        assert report["valid"] is True
        assert report["errors"] == []
        assert report["warnings"] == []


class TestCheckDuplicates:
    def test_no_duplicates(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_duplicates(arr, report)
        assert report["warnings"] == []

    def test_duplicates_detected(self):
        arr = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_duplicates(arr, report)
        assert any("duplicate" in w for w in report["warnings"])
        assert "1 duplicate" in str(report["warnings"])

    def test_all_duplicates(self):
        arr = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_duplicates(arr, report)
        assert any("2 duplicate" in w for w in report["warnings"])


class TestCheckVarianceAndRange:
    def test_zero_variance_warns(self):
        arr = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_variance_and_range(arr, report)
        assert any("zero variance" in w for w in report["warnings"])

    def test_large_range_warns(self):
        arr = np.array([[0.0, 0.0], [1e7, 0.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_variance_and_range(arr, report)
        assert any("Large coordinate range" in w for w in report["warnings"])

    def test_large_range_env_threshold(self):
        arr = np.array([[0.0, 0.0], [1000.0, 0.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        with patch.dict(os.environ, {"NERVE_COORD_RANGE_WARN": "500"}):
            _check_variance_and_range(arr, report)
        assert any("Large coordinate range" in w for w in report["warnings"])

    def test_range_below_env_threshold_no_warn(self):
        arr = np.array([[0.0, 0.0], [100.0, 0.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        with patch.dict(os.environ, {"NERVE_COORD_RANGE_WARN": "1e6"}):
            _check_variance_and_range(arr, report)
        assert not any("Large coordinate range" in w for w in report["warnings"])

    def test_all_columns_vary_no_warn(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_variance_and_range(arr, report)
        assert not any("zero variance" in w for w in report["warnings"])


class TestCheckSize:
    def test_small_dataset_no_warn(self):
        arr = np.zeros((10, 3))
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_size(arr, report)
        assert report["warnings"] == []

    def test_large_dataset_warns(self):
        arr = np.zeros((60000, 3))
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_size(arr, report)
        assert any("Large dataset" in w for w in report["warnings"])

    def test_large_dataset_env_threshold(self):
        arr = np.zeros((100, 3))
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        with patch.dict(os.environ, {"NERVE_NPOINTS_WARN": "50"}):
            _check_size(arr, report)
        assert any("Large dataset" in w for w in report["warnings"])

    def test_1d_array_zero_points(self):
        arr = np.array([1.0, 2.0])
        report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}
        _check_size(arr, report)
        assert report["warnings"] == []


class TestCheckDataQualityTorch:
    def test_torch_tensor_accepted(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = check_data_quality(t)
        assert result["valid"] is True

    def test_torch_empty_tensor_fails(self):
        torch = pytest.importorskip("torch")
        t = torch.empty(0, 3)
        result = check_data_quality(t)
        assert result["valid"] is False

    def test_unsupported_object_without_attributes(self):
        result = check_data_quality(object())
        assert result["valid"] is False
        assert any("numpy array" in e for e in result["errors"])

    def test_env_var_large_range_warns(self):
        arr = np.array([[0.0, 0.0], [2000.0, 0.0]])
        with patch.dict(os.environ, {"NERVE_COORD_RANGE_WARN": "1000"}):
            result = check_data_quality(arr)
        assert any("Large coordinate range" in w for w in result["warnings"])

    def test_env_var_large_dataset_warns(self):
        arr = np.zeros((200, 3))
        with patch.dict(os.environ, {"NERVE_NPOINTS_WARN": "100"}):
            result = check_data_quality(arr)
        assert any("Large dataset" in w for w in result["warnings"])

    def test_integer_array_passes(self):
        arr = np.array([[1, 2], [3, 4]])
        result = check_data_quality(arr)
        assert result["valid"] is True


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

    def test_non_callable_raises(self):
        pytest.importorskip("psutil")
        with pytest.raises(TypeError, match="callable"):
            profile_memory(42)

    def test_import_error_when_psutil_missing(self):
        with (
            patch.dict("sys.modules", {"psutil": None}),
            patch("importlib.import_module", side_effect=ImportError("no psutil")),
            pytest.raises(ImportError, match="psutil is required"),
        ):
            profile_memory(lambda: None)

    def test_kwargs_forwarded(self):
        pytest.importorskip("psutil")

        def add(a, b=0):
            return a + b

        result, _stats = profile_memory(add, 1, b=2)
        assert result == 3


class TestDebugMode:
    def test_enter_exit(self):
        dm = DebugMode()
        with dm as ctx:
            assert ctx is dm

    def test_print_intermediate_true(self):
        with DebugMode(print_intermediate=True):
            pass

    def test_print_intermediate_false(self):
        with DebugMode(print_intermediate=False):
            pass

    def test_stream_none_no_write(self):
        with DebugMode(stream=None) as dm:
            dm._write("invisible")

    def test_stream_stringio(self):
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
        assert "ValueError" in stream.getvalue()
        assert "Exception caught" in stream.getvalue()

    def test_stream_captures_exception_no_stream(self):
        with pytest.raises(RuntimeError), DebugMode(stream=None) as dm:
            raise RuntimeError("silent")
        assert isinstance(dm, DebugMode)

    def test_repr(self):
        dm = DebugMode(print_intermediate=True)
        r = repr(dm)
        assert "DebugMode" in r
        assert "True" in r

    def test_repr_no_print_intermediate(self):
        dm = DebugMode(print_intermediate=False)
        r = repr(dm)
        assert "False" in r

    def test_repr_stream_none(self):
        dm = DebugMode(stream=None)
        r = repr(dm)
        assert "None" in r

    def test_repr_stream_present(self):
        import io

        dm = DebugMode(stream=io.StringIO())
        r = repr(dm)
        assert "StringIO" in r or "StringI" in r

    def test_non_bool_raises(self):
        with pytest.raises(TypeError, match="boolean"):
            DebugMode(print_intermediate=42)

    def test_invalid_stream_raises(self):
        with pytest.raises(TypeError, match="stream"):
            DebugMode(stream=42)


class TestCheckGPUAvailability:
    def test_returns_dict(self):
        result = check_gpu_availability()
        assert isinstance(result, dict)
        assert "cuda_available" in result
        assert "device_count" in result
        assert "devices" in result
        assert "cuda_version" in result

    def test_defaults_when_no_cupy(self):
        with (
            patch.dict("sys.modules", {"cupy": None}),
            patch("importlib.import_module", side_effect=ImportError("no cupy")),
        ):
            result = check_gpu_availability()
            assert result["cuda_available"] is False
            assert result["cuda_version"] is None
            assert result["device_count"] == 0
            assert result["devices"] == []

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
        assert "processor" in result

    def test_cpu_count_is_int_or_none(self):
        result = system_info()
        assert result["cpu_count"] is None or isinstance(result["cpu_count"], int)

    def test_pynerve_version_when_importable(self):
        result = system_info()
        assert "pynerve_version" in result
        assert isinstance(result["pynerve_version"], str)

    def test_pynerve_version_when_not_importable(self):
        with patch("builtins.__import__", side_effect=ImportError):
            result = system_info()
            assert result["gpu_info"] is not None


class TestValidationError:
    def test_simple_message(self):
        e = ValidationError("bad input")
        assert "bad input" in str(e)
        assert e.parameter is None
        assert e.expected is None
        assert e.actual is None

    def test_with_parameter(self):
        e = ValidationError("bad input", parameter="x")
        assert e.parameter == "x"

    def test_with_all_fields(self):
        e = ValidationError("bad input", parameter="x", expected="int", actual="str")
        assert e.parameter == "x"
        assert e.expected == "int"
        assert e.actual == "str"

    def test_empty_parameter_raises(self):
        with pytest.raises(TypeError, match="parameter must be a non-empty string"):
            ValidationError("msg", parameter="")

    def test_non_string_parameter_raises(self):
        with pytest.raises(TypeError, match="parameter must be a non-empty string"):
            ValidationError("msg", parameter=42)

    def test_empty_expected_raises(self):
        with pytest.raises(TypeError, match="expected must be a non-empty string"):
            ValidationError("msg", expected="")

    def test_non_string_expected_raises(self):
        with pytest.raises(TypeError, match="expected must be a non-empty string"):
            ValidationError("msg", expected=42)

    def test_empty_actual_raises(self):
        with pytest.raises(TypeError, match="actual must be a non-empty string"):
            ValidationError("msg", actual="")

    def test_non_string_actual_raises(self):
        with pytest.raises(TypeError, match="actual must be a non-empty string"):
            ValidationError("msg", actual=42)

    def test_repr_simple(self):
        e = ValidationError("bad input")
        r = repr(e)
        assert "ValidationError" in r
        assert "category=" in r
        assert "code=" in r

    def test_repr_with_parameter(self):
        e = ValidationError("bad input", parameter="x")
        r = repr(e)
        assert "parameter='x'" in r

    def test_repr_with_expected_and_actual(self):
        e = ValidationError("bad input", expected="int", actual="str")
        r = repr(e)
        assert "expected='int'" in r
        assert "actual='str'" in r

    def test_repr_with_all_fields(self):
        e = ValidationError("bad input", parameter="x", expected="int", actual="str")
        r = repr(e)
        assert "parameter='x'" in r
        assert "expected='int'" in r
        assert "actual='str'" in r

    def test_is_nerve_error(self):
        e = ValidationError("bad input")
        assert isinstance(e, NerveError)

    def test_error_code(self):
        e = ValidationError("bad input")
        assert e.error_code == 0x00000A00

    def test_error_category(self):
        e = ValidationError("bad input")
        assert e.error_category == ErrorCategory.OPERATIONAL

    def test_str_includes_category_and_code(self):
        e = ValidationError("bad input")
        s = str(e)
        assert "bad input" in s
        assert "operational" in s.lower()
        assert "0x" in s


class TestShapeError:
    def test_simple(self):
        e = ShapeError("bad shape")
        assert "bad shape" in str(e)
        assert e.expected_shape is None
        assert e.actual_shape is None
        assert e.expected_ndim is None
        assert e.actual_ndim is None

    def test_with_shapes(self):
        e = ShapeError("bad shape", expected_shape=(10, 3), actual_shape=(5,))
        assert e.expected_shape == (10, 3)
        assert e.actual_shape == (5,)

    def test_with_ndims(self):
        e = ShapeError("bad shape", expected_ndim=2, actual_ndim=1)
        assert e.expected_ndim == 2
        assert e.actual_ndim == 1

    def test_with_all_fields(self):
        e = ShapeError(
            "bad shape",
            parameter="data",
            expected_shape=(100, 3),
            actual_shape=(100,),
            expected_ndim=2,
            actual_ndim=1,
        )
        assert e.parameter == "data"
        assert e.expected_shape == (100, 3)
        assert e.actual_shape == (100,)
        assert e.expected_ndim == 2
        assert e.actual_ndim == 1

    def test_expected_shape_none_valid(self):
        e = ShapeError("bad shape", expected_shape=None)
        assert e.expected_shape is None

    def test_actual_shape_none_valid(self):
        e = ShapeError("bad shape", actual_shape=None)
        assert e.actual_shape is None

    def test_expected_ndim_none_valid(self):
        e = ShapeError("bad shape", expected_ndim=None)
        assert e.expected_ndim is None

    def test_actual_ndim_none_valid(self):
        e = ShapeError("bad shape", actual_ndim=None)
        assert e.actual_ndim is None

    def test_negative_ndim_raises(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            ShapeError("bad shape", expected_ndim=-1)

    def test_repr_with_shapes(self):
        e = ShapeError("bad shape", expected_shape=(10, 3), actual_shape=(5,))
        r = repr(e)
        assert "expected_shape=(10, 3)" in r
        assert "actual_shape=(5,)" in r

    def test_repr_with_ndims(self):
        e = ShapeError("bad shape", expected_ndim=2, actual_ndim=1)
        r = repr(e)
        assert "expected_ndim=2" in r
        assert "actual_ndim=1" in r

    def test_repr_no_extra_parts(self):
        e = ShapeError("bad shape")
        r = repr(e)
        assert "ShapeError" in r
        assert "category=" in r

    def test_is_validation_error(self):
        e = ShapeError("bad shape")
        assert isinstance(e, ValidationError)

    def test_error_code(self):
        e = ShapeError("bad shape")
        assert e.error_code == E91_SHAPE_ERROR


class TestDtypeError:
    def test_simple(self):
        e = DtypeError("bad dtype")
        assert "bad dtype" in str(e)
        assert e.expected_dtypes == []
        assert e.actual_dtype is None

    def test_with_expected_dtypes(self):
        e = DtypeError("bad dtype", expected_dtypes=["float32", "float64"])
        assert e.expected_dtypes == ["float32", "float64"]

    def test_with_actual_dtype(self):
        e = DtypeError("bad dtype", actual_dtype="int32")
        assert e.actual_dtype == "int32"

    def test_expected_dtypes_none(self):
        e = DtypeError("bad dtype", expected_dtypes=None)
        assert e.expected_dtypes == []

    def test_expected_dtypes_empty_list(self):
        e = DtypeError("bad dtype", expected_dtypes=[])
        assert e.expected_dtypes == []

    def test_expected_dtypes_invalid(self):
        with pytest.raises(ValidationError, match="must contain non-empty strings"):
            DtypeError("bad dtype", expected_dtypes=[1, 2])

    def test_repr_with_dtypes(self):
        e = DtypeError("bad dtype", expected_dtypes=["float32"], actual_dtype="int32")
        r = repr(e)
        assert "expected_dtypes=['float32']" in r
        assert "actual_dtype='int32'" in r

    def test_repr_no_dtypes(self):
        e = DtypeError("bad dtype")
        r = repr(e)
        assert "DtypeError" in r
        assert "category=" in r

    def test_repr_empty_expected_dtypes(self):
        e = DtypeError("bad dtype", expected_dtypes=[])
        r = repr(e)
        assert "expected_dtypes" not in r

    def test_is_validation_error(self):
        e = DtypeError("bad dtype")
        assert isinstance(e, ValidationError)

    def test_error_code(self):
        e = DtypeError("bad dtype")
        assert e.error_code == 0x00000A02


class TestDeviceError:
    def test_simple(self):
        e = DeviceError("no device")
        assert e.requested_device is None
        assert e.available_devices == []

    def test_with_requested_device(self):
        e = DeviceError("no device", requested_device="cuda:0")
        assert e.requested_device == "cuda:0"

    def test_with_available_devices(self):
        e = DeviceError("no device", available_devices=["cpu", "cuda:0"])
        assert e.available_devices == ["cpu", "cuda:0"]

    def test_available_devices_none(self):
        e = DeviceError("no device", available_devices=None)
        assert e.available_devices == []

    def test_empty_requested_device_raises(self):
        with pytest.raises(TypeError, match="requested_device must be a non-empty string"):
            DeviceError("msg", requested_device="")

    def test_non_string_requested_device_raises(self):
        with pytest.raises(TypeError, match="requested_device must be a non-empty string"):
            DeviceError("msg", requested_device=42)

    def test_invalid_available_devices_raises(self):
        with pytest.raises(ValidationError, match="must contain non-empty strings"):
            DeviceError("msg", available_devices=[1, 2])

    def test_repr_with_devices(self):
        e = DeviceError("no device", requested_device="cuda:0", available_devices=["cpu"])
        r = repr(e)
        assert "requested_device='cuda:0'" in r
        assert "available_devices=['cpu']" in r

    def test_repr_no_devices(self):
        e = DeviceError("no device")
        r = repr(e)
        assert "DeviceError" in r
        assert "category=" in r

    def test_is_validation_error(self):
        e = DeviceError("no device")
        assert isinstance(e, ValidationError)

    def test_error_code(self):
        e = DeviceError("no device")
        assert e.error_code == 0x00000A03


class TestBackendRequiredError:
    def test_simple(self):
        e = BackendRequiredError("no backend")
        assert e.backend is None
        assert e.installation_hint is None

    def test_with_backend(self):
        e = BackendRequiredError("no backend", backend="gpu")
        assert e.backend == "gpu"

    def test_with_installation_hint(self):
        e = BackendRequiredError("no backend", installation_hint="pip install pynerve[cuda]")
        assert e.installation_hint == "pip install pynerve[cuda]"

    def test_with_both(self):
        e = BackendRequiredError("no backend", backend="gpu", installation_hint="pip install ...")
        assert e.backend == "gpu"
        assert e.installation_hint == "pip install ..."

    def test_empty_backend_raises(self):
        with pytest.raises(ValidationError, match="backend must be a non-empty string"):
            BackendRequiredError("msg", backend="")

    def test_non_string_backend_raises(self):
        with pytest.raises(ValidationError, match="backend must be a non-empty string"):
            BackendRequiredError("msg", backend=42)

    def test_empty_installation_hint_raises(self):
        with pytest.raises(ValidationError, match="installation_hint must be a non-empty string"):
            BackendRequiredError("msg", installation_hint="")

    def test_non_string_installation_hint_raises(self):
        with pytest.raises(ValidationError, match="installation_hint must be a non-empty string"):
            BackendRequiredError("msg", installation_hint=42)

    def test_extra_kwargs_passed_to_validation_error(self):
        e = BackendRequiredError("no backend", parameter="x", expected="str")
        assert e.parameter == "x"
        assert e.expected == "str"

    def test_repr_with_backend(self):
        e = BackendRequiredError("no backend", backend="gpu")
        r = repr(e)
        assert "backend='gpu'" in r

    def test_repr_with_installation_hint(self):
        e = BackendRequiredError("no backend", installation_hint="pip install x")
        r = repr(e)
        assert "installation_hint=" in r

    def test_repr_empty(self):
        e = BackendRequiredError("no backend")
        r = repr(e)
        assert "BackendRequiredError" in r
        assert "category=" in r

    def test_is_validation_error(self):
        e = BackendRequiredError("no backend")
        assert isinstance(e, ValidationError)

    def test_error_code(self):
        e = BackendRequiredError("no backend")
        assert e.error_code == 0x00000A04


class TestPersistenceError:
    def test_simple(self):
        e = PersistenceError("ph failed")
        assert "ph failed" in str(e)
        assert e.backend is None
        assert e.operation is None

    def test_with_backend(self):
        e = PersistenceError("ph failed", backend="cpu")
        assert e.backend == "cpu"

    def test_with_operation(self):
        e = PersistenceError("ph failed", operation="reduce")
        assert e.operation == "reduce"

    def test_with_both(self):
        e = PersistenceError("ph failed", backend="gpu", operation="compute")
        assert e.backend == "gpu"
        assert e.operation == "compute"

    def test_empty_backend_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            PersistenceError("msg", backend="")

    def test_non_string_backend_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            PersistenceError("msg", backend=42)

    def test_empty_operation_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            PersistenceError("msg", operation="")

    def test_non_string_operation_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            PersistenceError("msg", operation=42)

    def test_extra_kwargs(self):
        e = PersistenceError("ph failed", details={"info": "x"})
        assert e.details == {"info": "x"}

    def test_repr_with_backend(self):
        e = PersistenceError("ph failed", backend="gpu")
        r = repr(e)
        assert "backend='gpu'" in r

    def test_repr_with_operation(self):
        e = PersistenceError("ph failed", operation="compute")
        r = repr(e)
        assert "operation='compute'" in r

    def test_repr_with_both(self):
        e = PersistenceError("ph failed", backend="gpu", operation="compute")
        r = repr(e)
        assert "backend='gpu'" in r
        assert "operation='compute'" in r

    def test_repr_no_extras(self):
        e = PersistenceError("ph failed")
        r = repr(e)
        assert "PersistenceError" in r
        assert "category=" in r
        assert "code=" in r

    def test_is_nerve_error(self):
        e = PersistenceError("ph failed")
        assert isinstance(e, NerveError)

    def test_error_code(self):
        e = PersistenceError("ph failed")
        assert e.error_code == E50_PH_ABORT

    def test_error_category(self):
        e = PersistenceError("ph failed")
        assert e.error_category == ErrorCategory.ALGORITHMIC


class TestShapeMismatchError:
    def test_instantiation(self):
        e = ShapeMismatchError("shape mismatch")
        assert isinstance(e, NerveError)
        assert e.error_code == E91_SHAPE_ERROR
        assert e.error_category == ErrorCategory.ALGORITHMIC

    def test_with_details(self):
        e = ShapeMismatchError("shape mismatch", details={"expected": (10, 3), "actual": (5,)})
        assert e.details["expected"] == (10, 3)

    def test_str_includes_code(self):
        e = ShapeMismatchError("shape mismatch")
        assert "0x00000A01" in str(e)

    def test_repr_includes_category(self):
        e = ShapeMismatchError("shape mismatch")
        assert "ShapeMismatchError" in repr(e)
        assert "algorithm" in repr(e).lower()


class TestDimensionError:
    def test_instantiation(self):
        e = DimensionError("bad dimension")
        assert isinstance(e, NerveError)
        assert e.error_code == E91_SHAPE_ERROR
        assert e.error_category == ErrorCategory.ALGORITHMIC


class TestTypeMismatchError:
    def test_instantiation(self):
        e = TypeMismatchError("type mismatch")
        assert isinstance(e, NerveError)
        assert e.error_code == E54_PH4_INVALID_INPUT
        assert e.error_category == ErrorCategory.PH4_RESEARCH


class TestInvalidSimplexError:
    def test_instantiation(self):
        e = InvalidSimplexError("bad simplex")
        assert isinstance(e, NerveError)
        assert e.error_code == E88_INVALID_SIMPLICES
        assert e.error_category == ErrorCategory.ALGORITHMIC


class TestMatrixStructureError:
    def test_instantiation(self):
        e = MatrixStructureError("bad matrix")
        assert isinstance(e, NerveError)
        assert e.error_code == E85_MATRIX_STRUCTURE
        assert e.error_category == ErrorCategory.ALGORITHMIC


class TestInvalidArgumentError:
    def test_simple(self):
        e = InvalidArgumentError("bad arg")
        assert "bad arg" in str(e)
        assert e.parameter is None
        assert e.expected is None
        assert e.actual is None

    def test_with_parameter(self):
        e = InvalidArgumentError("bad arg", parameter="x")
        assert e.parameter == "x"

    def test_with_expected(self):
        e = InvalidArgumentError("bad arg", expected="int")
        assert e.expected == "int"

    def test_with_actual(self):
        e = InvalidArgumentError("bad arg", actual="str")
        assert e.actual == "str"

    def test_with_all_fields(self):
        e = InvalidArgumentError("bad arg", parameter="x", expected="int", actual="str")
        assert e.parameter == "x"
        assert e.expected == "int"
        assert e.actual == "str"

    def test_extra_kwargs(self):
        e = InvalidArgumentError("bad arg", details={"k": "v"})
        assert e.details == {"k": "v"}

    def test_repr_simple(self):
        e = InvalidArgumentError("bad arg")
        r = repr(e)
        assert "InvalidArgumentError" in r

    def test_repr_with_parameter(self):
        e = InvalidArgumentError("bad arg", parameter="x")
        r = repr(e)
        assert "parameter='x'" in r

    def test_repr_with_all_fields(self):
        e = InvalidArgumentError("bad arg", parameter="x", expected="int", actual="str")
        r = repr(e)
        assert "parameter='x'" in r
        assert "expected='int'" in r
        assert "actual='str'" in r

    def test_is_nerve_error(self):
        e = InvalidArgumentError("bad arg")
        assert isinstance(e, NerveError)

    def test_error_code(self):
        e = InvalidArgumentError("bad arg")
        assert e.error_code == E54_PH4_INVALID_INPUT

    def test_error_category(self):
        e = InvalidArgumentError("bad arg")
        assert e.error_category == ErrorCategory.PH4_RESEARCH


class TestBudgetExceededError:
    def test_instantiation(self):
        e = BudgetExceededError("budget exceeded")
        assert isinstance(e, NerveError)
        assert e.error_code == E53_PH4_BUDGET_EXCEEDED
        assert e.error_category == ErrorCategory.PH4_RESEARCH


class TestNerveIOError:
    def test_instantiation(self):
        e = NerveIOError("io error")
        assert isinstance(e, NerveError)
        assert e.error_code == E00_IO_TIMEOUT
        assert e.error_category == ErrorCategory.IO_INFRA


class TestDeterminismError:
    def test_instantiation(self):
        e = DeterminismError("nondeterministic")
        assert isinstance(e, NerveError)
        assert e.error_code == E30_DET_MISMATCH
        assert e.error_category == ErrorCategory.DETERMINISM


class TestNUMAError:
    def test_instantiation(self):
        e = NUMAError("numa bind failed")
        assert isinstance(e, NerveError)
        assert e.error_code == E60_NUMA_BIND_FAIL
        assert e.error_category == ErrorCategory.NUMA_AFFINITY


class TestBettiError:
    def test_instantiation(self):
        e = BettiError("invalid betti numbers")
        assert isinstance(e, NerveError)
        assert e.error_code == E87_INVALID_BETTI_NUMBERS
        assert e.error_category == ErrorCategory.ALGORITHMIC
