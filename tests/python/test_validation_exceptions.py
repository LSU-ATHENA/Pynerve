"""Tests for exceptions/_validation.py -- ValidationError hierarchy."""

from __future__ import annotations

import pytest

from pynerve._error_codes import (
    E90_VALIDATION_ERROR,
    E91_SHAPE_ERROR,
    E92_DTYPE_ERROR,
    E93_DEVICE_ERROR,
    E94_BACKEND_REQUIRED,
    ErrorCategory,
)
from pynerve.exceptions._base import NerveError
from pynerve.exceptions._validation import (
    BackendRequiredError,
    DeviceError,
    DtypeError,
    ShapeError,
    ValidationError,
)


# ValidationError 


class TestValidationError:
    def test_basic(self):
        e = ValidationError("invalid input")
        assert "invalid input" in str(e)
        assert e.error_code == E90_VALIDATION_ERROR
        assert e.error_category == ErrorCategory.OPERATIONAL

    def test_is_nerve_error(self):
        assert issubclass(ValidationError, NerveError)

    def test_with_parameter(self):
        e = ValidationError("bad", parameter="x")
        assert e.parameter == "x"

    def test_with_expected_actual(self):
        e = ValidationError("bad", expected="int", actual="str")
        assert e.expected == "int"
        assert e.actual == "str"

    def test_repr_minimal(self):
        e = ValidationError("msg")
        r = repr(e)
        assert "ValidationError" in r

    def test_repr_with_fields(self):
        e = ValidationError("msg", parameter="x", expected="int", actual="str")
        r = repr(e)
        assert "parameter='x'" in r
        assert "expected='int'" in r
        assert "actual='str'" in r

    def test_invalid_parameter_type(self):
        with pytest.raises(TypeError, match="non-empty"):
            ValidationError("msg", parameter=123)  # type: ignore[arg-type]

    def test_invalid_parameter_empty(self):
        with pytest.raises(TypeError, match="non-empty"):
            ValidationError("msg", parameter="")

    def test_invalid_expected_type(self):
        with pytest.raises(TypeError, match="non-empty"):
            ValidationError("msg", expected=123)  # type: ignore[arg-type]

    def test_invalid_actual_type(self):
        with pytest.raises(TypeError, match="non-empty"):
            ValidationError("msg", actual=123)  # type: ignore[arg-type]

    def test_parameter_none_allowed(self):
        e = ValidationError("msg", parameter=None)
        assert e.parameter is None


# ShapeError 


class TestShapeError:
    def test_basic(self):
        e = ShapeError("wrong shape")
        assert e.error_code == E91_SHAPE_ERROR
        assert isinstance(e, ValidationError)

    def test_with_shapes(self):
        e = ShapeError("wrong shape", expected_shape=(10, 20), actual_shape=(5, 10))
        assert e.expected_shape == (10, 20)
        assert e.actual_shape == (5, 10)

    def test_with_ndim(self):
        e = ShapeError("wrong ndim", expected_ndim=2, actual_ndim=3)
        assert e.expected_ndim == 2
        assert e.actual_ndim == 3

    def test_repr_with_shapes(self):
        e = ShapeError("msg", expected_shape=(2, 3), actual_shape=(4,))
        r = repr(e)
        assert "expected_shape=(2, 3)" in r
        assert "actual_shape=(4,)" in r

    def test_repr_with_ndim(self):
        e = ShapeError("msg", expected_ndim=2, actual_ndim=1)
        r = repr(e)
        assert "expected_ndim=2" in r
        assert "actual_ndim=1" in r

    def test_none_shapes(self):
        e = ShapeError("msg", expected_shape=None, actual_shape=None)
        assert e.expected_shape is None
        assert e.actual_shape is None

    def test_invalid_shape_tuple(self):
        with pytest.raises(Exception):
            ShapeError("msg", expected_shape="not a tuple")  # type: ignore[arg-type]

    def test_parameter_field(self):
        e = ShapeError("msg", parameter="tensor")
        assert e.parameter == "tensor"


# DtypeError 


class TestDtypeError:
    def test_basic(self):
        e = DtypeError("wrong dtype")
        assert e.error_code == E92_DTYPE_ERROR
        assert isinstance(e, ValidationError)

    def test_with_fields(self):
        e = DtypeError("wrong dtype", expected_dtypes=["float32", "float64"], actual_dtype="int32")
        assert e.expected_dtypes == ["float32", "float64"]
        assert e.actual_dtype == "int32"

    def test_repr_with_fields(self):
        e = DtypeError("msg", expected_dtypes=["float32"], actual_dtype="int64")
        r = repr(e)
        assert "expected_dtypes=['float32']" in r
        assert "actual_dtype='int64'" in r

    def test_empty_expected_dtypes(self):
        e = DtypeError("msg", expected_dtypes=[])
        assert e.expected_dtypes == []

    def test_actual_dtype_string_none(self):
        e = DtypeError("msg", actual_dtype=None)
        assert e.actual_dtype is None

    def test_none_expected_dtypes(self):
        e = DtypeError("msg", expected_dtypes=None)
        assert e.expected_dtypes == []


# DeviceError 


class TestDeviceError:
    def test_basic(self):
        e = DeviceError("no cuda")
        assert e.error_code == E93_DEVICE_ERROR
        assert isinstance(e, ValidationError)

    def test_with_requested_device(self):
        e = DeviceError("no cuda", requested_device="cuda:0")
        assert e.requested_device == "cuda:0"

    def test_with_available_devices(self):
        e = DeviceError("no cuda", available_devices=["cpu"])
        assert e.available_devices == ["cpu"]

    def test_repr_with_fields(self):
        e = DeviceError("msg", requested_device="cuda", available_devices=["cpu", "mps"])
        r = repr(e)
        assert "requested_device='cuda'" in r
        assert "available_devices=['cpu', 'mps']" in r

    def test_invalid_requested_device(self):
        with pytest.raises(TypeError, match="non-empty"):
            DeviceError("msg", requested_device="")

    def test_invalid_requested_device_type(self):
        with pytest.raises(TypeError, match="non-empty"):
            DeviceError("msg", requested_device=123)  # type: ignore[arg-type]

    def test_requested_device_none(self):
        e = DeviceError("msg", requested_device=None)
        assert e.requested_device is None

    def test_available_devices_none(self):
        e = DeviceError("msg", available_devices=None)
        assert e.available_devices == []


# BackendRequiredError 


class TestBackendRequiredError:
    def test_basic(self):
        e = BackendRequiredError("backend needed")
        assert e.error_code == E94_BACKEND_REQUIRED
        assert isinstance(e, ValidationError)

    def test_with_backend(self):
        e = BackendRequiredError("need cuda", backend="cuda")
        assert e.backend == "cuda"

    def test_with_installation_hint(self):
        e = BackendRequiredError("need torch", installation_hint="pip install torch")
        assert e.installation_hint == "pip install torch"

    def test_repr_with_fields(self):
        e = BackendRequiredError("msg", backend="torch", installation_hint="pip install torch")
        r = repr(e)
        assert "backend='torch'" in r
        assert "installation_hint='pip install torch'" in r

    def test_invalid_backend_type(self):
        with pytest.raises(Exception):
            BackendRequiredError("msg", backend="")

    def test_invalid_installation_hint_type(self):
        with pytest.raises(Exception):
            BackendRequiredError("msg", installation_hint="")

    def test_backend_none(self):
        e = BackendRequiredError("msg", backend=None)
        assert e.backend is None

    def test_installation_hint_none(self):
        e = BackendRequiredError("msg", installation_hint=None)
        assert e.installation_hint is None

    def test_extra_kwargs(self):
        e = BackendRequiredError("msg", parameter="x")
        assert e.parameter == "x"
