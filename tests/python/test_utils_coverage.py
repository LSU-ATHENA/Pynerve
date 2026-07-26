"""Additional tests for _utils.py -- tensor validation, device matching, dtype checks."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pynerve._utils import (
    _get_dtype,
    _get_torch,
    _validate_name,
    _validate_tensor,
    ensure_batch_dim,
    is_tensor,
    remove_batch_dim,
    suppress_warnings,
    validate_devices_match,
    validate_dtype,
    validate_positive,
    validate_range,
    validate_tensor_shape,
)
from pynerve.exceptions import DeviceError, DtypeError, ShapeError, ValidationError


class TestGetTorch:
    def test_returns_torch_module(self):
        result = _get_torch()
        assert result is torch

    def test_torch_is_tensor_with_tensor(self):
        t = torch.tensor([1.0, 2.0])
        assert is_tensor(t) is True

    def test_torch_gpu_tensor(self):
        if torch.cuda.is_available():
            t = torch.tensor([1.0], device="cuda")
            assert is_tensor(t) is True


class TestValidateTensor:
    def test_accepts_tensor(self):
        t = torch.tensor([1.0])
        result = _validate_tensor(t, "x")
        assert result is t

    def test_rejects_numpy(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _validate_tensor(np.array([1.0]), "x")

    def test_rejects_list(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _validate_tensor([1, 2, 3], "x")

    def test_rejects_none(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _validate_tensor(None, "x")

    def test_default_name(self):
        t = torch.tensor([1.0])
        result = _validate_tensor(t)
        assert result is t


class TestValidateName:
    def test_accepts_nonempty(self):
        assert _validate_name("hello") == "hello"
        assert _validate_name("test", "param") == "test"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_name("")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_name(42)

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_name(None)


class TestGetDtype:
    def test_float32_tensor(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        assert _get_dtype(t) == torch.float32

    def test_float64_tensor(self):
        t = torch.tensor([1.0], dtype=torch.float64)
        assert _get_dtype(t) == torch.float64

    def test_int_tensor(self):
        t = torch.tensor([1, 2, 3])
        assert _get_dtype(t) == torch.int64


class TestEnsureBatchDim:
    def test_adds_batch_to_1d(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result, was_single = ensure_batch_dim(t, 2)
        assert result.shape == (1, 3)
        assert was_single is True

    def test_no_change_for_2d(self):
        t = torch.randn(5, 3)
        result, was_single = ensure_batch_dim(t, 2)
        assert result.shape == (5, 3)
        assert was_single is False

    def test_wrong_dims_raises(self):
        t = torch.randn(2, 3, 4)
        with pytest.raises(ShapeError, match="Expected"):
            ensure_batch_dim(t, 2)

    def test_expected_ndim_3_with_2d_input(self):
        t = torch.randn(5, 3)
        result, was_single = ensure_batch_dim(t, 3)
        assert result.shape == (1, 5, 3)
        assert was_single is True

    def test_expected_ndim_3_with_3d_input(self):
        t = torch.randn(2, 5, 3)
        result, was_single = ensure_batch_dim(t, 3)
        assert result.shape == (2, 5, 3)
        assert was_single is False


class TestRemoveBatchDim:
    def test_removes_when_single(self):
        t = torch.tensor([[1.0, 2.0, 3.0]])
        result = remove_batch_dim(t, True)
        assert result.shape == (3,)

    def test_keeps_when_not_single(self):
        t = torch.randn(5, 3)
        result = remove_batch_dim(t, False)
        assert result.shape == (5, 3)

    def test_size_not_one_keeps(self):
        t = torch.randn(3, 3)
        result = remove_batch_dim(t, True)
        assert result.shape == (3, 3)

    def test_0d_tensor_keeps(self):
        t = torch.tensor(5.0)
        result = remove_batch_dim(t, True)
        assert result.dim() == 0


class TestValidateTensorShape:
    def test_correct_ndim_passes(self):
        t = torch.randn(10, 3)
        validate_tensor_shape(t, 2, "data")

    def test_wrong_ndim_raises(self):
        t = torch.randn(10, 3, 2)
        with pytest.raises(ShapeError, match="Expected"):
            validate_tensor_shape(t, 2, "data")

    def test_1d_tensor_raises_for_2d_expected(self):
        t = torch.tensor([1.0, 2.0])
        with pytest.raises(ShapeError):
            validate_tensor_shape(t, 2, "data")


class TestValidateDtype:
    def test_correct_dtype_passes(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        validate_dtype(t, {torch.float32, torch.float64}, "data")

    def test_wrong_dtype_raises(self):
        t = torch.tensor([1], dtype=torch.int64)
        with pytest.raises(DtypeError, match="Unsupported dtype"):
            validate_dtype(t, {torch.float32}, "data")

    def test_empty_set_raises(self):
        t = torch.tensor([1.0])
        with pytest.raises(ValueError, match="non-empty set"):
            validate_dtype(t, set(), "data")

    def test_non_dtype_in_set_raises(self):
        t = torch.tensor([1.0])
        with pytest.raises(TypeError, match="torch.dtype"):
            validate_dtype(t, {42}, "data")


class TestValidateDevicesMatch:
    def test_all_same_device(self):
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        validate_devices_match([t1, t2], ["a", "b"])

    def test_mismatched_devices(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")
        t1 = torch.tensor([1.0], device="cuda")
        t2 = torch.tensor([2.0], device="cpu")
        with pytest.raises(DeviceError, match="Device mismatch"):
            validate_devices_match([t1, t2], ["a", "b"])

    def test_single_tensor_ok(self):
        t = torch.tensor([1.0])
        validate_devices_match([t], ["a"])

    def test_cpu_and_meta_devices(self):
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        t2_cpu = t2.to("cpu")
        validate_devices_match([t1, t2_cpu], ["a", "b"])


class TestSuppressWarnings:
    def test_suppresses_user_warning(self):
        with suppress_warnings(UserWarning):
            warnings.warn("hidden", UserWarning)

    def test_rejects_non_warning_class(self):
        with pytest.raises(TypeError, match="Warning subclass"):
            with suppress_warnings(int):
                pass

    def test_default_suppresses_all(self):
        with suppress_warnings():
            warnings.warn("hidden", DeprecationWarning)
            warnings.warn("hidden too", UserWarning)


class TestValidatePositive:
    def test_accepts_positive_float(self):
        validate_positive(3.14, "x")

    def test_accepts_positive_int(self):
        validate_positive(5, "x")

    def test_rejects_zero(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive(0.0, "x")

    def test_rejects_inf(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_positive(float("inf"), "x")


class TestValidateRange:
    def test_accepts_boundary(self):
        validate_range(0.0, 0.0, 10.0)
        validate_range(10.0, 0.0, 10.0)

    def test_rejects_nan(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_range(float("nan"), 0.0, 10.0)

    def test_rejects_nan_min(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_range(5.0, float("nan"), 10.0)

    def test_rejects_nan_max(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_range(5.0, 0.0, float("nan"))
