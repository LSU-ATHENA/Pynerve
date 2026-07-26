"""Edge case tests for _utils.py -- remaining uncovered paths."""

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
    is_numpy_array,
    is_tensor,
    remove_batch_dim,
    suppress_warnings,
    validate_devices_match,
    validate_dtype,
    validate_positive,
    validate_range,
    validate_tensor_shape,
)
from pynerve.exceptions import ValidationError


class TestIsNumpyArray:
    def test_numpy_returns_true(self):
        assert is_numpy_array(np.array([1.0])) is True
        assert is_numpy_array(np.array([[1, 2]])) is True

    def test_non_numpy_returns_false(self):
        assert is_numpy_array(42) is False
        assert is_numpy_array([1, 2, 3]) is False
        assert is_numpy_array(None) is False

    def test_tensor_returns_false(self):
        t = torch.tensor([1.0])
        assert is_numpy_array(t) is False


class TestIsTensor:
    def test_tensor_returns_true(self):
        t = torch.tensor([1.0])
        assert is_tensor(t) is True

    def test_non_tensor_returns_false(self):
        assert is_tensor(np.array([1.0])) is False
        assert is_tensor(42) is False


class TestGetDtype:
    def test_float32(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        assert _get_dtype(t) == torch.float32

    def test_int64(self):
        t = torch.tensor([1, 2, 3])
        assert _get_dtype(t) == torch.int64


class TestValidateDevicesMatchEdge:
    def test_empty_list_noop(self):
        validate_devices_match([], [])

    def test_length_mismatch_raises(self):
        t = torch.tensor([1.0])
        with pytest.raises(ValueError, match="matching lengths"):
            validate_devices_match([t], [])


class TestValidatePositiveEdge:
    def test_valid_positive(self):
        validate_positive(1.0, "x")
        validate_positive(0.1, "x")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive(-1.0, "x")


class TestValidateRangeEdge:
    def test_valid_in_range(self):
        validate_range(5.0, 0.0, 10.0)

    def test_below_range_raises(self):
        with pytest.raises(ValidationError, match="in range"):
            validate_range(-1.0, 0.0, 10.0)

    def test_above_range_raises(self):
        with pytest.raises(ValidationError, match="in range"):
            validate_range(11.0, 0.0, 10.0)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="less than"):
            validate_range(5.0, 10.0, 0.0)


class TestEnsureBatchDimEdge:
    def test_0d_tensor(self):
        t = torch.tensor(5.0)
        result, was_single = ensure_batch_dim(t, 1)
        assert was_single is True
        assert result.dim() == 1
        assert result.item() == 5.0

    def test_wrong_ndim_raises(self):
        t = torch.randn(2, 3, 4)
        with pytest.raises(Exception):
            ensure_batch_dim(t, 2)


class TestRemoveBatchDimEdge:
    def test_0d_tensor(self):
        t = torch.tensor(5.0)
        result = remove_batch_dim(t, True)
        assert result.dim() == 0

    def test_was_single_false(self):
        t = torch.randn(5, 3)
        result = remove_batch_dim(t, False)
        assert result.shape == (5, 3)

    def test_non_bool_was_single_raises(self):
        t = torch.tensor([1.0])
        with pytest.raises(TypeError, match="boolean"):
            remove_batch_dim(t, "yes")  # type: ignore[arg-type]


class TestSuppressWarningsEdge:
    def test_suppresses_deprecation(self):
        with suppress_warnings(DeprecationWarning):
            warnings.warn("hidden", DeprecationWarning)

    def test_non_warning_raises(self):
        with pytest.raises(TypeError, match="Warning subclass"):
            with suppress_warnings(int):  # type: ignore[arg-type]
                pass

    def test_default_category(self):
        with suppress_warnings():
            warnings.warn("hidden", UserWarning)
