"""Tests for internal utilities (requires PyTorch for some tensor helpers)."""

from __future__ import annotations

import pytest
from pynerve._utils import (
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


class TestIsTensor:
    def test_returns_false_for_numpy(self):
        import numpy as np

        assert is_tensor(np.array([1, 2, 3])) is False

    def test_returns_false_for_list(self):
        assert is_tensor([1, 2, 3]) is False

    def test_returns_false_for_scalar(self):
        assert is_tensor(42) is False


class TestIsNumpyArray:
    def test_returns_true_for_array(self):
        import numpy as np

        assert is_numpy_array(np.array([1, 2, 3])) is True

    def test_returns_false_for_list(self):
        assert is_numpy_array([1, 2, 3]) is False

    def test_returns_false_for_scalar(self):
        assert is_numpy_array(42) is False


class TestValidatePositive:
    def test_accepts_positive(self):
        assert validate_positive(5.0, "val") is None

    def test_rejects_zero(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive(0.0, "val")

    def test_rejects_negative(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive(-1.0, "val")

    def test_rejects_nan(self):

        with pytest.raises(ValidationError, match="finite"):
            validate_positive(float("nan"), "val")


class TestValidateRange:
    def test_accepts_value_in_range(self):
        assert validate_range(5.0, 0.0, 10.0, "val") is None

    def test_rejects_below_min(self):
        with pytest.raises(ValidationError, match="range"):
            validate_range(-1.0, 0.0, 10.0, "val")

    def test_rejects_above_max(self):
        with pytest.raises(ValidationError, match="range"):
            validate_range(11.0, 0.0, 10.0, "val")

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="min_val"):
            validate_range(5.0, 10.0, 0.0, "val")


class TestSuppressWarnings:
    def test_suppresses_given_warning(self):
        import warnings

        with suppress_warnings(UserWarning):
            warnings.warn("test warning", UserWarning, stacklevel=2)

    def test_does_not_suppress_other_warnings(self):
        import warnings

        with pytest.warns(RuntimeWarning), suppress_warnings(UserWarning):
            warnings.warn("runtime", RuntimeWarning, stacklevel=2)

    def test_rejects_non_warning_type(self):
        with pytest.raises((TypeError, Exception)), suppress_warnings(int):
            pass


class TestEnsureAndRemoveBatchDim:
    def test_ensure_batch_dim_on_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            ensure_batch_dim([1, 2, 3], 2)

    def test_remove_batch_dim_non_bool(self):
        with pytest.raises((TypeError, ValueError)):
            remove_batch_dim(None, "not_bool")


class TestValidateDevicesMatch:
    def test_empty_list_ok(self):
        assert validate_devices_match([], []) is None

    def test_mismatched_list_lengths(self):
        with pytest.raises(ValueError, match="matching lengths"):
            validate_devices_match([42], ["a", "b"])


class TestValidateDtype:
    def test_none_tensor_raises(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            validate_dtype(42, set(), "val")

    def test_empty_supported_set_raises(self):
        with pytest.raises((TypeError, ValueError)):
            validate_dtype(None, set(), "val")


class TestValidateTensorShape:
    def test_non_tensor_raises(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            validate_tensor_shape(42, 2, "val")
