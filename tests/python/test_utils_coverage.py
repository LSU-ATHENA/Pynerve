from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pytest
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
from pynerve.exceptions import (
    DeviceError,
    DtypeError,
    ShapeError,
    ValidationError,
)

torch = pytest.importorskip("torch", reason="PyTorch required for tensor tests")


class TestValidateName:
    def test_accepts_non_empty_string(self):
        assert _validate_name("abc") == "abc"
        assert _validate_name("x", "label") == "x"

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            _validate_name(42)
        with pytest.raises(ValueError):
            _validate_name([1, 2])

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            _validate_name("")
        with pytest.raises(ValueError):
            _validate_name("", "custom")


class TestGetTorch:
    def test_returns_torch_when_available(self):
        result = _get_torch()
        assert result is torch

    def test_raises_when_torch_not_installed(self):
        with patch("pynerve._utils._has_torch", False):
            with patch("pynerve._utils._torch_module", None):
                with pytest.raises(ImportError, match="PyTorch is required"):
                    _get_torch()


class TestValidateTensor:
    def test_accepts_torch_tensor(self):
        t = torch.tensor([1.0])
        result = _validate_tensor(t)
        assert result is t

    def test_accepts_named_tensor(self):
        t = torch.zeros(3, 4)
        result = _validate_tensor(t, "weights")
        assert result is t

    def test_rejects_numpy_array(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _validate_tensor(np.array([1, 2, 3]))

    def test_rejects_list(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _validate_tensor([1, 2, 3])

    def test_rejects_none(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _validate_tensor(None)

    def test_rejects_scalar(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _validate_tensor(42)

    def test_uses_name_in_error(self):
        with pytest.raises(TypeError, match="my_tensor"):
            _validate_tensor(42, "my_tensor")

    def test_rejects_tensor_like_str_when_torch_not_installed(self):
        Tensor = type("Tensor", (), {})
        obj = Tensor()
        with patch("pynerve._utils._has_torch", False):
            with patch("pynerve._utils._torch_module", None):
                with pytest.raises(ImportError, match="PyTorch is required"):
                    _validate_tensor(obj)

    def test_rejects_non_tensor_like_when_torch_not_installed(self):
        with patch("pynerve._utils._has_torch", False):
            with patch("pynerve._utils._torch_module", None):
                with pytest.raises(TypeError, match="torch.Tensor"):
                    _validate_tensor(42, "val")


class TestGetDtype:
    def test_returns_dtype_of_float_tensor(self):
        t = torch.tensor([1.0, 2.0])
        assert _get_dtype(t) == torch.float32

    def test_returns_dtype_of_int_tensor(self):
        t = torch.tensor([1, 2])
        assert _get_dtype(t) == torch.int64

    def test_returns_dtype_of_double_tensor(self):
        t = torch.tensor([1.0], dtype=torch.float64)
        assert _get_dtype(t) == torch.float64


class TestEnsureBatchDim:
    def test_adds_batch_dim_to_2d_tensor_expecting_3d(self):
        t = torch.randn(4, 5)
        result, was_single = ensure_batch_dim(t, 3)
        assert result.shape == (1, 4, 5)
        assert was_single is True

    def test_passes_through_3d_tensor_expecting_3d(self):
        t = torch.randn(2, 4, 5)
        result, was_single = ensure_batch_dim(t, 3)
        assert result.shape == (2, 4, 5)
        assert was_single is False

    def test_adds_batch_dim_to_1d_scalar_tensor_expecting_2d(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result, was_single = ensure_batch_dim(t, 2)
        assert result.shape == (1, 3)
        assert was_single is True

    def test_passes_through_2d_tensor_expecting_2d(self):
        t = torch.randn(3, 4)
        result, was_single = ensure_batch_dim(t, 2)
        assert result.shape == (3, 4)
        assert was_single is False

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            ensure_batch_dim([1, 2, 3], 2)

    def test_rejects_wrong_dimensionality(self):
        t = torch.randn(2, 3, 4)
        with pytest.raises(ShapeError, match="Expected"):
            ensure_batch_dim(t, 2)

    def test_rejects_too_few_dims(self):
        t = torch.randn(3, 4)
        with pytest.raises(ShapeError, match="Expected"):
            ensure_batch_dim(t, 5)

    def test_rejects_negative_expected_ndim(self):
        t = torch.randn(3)
        with pytest.raises((TypeError, ValueError, ValidationError)):
            ensure_batch_dim(t, -1)

    def test_rejects_zero_expected_ndim(self):
        t = torch.randn(3)
        with pytest.raises((TypeError, ValueError, ValidationError)):
            ensure_batch_dim(t, 0)

    def test_rejects_non_integer_expected_ndim(self):
        t = torch.randn(3)
        with pytest.raises((TypeError, ValueError, ValidationError)):
            ensure_batch_dim(t, 1.5)


class TestRemoveBatchDim:
    def test_squeezes_single_batch_dimension(self):
        t = torch.randn(1, 4, 5)
        result = remove_batch_dim(t, True)
        assert result.shape == (4, 5)

    def test_preserves_tensor_when_was_single_false(self):
        t = torch.randn(2, 4, 5)
        result = remove_batch_dim(t, False)
        assert result.shape == (2, 4, 5)
        assert torch.equal(result, t)

    def test_no_squeeze_when_dim_zero(self):
        t = torch.tensor(1.0)
        result = remove_batch_dim(t, True)
        assert result.shape == ()
        assert torch.equal(result, t)

    def test_no_squeeze_when_batch_not_size_1(self):
        t = torch.randn(3, 4)
        result = remove_batch_dim(t, True)
        assert result.shape == (3, 4)

    def test_squeeze_keeps_values(self):
        t = torch.randn(1, 5)
        result = remove_batch_dim(t, True)
        assert result.shape == (5,)
        assert torch.allclose(result, t.squeeze(0))

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            remove_batch_dim(None, False)

    def test_rejects_non_bool_was_single(self):
        t = torch.randn(1, 3)
        with pytest.raises(TypeError, match="boolean"):
            remove_batch_dim(t, "not_bool")

    def test_rejects_int_was_single(self):
        t = torch.randn(1, 3)
        with pytest.raises(TypeError, match="boolean"):
            remove_batch_dim(t, 1)

    def test_rejects_none_was_single(self):
        t = torch.randn(1, 3)
        with pytest.raises(TypeError, match="boolean"):
            remove_batch_dim(t, None)


class TestValidateTensorShape:
    def test_accepts_correct_shape(self):
        t = torch.randn(2, 3)
        validate_tensor_shape(t, 2)

    def test_accepts_correct_shape_named(self):
        t = torch.randn(4, 5, 6)
        validate_tensor_shape(t, 3, "features")

    def test_rejects_wrong_dimensionality(self):
        t = torch.randn(2, 3)
        with pytest.raises(ShapeError, match="Expected"):
            validate_tensor_shape(t, 3, "input")

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            validate_tensor_shape(42, 2)

    def test_rejects_zero_expected_ndim(self):
        t = torch.randn(3)
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_tensor_shape(t, 0)

    def test_rejects_empty_name(self):
        t = torch.randn(2, 3)
        with pytest.raises(ValueError):
            validate_tensor_shape(t, 2, "")

    def test_error_message_includes_name(self):
        t = torch.randn(2, 3)
        with pytest.raises(ShapeError, match="my_input"):
            validate_tensor_shape(t, 3, "my_input")


class TestValidateDtype:
    def test_accepts_float32_in_supported(self):
        t = torch.tensor([1.0, 2.0])
        validate_dtype(t, {torch.float32, torch.float64})

    def test_accepts_float64_in_supported(self):
        t = torch.tensor([1.0], dtype=torch.float64)
        validate_dtype(t, {torch.float32, torch.float64})

    def test_accepts_int_in_supported(self):
        t = torch.tensor([1, 2])
        validate_dtype(t, {torch.int32, torch.int64})

    def test_accepts_named_tensor(self):
        t = torch.tensor([1.0])
        validate_dtype(t, {torch.float32}, "weights")

    def test_rejects_unsupported_dtype(self):
        t = torch.tensor([1, 2])
        with pytest.raises(DtypeError, match="Unsupported dtype"):
            validate_dtype(t, {torch.float32})

    def test_dtype_error_includes_name(self):
        t = torch.tensor([1, 2])
        with pytest.raises(DtypeError, match="features"):
            validate_dtype(t, {torch.float32}, "features")

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            validate_dtype(42, {torch.float32})

    def test_rejects_empty_supported_set(self):
        t = torch.tensor([1.0])
        with pytest.raises(ValueError, match="non-empty set"):
            validate_dtype(t, set())

    def test_rejects_non_set(self):
        t = torch.tensor([1.0])
        with pytest.raises(ValueError, match="non-empty set"):
            validate_dtype(t, [torch.float32])

    def test_rejects_non_dtype_items(self):
        t = torch.tensor([1.0])
        with pytest.raises(TypeError, match="torch.dtype"):
            validate_dtype(t, {torch.float32, "bad"})

    def test_rejects_empty_name(self):
        t = torch.tensor([1.0])
        with pytest.raises(ValueError):
            validate_dtype(t, {torch.float32}, "")


class TestValidatePositive:
    def test_accepts_positive_float(self):
        validate_positive(5.0, "val")

    def test_accepts_positive_int(self):
        validate_positive(1)

    def test_accepts_large_positive(self):
        validate_positive(1e100, "big")

    def test_accepts_small_positive(self):
        validate_positive(1e-10, "tiny")

    def test_rejects_zero(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive(0.0)

    def test_rejects_negative(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive(-1.0)

    def test_rejects_nan(self):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_positive(float("nan"))

    def test_rejects_inf(self):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_positive(float("inf"))

    def test_rejects_neg_inf(self):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_positive(float("-inf"))

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            validate_positive(5.0, "")


class TestValidateRange:
    def test_accepts_value_in_range(self):
        validate_range(5.0, 0.0, 10.0)

    def test_accepts_value_at_min(self):
        validate_range(0.0, 0.0, 10.0)

    def test_accepts_value_at_max(self):
        validate_range(10.0, 0.0, 10.0)

    def test_accepts_single_point_range(self):
        validate_range(5.0, 5.0, 5.0)

    def test_accepts_named_value(self):
        validate_range(0.5, 0.0, 1.0, "prob")

    def test_rejects_below_min(self):
        with pytest.raises(ValidationError, match="range"):
            validate_range(-1.0, 0.0, 10.0)

    def test_rejects_above_max(self):
        with pytest.raises(ValidationError, match="range"):
            validate_range(11.0, 0.0, 10.0)

    def test_rejects_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min_val"):
            validate_range(5.0, 10.0, 0.0)

    def test_rejects_nan_value(self):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_range(float("nan"), 0.0, 10.0)

    def test_rejects_inf_value(self):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_range(float("inf"), 0.0, 10.0)

    def test_rejects_nan_min(self):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_range(5.0, float("nan"), 10.0)

    def test_rejects_inf_max(self):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            validate_range(5.0, 0.0, float("inf"))

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            validate_range(5.0, 0.0, 10.0, "")


class TestValidateDevicesMatch:
    def test_empty_lists_noop(self):
        validate_devices_match([], [])

    def test_single_tensor_noop(self):
        t = torch.tensor([1.0])
        validate_devices_match([t], ["a"])

    def test_matching_devices(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        validate_devices_match([a, b], ["first", "second"])

    def test_matching_devices_multi_element(self):
        a = torch.randn(2, 3)
        b = torch.randn(4, 5)
        c = torch.randn(1)
        validate_devices_match([a, b, c], ["x", "y", "z"])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="matching lengths"):
            validate_devices_match([torch.tensor(1.0)], ["a", "b"])

    def test_mismatched_lengths_more_names(self):
        with pytest.raises(ValueError, match="matching lengths"):
            validate_devices_match([torch.tensor(1.0), torch.tensor(2.0)], ["a"])

    def test_mismatched_devices(self):
        cpu_t = torch.tensor([1.0], device="cpu")
        meta_t = torch.tensor([2.0], device="meta")
        with pytest.raises(DeviceError, match="Device mismatch"):
            validate_devices_match([cpu_t, meta_t], ["a", "b"])

    def test_device_mismatch_error_includes_names(self):
        cpu_t = torch.tensor([1.0], device="cpu")
        meta_t = torch.tensor([2.0], device="meta")
        with pytest.raises(DeviceError, match="first"):
            validate_devices_match([cpu_t, meta_t], ["first", "second"])

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            validate_devices_match([42], ["a"])

    def test_rejects_non_tensor_second_element(self):
        t = torch.tensor([1.0])
        with pytest.raises(TypeError, match="torch.Tensor"):
            validate_devices_match([t, 42], ["a", "b"])

    def test_rejects_empty_name(self):
        t = torch.tensor([1.0])
        with pytest.raises(ValueError):
            validate_devices_match([t], [""])


class TestSuppressWarnings:
    def test_suppresses_deprecation_warning(self):
        with suppress_warnings(DeprecationWarning):
            warnings.warn("old", DeprecationWarning)

    def test_suppresses_user_warning(self):
        with suppress_warnings(UserWarning):
            warnings.warn("user", UserWarning)

    def test_suppresses_all_warnings_by_default(self):
        with suppress_warnings():
            warnings.warn("dep", DeprecationWarning)
            warnings.warn("user", UserWarning)

    def test_does_not_suppress_other_categories(self):
        with pytest.warns(RuntimeWarning), suppress_warnings(UserWarning):
            warnings.warn("runtime", RuntimeWarning)

    def test_does_not_suppress_when_no_match(self):
        with pytest.warns(DeprecationWarning), suppress_warnings(UserWarning):
            warnings.warn("deprecated", DeprecationWarning)

    def test_rejects_non_warning_type(self):
        with pytest.raises((TypeError, Exception)), suppress_warnings(int):
            pass

    def test_rejects_non_type(self):
        with pytest.raises((TypeError, Exception)), suppress_warnings(42):
            pass

    def test_rejects_plain_object(self):
        with pytest.raises((TypeError, Exception)), suppress_warnings(object()):
            pass

    def test_context_manager_yields(self):
        entered = False
        with suppress_warnings():
            entered = True
        assert entered

    def test_nested_suppressions(self):
        with suppress_warnings(UserWarning):
            with suppress_warnings(DeprecationWarning):
                warnings.warn("dep", DeprecationWarning)
            warnings.warn("user", UserWarning)


class TestIsTensor:
    def test_returns_true_for_tensor(self):
        assert is_tensor(torch.tensor([1.0])) is True

    def test_returns_true_for_randn_tensor(self):
        assert is_tensor(torch.randn(2, 3)) is True

    def test_returns_true_for_zeros_tensor(self):
        assert is_tensor(torch.zeros(5)) is True

    def test_returns_false_for_numpy(self):
        assert is_tensor(np.array([1, 2, 3])) is False

    def test_returns_false_for_list(self):
        assert is_tensor([1, 2, 3]) is False

    def test_returns_false_for_scalar(self):
        assert is_tensor(42) is False

    def test_returns_false_for_none(self):
        assert is_tensor(None) is False

    def test_returns_false_when_torch_not_installed(self):
        with patch("pynerve._utils._has_torch", False):
            with patch("pynerve._utils._torch_module", None):
                assert is_tensor(torch.tensor([1.0])) is False

    def test_returns_false_for_tensor_subclass_when_no_torch(self):
        obj = type("FakeTensor", (), {})()
        with patch("pynerve._utils._has_torch", False):
            with patch("pynerve._utils._torch_module", None):
                assert is_tensor(obj) is False


class TestIsNumpyArray:
    def test_returns_true_for_array(self):
        assert is_numpy_array(np.array([1, 2, 3])) is True

    def test_returns_true_for_2d_array(self):
        assert is_numpy_array(np.ones((2, 3))) is True

    def test_returns_true_for_empty_array(self):
        assert is_numpy_array(np.array([])) is True

    def test_returns_true_for_scalar_array(self):
        assert is_numpy_array(np.array(1.0)) is True

    def test_returns_false_for_list(self):
        assert is_numpy_array([1, 2, 3]) is False

    def test_returns_false_for_scalar(self):
        assert is_numpy_array(42) is False

    def test_returns_false_for_none(self):
        assert is_numpy_array(None) is False

    def test_returns_false_for_tensor(self):
        assert is_numpy_array(torch.tensor([1, 2, 3])) is False


class TestEnsureBatchDimRemoveBatchDimRoundtrip:
    def test_add_then_remove_2d_to_3d(self):
        t = torch.randn(4, 5)
        with_batch, was_single = ensure_batch_dim(t, 3)
        assert was_single is True
        result = remove_batch_dim(with_batch, was_single)
        assert result.shape == t.shape
        assert torch.equal(result, t)

    def test_add_then_remove_1d_to_2d(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        with_batch, was_single = ensure_batch_dim(t, 2)
        assert was_single is True
        result = remove_batch_dim(with_batch, was_single)
        assert result.shape == t.shape
        assert torch.equal(result, t)

    def test_noop_roundtrip_3d(self):
        t = torch.randn(2, 4, 5)
        with_batch, was_single = ensure_batch_dim(t, 3)
        assert was_single is False
        result = remove_batch_dim(with_batch, was_single)
        assert result.shape == t.shape
        assert torch.equal(result, t)


class TestEnsureBatchDimEdgeCases:
    def test_zero_dim_tensor_expecting_1d(self):
        t = torch.tensor(1.0)
        result, was_single = ensure_batch_dim(t, 1)
        assert was_single is True
        assert result.shape == (1,)
        assert result.item() == 1.0

    def test_ensure_dim_name_in_error(self):
        t = torch.randn(2, 3, 4, 5)
        with pytest.raises(ShapeError):
            ensure_batch_dim(t, 2)


class TestValidateDtypeEdgeCases:
    def test_single_element_set(self):
        t = torch.tensor([1.0])
        validate_dtype(t, {torch.float32}, "x")

    def test_error_message_includes_supported_list(self):
        t = torch.tensor([1, 2])
        with pytest.raises(DtypeError, match="torch.int64"):
            validate_dtype(t, {torch.float32, torch.float64})


class TestValidatePositiveEdgeCases:
    def test_different_types_positive(self):
        validate_positive(1, "int")
        validate_positive(0.5, "float")

    def test_error_message_includes_value(self):
        with pytest.raises(ValidationError, match="-3"):
            validate_positive(-3.0, "test")


class TestValidateRangeEdgeCases:
    def test_boundaries_float(self):
        validate_range(0.0, 0.0, 0.0, "zero")

    def test_negative_range_when_both_inverted(self):
        with pytest.raises(ValueError, match="min_val"):
            validate_range(-1.0, 5.0, -5.0)

    def test_error_message_includes_bounds(self):
        with pytest.raises(ValidationError, match="0"):
            validate_range(-1.0, 0.0, 10.0)


class TestValidateDevicesMatchEdgeCases:
    def test_all_same_device_across_many(self):
        tensors = [torch.randn(i + 1, 3) for i in range(5)]
        names = [f"t{i}" for i in range(5)]
        validate_devices_match(tensors, names)

    def test_error_includes_device_names(self):
        a = torch.tensor([1.0], device="cpu")
        b = torch.tensor([2.0], device="meta")
        with pytest.raises(DeviceError, match="left|right"):
            validate_devices_match([a, b], ["left", "right"])
