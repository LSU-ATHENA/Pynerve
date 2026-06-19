"""Tests for pynerve._compute_pipeline internal helpers."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pynerve._compute_pipeline import (
    _apply_option_overrides,
    _clone_options,
    _is_likely_distance_matrix,
    _resolve_options,
    _tensor_to_array,
    _to_point_array,
    _validate_array,
    _validate_error_tolerance,
    _validate_max_dim,
    _validate_max_radius,
    _validate_threads,
)
from pynerve._fallback_classes import PersistenceBackend, PersistenceMode, PersistenceOptions
from pynerve._persistence_result import _CAP_WARNED
from pynerve.exceptions import InvalidArgumentError, ShapeMismatchError, ValidationError

try:
    import torch as _torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    _torch = MagicMock()  # type: ignore[no-redef]


def _cuda_usable() -> bool:
    if not _HAS_TORCH or not _torch.cuda.is_available():
        return False
    try:
        _torch.empty(1, device="cuda")
        return True
    except (RuntimeError, _torch.cuda.DeferredCudaCallError):
        return False


class TestIsLikelyDistanceMatrix:
    def test_1d_not_matrix(self):
        assert _is_likely_distance_matrix(np.array([1.0, 2.0])) is False

    def test_3d_not_matrix(self):
        assert _is_likely_distance_matrix(np.zeros((2, 2, 2))) is False

    def test_non_square(self):
        assert _is_likely_distance_matrix(np.zeros((3, 2))) is False

    def test_too_small(self):
        assert _is_likely_distance_matrix(np.zeros((1, 1))) is False

    def test_nonzero_diagonal(self):
        arr = np.ones((3, 3))
        np.fill_diagonal(arr, 0.5)
        assert _is_likely_distance_matrix(arr) is False

    def test_not_symmetric(self):
        arr = np.zeros((3, 3))
        arr[0, 1] = 1.0
        assert _is_likely_distance_matrix(arr) is False

    def test_valid_distance_matrix(self):
        arr = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        assert _is_likely_distance_matrix(arr) is True

    def test_zero_matrix_is_distance(self):
        assert _is_likely_distance_matrix(np.zeros((5, 5))) is True

    def test_nearly_symmetric_passes(self):
        arr = np.array([[0.0, 1.0], [1.0 + 1e-16, 0.0]])
        assert _is_likely_distance_matrix(arr) is True


class TestCloneOptions:
    def test_none_returns_default(self):
        opts = _clone_options(None)
        assert isinstance(opts, PersistenceOptions)
        assert opts.max_dim == 2
        assert opts.mode == PersistenceMode.EXACT

    def test_clone_preserves_values(self):
        orig = PersistenceOptions(max_dim=5, threads=4, max_radius=1.0)
        cloned = _clone_options(orig)
        assert cloned.max_dim == 5
        assert cloned.threads == 4
        assert cloned.max_radius == 1.0

    def test_clone_is_independent(self):
        orig = PersistenceOptions(max_dim=3)
        cloned = _clone_options(orig)
        cloned = dataclasses.replace(cloned, max_dim=7)
        assert orig.max_dim == 3
        assert cloned.max_dim == 7


class TestValidateMaxDim:
    def test_positive(self):
        assert _validate_max_dim(3) == 3

    def test_zero(self):
        assert _validate_max_dim(0) == 0

    def test_float_converted(self):
        assert _validate_max_dim(2.0) == 2

    def test_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="max_dim"):
            _validate_max_dim(-1)

    def test_none_raises_type_error_before_int_conversion(self):
        with pytest.raises((TypeError, ValueError)):
            _validate_max_dim(None)  # type: ignore[arg-type]


class TestValidateMaxRadius:
    def test_finite_positive(self):
        assert _validate_max_radius(1.5) == 1.5

    def test_zero(self):
        assert _validate_max_radius(0.0) == 0.0

    def test_explicit_cap_truncates_inf(self):
        _CAP_WARNED[0] = False
        with pytest.warns(UserWarning):
            result = _validate_max_radius(float("inf"), cap=100.0)
        assert result == 100.0

    def test_inf_uses_default_cap_with_warning(self):
        _CAP_WARNED[0] = False
        with pytest.warns(UserWarning):
            result = _validate_max_radius(float("inf"))
            assert np.isfinite(result)

    def test_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="max_radius"):
            _validate_max_radius(-1.0)

    def test_nan_raises(self):
        with pytest.raises(InvalidArgumentError, match="max_radius"):
            _validate_max_radius(float("nan"))

    def test_argument_is_int(self):
        assert _validate_max_radius(5) == 5.0


class TestValidateThreads:
    def test_positive(self):
        assert _validate_threads(4) == 4

    def test_one(self):
        assert _validate_threads(1) == 1

    def test_float_truncated(self):
        assert _validate_threads(2.9) == 2

    def test_zero_raises(self):
        with pytest.raises(InvalidArgumentError, match="threads"):
            _validate_threads(0)

    def test_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="threads"):
            _validate_threads(-3)


class TestValidateErrorTolerance:
    def test_zero(self):
        assert _validate_error_tolerance(0.0) == 0.0

    def test_positive(self):
        assert _validate_error_tolerance(1e-6) == 1e-6

    def test_int_converted(self):
        assert _validate_error_tolerance(1) == 1.0

    def test_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="error_tolerance"):
            _validate_error_tolerance(-0.1)

    def test_nan_raises(self):
        with pytest.raises(InvalidArgumentError, match="error_tolerance"):
            _validate_error_tolerance(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(InvalidArgumentError, match="error_tolerance"):
            _validate_error_tolerance(float("inf"))


class TestApplyOptionOverrides:
    def test_no_overrides_returns_unchanged(self):
        opts = PersistenceOptions(max_dim=3)
        result = _apply_option_overrides(opts)
        assert result.max_dim == 3

    def test_max_dim_override(self):
        opts = PersistenceOptions(max_dim=2)
        result = _apply_option_overrides(opts, max_dim=5)
        assert result.max_dim == 5

    def test_max_radius_override(self):
        opts = PersistenceOptions(max_radius=None)
        result = _apply_option_overrides(opts, max_radius=2.0)
        assert result.max_radius == 2.0

    def test_max_radius_override_with_cap(self):
        opts = PersistenceOptions(max_radius=None)
        _CAP_WARNED[0] = False
        with pytest.warns(UserWarning):
            result = _apply_option_overrides(opts, max_radius=float("inf"), max_radius_cap=50.0)
        assert result.max_radius == 50.0

    def test_mode_override(self):
        opts = PersistenceOptions(mode=PersistenceMode.EXACT)
        result = _apply_option_overrides(opts, mode=PersistenceMode.APPROX)
        assert result.mode == PersistenceMode.APPROX

    def test_backend_override(self):
        opts = PersistenceOptions(backend=PersistenceBackend.CPU_ADAPTIVE_ACCELERATION)
        result = _apply_option_overrides(opts, backend=PersistenceBackend.CPU_EXACT)
        assert result.backend == PersistenceBackend.CPU_EXACT

    def test_threads_override(self):
        opts = PersistenceOptions(threads=0)
        result = _apply_option_overrides(opts, threads=8)
        assert result.threads == 8

    def test_error_tolerance_override(self):
        opts = PersistenceOptions(error_tolerance=0.0)
        result = _apply_option_overrides(opts, error_tolerance=0.001)
        assert result.error_tolerance == 0.001

    def test_device_trumps_backend(self):
        opts = PersistenceOptions(backend=PersistenceBackend.CPU_EXACT)
        result = _apply_option_overrides(opts, device="cuda")
        assert result.backend == PersistenceBackend.CUDA_HYBRID

    def test_device_cpu_maps_to_adaptive(self):
        opts = PersistenceOptions(backend=PersistenceBackend.CPU_EXACT)
        result = _apply_option_overrides(opts, device="cpu")
        assert result.backend == PersistenceBackend.CPU_ADAPTIVE_ACCELERATION

    def test_both_device_and_backend_device_wins(self):
        opts = PersistenceOptions()
        result = _apply_option_overrides(
            opts,
            device="cuda",
            backend=PersistenceBackend.CPU_EXACT,
        )
        assert result.backend == PersistenceBackend.CUDA_HYBRID

    def test_seed_does_not_change_options_fields(self):
        opts = PersistenceOptions(max_dim=1)
        result = _apply_option_overrides(opts, seed=42)
        assert result.max_dim == 1

    def test_seed_negative_raises(self):
        opts = PersistenceOptions()
        with pytest.raises(InvalidArgumentError, match="seed"):
            _apply_option_overrides(opts, seed=-1)

    def test_multiple_overrides_at_once(self):
        opts = PersistenceOptions()
        result = _apply_option_overrides(
            opts,
            max_dim=4,
            max_radius=5.0,
            threads=3,
            error_tolerance=0.01,
        )
        assert result.max_dim == 4
        assert result.max_radius == 5.0
        assert result.threads == 3
        assert result.error_tolerance == 0.01

    def test_unknown_device_raises(self):
        opts = PersistenceOptions()
        with pytest.raises(ValueError, match="Unknown device"):
            _apply_option_overrides(opts, device="quantum_gpu")


class TestValidateArray:
    def test_valid_2d(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_array(arr)
        assert result is arr
        assert result.flags.c_contiguous

    def test_1d_raises(self):
        with pytest.raises(ShapeMismatchError, match="2D array"):
            _validate_array(np.array([1.0, 2.0, 3.0]))

    def test_3d_raises(self):
        with pytest.raises(ShapeMismatchError, match="2D array"):
            _validate_array(np.zeros((2, 2, 2)))

    def test_empty_rows(self):
        with pytest.raises(InvalidArgumentError, match="cannot be empty"):
            _validate_array(np.empty((0, 3)))

    def test_empty_cols(self):
        with pytest.raises(InvalidArgumentError, match="cannot be empty"):
            _validate_array(np.empty((3, 0)))

    def test_nan_values_raises(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(InvalidArgumentError, match="NaN or infinite"):
            _validate_array(arr)

    def test_inf_values_raises(self):
        arr = np.array([[1.0, 2.0], [float("inf"), 4.0]])
        with pytest.raises(InvalidArgumentError, match="NaN or infinite"):
            _validate_array(arr)

    def test_neg_inf_raises(self):
        arr = np.array([[1.0, 2.0], [float("-inf"), 4.0]])
        with pytest.raises(InvalidArgumentError, match="NaN or infinite"):
            _validate_array(arr)

    def test_fortran_order_converted_to_c(self):
        arr = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert not arr.flags.c_contiguous
        result = _validate_array(arr)
        assert result.flags.c_contiguous
        assert result is not arr


class TestTensorToArray:
    @pytest.mark.torch
    def test_basic_conversion(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        t = _torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=_torch.float64)
        arr = _tensor_to_array(t, _torch)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))

    @pytest.mark.torch
    def test_1d_tensor_raises(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        t = _torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ShapeMismatchError, match="2D array"):
            _tensor_to_array(t, _torch)

    @pytest.mark.torch
    def test_empty_tensor_raises(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        t = _torch.empty((0, 3))
        with pytest.raises(InvalidArgumentError, match="cannot be empty"):
            _tensor_to_array(t, _torch)

    @pytest.mark.torch
    def test_detaches_gradient(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        t = _torch.tensor([[1.0, 2.0]], dtype=_torch.float64, requires_grad=True)
        arr = _tensor_to_array(t, _torch)
        assert not _torch.is_tensor(arr) or not getattr(arr, "requires_grad", False)
        np.testing.assert_array_equal(arr, np.array([[1.0, 2.0]]))

    @pytest.mark.torch
    def test_cuda_warns_and_moves_to_cpu(self):
        if not _cuda_usable():
            pytest.skip("CUDA not available or incompatible compute capability")
        t = _torch.tensor([[1.0, 2.0]], dtype=_torch.float64, device="cuda")
        with pytest.warns(UserWarning, match="GPU tensor moved to CPU"):
            arr = _tensor_to_array(t, _torch)
        np.testing.assert_array_equal(arr, np.array([[1.0, 2.0]]))

    @pytest.mark.torch
    def test_float32_to_float64_conversion(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        t = _torch.tensor([[1.0, 2.0]], dtype=_torch.float32)
        arr = _tensor_to_array(t, _torch, dtype="float64")
        assert arr.dtype == np.float64

    @pytest.mark.torch
    def test_explicit_dtype_override(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        t = _torch.tensor([[1, 2], [3, 4]], dtype=_torch.int64)
        arr = _tensor_to_array(t, _torch, dtype="float32")
        assert arr.dtype == np.float32

    @pytest.mark.torch
    def test_non_contiguous_tensor(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        t = _torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=_torch.float64)
        t = t.transpose(0, 1)
        assert not t.is_contiguous()
        arr = _tensor_to_array(t, _torch)
        np.testing.assert_array_equal(arr, np.array([[1.0, 3.0], [2.0, 4.0]]))


class TestToPointArray:
    def test_numpy_2d(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = _to_point_array(pts)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, pts.astype(np.float64))

    def test_numpy_with_explicit_dtype(self):
        pts = np.float32([[1.0, 2.0], [3.0, 4.0]])
        result = _to_point_array(pts, dtype="float32")
        assert result.dtype == np.float32

    def test_list_of_lists(self):
        pts = [[1.0, 2.0], [3.0, 4.0]]
        result = _to_point_array(pts)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_tuple_of_tuples(self):
        pts = ((1.0, 2.0), (3.0, 4.0))
        result = _to_point_array(pts)
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_list_int_to_float(self):
        pts = [[1, 2], [3, 4]]
        result = _to_point_array(pts)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_jagged_list_raises(self):
        with pytest.raises((ValueError, InvalidArgumentError)):
            _to_point_array([[1, 2], [3, 4, 5]])

    def test_non_numeric_object_raises(self):
        with pytest.raises((TypeError, ValueError, InvalidArgumentError)):
            _to_point_array([["a", "b"], ["c", "d"]])

    def test_1d_raises(self):
        with pytest.raises(ShapeMismatchError, match="2D array"):
            _to_point_array([1.0, 2.0, 3.0])

    def test_empty_raises(self):
        with pytest.raises((ShapeMismatchError, InvalidArgumentError)):
            _to_point_array([])

    def test_nan_raises(self):
        with pytest.raises(InvalidArgumentError, match="NaN"):
            _to_point_array([[1.0, np.nan]])

    def test_inf_raises(self):
        with pytest.raises(InvalidArgumentError, match="NaN"):
            _to_point_array([[1.0, float("inf")]])

    @pytest.mark.torch
    def test_torch_tensor_cpu(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        with patch(
            "pynerve._compute_pipeline._nerve_state",
            return_value=(None, None, _torch),
        ):
            t = _torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=_torch.float64)
            result = _to_point_array(t)
            np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    @pytest.mark.torch
    def test_torch_tensor_cuda_auto_backend(self):
        if not _cuda_usable():
            pytest.skip("CUDA not available or incompatible compute capability")
        with patch(
            "pynerve._compute_pipeline._nerve_state",
            return_value=(None, None, _torch),
        ):
            t = _torch.tensor([[1.0, 2.0]], dtype=_torch.float64, device="cuda")
            with pytest.warns(UserWarning, match="GPU tensor moved to CPU"):
                result = _to_point_array(t)
            np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))

    def test_validation_error_on_options_type(self):
        with pytest.raises(ValidationError):
            PersistenceOptions(max_dim="two")  # type: ignore[arg-type]

    def test_2d_empty_cols_array_raises(self):
        with pytest.raises(InvalidArgumentError, match="cannot be empty"):
            _to_point_array(np.empty((3, 0)))


class TestResolveOptions:
    def test_defaults(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _resolve_options(pts, None)
        assert isinstance(result, PersistenceOptions)
        assert result.max_dim == 2

    def test_custom_options_preserved(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        opts = PersistenceOptions(max_dim=3, threads=4)
        result = _resolve_options(pts, opts)
        assert result.max_dim == 3
        assert result.threads == 4

    def test_overrides_applied(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        opts = PersistenceOptions(max_dim=2)
        result = _resolve_options(pts, opts, max_dim=5)
        assert result.max_dim == 5

    def test_tensor_cpu_no_auto_backend_switch(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        opts = PersistenceOptions(backend=PersistenceBackend.CPU_EXACT)
        result = _resolve_options(pts, opts)
        assert result.backend == PersistenceBackend.CPU_EXACT

    @pytest.mark.torch
    def test_cuda_tensor_triggers_cuda_backend(self):
        if not _cuda_usable():
            pytest.skip("CUDA not available or incompatible compute capability")
        with patch(
            "pynerve._compute_pipeline._nerve_state",
            return_value=(None, None, _torch),
        ):
            t = _torch.tensor([[1.0, 2.0]], dtype=_torch.float64, device="cuda")
            opts = PersistenceOptions(backend=PersistenceBackend.CPU_EXACT)
            result = _resolve_options(t, opts)
            assert result.backend == PersistenceBackend.CUDA_HYBRID

    @pytest.mark.torch
    def test_cpu_tensor_keeps_cpu_backend(self):
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        with patch(
            "pynerve._compute_pipeline._nerve_state",
            return_value=(None, None, _torch),
        ):
            t = _torch.tensor([[1.0, 2.0]], dtype=_torch.float64)
            opts = PersistenceOptions(backend=PersistenceBackend.CPU_EXACT)
            result = _resolve_options(t, opts)
            assert result.backend == PersistenceBackend.CPU_EXACT

    def test_without_torch_installed_no_switch(self):
        with patch(
            "pynerve._compute_pipeline._nerve_state",
            return_value=(None, None, None),
        ):
            pts = np.array([[1.0, 2.0]])
            opts = PersistenceOptions(backend=PersistenceBackend.CPU_EXACT)
            result = _resolve_options(pts, opts)
            assert result.backend == PersistenceBackend.CPU_EXACT
