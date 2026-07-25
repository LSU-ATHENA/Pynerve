"""Tests for _compute_pipeline.py and _compute_engine.py — validation, tensor conversion, engine selection."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

import torch

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestIsLikelyDistanceMatrix:
    """Covers _compute_pipeline._is_likely_distance_matrix."""

    def test_valid_distance_matrix(self):
        from pynerve._compute_pipeline import _is_likely_distance_matrix
        dm = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        assert _is_likely_distance_matrix(dm) is True

    def test_non_square(self):
        from pynerve._compute_pipeline import _is_likely_distance_matrix
        assert _is_likely_distance_matrix(np.zeros((3, 4))) is False

    def test_non_zero_diagonal(self):
        from pynerve._compute_pipeline import _is_likely_distance_matrix
        dm = np.array([[1, 2], [2, 0]], dtype=float)
        assert _is_likely_distance_matrix(dm) is False

    def test_non_symmetric(self):
        from pynerve._compute_pipeline import _is_likely_distance_matrix
        dm = np.array([[0, 1], [2, 0]], dtype=float)
        assert _is_likely_distance_matrix(dm) is False

    def test_too_small(self):
        from pynerve._compute_pipeline import _is_likely_distance_matrix
        assert _is_likely_distance_matrix(np.zeros((1, 1))) is False

    def test_non_2d(self):
        from pynerve._compute_pipeline import _is_likely_distance_matrix
        assert _is_likely_distance_matrix(np.zeros((2, 2, 2))) is False


class TestPipelineValidators:
    """Covers _compute_pipeline validation helpers."""

    def test_validate_max_dim_valid(self):
        from pynerve._compute_pipeline import _validate_max_dim
        assert _validate_max_dim(2) == 2
        assert _validate_max_dim(0) == 0

    def test_validate_max_dim_negative(self):
        from pynerve._compute_pipeline import _validate_max_dim
        with pytest.raises(Exception, match="non-negative"):
            _validate_max_dim(-1)

    def test_validate_max_radius_valid(self):
        from pynerve._compute_pipeline import _validate_max_radius
        assert _validate_max_radius(1.0) == 1.0

    def test_validate_max_radius_inf(self):
        from pynerve._compute_pipeline import _validate_max_radius
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _validate_max_radius(float("inf"))
        assert result > 0
        assert np.isfinite(result)

    def test_validate_max_radius_negative(self):
        from pynerve._compute_pipeline import _validate_max_radius
        with pytest.raises(Exception, match="non-negative"):
            _validate_max_radius(-1.0)

    def test_validate_max_radius_nan(self):
        from pynerve._compute_pipeline import _validate_max_radius
        with pytest.raises(Exception, match="finite"):
            _validate_max_radius(float("nan"))

    def test_validate_threads_valid(self):
        from pynerve._compute_pipeline import _validate_threads
        assert _validate_threads(4) == 4

    def test_validate_threads_zero(self):
        from pynerve._compute_pipeline import _validate_threads
        with pytest.raises(Exception, match="positive"):
            _validate_threads(0)

    def test_validate_error_tolerance_valid(self):
        from pynerve._compute_pipeline import _validate_error_tolerance
        assert _validate_error_tolerance(0.01) == 0.01

    def test_validate_error_tolerance_negative(self):
        from pynerve._compute_pipeline import _validate_error_tolerance
        with pytest.raises(Exception, match="non-negative"):
            _validate_error_tolerance(-0.01)

    def test_validate_error_tolerance_nan(self):
        from pynerve._compute_pipeline import _validate_error_tolerance
        with pytest.raises(Exception, match="finite"):
            _validate_error_tolerance(float("nan"))


class TestValidateArray:
    """Covers _compute_pipeline._validate_array."""

    def test_valid_2d(self):
        from pynerve._compute_pipeline import _validate_array
        arr = np.random.rand(10, 3)
        result = _validate_array(arr)
        assert result.shape == (10, 3)

    def test_non_2d(self):
        from pynerve._compute_pipeline import _validate_array
        with pytest.raises(Exception, match="2D"):
            _validate_array(np.zeros((2, 2, 2)))

    def test_empty_rows(self):
        from pynerve._compute_pipeline import _validate_array
        with pytest.raises(Exception, match="empty"):
            _validate_array(np.empty((0, 3)))

    def test_empty_cols(self):
        from pynerve._compute_pipeline import _validate_array
        with pytest.raises(Exception, match="empty"):
            _validate_array(np.empty((3, 0)))

    def test_nan_values(self):
        from pynerve._compute_pipeline import _validate_array
        arr = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with pytest.raises(Exception, match="NaN"):
            _validate_array(arr)

    def test_inf_values(self):
        from pynerve._compute_pipeline import _validate_array
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])
        with pytest.raises(Exception, match="NaN|infinite"):
            _validate_array(arr)

    def test_non_contiguous(self):
        from pynerve._compute_pipeline import _validate_array
        arr = np.asfortranarray(np.random.rand(10, 3))
        result = _validate_array(arr)
        assert result.flags.c_contiguous


class TestTensorToArray:
    """Covers _compute_pipeline._tensor_to_array."""

    def test_valid_tensor(self):
        from pynerve._compute_pipeline import _tensor_to_array
        t = torch.rand(10, 3)
        result = _tensor_to_array(t, torch, dtype=None)
        assert result.shape == (10, 3)

    def test_non_2d_tensor(self):
        from pynerve._compute_pipeline import _tensor_to_array
        t = torch.rand(2, 3, 4)
        with pytest.raises(Exception, match="2D"):
            _tensor_to_array(t, torch, dtype=None)

    def test_empty_tensor(self):
        from pynerve._compute_pipeline import _tensor_to_array
        t = torch.empty(0, 3)
        with pytest.raises(Exception, match="empty"):
            _tensor_to_array(t, torch, dtype=None)

    def test_dtype_conversion(self):
        from pynerve._compute_pipeline import _tensor_to_array
        t = torch.rand(10, 3, dtype=torch.float32)
        result = _tensor_to_array(t, torch, dtype="float64")
        assert result.dtype == np.float64


class TestToPointArray:
    """Covers _compute_pipeline._to_point_array."""

    def test_from_numpy(self):
        from pynerve._compute_pipeline import _to_point_array
        arr = np.random.rand(10, 3)
        result = _to_point_array(arr)
        assert result.shape == (10, 3)

    def test_from_list(self):
        from pynerve._compute_pipeline import _to_point_array
        result = _to_point_array([[1, 2, 3], [4, 5, 6]])
        assert result.shape == (2, 3)

    def test_from_tensor(self):
        from pynerve._compute_pipeline import _to_point_array
        t = torch.rand(10, 3)
        result = _to_point_array(t)
        assert result.shape == (10, 3)

    def test_jagged_list_raises(self):
        from pynerve._compute_pipeline import _to_point_array
        with pytest.raises((ValueError, Exception)):
            _to_point_array([[1, 2], [3, 4, 5]])


class TestCloneOptions:
    """Covers _compute_pipeline._clone_options."""

    def test_clone_none(self):
        from pynerve._compute_pipeline import _clone_options
        opts = _clone_options(None)
        assert opts is not None

    def test_clone_returns_copy(self):
        from pynerve._compute_pipeline import _clone_options
        from pynerve._fallback_classes import PersistenceOptions
        opts = PersistenceOptions(max_dim=3)
        cloned = _clone_options(opts)
        assert cloned.max_dim == 3
        assert cloned is not opts


class TestEngineSelection:
    """Covers _compute_engine._auto_select_engine."""

    def test_small_dataset(self):
        from pynerve._compute_engine import _auto_select_engine
        from pynerve._fallback_classes import PersistenceEngine
        result = _auto_select_engine(500, 3, None, None, None)
        assert result == PersistenceEngine.PH0

    def test_small_dataset_high_dim(self):
        from pynerve._compute_engine import _auto_select_engine
        from pynerve._fallback_classes import PersistenceEngine
        result = _auto_select_engine(500, 4, None, None, None)
        assert result == PersistenceEngine.PH3

    def test_medium_dataset(self):
        from pynerve._compute_engine import _auto_select_engine
        from pynerve._fallback_classes import PersistenceEngine
        result = _auto_select_engine(5000, 3, None, None, None)
        assert result == PersistenceEngine.PH3

    def test_large_dataset(self):
        from pynerve._compute_engine import _auto_select_engine
        from pynerve._fallback_classes import PersistenceEngine
        result = _auto_select_engine(50000, 3, None, None, None)
        assert result == PersistenceEngine.PH4

    def test_very_large_dataset(self):
        from pynerve._compute_engine import _auto_select_engine
        from pynerve._fallback_classes import PersistenceEngine
        result = _auto_select_engine(500000, 3, None, None, None)
        assert result == PersistenceEngine.PH5

    def test_huge_dataset(self):
        from pynerve._compute_engine import _auto_select_engine
        from pynerve._fallback_classes import PersistenceEngine
        result = _auto_select_engine(2000000, 3, None, None, None)
        assert result == PersistenceEngine.PH6

    def test_approx_mode(self):
        from pynerve._compute_engine import _auto_select_engine
        from pynerve._fallback_classes import PersistenceEngine, PersistenceMode
        result = _auto_select_engine(100, 3, None, PersistenceMode.APPROX, None)
        assert result == PersistenceEngine.PH5

    def test_cuda_device_no_cuda(self):
        import torch
        from pynerve._compute_engine import _auto_select_engine
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine")
        with pytest.raises(RuntimeError, match="CUDA"):
            _auto_select_engine(100, 3, "cuda:0", None, None)


class TestRequireCore:
    """Covers _compute_engine._require_core."""

    def test_raises_without_core(self):
        from pynerve._compute_engine import _require_core
        from pynerve._persistence_result import _nerve_state
        # _nerve_state returns (core, import_error, pytorch); with mocks, core is a MagicMock
        core, _, _ = _nerve_state()
        if core is not None:
            pytest.skip("C++ core is available (mocked or real)")
        with pytest.raises(Exception, match="BackendRequired|pynerve_internal"):
            _require_core()


class TestResolveEngineFunc:
    """Covers _compute_engine._resolve_engine_func."""

    def test_auto_engine(self):
        from pynerve._compute_engine import _resolve_engine_func
        from pynerve._fallback_classes import PersistenceEngine
        core = MagicMock()
        core.compute_persistence = lambda *a, **k: {}
        result = _resolve_engine_func(core, PersistenceEngine.AUTO, n_points=100, dim=3)
        assert callable(result)

    def test_specific_engine_ph4(self):
        from pynerve._compute_engine import _resolve_engine_func
        from pynerve._fallback_classes import PersistenceEngine
        core = MagicMock()
        core.compute_persistence_up_to_dim_4 = lambda *a, **k: {}
        result = _resolve_engine_func(core, PersistenceEngine.PH4)
        assert callable(result)

    def test_fallback_to_compute_persistence(self):
        from pynerve._compute_engine import _resolve_engine_func
        from pynerve._fallback_classes import PersistenceEngine
        core = MagicMock()
        core.compute_persistence = lambda *a, **k: {}
        result = _resolve_engine_func(core, PersistenceEngine.PH4)
        assert callable(result)


class TestComputeBackendHelpers:
    """Covers _compute_backend.py helper functions."""

    def test_resolve_device_cpu(self):
        from pynerve._compute_backend import _resolve_device_to_backend
        from pynerve._fallback_classes import PersistenceBackend
        result = _resolve_device_to_backend("cpu")
        assert result == PersistenceBackend.CPU_ADAPTIVE_ACCELERATION

    def test_resolve_device_cuda(self):
        from pynerve._compute_backend import _resolve_device_to_backend
        from pynerve._fallback_classes import PersistenceBackend
        result = _resolve_device_to_backend("cuda")
        assert result == PersistenceBackend.CUDA_HYBRID

    def test_resolve_device_cuda_with_id(self):
        from pynerve._compute_backend import _resolve_device_to_backend
        from pynerve._fallback_classes import PersistenceBackend
        result = _resolve_device_to_backend("cuda:1")
        assert result == PersistenceBackend.CUDA_HYBRID

    def test_resolve_device_unknown(self):
        from pynerve._compute_backend import _resolve_device_to_backend
        with pytest.raises(ValueError, match="Unknown device"):
            _resolve_device_to_backend("tpu")

    def test_seed_rng_valid(self):
        from pynerve._compute_backend import _seed_rng
        _seed_rng(42)

    def test_seed_rng_negative(self):
        from pynerve._compute_backend import _seed_rng
        with pytest.raises(Exception, match="non-negative"):
            _seed_rng(-1)

    def test_to_events_list_with_strings(self):
        from pynerve._compute_backend import _to_events_list
        events = [("add", [0, 1, 2]), ("remove", [1])]
        result = _to_events_list(events)
        assert result == [("add", [0, 1, 2]), ("remove", [1])]

    def test_to_events_list_with_enum(self):
        from pynerve._compute_backend import _to_events_list
        from pynerve._fallback_classes import EventType
        events = [(EventType.ADD, [0, 1]), (EventType.REMOVE, [2])]
        result = _to_events_list(events)
        assert result[0][0] == "add"
        assert result[1][0] == "remove"

    def test_to_events_list_invalid_type(self):
        from pynerve._compute_backend import _to_events_list
        with pytest.raises(Exception, match="event type"):
            _to_events_list([("invalid", [0])])

    def test_try_as_ndarray_numpy(self):
        from pynerve._compute_backend import _try_as_ndarray
        arr = np.random.rand(10, 3)
        result = _try_as_ndarray(arr)
        assert result is not None
        assert result.shape == (10, 3)

    def test_try_as_ndarray_tensor(self):
        from pynerve._compute_backend import _try_as_ndarray
        t = torch.rand(10, 3)
        result = _try_as_ndarray(t)
        assert result is not None
        assert result.shape == (10, 3)

    def test_try_as_ndarray_invalid(self):
        from pynerve._compute_backend import _try_as_ndarray
        result = _try_as_ndarray("not an array")
        assert result is None

    def test_try_as_ndarray_1d(self):
        from pynerve._compute_backend import _try_as_ndarray
        # _try_as_ndarray returns ndarray inputs directly without dimension check
        result = _try_as_ndarray(np.array([1, 2, 3]))
        assert result is not None  # 1D ndarray is returned as-is
        assert result.ndim == 1

    def test_warn_device_overrides(self):
        from pynerve._compute_backend import _warn_device_overrides_backend
        from pynerve._fallback_classes import PersistenceBackend
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_device_overrides_backend("cpu", PersistenceBackend.CUDA_HYBRID)
            assert len(w) == 1
            assert "device" in str(w[0].message).lower()
