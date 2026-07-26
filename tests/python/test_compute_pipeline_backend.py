"""Tests for pynerve/_compute_pipeline.py and _compute_backend.py -- pipeline helpers, backend dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from pynerve._compute_backend import (
    _DEVICE_TO_BACKEND,
    _resolve_device_to_backend,
    _seed_rng,
    _to_events_list,
    _to_internal_options,
    _warn_device_overrides_backend,
)
from pynerve._compute_pipeline import (
    _clone_options,
    _is_likely_distance_matrix,
    _to_point_array,
    _validate_array,
    _validate_error_tolerance,
    _validate_max_dim,
    _validate_max_radius,
    _validate_threads,
)
from pynerve._fallback_classes import EventType, PersistenceBackend, PersistenceMode, PersistenceOptions
from pynerve.exceptions import InvalidArgumentError, ShapeMismatchError


class TestIsLikelyDistanceMatrix:
    def test_perfect_distance_matrix(self):
        dm = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        assert _is_likely_distance_matrix(dm) is True

    def test_non_square(self):
        dm = np.array([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
        assert _is_likely_distance_matrix(dm) is False

    def test_1d_returns_false(self):
        assert _is_likely_distance_matrix(np.array([1.0, 2.0])) is False

    def test_nonzero_diagonal(self):
        dm = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert _is_likely_distance_matrix(dm) is False

    def test_asymmetric(self):
        dm = np.array([[0.0, 1.0], [2.0, 0.0]])
        assert _is_likely_distance_matrix(dm) is False


class TestCloneOptions:
    def test_clone(self):
        opts = PersistenceOptions(max_dim=3)
        cloned = _clone_options(opts)
        assert cloned.max_dim == 3
        assert cloned is not opts

    def test_none_creates_default(self):
        cloned = _clone_options(None)
        assert isinstance(cloned, PersistenceOptions)


class TestValidateMaxDim:
    def test_valid(self):
        assert _validate_max_dim(0) == 0
        assert _validate_max_dim(5) == 5

    def test_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="non-negative"):
            _validate_max_dim(-1)

    def test_float_converts(self):
        assert _validate_max_dim(3.0) == 3


class TestValidateMaxRadius:
    def test_basic(self):
        assert _validate_max_radius(5.0) == 5.0

    def test_inf_capped(self):
        result = _validate_max_radius(float("inf"), cap=100.0)
        assert np.isfinite(result)

    def test_non_finite_raises(self):
        with pytest.raises(InvalidArgumentError, match="finite"):
            _validate_max_radius(float("nan"))

    def test_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="non-negative"):
            _validate_max_radius(-1.0)


class TestValidateThreads:
    def test_valid(self):
        assert _validate_threads(4) == 4

    def test_zero_raises(self):
        with pytest.raises(InvalidArgumentError, match="positive"):
            _validate_threads(0)


class TestValidateErrorTolerance:
    def test_valid(self):
        assert _validate_error_tolerance(0.001) == 0.001

    def test_invalid_raises(self):
        with pytest.raises(InvalidArgumentError, match="finite"):
            _validate_error_tolerance(-0.1)


class TestValidateArray:
    def test_valid(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_array(arr)
        assert result.shape == (2, 2)

    def test_non_2d_raises(self):
        with pytest.raises(ShapeMismatchError):
            _validate_array(np.array([1.0, 2.0]))

    def test_empty_raises(self):
        with pytest.raises(InvalidArgumentError, match="empty"):
            _validate_array(np.array([[]]))


class TestToPointArray:
    def test_numpy(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _to_point_array(arr)
        assert result.shape == (2, 2)

    def test_list(self):
        result = _to_point_array([[1.0, 2.0], [3.0, 4.0]])
        assert result.shape == (2, 2)


class TestToEventsList:
    def test_event_type(self):
        events = [(EventType.ADD, [0, 1])]
        result = _to_events_list(events)
        assert result == [("add", [0, 1])]

    def test_string_type(self):
        events = [("remove", [0])]
        result = _to_events_list(events)
        assert result == [("remove", [0])]

    def test_invalid_type_raises(self):
        with pytest.raises(Exception, match="event type"):
            _to_events_list([("bad", [0])])


class TestResolveDeviceToBackend:
    def test_cpu(self):
        backend = _resolve_device_to_backend("cpu")
        assert backend == PersistenceBackend.CPU_ADAPTIVE_ACCELERATION

    def test_cuda(self):
        backend = _resolve_device_to_backend("cuda")
        assert backend == PersistenceBackend.CUDA_HYBRID

    def test_cuda_with_id(self):
        backend = _resolve_device_to_backend("cuda:0")
        assert backend == PersistenceBackend.CUDA_HYBRID

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError, match="Unknown device"):
            _resolve_device_to_backend("tpu")


class TestWarnDeviceOverridesBackend:
    def test_emits_warning(self):
        with pytest.warns(UserWarning, match="device.*precedence"):
            _warn_device_overrides_backend("cuda", PersistenceBackend.CPU_EXACT)


class TestSeedRng:
    def test_valid(self):
        _seed_rng(42)

    def test_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="non-negative"):
            _seed_rng(-1)
