"""Tests for _image_utils.py -- persistence image computation."""

from __future__ import annotations

import numpy as np
import pytest
from pynerve._image_utils import (
    _finite_range,
    _normalize_image_resolution,
    _to_diagram_array,
    persistence_image,
)
from pynerve.exceptions import InvalidArgumentError


class TestToDiagramArray:
    def test_numpy_2d_array(self):
        arr = np.array([[0.0, 1.0, 0], [1.0, 2.0, 1]], dtype=np.float64)
        result = _to_diagram_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_list_converted(self):
        data = [[0.0, 1.0, 0], [1.0, 2.0, 1]]
        result = _to_diagram_array(data)
        assert result.shape == (2, 3)
        assert result.dtype == np.float64

    def test_empty_array_returns_empty(self):
        arr = np.empty((0, 3), dtype=np.float64)
        result = _to_diagram_array(arr)
        assert result.shape == (0, 3)

    def test_empty_list_returns_empty(self):
        result = _to_diagram_array([])
        assert result.shape == (0, 3)

    def test_scalar_raises(self):
        with pytest.raises(Exception):
            _to_diagram_array(42)

    def test_none_raises(self):
        with pytest.raises(Exception):
            _to_diagram_array(None)


class TestNormalizeImageResolution:
    def test_int_resolution(self):
        h, w = _normalize_image_resolution(10)
        assert h == 10
        assert w == 10

    def test_tuple_resolution(self):
        h, w = _normalize_image_resolution((10, 20))
        assert h == 10
        assert w == 20

    def test_zero_raises(self):
        with pytest.raises(InvalidArgumentError, match="positive"):
            _normalize_image_resolution(0)

    def test_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="positive"):
            _normalize_image_resolution(-1)

    def test_tuple_zero_raises(self):
        with pytest.raises(InvalidArgumentError, match="positive"):
            _normalize_image_resolution((0, 10))

    def test_tuple_negative_raises(self):
        with pytest.raises(InvalidArgumentError, match="positive"):
            _normalize_image_resolution((10, -5))

    def test_wrong_length_tuple_raises(self):
        with pytest.raises(InvalidArgumentError, match="tuple"):
            _normalize_image_resolution((10, 20, 30))

    def test_single_element_tuple_raises(self):
        with pytest.raises(InvalidArgumentError, match="tuple"):
            _normalize_image_resolution((10,))


class TestFiniteRange:
    def test_explicit_values(self):
        low, high = _finite_range(np.array([]), (0.0, 1.0))
        assert low == 0.0
        assert high == 1.0

    def test_from_values(self):
        arr = np.array([1.0, 5.0, 3.0])
        low, high = _finite_range(arr, None)
        assert low == 1.0
        assert high == 5.0

    def test_empty_values_without_explicit(self):
        low, high = _finite_range(np.array([]), None)
        assert low == 0.0
        assert high == 1.0

    def test_equal_bounds_expanded(self):
        arr = np.array([5.0, 5.0])
        low, high = _finite_range(arr, None)
        assert high == low + 1.0

    def test_explicit_equal_bounds_expanded(self):
        low, high = _finite_range(np.array([]), (3.0, 3.0))
        assert high == low + 1.0

    def test_reversed_bounds_raises(self):
        with pytest.raises(InvalidArgumentError, match="ordered"):
            _finite_range(np.array([]), (5.0, 1.0))

    def test_nan_bounds_raises(self):
        with pytest.raises(InvalidArgumentError, match="finite"):
            _finite_range(np.array([]), (float("nan"), 1.0))


class TestPersistenceImage:
    def test_basic_output_shape(self):
        diagram = np.array([[0.0, 1.0, 0], [0.5, 2.0, 0]], dtype=np.float64)
        image = persistence_image(diagram, resolution=10, sigma=1.0)
        assert image.shape == (10, 10)

    def test_rectangular_resolution(self):
        diagram = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        image = persistence_image(diagram, resolution=(8, 12), sigma=1.0)
        assert image.shape == (8, 12)

    def test_empty_diagram_returns_zeros(self):
        diagram = np.empty((0, 3), dtype=np.float64)
        image = persistence_image(diagram, resolution=5, sigma=1.0)
        assert image.shape == (5, 5)
        assert np.all(image == 0.0)

    def test_no_finite_pairs_returns_zeros(self):
        diagram = np.array([[0.0, float("inf"), 0]], dtype=np.float64)
        image = persistence_image(diagram, resolution=5, sigma=1.0)
        assert np.all(image == 0.0)

    def test_persistence_weight(self):
        diagram = np.array([[0.0, 1.0, 0], [0.0, 10.0, 0]], dtype=np.float64)
        image_p = persistence_image(diagram, resolution=10, sigma=1.0, weight="persistence")
        image_u = persistence_image(diagram, resolution=10, sigma=1.0, weight="uniform")
        assert image_p.shape == image_u.shape

    def test_invalid_weight_raises(self):
        diagram = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        with pytest.raises(InvalidArgumentError, match="weight"):
            persistence_image(diagram, weight="invalid")

    def test_nan_sigma_raises(self):
        diagram = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        with pytest.raises(InvalidArgumentError, match="sigma"):
            persistence_image(diagram, sigma=float("nan"))

    def test_negative_sigma_raises(self):
        diagram = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        with pytest.raises(InvalidArgumentError, match="sigma"):
            persistence_image(diagram, sigma=-1.0)

    def test_zero_sigma_raises(self):
        diagram = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        with pytest.raises(InvalidArgumentError, match="sigma"):
            persistence_image(diagram, sigma=0.0)

    def test_default_resolution(self):
        diagram = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        image = persistence_image(diagram, sigma=1.0)
        assert image.shape == (20, 20)

    def test_birth_range_restricts(self):
        diagram = np.array([[0.0, 1.0, 0], [5.0, 6.0, 0]], dtype=np.float64)
        image = persistence_image(diagram, resolution=10, sigma=0.5, birth_range=(0.0, 1.0))
        assert image.shape == (10, 10)

    def test_persistence_range_restricts(self):
        diagram = np.array([[0.0, 0.5, 0], [0.0, 10.0, 0]], dtype=np.float64)
        image = persistence_image(diagram, resolution=10, sigma=1.0, persistence_range=(0.0, 1.0))
        assert image.shape == (10, 10)

    def test_image_is_float64(self):
        diagram = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        image = persistence_image(diagram, sigma=1.0)
        assert image.dtype == np.float64

    def test_list_input(self):
        diagram = [[0.0, 1.0, 0], [0.5, 2.0, 1]]
        image = persistence_image(diagram, resolution=8, sigma=1.0)
        assert image.shape == (8, 8)
