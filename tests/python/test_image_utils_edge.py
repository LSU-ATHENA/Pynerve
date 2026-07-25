"""Edge case tests for pynerve/_image_utils.py — persistence image computation."""

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
    def test_numpy_2d(self):
        arr = np.array([[0.0, 1.0, 0], [0.5, 2.0, 0]])
        result = _to_diagram_array(arr)
        assert result.shape == (2, 3)

    def test_empty_array(self):
        arr = np.empty((0, 3))
        result = _to_diagram_array(arr)
        assert result.shape == (0, 3)

    def test_has_pairs_array(self):
        class Fake:
            pairs_array = np.array([[0.0, 1.0, 0]])
        result = _to_diagram_array(Fake())
        assert result.shape == (1, 3)

    def test_has_pairs(self):
        class Fake:
            pairs = np.array([[0.0, 1.0, 0]])
        result = _to_diagram_array(Fake())
        assert result.shape == (1, 3)

    def test_generic(self):
        result = _to_diagram_array([[0.0, 1.0, 0]])
        assert result.shape == (1, 3)


class TestNormalizeImageResolution:
    def test_int(self):
        h, w = _normalize_image_resolution(50)
        assert h == 50 and w == 50

    def test_tuple(self):
        h, w = _normalize_image_resolution((30, 40))
        assert h == 30 and w == 40

    def test_invalid_tuple_raises(self):
        with pytest.raises(InvalidArgumentError, match="resolution tuple"):
            _normalize_image_resolution((10,))

    def test_zero_raises(self):
        with pytest.raises(InvalidArgumentError, match="positive"):
            _normalize_image_resolution(0)


class TestFiniteRange:
    def test_from_values(self):
        vals = np.array([1.0, 5.0])
        low, high = _finite_range(vals, None)
        assert low == 1.0 and high == 5.0

    def test_explicit(self):
        vals = np.array([1.0])
        low, high = _finite_range(vals, (0.0, 10.0))
        assert low == 0.0 and high == 10.0

    def test_empty_values(self):
        vals = np.array([])
        low, high = _finite_range(vals, None)
        assert low == 0.0 and high == 1.0

    def test_equal_bounds_expanded(self):
        vals = np.array([5.0, 5.0])
        low, high = _finite_range(vals, (5.0, 5.0))
        assert high == low + 1.0


class TestPersistenceImageEdge:
    def test_empty_diagram(self):
        result = persistence_image(np.empty((0, 3)))
        assert result.shape == (20, 20)
        assert np.all(result == 0.0)

    def test_custom_resolution_int(self):
        diagram = np.array([[0.0, 1.0, 0]])
        result = persistence_image(diagram, resolution=32)
        assert result.shape == (32, 32)

    def test_custom_sigma(self):
        diagram = np.array([[0.0, 1.0, 0], [0.5, 2.0, 0]])
        result = persistence_image(diagram, sigma=0.5)
        assert result.shape == (20, 20)

    def test_weight_uniform(self):
        diagram = np.array([[0.0, 1.0, 0]])
        result = persistence_image(diagram, weight="uniform")
        assert result.shape == (20, 20)

    def test_invalid_weight_raises(self):
        with pytest.raises(InvalidArgumentError, match="weight"):
            persistence_image(np.array([[0.0, 1.0, 0]]), weight="bad")

    def test_invalid_sigma_raises(self):
        with pytest.raises(InvalidArgumentError, match="sigma"):
            persistence_image(np.array([[0.0, 1.0, 0]]), sigma=-1.0)

    def test_inf_deaths_filtered(self):
        diagram = np.array([[0.0, np.inf, 0], [1.0, 2.0, 0]])
        result = persistence_image(diagram)
        assert result.shape == (20, 20)

    def test_birth_range(self):
        diagram = np.array([[0.0, 1.0, 0], [2.0, 5.0, 0]])
        result = persistence_image(diagram, birth_range=(0.0, 3.0))
        assert result.shape == (20, 20)
