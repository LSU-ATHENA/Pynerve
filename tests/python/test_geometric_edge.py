"""Tests for _validation/_geometric.py -- remaining edge cases."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pynerve._validation._geometric import (
    validate_device_spec,
    validate_diagram_array,
    validate_finite_deaths,
    validate_shape,
    validate_shape_tuple,
)
from pynerve.exceptions import ShapeError, ValidationError


class TestValidateFiniteDeathsEdge:
    def test_empty_array(self):
        validate_finite_deaths(np.array([], dtype=np.float64))

    def test_mixed_valid(self):
        deaths = np.array([1.0, float("inf"), 2.0])
        validate_finite_deaths(deaths)

    def test_all_inf(self):
        deaths = np.array([float("inf"), float("inf")])
        validate_finite_deaths(deaths)


class TestValidateDiagramArrayEdge:
    def test_empty_array_returns_empty(self):
        arr = np.empty((0, 3), dtype=np.float64)
        result = validate_diagram_array(arr)
        assert result.shape == (0, 3)

    def test_requires_dims_with_3_cols(self):
        arr = np.array([[0.0, 1.0, 0]], dtype=np.float64)
        result = validate_diagram_array(arr, require_dims=True)
        assert result.shape == (1, 3)

    def test_requires_dims_with_2_cols_raises(self):
        arr = np.array([[0.0, 1.0]], dtype=np.float64)
        with pytest.raises(ShapeError, match="at least 3 columns"):
            validate_diagram_array(arr, require_dims=True)

    def test_non_contiguous_array(self):
        arr = np.array([[0.0, 1.0, 0], [1.0, 2.0, 0]], dtype=np.float64)
        non_contig = arr[:, ::-1].copy()
        non_contig = np.asfortranarray(non_contig)
        result = validate_diagram_array(non_contig[:, ::-1])
        assert result.flags.c_contiguous

    def test_float_dimensions_accepted(self):
        arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        result = validate_diagram_array(arr)
        assert result.shape == (1, 3)


class TestValidateShapeEdge:
    def test_single_integer(self):
        result = validate_shape(5)
        assert result == (5,)

    def test_sequence_of_ints(self):
        result = validate_shape((10, 20, 30))
        assert result == (10, 20, 30)

    def test_allow_infer_with_neg_one(self):
        result = validate_shape((10, -1), allow_infer=True)
        assert result == (10, -1)

    def test_allow_infer_multiple_neg_ones_raises(self):
        with pytest.raises(ShapeError, match="at most one"):
            validate_shape((-1, -1), allow_infer=True)

    def test_negative_dim_raises(self):
        with pytest.raises(ShapeError, match="non-negative"):
            validate_shape((-5,))

    def test_empty_sequence_raises(self):
        with pytest.raises(ShapeError, match="at least one"):
            validate_shape(())

    def test_bool_raises(self):
        with pytest.raises(ShapeError, match="integer"):
            validate_shape(True)

    def test_string_raises(self):
        with pytest.raises(ShapeError, match="integer or sequence"):
            validate_shape("abc")


class TestValidateShapeTupleEdge:
    def test_none_returns_none(self):
        assert validate_shape_tuple(None, "x") is None

    def test_valid_tuple(self):
        result = validate_shape_tuple((10, 20), "x")
        assert result == (10, 20)

    def test_negative_raises(self):
        with pytest.raises(ShapeError, match="non-negative"):
            validate_shape_tuple((-1,), "x")

    def test_bool_in_tuple_raises(self):
        with pytest.raises(ShapeError, match="integer"):
            validate_shape_tuple((True,), "x")


class TestValidateDeviceSpecEdge:
    def test_cpu_valid(self):
        validate_device_spec("cpu")

    def test_cuda_valid(self):
        validate_device_spec("cuda")

    def test_cuda_with_index(self):
        validate_device_spec("cuda:0")
        validate_device_spec("cuda:3")

    def test_empty_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            validate_device_spec("")

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValidationError, match="Unknown"):
            validate_device_spec("metal:0")

    def test_negative_index_raises(self):
        with pytest.raises(ValidationError, match="Invalid"):
            validate_device_spec("cuda:-1")

    def test_non_integer_index_raises(self):
        with pytest.raises(ValidationError, match="Invalid"):
            validate_device_spec("cuda:abc")
