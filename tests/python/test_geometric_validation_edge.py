"""Edge case tests for _validation/_geometric.py -- validate_shape, validate_device_spec, validate_shape_tuple."""

from __future__ import annotations

import numpy as np
import pytest

from pynerve._validation._geometric import (
    validate_device_spec,
    validate_diagram_array,
    validate_finite_deaths,
    validate_points,
    validate_shape,
    validate_shape_tuple,
)
from pynerve.exceptions import ShapeError, ValidationError


class TestValidateFiniteDeathsEdge:
    def test_accepts_valid(self):
        deaths = np.array([1.0, 2.0, float("inf")])
        validate_finite_deaths(deaths)  # should not raise

    def test_empty_array(self):
        validate_finite_deaths(np.array([]))

    def test_negative_inf_raises(self):
        deaths = np.array([float("-inf")])
        with pytest.raises(ValidationError, match="positive infinity"):
            validate_finite_deaths(deaths)

    def test_nan_raises(self):
        deaths = np.array([float("nan")])
        with pytest.raises(ValidationError, match="NaN"):
            validate_finite_deaths(deaths)


class TestValidatePointsEdge:
    def test_torch_tensor_accepts(self):
        torch = pytest.importorskip("torch")
        pts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = validate_points(pts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_torch_1d_raises(self):
        torch = pytest.importorskip("torch")
        pts = torch.tensor([1.0, 2.0])
        with pytest.raises(ShapeError, match="2D"):
            validate_points(pts)

    def test_torch_zero_cols_raises(self):
        torch = pytest.importorskip("torch")
        pts = torch.empty((3, 0))
        with pytest.raises(ShapeError, match="coordinate"):
            validate_points(pts)

    def test_torch_nan_raises(self):
        torch = pytest.importorskip("torch")
        pts = torch.tensor([[float("nan"), 2.0]])
        with pytest.raises(ValidationError, match="finite"):
            validate_points(pts)

    def test_numpy_non_2d_raises(self):
        with pytest.raises(ShapeError, match="2D"):
            validate_points(np.array([1.0, 2.0]))

    def test_numpy_zero_cols_raises(self):
        with pytest.raises(ShapeError, match="coordinate"):
            validate_points(np.empty((3, 0)))

    def test_numpy_nan_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_points(np.array([[float("nan"), 2.0]]))


class TestValidateShapeEdge:
    def test_integer_shape(self):
        result = validate_shape(5)
        assert result == (5,)

    def test_sequence_shape(self):
        result = validate_shape([2, 3, 4])
        assert result == (2, 3, 4)

    def test_tuple_shape(self):
        result = validate_shape((1, 2))
        assert result == (1, 2)

    def test_np_int_accepts(self):
        result = validate_shape(np.int64(5))
        assert result == (5,)

    def test_allow_infer(self):
        result = validate_shape((3, -1, 5), allow_infer=True)
        assert result == (3, -1, 5)

    def test_allow_infer_none_used(self):
        result = validate_shape((3, 4), allow_infer=True)
        assert result == (3, 4)

    def test_multiple_infer_raises(self):
        with pytest.raises(ShapeError, match="at most one inferred"):
            validate_shape((-1, -1), allow_infer=True)

    def test_negative_dimension_raises(self):
        with pytest.raises(ShapeError, match="non-negative"):
            validate_shape((-2, 3))

    def test_empty_sequence_raises(self):
        with pytest.raises(ShapeError, match="at least one"):
            validate_shape(())

    def test_string_raises(self):
        with pytest.raises(ShapeError, match="integer"):
            validate_shape("abc")  # type: ignore[arg-type]

    def test_bool_in_sequence_raises(self):
        with pytest.raises(ShapeError, match="integers"):
            validate_shape((True, 5))  # type: ignore[list-item]

    def test_infer_without_allow_raises(self):
        with pytest.raises(ShapeError, match="non-negative"):
            validate_shape((-1, 3))

    def test_bool_as_shape_raises(self):
        with pytest.raises(ShapeError, match="integer"):
            validate_shape(True)  # type: ignore[arg-type]


class TestValidateShapeTupleEdge:
    def test_none_returns_none(self):
        assert validate_shape_tuple(None, "shape") is None

    def test_valid_tuple(self):
        result = validate_shape_tuple((1, 2, 3), "shape")
        assert result == (1, 2, 3)

    def test_empty_tuple(self):
        result = validate_shape_tuple((), "shape")
        assert result == ()

    def test_negative_raises(self):
        with pytest.raises(ShapeError, match="non-negative"):
            validate_shape_tuple((-1,), "shape")

    def test_non_sequence_raises(self):
        with pytest.raises(ShapeError, match="sequence"):
            validate_shape_tuple(42, "shape")  # type: ignore[arg-type]

    def test_bool_in_tuple_raises(self):
        with pytest.raises(ShapeError, match="integers"):
            validate_shape_tuple((True,), "shape")  # type: ignore[list-item]


class TestValidateDeviceSpecEdge:
    def test_cpu(self):
        validate_device_spec("cpu")

    def test_cuda(self):
        validate_device_spec("cuda")

    def test_cuda_with_id(self):
        validate_device_spec("cuda:0")
        validate_device_spec("cuda:3")

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            validate_device_spec("")

    def test_non_string_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            validate_device_spec(42)  # type: ignore[arg-type]

    def test_negative_device_id_raises(self):
        with pytest.raises(ValidationError, match="Invalid"):
            validate_device_spec("cuda:-1")

    def test_non_numeric_device_id_raises(self):
        with pytest.raises(ValidationError, match="Invalid"):
            validate_device_spec("cuda:abc")

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValidationError, match="Unknown"):
            validate_device_spec("rocm:0")


class TestValidateDiagramArrayEdge:
    def test_valid_2d(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = validate_diagram_array(arr)
        assert result.shape == (2, 2)

    def test_empty_array_returns_empty_3col(self):
        arr = np.empty((0, 5))
        result = validate_diagram_array(arr)
        assert result.shape == (0, 3)

    def test_1d_raises(self):
        with pytest.raises(ShapeError, match="2"):
            validate_diagram_array(np.array([1.0, 2.0]))

    def test_nan_birth_raises(self):
        arr = np.array([[float("nan"), 1.0]])
        with pytest.raises(ValidationError, match="births"):
            validate_diagram_array(arr)

    def test_nan_death_raises(self):
        arr = np.array([[0.0, float("nan")]])
        with pytest.raises(ValidationError, match="finite"):
            validate_diagram_array(arr)

    def test_death_less_than_birth_raises(self):
        arr = np.array([[5.0, 1.0]])
        with pytest.raises(ValidationError, match=">= births"):
            validate_diagram_array(arr)

    def test_invalid_dims_raises(self):
        arr = np.array([[0.0, 1.0, float("nan")]])
        with pytest.raises(ValidationError, match="finite non-negative"):
            validate_diagram_array(arr)

    def test_require_dims_no_dim_raises(self):
        arr = np.array([[0.0, 1.0]])
        with pytest.raises(ShapeError, match="3 columns"):
            validate_diagram_array(arr, require_dims=True)

    def test_require_dims_with_dim(self):
        arr = np.array([[0.0, 1.0, 0]])
        result = validate_diagram_array(arr, require_dims=True)
        assert result.shape == (1, 3)

    def test_non_contiguous_returns_contiguous(self):
        arr = np.array([[0.0, 1.0, 0], [2.0, 3.0, 1]]).T  # (3, 2) non-contiguous
        contig = np.ascontiguousarray(arr.T)
        result = validate_diagram_array(contig)
        assert result.flags.c_contiguous
