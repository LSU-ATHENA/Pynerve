from __future__ import annotations

import math

import numpy as np
import pytest
from pynerve._validation import (
    parse_nonnegative_int,
    validate_bool,
    validate_device_id,
    validate_device_spec,
    validate_diagram,
    validate_diagram_array,
    validate_finite_deaths,
    validate_finite_scalar,
    validate_finite_tensor,
    validate_floating_tensor,
    validate_max_dist,
    validate_max_radius,
    validate_nonempty_string,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_optional_finite,
    validate_optional_nonnegative_int,
    validate_optional_positive_int,
    validate_optional_string,
    validate_points,
    validate_positive_finite,
    validate_positive_int,
    validate_probability,
    validate_seed,
    validate_shape,
    validate_shape_tuple,
    validate_string_list,
)
from pynerve.exceptions import DtypeError, ShapeError, ValidationError

torch = pytest.importorskip("torch")


class TestValidatePositiveInt:
    def test_valid(self):
        assert validate_positive_int(1, "x") == 1
        assert validate_positive_int(100, "x") == 100
        assert validate_positive_int(np.int64(3), "x") == 3

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_int(0, "x")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_int(-1, "x")

    def test_bool_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int(True, "x")

    def test_float_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int(3.14, "x")

    def test_none_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int(None, "x")

    def test_str_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int("abc", "x")


class TestValidateNonnegativeInt:
    def test_positive_valid(self):
        assert validate_nonnegative_int(5, "x") == 5

    def test_zero_valid(self):
        assert validate_nonnegative_int(0, "x") == 0

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_nonnegative_int(-1, "x")

    def test_bool_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_nonnegative_int(False, "x")

    def test_none_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_nonnegative_int(None, "x")


class TestValidatePositiveFinite:
    def test_valid(self):
        assert validate_positive_finite(1.0, "x") == 1.0
        assert validate_positive_finite(0.5, "x") == 0.5

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="must be finite and positive"):
            validate_positive_finite(0.0, "x")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be finite and positive"):
            validate_positive_finite(-1.0, "x")

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="must be finite and positive"):
            validate_positive_finite(float("inf"), "x")

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="must be finite and positive"):
            validate_positive_finite(float("nan"), "x")

    def test_neginf_raises(self):
        with pytest.raises(ValidationError, match="must be finite and positive"):
            validate_positive_finite(float("-inf"), "x")


class TestValidateNonnegativeFinite:
    def test_positive_valid(self):
        assert validate_nonnegative_finite(3.14, "x") == 3.14

    def test_zero_valid(self):
        assert validate_nonnegative_finite(0.0, "x") == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be finite and non-negative"):
            validate_nonnegative_finite(-0.5, "x")

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="must be finite and non-negative"):
            validate_nonnegative_finite(float("inf"), "x")

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="must be finite and non-negative"):
            validate_nonnegative_finite(float("nan"), "x")


class TestValidateNonemptyString:
    def test_valid(self):
        assert validate_nonempty_string("hello", "x") == "hello"

    def test_empty_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_nonempty_string("", "x")

    def test_none_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_nonempty_string(None, "x")

    def test_int_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_nonempty_string(123, "x")


class TestValidateBool:
    def test_true_valid(self):
        assert validate_bool(True, "x") is True

    def test_false_valid(self):
        assert validate_bool(False, "x") is False

    def test_int_zero_raises(self):
        with pytest.raises(ValidationError, match="must be a boolean"):
            validate_bool(0, "x")

    def test_int_one_raises(self):
        with pytest.raises(ValidationError, match="must be a boolean"):
            validate_bool(1, "x")

    def test_none_raises(self):
        with pytest.raises(ValidationError, match="must be a boolean"):
            validate_bool(None, "x")


class TestValidateProbability:
    def test_zero_valid(self):
        assert validate_probability(0.0, "x") == 0.0

    def test_one_valid(self):
        assert validate_probability(1.0, "x") == 1.0

    def test_mid_valid(self):
        assert validate_probability(0.5, "x") == 0.5

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match=r"must be in \[0, 1\]"):
            validate_probability(-0.1, "x")

    def test_above_one_raises(self):
        with pytest.raises(ValidationError, match=r"must be in \[0, 1\]"):
            validate_probability(1.1, "x")

    def test_nan_passes_comparison(self):
        result = validate_probability(float("nan"), "x")
        assert math.isnan(result)


class TestValidateFiniteScalar:
    def test_valid(self):
        assert validate_finite_scalar(0.0, "x") == 0.0
        assert validate_finite_scalar(-5.0, "x") == -5.0
        assert validate_finite_scalar(3.14, "x") == 3.14

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="must be finite"):
            validate_finite_scalar(float("inf"), "x")

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="must be finite"):
            validate_finite_scalar(float("nan"), "x")


class TestValidateDeviceId:
    def test_valid(self):
        assert validate_device_id(0, "x") == 0
        assert validate_device_id(5, "x") == 5

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_device_id(-1, "x")

    def test_bool_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_device_id(True, "x")

    def test_float_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_device_id(1.0, "x")


class TestValidateOptionalPositiveInt:
    def test_valid_int(self):
        assert validate_optional_positive_int(42, "x") == 42

    def test_valid_none(self):
        assert validate_optional_positive_int(None, "x") is None

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_optional_positive_int(0, "x")

    def test_bool_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_optional_positive_int(True, "x")


class TestValidateOptionalNonnegativeInt:
    def test_valid_int(self):
        assert validate_optional_nonnegative_int(0, "x") == 0
        assert validate_optional_nonnegative_int(10, "x") == 10

    def test_valid_none(self):
        assert validate_optional_nonnegative_int(None, "x") is None

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_optional_nonnegative_int(-1, "x")


class TestValidateOptionalFinite:
    def test_valid_float(self):
        assert validate_optional_finite(3.14, "x") == 3.14

    def test_valid_none(self):
        assert validate_optional_finite(None, "x") is None

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="must be finite"):
            validate_optional_finite(float("inf"), "x")


class TestValidateSeed:
    def test_valid_int(self):
        assert validate_seed(42, "x") == 42
        assert validate_seed(0, "x") == 0

    def test_valid_none(self):
        assert validate_seed(None, "x") is None

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_seed(-1, "x")

    def test_bool_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_seed(True, "x")

    def test_float_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_seed(2.5, "x")


class TestValidateOptionalString:
    def test_valid_string(self):
        assert validate_optional_string("abc", "x") == "abc"

    def test_valid_none(self):
        assert validate_optional_string(None, "x") is None

    def test_empty_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_optional_string("", "x")

    def test_int_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_optional_string(123, "x")


class TestValidateStringList:
    def test_valid_list(self):
        assert validate_string_list(["a", "b"], "x") == ["a", "b"]

    def test_valid_empty_list(self):
        assert validate_string_list([], "x") == []

    def test_none_returns_empty(self):
        assert validate_string_list(None, "x") == []

    def test_string_raises(self):
        with pytest.raises(ValidationError, match="must be a sequence of strings"):
            validate_string_list("abc", "x")

    def test_bytes_raises(self):
        with pytest.raises(ValidationError, match="must be a sequence of strings"):
            validate_string_list(b"abc", "x")

    def test_empty_string_in_list_raises(self):
        with pytest.raises(ValidationError, match="must contain non-empty strings"):
            validate_string_list(["a", ""], "x")

    def test_non_string_in_list_raises(self):
        with pytest.raises(ValidationError, match="must contain non-empty strings"):
            validate_string_list([1, 2], "x")


class TestValidateMaxDist:
    def test_valid_float(self):
        assert validate_max_dist(5.0, "x") == 5.0
        assert validate_max_dist(0.0, "x") == 0.0

    def test_valid_none(self):
        assert validate_max_dist(None, "x") is None

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be finite and non-negative"):
            validate_max_dist(-1.0, "x")

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="must be finite and non-negative"):
            validate_max_dist(float("inf"), "x")

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="must be finite and non-negative"):
            validate_max_dist(float("nan"), "x")


class TestValidateMaxRadius:
    def test_valid_float(self):
        assert validate_max_radius(10.0, "x") == 10.0
        assert validate_max_radius(0.0, "x") == 0.0

    def test_none_returns_inf(self):
        assert validate_max_radius(None, "x") == float("inf")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be finite and non-negative"):
            validate_max_radius(-1.0, "x")

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="must be finite and non-negative"):
            validate_max_radius(float("inf"), "x")


class TestParseNonnegativeInt:
    def test_valid_int(self):
        assert parse_nonnegative_int(0, "x") == 0
        assert parse_nonnegative_int(42, "x") == 42

    def test_valid_str(self):
        assert parse_nonnegative_int("0", "x") == 0
        assert parse_nonnegative_int("42", "x") == 42

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            parse_nonnegative_int(-1, "x")

    def test_negative_str_raises(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            parse_nonnegative_int("-1", "x")

    def test_bool_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            parse_nonnegative_int(True, "x")
        with pytest.raises(ValidationError, match="must be an integer"):
            parse_nonnegative_int(False, "x")

    def test_float_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            parse_nonnegative_int(3.14, "x")

    def test_invalid_str_raises(self):
        with pytest.raises(ValueError):
            parse_nonnegative_int("abc", "x")


class TestValidateFiniteDeaths:
    def test_finite_valid(self):
        arr = np.array([1.0, 2.0, 3.0])
        validate_finite_deaths(arr)

    def test_posinf_valid(self):
        arr = np.array([1.0, float("inf"), 2.0])
        validate_finite_deaths(arr)

    def test_all_posinf_valid(self):
        arr = np.array([float("inf"), float("inf")])
        validate_finite_deaths(arr)

    def test_nan_raises(self):
        arr = np.array([1.0, float("nan"), 2.0])
        with pytest.raises(ValidationError, match="must not be NaN"):
            validate_finite_deaths(arr)

    def test_neginf_raises(self):
        arr = np.array([1.0, float("-inf"), 2.0])
        with pytest.raises(ValidationError, match="must be finite or positive infinity"):
            validate_finite_deaths(arr)


class TestValidateFiniteTensor:
    def test_valid(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        validate_finite_tensor(t)

    def test_non_tensor_raises(self):
        with pytest.raises(DtypeError, match="must be a torch.Tensor"):
            validate_finite_tensor(np.array([1.0]))

    def test_nan_raises(self):
        t = torch.tensor([1.0, float("nan")])
        with pytest.raises(ValidationError, match="must contain only finite values"):
            validate_finite_tensor(t)

    def test_inf_raises(self):
        t = torch.tensor([float("inf"), 1.0])
        with pytest.raises(ValidationError, match="must contain only finite values"):
            validate_finite_tensor(t)


class TestValidateFloatingTensor:
    def test_float32_valid(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        validate_floating_tensor(t)

    def test_float64_valid(self):
        t = torch.tensor([1.0], dtype=torch.float64)
        validate_floating_tensor(t)

    def test_non_tensor_raises(self):
        with pytest.raises(DtypeError, match="must be a torch.Tensor"):
            validate_floating_tensor(np.array([1.0]))

    def test_integer_tensor_raises(self):
        t = torch.tensor([1, 2, 3])
        with pytest.raises(DtypeError, match="must use a floating-point dtype"):
            validate_floating_tensor(t)


class TestValidateDiagram:
    def test_valid_simple(self):
        d = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        result = validate_diagram(d)
        assert result is d

    def test_valid_with_inf_death(self):
        d = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        result = validate_diagram(d)
        assert result is d

    def test_valid_extra_columns(self):
        d = torch.tensor([[0.0, 1.0, 0.0, 5.0]], dtype=torch.float64)
        result = validate_diagram(d)
        assert result is d

    def test_empty_valid(self):
        d = torch.empty(0, 3, dtype=torch.float64)
        result = validate_diagram(d)
        assert result is d

    def test_one_d_raises(self):
        d = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(ShapeError, match="must be a 2D tensor"):
            validate_diagram(d)

    def test_three_d_raises(self):
        d = torch.zeros(2, 2, 2, dtype=torch.float64)
        with pytest.raises(ShapeError, match="must be a 2D tensor"):
            validate_diagram(d)

    def test_too_few_columns_raises(self):
        d = torch.tensor([[1.0]], dtype=torch.float64)
        with pytest.raises(ShapeError, match="must have at least 2 columns"):
            validate_diagram(d)

    def test_non_float_raises(self):
        d = torch.tensor([[0, 1], [1, 2]])
        with pytest.raises(DtypeError, match="must use a floating-point dtype"):
            validate_diagram(d)

    def test_nan_birth_raises(self):
        d = torch.tensor([[float("nan"), 1.0]], dtype=torch.float64)
        with pytest.raises(ValidationError, match="births must be finite"):
            validate_diagram(d)

    def test_nan_death_raises(self):
        d = torch.tensor([[0.0, float("nan")]], dtype=torch.float64)
        with pytest.raises(ValidationError, match="deaths must not be NaN"):
            validate_diagram(d)

    def test_death_less_than_birth_raises(self):
        d = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
        with pytest.raises(ValidationError, match="finite deaths must be >= births"):
            validate_diagram(d)

    def test_all_inf_deaths_bypass_comparison(self):
        d = torch.tensor([[0.0, float("inf")], [-5.0, float("inf")]], dtype=torch.float64)
        validate_diagram(d)


class TestValidatePoints:
    def test_numpy_valid(self):
        pts = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = validate_points(pts)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, pts)

    def test_numpy_empty_points_valid(self):
        pts = np.empty((0, 3))
        result = validate_points(pts)
        assert result.shape == (0, 3)

    def test_numpy_one_d_raises(self):
        pts = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ShapeError, match="must be a 2D array"):
            validate_points(pts)

    def test_numpy_three_d_raises(self):
        pts = np.zeros((2, 2, 2))
        with pytest.raises(ShapeError, match="must be a 2D array"):
            validate_points(pts)

    def test_numpy_zero_coords_raises(self):
        pts = np.empty((5, 0))
        with pytest.raises(ShapeError, match="must contain at least one coordinate"):
            validate_points(pts)

    def test_numpy_nonfinite_raises(self):
        pts = np.array([[0.0, 1.0], [float("nan"), 3.0]])
        with pytest.raises(ValidationError, match="must contain only finite coordinates"):
            validate_points(pts)

    def test_tensor_valid(self):
        pts = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        result = validate_points(pts)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, np.array([[0.0, 1.0], [2.0, 3.0]]))

    def test_tensor_empty_valid(self):
        pts = torch.empty(0, 3)
        result = validate_points(pts)
        assert result.shape == (0, 3)

    def test_tensor_one_d_raises(self):
        pts = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ShapeError, match="must be a 2D array"):
            validate_points(pts)

    def test_tensor_zero_coords_raises(self):
        pts = torch.empty(5, 0)
        with pytest.raises(ShapeError, match="must contain at least one coordinate"):
            validate_points(pts)

    def test_tensor_nonfinite_raises(self):
        pts = torch.tensor([[0.0, 1.0], [float("inf"), 3.0]])
        with pytest.raises(ValidationError, match="must contain only finite coordinates"):
            validate_points(pts)

    def test_tensor_empty_zero_coords_passes_shape_check(self):
        pts = torch.empty(0, 0)
        with pytest.raises(ShapeError, match="must contain at least one coordinate"):
            validate_points(pts)


class TestValidateShape:
    def test_single_int(self):
        assert validate_shape(5) == (5,)

    def test_int_list(self):
        assert validate_shape([2, 3, 4]) == (2, 3, 4)

    def test_int_tuple(self):
        assert validate_shape((10, 20)) == (10, 20)

    def test_numpy_int_valid(self):
        assert validate_shape(np.int64(5)) == (5,)

    def test_allow_infer_single_minus_one(self):
        assert validate_shape((3, -1), allow_infer=True) == (3, -1)

    def test_allow_infer_false_rejects_minus_one(self):
        with pytest.raises(ShapeError, match="dimensions must be non-negative"):
            validate_shape((3, -1), allow_infer=False)

    def test_allow_infer_true_rejects_two_minus_one(self):
        with pytest.raises(ShapeError, match="at most one inferred dimension"):
            validate_shape((-1, -1), allow_infer=True)

    def test_negative_dim_raises(self):
        with pytest.raises(ShapeError, match="dimensions must be non-negative"):
            validate_shape((3, -2))

    def test_empty_sequence_raises(self):
        with pytest.raises(ShapeError, match="must contain at least one dimension"):
            validate_shape([])

    def test_bool_raises(self):
        with pytest.raises(ShapeError, match="must be an integer or sequence of integers"):
            validate_shape(True)

    def test_string_raises(self):
        with pytest.raises(ShapeError, match="must be an integer or sequence of integers"):
            validate_shape("abc")

    def test_bytes_raises(self):
        with pytest.raises(ShapeError, match="must be an integer or sequence of integers"):
            validate_shape(b"abc")

    def test_none_not_sequence(self):
        with pytest.raises(ShapeError, match="must be an integer or sequence of integers"):
            validate_shape(None)

    def test_non_integer_dim_raises(self):
        with pytest.raises(ShapeError, match="dimensions must be integers"):
            validate_shape((3, 1.5))

    def test_bool_in_sequence_raises(self):
        with pytest.raises(ShapeError, match="dimensions must be integers"):
            validate_shape((3, True))


class TestValidateShapeTuple:
    def test_valid_tuple(self):
        assert validate_shape_tuple((2, 3), "x") == (2, 3)

    def test_empty_tuple_valid(self):
        assert validate_shape_tuple((), "x") == ()

    def test_none_valid(self):
        assert validate_shape_tuple(None, "x") is None

    def test_negative_raises(self):
        with pytest.raises(ShapeError, match="dimensions must be non-negative"):
            validate_shape_tuple((2, -1), "x")

    def test_string_raises(self):
        with pytest.raises(ShapeError, match="must be a sequence of dimensions"):
            validate_shape_tuple("abc", "x")

    def test_bytes_raises(self):
        with pytest.raises(ShapeError, match="must be a sequence of dimensions"):
            validate_shape_tuple(b"abc", "x")

    def test_bool_dim_raises(self):
        with pytest.raises(ShapeError, match="dimensions must be integers"):
            validate_shape_tuple((3, True), "x")


class TestValidateDiagramArray:
    def test_valid_simple(self):
        arr = np.array([[0.0, 1.0], [1.0, 2.0]])
        result = validate_diagram_array(arr)
        np.testing.assert_array_equal(result, arr[:, :2])

    def test_valid_with_inf_death(self):
        arr = np.array([[0.0, float("inf")]])
        result = validate_diagram_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_valid_with_dims(self):
        arr = np.array([[0.0, 1.0, 0], [1.0, 2.0, 1]])
        result = validate_diagram_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_empty_returns_empty_three_col(self):
        arr = np.empty((0, 2))
        result = validate_diagram_array(arr)
        assert result.shape == (0, 3)

    def test_one_d_raises(self):
        arr = np.array([1.0, 2.0])
        with pytest.raises(ShapeError, match="must have shape"):
            validate_diagram_array(arr)

    def test_single_column_raises(self):
        arr = np.array([[1.0]])
        with pytest.raises(ShapeError, match="must have shape"):
            validate_diagram_array(arr)

    def test_nan_birth_raises(self):
        arr = np.array([[float("nan"), 1.0]])
        with pytest.raises(ValidationError, match="births must be finite"):
            validate_diagram_array(arr)

    def test_nan_death_raises(self):
        arr = np.array([[0.0, float("nan")]])
        with pytest.raises(ValidationError, match="deaths must be finite or \\+inf"):
            validate_diagram_array(arr)

    def test_neginf_death_raises(self):
        arr = np.array([[0.0, float("-inf")]])
        with pytest.raises(ValidationError, match="deaths must be finite or \\+inf"):
            validate_diagram_array(arr)

    def test_death_less_than_birth_raises(self):
        arr = np.array([[1.0, 0.5]])
        with pytest.raises(ValidationError, match="finite deaths must be >= births"):
            validate_diagram_array(arr)

    def test_invalid_dim_negative_raises(self):
        arr = np.array([[0.0, 1.0, -1]])
        with pytest.raises(
            ValidationError, match="dimensions must be finite non-negative integers"
        ):
            validate_diagram_array(arr)

    def test_invalid_dim_float_raises(self):
        arr = np.array([[0.0, 1.0, 1.5]])
        with pytest.raises(
            ValidationError, match="dimensions must be finite non-negative integers"
        ):
            validate_diagram_array(arr)

    def test_invalid_dim_nan_raises(self):
        arr = np.array([[0.0, 1.0, float("nan")]])
        with pytest.raises(
            ValidationError, match="dimensions must be finite non-negative integers"
        ):
            validate_diagram_array(arr)

    def test_require_dims_missing_raises(self):
        arr = np.array([[0.0, 1.0]])
        with pytest.raises(ShapeError, match="must have at least 3 columns"):
            validate_diagram_array(arr, require_dims=True)

    def test_require_dims_present_valid(self):
        arr = np.array([[0.0, 1.0, 0]])
        result = validate_diagram_array(arr, require_dims=True)
        np.testing.assert_array_equal(result, arr)

    def test_non_contiguous_returns_contiguous(self):
        arr = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        noncontig = arr.T
        result = validate_diagram_array(noncontig)
        assert result.flags.c_contiguous


class TestValidateDeviceSpec:
    def test_cpu_valid(self):
        validate_device_spec("cpu")

    def test_cuda_valid(self):
        validate_device_spec("cuda")

    def test_cuda_with_id_valid(self):
        validate_device_spec("cuda:0")
        validate_device_spec("cuda:5")

    def test_mps_valid(self):
        validate_device_spec("mps")

    def test_hip_valid(self):
        validate_device_spec("hip")

    def test_rocm_maps_to_hip(self):
        validate_device_spec("rocm")

    def test_xpu_valid(self):
        validate_device_spec("xpu")

    def test_empty_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_device_spec("")

    def test_none_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_device_spec(None)

    def test_int_raises(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_device_spec(123)

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValidationError, match="Unknown device"):
            validate_device_spec("metal:0")

    def test_negative_index_raises(self):
        with pytest.raises(ValidationError, match="Invalid device"):
            validate_device_spec("cuda:-1")

    def test_non_integer_index_raises(self):
        with pytest.raises(ValidationError, match="Invalid device"):
            validate_device_spec("cuda:abc")

    def test_colon_no_index_raises(self):
        with pytest.raises(ValidationError, match="Invalid device"):
            validate_device_spec("cuda:")


class TestCustomParamNames:
    def test_finite_deaths_custom_name_in_error(self):
        with pytest.raises(ValidationError, match="mydiag must not be NaN"):
            validate_finite_deaths(np.array([float("nan")]), name="mydiag")


class TestValidationReturns:
    def test_validate_device_spec_returns_none_on_cpu(self):
        assert validate_device_spec("cpu") is None

    def test_validate_diagram_array_preserves_dtype(self):
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        result = validate_diagram_array(arr)
        assert result.dtype == np.float32

    def test_diagram_death_eq_birth_valid(self):
        d = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        validate_diagram(d)
