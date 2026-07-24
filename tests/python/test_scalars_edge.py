"""Tests for _validation/_scalars.py — remaining edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pynerve._validation._scalars import (
    parse_nonnegative_int,
    validate_bool,
    validate_device_id,
    validate_finite_scalar,
    validate_max_dist,
    validate_max_radius,
    validate_nonempty_string,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_optional_finite,
    validate_optional_nonnegative_int,
    validate_optional_positive_int,
    validate_optional_string,
    validate_positive_finite,
    validate_positive_int,
    validate_probability,
    validate_seed,
    validate_string_list,
)
from pynerve.exceptions import ValidationError


class TestValidateBoolEdge:
    def test_rejects_int_zero(self):
        with pytest.raises(ValidationError, match="boolean"):
            validate_bool(0, "x")

    def test_rejects_int_one(self):
        with pytest.raises(ValidationError, match="boolean"):
            validate_bool(1, "x")


class TestValidateProbabilityEdge:
    def test_nan_passes_range_check(self):
        result = validate_probability(float("nan"), "x")
        assert math.isnan(result)


class TestValidateFiniteScalarEdge:
    def test_accepts_large_value(self):
        assert validate_finite_scalar(1e100, "x") == 1e100

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_finite_scalar(float("nan"), "x")


class TestValidateNonnegativeFiniteEdge:
    def test_accepts_large(self):
        assert validate_nonnegative_finite(1e50, "x") == 1e50


class TestValidateDeviceIdEdge:
    def test_accepts_np_int(self):
        assert validate_device_id(np.int64(3), "x") == 3

    def test_rejects_float(self):
        with pytest.raises(ValidationError, match="integer"):
            validate_device_id(1.5, "x")


class TestValidateOptionalStringEdge:
    def test_accepts_unicode(self):
        assert validate_optional_string("héllo", "x") == "héllo"


class TestValidateStringListEdge:
    def test_accepts_mixed_strings(self):
        assert validate_string_list(["a", "b", "c"], "x") == ["a", "b", "c"]


class TestValidateMaxDistEdge:
    def test_accepts_small_value(self):
        assert validate_max_dist(1e-10, "x") == 1e-10


class TestValidateMaxRadiusEdge:
    def test_none_returns_inf(self):
        assert math.isinf(validate_max_radius(None, "x"))

    def test_accepts_small_value(self):
        assert validate_max_radius(1e-10, "x") == 1e-10


class TestParseNonnegativeIntEdge:
    def test_accepts_np_int(self):
        assert parse_nonnegative_int(int(np.int64(5)), "x") == 5

    def test_rejects_none(self):
        with pytest.raises(ValidationError, match="integer"):
            parse_nonnegative_int(None, "x")


class TestValidateSeedEdge:
    def test_accepts_np_int(self):
        assert validate_seed(np.int64(42), "x") == 42
