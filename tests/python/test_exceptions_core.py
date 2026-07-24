"""Tests for exceptions/_base.py, _memory.py, and _numerical.py."""

from __future__ import annotations

import pytest

from pynerve._error_codes import (
    E20_NUM_NAN,
    E21_NUM_NO_CONVERGE,
    E41_RESOURCE_LIMIT,
    E71_PRECISION_LOSS,
    E73_PRECISION_CATASTROPHIC,
    UNKNOWN,
    ErrorCategory,
    ErrorSeverity,
)
from pynerve.exceptions._base import NerveError
from pynerve.exceptions._memory import AllocationError, NerveMemoryError, OutOfMemoryError
from pynerve.exceptions._numerical import (
    ConvergenceError,
    NumericalError,
    NumericalInstabilityError,
    PrecisionError,
)


# NerveError base 


class TestNerveError:
    def test_basic_construction(self):
        e = NerveError("something broke")
        assert "something broke" in str(e)
        assert e.error_code == UNKNOWN
        assert e.error_category == ErrorCategory.UNKNOWN_CATEGORY
        assert e.severity == ErrorSeverity.ERROR

    def test_empty_message(self):
        e = NerveError()
        assert "Nerve error" in str(e)

    def test_with_details(self):
        e = NerveError("error", details={"key": "value"})
        assert e.details == {"key": "value"}

    def test_details_none(self):
        e = NerveError("error", details=None)
        assert e.details == {}

    def test_details_not_mapping(self):
        with pytest.raises(TypeError, match="mapping"):
            NerveError("error", details="not a dict")  # type: ignore[arg-type]

    def test_with_cpp_message(self):
        e = NerveError("python msg", cpp_message="cpp msg")
        assert e.cpp_message == "cpp msg"
        assert "python msg" in str(e)

    def test_cpp_message_none(self):
        e = NerveError("msg", cpp_message=None)
        assert e.cpp_message == "msg"

    def test_cpp_message_empty(self):
        # empty string is falsy, so cpp_message falls back to message
        e = NerveError("msg", cpp_message="")
        assert e.cpp_message == "msg"

    def test_repr(self):
        e = NerveError("test")
        r = repr(e)
        assert "NerveError" in r
        assert "category=" in r
        assert "code=" in r

    def test_formatted_message(self):
        e = NerveError("value is bad")
        assert "[NerveError]" in str(e)
        assert "value is bad" in str(e)

    def test_category_in_formatted_message(self):
        e = NerveError("test")
        assert "category=unknown" in str(e)

    def test_isinstance_value_error(self):
        assert isinstance(NerveError("test"), ValueError)

    def test_details_preserves_types(self):
        e = NerveError("x", details={"a": 1, "b": 2.0, "c": [3]})
        assert e.details["a"] == 1
        assert e.details["b"] == 2.0
        assert e.details["c"] == [3]


# Memory exceptions 


class TestMemoryErrors:
    def test_nerve_memory_error_is_nerve_error(self):
        assert issubclass(NerveMemoryError, NerveError)

    def test_nerve_memory_error_code(self):
        assert NerveMemoryError.error_code == E41_RESOURCE_LIMIT

    def test_nerve_memory_error_category(self):
        assert NerveMemoryError.error_category == ErrorCategory.CAPACITY

    def test_out_of_memory_is_memory_error(self):
        assert issubclass(OutOfMemoryError, NerveMemoryError)
        e = OutOfMemoryError("oom")
        assert isinstance(e, NerveMemoryError)
        assert isinstance(e, NerveError)

    def test_allocation_error_is_memory_error(self):
        assert issubclass(AllocationError, NerveMemoryError)
        e = AllocationError("alloc failed")
        assert isinstance(e, NerveMemoryError)

    def test_out_of_memory_str(self):
        e = OutOfMemoryError("out of memory")
        assert "out of memory" in str(e)

    def test_allocation_error_str(self):
        e = AllocationError("bad alloc")
        assert "bad alloc" in str(e)


# Numerical exceptions 


class TestNumericalErrors:
    def test_numerical_error_is_nerve_error(self):
        assert issubclass(NumericalError, NerveError)

    def test_numerical_error_code(self):
        assert NumericalError.error_code == E20_NUM_NAN

    def test_numerical_error_category(self):
        assert NumericalError.error_category == ErrorCategory.NUMERICAL

    def test_convergence_error_is_numerical(self):
        assert issubclass(ConvergenceError, NumericalError)
        e = ConvergenceError("no convergence")
        assert e.error_code == E21_NUM_NO_CONVERGE

    def test_precision_error_is_numerical(self):
        assert issubclass(PrecisionError, NumericalError)
        e = PrecisionError("precision loss")
        assert e.error_code == E71_PRECISION_LOSS
        assert e.error_category == ErrorCategory.PRECISION

    def test_numerical_instability_is_numerical(self):
        assert issubclass(NumericalInstabilityError, NumericalError)
        e = NumericalInstabilityError("catastrophic")
        assert e.error_code == E73_PRECISION_CATASTROPHIC
        assert e.error_category == ErrorCategory.PRECISION

    def test_convergence_str(self):
        e = ConvergenceError("did not converge")
        assert "did not converge" in str(e)

    def test_precision_str(self):
        e = PrecisionError("lost precision")
        assert "lost precision" in str(e)

    def test_instability_str(self):
        e = NumericalInstabilityError("unstable")
        assert "unstable" in str(e)
