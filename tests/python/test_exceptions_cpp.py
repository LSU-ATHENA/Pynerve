"""Tests for remaining exception classes in exceptions/_cpp.py."""

from __future__ import annotations

from pynerve.exceptions._cpp import (
    BettiError,
    BudgetExceededError,
    DeterminismError,
    DimensionError,
    InvalidArgumentError,
    InvalidSimplexError,
    MatrixStructureError,
    NerveIOError,
    NUMAError,
    PersistenceError,
    ShapeMismatchError,
    TypeMismatchError,
)
from pynerve._error_codes import (
    E00_IO_TIMEOUT,
    E30_DET_MISMATCH,
    E50_PH_ABORT,
    E53_PH4_BUDGET_EXCEEDED,
    E54_PH4_INVALID_INPUT,
    E60_NUMA_BIND_FAIL,
    E85_MATRIX_STRUCTURE,
    E87_INVALID_BETTI_NUMBERS,
    E88_INVALID_SIMPLICES,
    E91_SHAPE_ERROR,
    ErrorCategory,
)


class TestPersistenceError:
    def test_basic(self):
        e = PersistenceError("persistence failed")
        assert "persistence failed" in str(e)
        assert "PersistenceError" in str(e)
        assert e.error_code == E50_PH_ABORT
        assert e.error_category == ErrorCategory.ALGORITHMIC

    def test_with_backend(self):
        e = PersistenceError("err", backend="ph5")
        assert e.backend == "ph5"
        assert "backend='ph5'" in repr(e)

    def test_with_operation(self):
        e = PersistenceError("err", operation="compute")
        assert e.operation == "compute"
        assert "operation='compute'" in repr(e)

    def test_with_both(self):
        e = PersistenceError("err", backend="ph6", operation="filter")
        assert e.backend == "ph6"
        assert e.operation == "filter"
        r = repr(e)
        assert "backend='ph6'" in r
        assert "operation='filter'" in r

    def test_repr_no_extras(self):
        e = PersistenceError("err")
        r = repr(e)
        assert "PersistenceError" in r


class TestShapeMismatchError:
    def test_basic(self):
        e = ShapeMismatchError("bad shape")
        assert e.error_code == E91_SHAPE_ERROR
        assert e.error_category == ErrorCategory.ALGORITHMIC


class TestDimensionError:
    def test_basic(self):
        e = DimensionError("dim error")
        assert e.error_code == E91_SHAPE_ERROR


class TestTypeMismatchError:
    def test_basic(self):
        e = TypeMismatchError("type error")
        assert e.error_code == E54_PH4_INVALID_INPUT
        assert e.error_category == ErrorCategory.PH4_RESEARCH


class TestInvalidSimplexError:
    def test_basic(self):
        e = InvalidSimplexError("bad simplex")
        assert e.error_code == E88_INVALID_SIMPLICES


class TestMatrixStructureError:
    def test_basic(self):
        e = MatrixStructureError("bad matrix")
        assert e.error_code == E85_MATRIX_STRUCTURE


class TestInvalidArgumentError:
    def test_basic(self):
        e = InvalidArgumentError("bad arg")
        assert e.error_code == E54_PH4_INVALID_INPUT

    def test_with_parameter(self):
        e = InvalidArgumentError("bad", parameter="x")
        assert e.parameter == "x"
        assert "parameter='x'" in repr(e)

    def test_with_expected_actual(self):
        e = InvalidArgumentError("bad", expected="int", actual="str")
        assert e.expected == "int"
        assert e.actual == "str"
        r = repr(e)
        assert "expected='int'" in r
        assert "actual='str'" in r

    def test_with_all(self):
        e = InvalidArgumentError("bad", parameter="dim", expected="int", actual="float")
        assert e.parameter == "dim"
        r = repr(e)
        assert "parameter='dim'" in r


class TestBudgetExceededError:
    def test_basic(self):
        e = BudgetExceededError("over budget")
        assert e.error_code == E53_PH4_BUDGET_EXCEEDED


class TestNerveIOError:
    def test_basic(self):
        e = NerveIOError("io fail")
        assert e.error_code == E00_IO_TIMEOUT
        assert e.error_category == ErrorCategory.IO_INFRA


class TestDeterminismError:
    def test_basic(self):
        e = DeterminismError("mismatch")
        assert e.error_code == E30_DET_MISMATCH
        assert e.error_category == ErrorCategory.DETERMINISM


class TestNUMAError:
    def test_basic(self):
        e = NUMAError("numa fail")
        assert e.error_code == E60_NUMA_BIND_FAIL
        assert e.error_category == ErrorCategory.NUMA_AFFINITY


class TestBettiError:
    def test_basic(self):
        e = BettiError("betti error")
        assert e.error_code == E87_INVALID_BETTI_NUMBERS
