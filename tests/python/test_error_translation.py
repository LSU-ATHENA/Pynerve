"""Tests for C++ to Python exception translation."""

from __future__ import annotations

import pytest
from pynerve._error_codes import (
    E00_IO_TIMEOUT,
    E01_IO_CORRUPT,
    E10_GPU_OOM,
    E11_GPU_LAUNCH_FAIL,
    E20_NUM_NAN,
    E21_NUM_NO_CONVERGE,
    E30_DET_MISMATCH,
    E41_RESOURCE_LIMIT,
    E50_PH_ABORT,
    E53_PH4_BUDGET_EXCEEDED,
    E54_PH4_INVALID_INPUT,
    E60_NUMA_BIND_FAIL,
    E71_PRECISION_LOSS,
    E73_PRECISION_CATASTROPHIC,
    E81_MATRIX_EMPTY,
    E88_INVALID_SIMPLICES,
    E92_BETTI_MISMATCH,
)
from pynerve._error_translation import translate_cpp_exception
from pynerve.exceptions import (
    BettiError,
    BudgetExceededError,
    ConvergenceError,
    DeterminismError,
    GPULaunchError,
    GPUMemoryError,
    InvalidSimplexError,
    NerveError,
    NerveIOError,
    NUMAError,
    NumericalError,
    NumericalInstabilityError,
    OutOfMemoryError,
    PersistenceError,
    PrecisionError,
    ShapeMismatchError,
    TypeMismatchError,
)


class TestTranslateCppException:
    def test_translates_io_timeout(self):
        exc = type("Fake", (Exception,), {"error_code": E00_IO_TIMEOUT})("timeout")
        result = translate_cpp_exception(exc)
        assert isinstance(result, NerveIOError)

    def test_translates_io_corrupt(self):
        exc = type("Fake", (Exception,), {"error_code": E01_IO_CORRUPT})("corrupt")
        result = translate_cpp_exception(exc)
        assert isinstance(result, NerveIOError)

    def test_translates_gpu_oom(self):
        exc = type("Fake", (Exception,), {"error_code": E10_GPU_OOM})("gpu oom")
        result = translate_cpp_exception(exc)
        assert isinstance(result, GPUMemoryError)

    def test_translates_gpu_launch_fail(self):
        exc = type("Fake", (Exception,), {"error_code": E11_GPU_LAUNCH_FAIL})("launch fail")
        result = translate_cpp_exception(exc)
        assert isinstance(result, GPULaunchError)

    def test_translates_numerical_nan(self):
        exc = type("Fake", (Exception,), {"error_code": E20_NUM_NAN})("nan")
        result = translate_cpp_exception(exc)
        assert isinstance(result, NumericalError)

    def test_translates_no_converge(self):
        exc = type("Fake", (Exception,), {"error_code": E21_NUM_NO_CONVERGE})("no converge")
        result = translate_cpp_exception(exc)
        assert isinstance(result, ConvergenceError)

    def test_translates_determinism_mismatch(self):
        exc = type("Fake", (Exception,), {"error_code": E30_DET_MISMATCH})("det mismatch")
        result = translate_cpp_exception(exc)
        assert isinstance(result, DeterminismError)

    def test_translates_resource_limit(self):
        exc = type("Fake", (Exception,), {"error_code": E41_RESOURCE_LIMIT})("oom")
        result = translate_cpp_exception(exc)
        assert isinstance(result, OutOfMemoryError)

    def test_translates_ph_abort(self):
        exc = type("Fake", (Exception,), {"error_code": E50_PH_ABORT})("abort")
        result = translate_cpp_exception(exc)
        assert isinstance(result, PersistenceError)

    def test_translates_budget_exceeded(self):
        exc = type("Fake", (Exception,), {"error_code": E53_PH4_BUDGET_EXCEEDED})("budget")
        result = translate_cpp_exception(exc)
        assert isinstance(result, BudgetExceededError)

    def test_translates_invalid_input(self):
        exc = type("Fake", (Exception,), {"error_code": E54_PH4_INVALID_INPUT})("invalid")
        result = translate_cpp_exception(exc)
        assert isinstance(result, TypeMismatchError)

    def test_translates_numa_bind_fail(self):
        exc = type("Fake", (Exception,), {"error_code": E60_NUMA_BIND_FAIL})("numa")
        result = translate_cpp_exception(exc)
        assert isinstance(result, NUMAError)

    def test_translates_precision_loss(self):
        exc = type("Fake", (Exception,), {"error_code": E71_PRECISION_LOSS})("precision")
        result = translate_cpp_exception(exc)
        assert isinstance(result, PrecisionError)

    def test_translates_precision_catastrophic(self):
        exc = type("Fake", (Exception,), {"error_code": E73_PRECISION_CATASTROPHIC})("catastrophic")
        result = translate_cpp_exception(exc)
        assert isinstance(result, NumericalInstabilityError)

    def test_translates_matrix_empty(self):
        exc = type("Fake", (Exception,), {"error_code": E81_MATRIX_EMPTY})("empty")
        result = translate_cpp_exception(exc)
        assert isinstance(result, ShapeMismatchError)

    def test_translates_invalid_simplices_to_invalid_simplex_error(self):
        exc = type("Fake", (Exception,), {"error_code": E88_INVALID_SIMPLICES})("bad simplex")
        result = translate_cpp_exception(exc)
        assert isinstance(result, InvalidSimplexError)

    def test_translates_betti_mismatch(self):
        exc = type("Fake", (Exception,), {"error_code": E92_BETTI_MISMATCH})("betti")
        result = translate_cpp_exception(exc)
        assert isinstance(result, BettiError)

    def test_translates_unknown_code_to_nerve_error(self):
        exc = type("Fake", (Exception,), {"error_code": 0xDEAD})("unknown")
        result = translate_cpp_exception(exc)
        assert isinstance(result, NerveError)
        assert "DEAD" in str(result) or "unknown" in str(result).lower()

    def test_translates_missing_error_code(self):
        exc = type("Fake", (Exception,), {})("no code")
        result = translate_cpp_exception(exc)
        assert isinstance(result, NerveError)

    def test_translates_none_error_code(self):
        exc = type("Fake", (Exception,), {"error_code": None})("none")
        result = translate_cpp_exception(exc)
        assert isinstance(result, NerveError)

    def test_rejects_non_exception(self):
        with pytest.raises(TypeError, match="must be an Exception"):
            translate_cpp_exception("not an exception")

    def test_preserves_message(self):
        exc = type("Fake", (Exception,), {"error_code": E50_PH_ABORT})("computation aborted")
        result = translate_cpp_exception(exc)
        assert "computation aborted" in str(result)

    def test_empty_message_falls_back_to_code(self):
        exc = type("Fake", (Exception,), {"error_code": E50_PH_ABORT})("")
        result = translate_cpp_exception(exc)
        assert "0x00000600" in str(result)
