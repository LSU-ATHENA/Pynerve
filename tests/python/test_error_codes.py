"""Tests for _error_codes.py and _error_translation.py."""

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
    E31_SCHEMA_VERSION,
    E41_RESOURCE_LIMIT,
    E50_PH_ABORT,
    E53_PH4_BUDGET_EXCEEDED,
    E54_PH4_INVALID_INPUT,
    E60_NUMA_BIND_FAIL,
    E61_NUMA_AFFINITY_FAIL,
    E62_NUMA_MIGRATION_ERROR,
    E70_PRECISION_DOWNGRADE,
    E71_PRECISION_LOSS,
    E72_PRECISION_UNDERFLOW,
    E73_PRECISION_CATASTROPHIC,
    E81_MATRIX_EMPTY,
    E82_MATRIX_SPARSE,
    E83_NO_PIVOTS_FOUND,
    E84_INSUFFICIENT_PIVOTS,
    E85_MATRIX_STRUCTURE,
    E86_NO_PERSISTENCE_PAIRS,
    E87_INVALID_BETTI_NUMBERS,
    E88_INVALID_SIMPLICES,
    E89_BOUNDARY_ERROR,
    E90_VALIDATION_ERROR,
    E91_SHAPE_ERROR,
    E92_DTYPE_ERROR,
    E93_DEVICE_ERROR,
    E94_BACKEND_REQUIRED,
    E95_EMPTY_COMPLEX,
    E99_COMPUTATION_TIMEOUT,
    E100_CONVERGENCE_FAILURE,
    SUCCESS,
    UNKNOWN,
    ErrorCategory,
    ErrorSeverity,
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
    MatrixStructureError,
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


# ── ErrorCategory ──────────────────────────────────────────────────────────


class TestErrorCategory:
    def test_all_values_distinct(self):
        values = [e.value for e in ErrorCategory]
        assert len(values) == len(set(values))

    def test_success_is_zero(self):
        assert ErrorCategory.SUCCESS == 0

    def test_unknown_is_255(self):
        assert ErrorCategory.UNKNOWN_CATEGORY == 255

    def test_str_and_repr(self):
        assert str(ErrorCategory.SUCCESS) == "ErrorCategory.SUCCESS"
        assert "ErrorCategory" in repr(ErrorCategory.GPU_COMPUTE)

    def test_int_equality(self):
        assert ErrorCategory.NUMERICAL == 3

    def test_membership(self):
        assert ErrorCategory(0) == ErrorCategory.SUCCESS
        assert ErrorCategory(255) == ErrorCategory.UNKNOWN_CATEGORY


# ── ErrorSeverity ──────────────────────────────────────────────────────────


class TestErrorSeverity:
    def test_all_values_distinct(self):
        values = [e.value for e in ErrorSeverity]
        assert len(values) == len(set(values))

    def test_info_is_zero(self):
        assert ErrorSeverity.INFO == 0

    def test_critical_is_highest(self):
        assert ErrorSeverity.CRITICAL == 3
        assert ErrorSeverity.CRITICAL > ErrorSeverity.ERROR

    def test_comparison(self):
        assert ErrorSeverity.WARNING < ErrorSeverity.ERROR
        assert ErrorSeverity.ERROR < ErrorSeverity.CRITICAL
        assert ErrorSeverity.INFO < ErrorSeverity.WARNING

    def test_ordering(self):
        severities = sorted(ErrorSeverity)
        assert severities == [
            ErrorSeverity.INFO,
            ErrorSeverity.WARNING,
            ErrorSeverity.ERROR,
            ErrorSeverity.CRITICAL,
        ]


# ── Error code constants ───────────────────────────────────────────────────


class TestErrorCodes:
    """Verify error codes are unique and have the expected bit patterns."""

    def test_success_is_zero(self):
        assert SUCCESS == 0x00000000

    def test_unknown_is_max(self):
        assert UNKNOWN == 0xFFFFFFFF

    def test_all_codes_unique(self):
        codes = [
            SUCCESS, UNKNOWN,
            E00_IO_TIMEOUT, E01_IO_CORRUPT,
            E10_GPU_OOM, E11_GPU_LAUNCH_FAIL,
            E20_NUM_NAN, E21_NUM_NO_CONVERGE,
            E30_DET_MISMATCH, E31_SCHEMA_VERSION,
            E41_RESOURCE_LIMIT,
            E50_PH_ABORT, E53_PH4_BUDGET_EXCEEDED, E54_PH4_INVALID_INPUT,
            E60_NUMA_BIND_FAIL, E61_NUMA_AFFINITY_FAIL, E62_NUMA_MIGRATION_ERROR,
            E70_PRECISION_DOWNGRADE, E71_PRECISION_LOSS, E72_PRECISION_UNDERFLOW,
            E73_PRECISION_CATASTROPHIC,
            E81_MATRIX_EMPTY, E82_MATRIX_SPARSE,
            E83_NO_PIVOTS_FOUND, E84_INSUFFICIENT_PIVOTS,
            E85_MATRIX_STRUCTURE, E86_NO_PERSISTENCE_PAIRS,
            E87_INVALID_BETTI_NUMBERS, E88_INVALID_SIMPLICES,
            E89_BOUNDARY_ERROR, E90_VALIDATION_ERROR,
            E91_SHAPE_ERROR, E92_DTYPE_ERROR,
            E93_DEVICE_ERROR, E94_BACKEND_REQUIRED,
            E95_EMPTY_COMPLEX,
            E99_COMPUTATION_TIMEOUT, E100_CONVERGENCE_FAILURE,
        ]
        assert len(codes) == len(set(codes))

    def test_category_byte_pattern(self):
        """Test that error code category nibbles map to correct ErrorCategory values."""
        # IO_INFRA = 1 → codes in 0x1xx range
        assert (E00_IO_TIMEOUT >> 8) & 0xF == ErrorCategory.IO_INFRA
        # GPU_COMPUTE = 2 → codes in 0x2xx
        assert (E10_GPU_OOM >> 8) & 0xF == ErrorCategory.GPU_COMPUTE
        # NUMERICAL = 3
        assert (E20_NUM_NAN >> 8) & 0xF == ErrorCategory.NUMERICAL
        # DETERMINISM = 4
        assert (E30_DET_MISMATCH >> 8) & 0xF == ErrorCategory.DETERMINISM
        # CAPACITY = 5
        assert (E41_RESOURCE_LIMIT >> 8) & 0xF == ErrorCategory.CAPACITY
        # ALGORITHMIC = 6
        assert (E50_PH_ABORT >> 8) & 0xF == ErrorCategory.ALGORITHMIC
        # OPERATIONAL = 7
        assert (E90_VALIDATION_ERROR >> 8) & 0xF == ErrorCategory.OPERATIONAL


# ── translate_cpp_exception ────────────────────────────────────────────────


class FakeCppException(Exception):
    """Simulates a C++ binding exception with an error_code attribute."""
    def __init__(self, msg: str = "", error_code: int = UNKNOWN):
        super().__init__(msg)
        self.error_code = error_code


class TestTranslateCppException:
    def test_known_code_maps_to_correct_exception(self):
        ex = FakeCppException("matrix empty", E81_MATRIX_EMPTY)
        result = translate_cpp_exception(ex)
        assert isinstance(result, ShapeMismatchError)

    def test_unknown_code_maps_to_nerve_error(self):
        ex = FakeCppException("something weird", 0xDEADBEEF)
        result = translate_cpp_exception(ex)
        assert isinstance(result, NerveError)

    def test_no_error_code_attribute(self):
        ex = Exception("plain exception")
        result = translate_cpp_exception(ex)
        assert isinstance(result, NerveError)
        assert "NerveError" in str(result)

    def test_none_error_code(self):
        ex = FakeCppException("test", None)  # type: ignore[arg-type]
        result = translate_cpp_exception(ex)
        assert isinstance(result, NerveError)

    def test_bool_error_code(self):
        ex = FakeCppException("test", True)  # type: ignore[arg-type]
        result = translate_cpp_exception(ex)
        assert isinstance(result, NerveError)

    def test_empty_message(self):
        ex = FakeCppException("", E10_GPU_OOM)
        result = translate_cpp_exception(ex)
        assert isinstance(result, GPUMemoryError)

    def test_not_an_exception(self):
        with pytest.raises(TypeError, match="must be an Exception"):
            translate_cpp_exception("not an exception")  # type: ignore[arg-type]

    def test_none_input(self):
        with pytest.raises(TypeError, match="must be an Exception"):
            translate_cpp_exception(None)  # type: ignore[arg-type]

    # ── Map each error code to its exception type ──

    def test_e00_io_timeout(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E00_IO_TIMEOUT)), NerveIOError)

    def test_e01_io_corrupt(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E01_IO_CORRUPT)), NerveIOError)

    def test_e10_gpu_oom(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E10_GPU_OOM)), GPUMemoryError)

    def test_e11_gpu_launch_fail(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E11_GPU_LAUNCH_FAIL)), GPULaunchError)

    def test_e20_num_nan(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E20_NUM_NAN)), NumericalError)

    def test_e21_num_no_converge(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E21_NUM_NO_CONVERGE)), ConvergenceError)

    def test_e30_det_mismatch(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E30_DET_MISMATCH)), DeterminismError)

    def test_e31_schema_version(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E31_SCHEMA_VERSION)), DeterminismError)

    def test_e41_resource_limit(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E41_RESOURCE_LIMIT)), OutOfMemoryError)

    def test_e50_ph_abort(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E50_PH_ABORT)), PersistenceError)

    def test_e53_ph4_budget(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E53_PH4_BUDGET_EXCEEDED)), BudgetExceededError)

    def test_e54_ph4_invalid(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E54_PH4_INVALID_INPUT)), TypeMismatchError)

    def test_e60_numa_bind(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E60_NUMA_BIND_FAIL)), NUMAError)

    def test_e61_numa_affinity(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E61_NUMA_AFFINITY_FAIL)), NUMAError)

    def test_e62_numa_migration(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E62_NUMA_MIGRATION_ERROR)), NUMAError)

    def test_e70_precision_downgrade(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E70_PRECISION_DOWNGRADE)), PrecisionError)

    def test_e71_precision_loss(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E71_PRECISION_LOSS)), PrecisionError)

    def test_e72_precision_underflow(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E72_PRECISION_UNDERFLOW)), PrecisionError)

    def test_e73_catastrophic(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E73_PRECISION_CATASTROPHIC)), NumericalInstabilityError)

    def test_e81_matrix_empty(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E81_MATRIX_EMPTY)), ShapeMismatchError)

    def test_e82_matrix_sparse(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E82_MATRIX_SPARSE)), MatrixStructureError)

    def test_e83_no_pivots(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E83_NO_PIVOTS_FOUND)), PersistenceError)

    def test_e84_insufficient_pivots(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E84_INSUFFICIENT_PIVOTS)), PersistenceError)

    def test_e85_matrix_structure(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E85_MATRIX_STRUCTURE)), MatrixStructureError)

    def test_e86_no_persistence_pairs(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E86_NO_PERSISTENCE_PAIRS)), PersistenceError)

    def test_e87_invalid_betti(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E87_INVALID_BETTI_NUMBERS)), BettiError)

    def test_e88_invalid_simplices(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E88_INVALID_SIMPLICES)), InvalidSimplexError)

    def test_e89_boundary_error(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E89_BOUNDARY_ERROR)), InvalidSimplexError)

    def test_e95_empty_complex(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E95_EMPTY_COMPLEX)), ShapeMismatchError)

    def test_e99_timeout(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E99_COMPUTATION_TIMEOUT)), ConvergenceError)

    def test_e100_convergence_failure(self):
        assert isinstance(translate_cpp_exception(FakeCppException("x", E100_CONVERGENCE_FAILURE)), ConvergenceError)

    # ── String message preservation ──

    def test_message_preserved(self):
        ex = FakeCppException("the GPU is on fire", E11_GPU_LAUNCH_FAIL)
        result = translate_cpp_exception(ex)
        assert "the GPU is on fire" in str(result)

    def test_code_in_formatted_message(self):
        ex = FakeCppException("bad", E82_MATRIX_SPARSE)
        result = translate_cpp_exception(ex)
        assert "0x00000901" in str(result)
