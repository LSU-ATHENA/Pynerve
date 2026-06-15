"""Translation from C++ binding exceptions to Python exceptions."""

from __future__ import annotations

from typing import cast

from ._error_codes import (
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
    E90_COMPLEX_ERROR,
    E91_REDUCED_HOMOLOGY,
    E92_BETTI_MISMATCH,
    E95_EMPTY_COMPLEX,
    E99_COMPUTATION_TIMEOUT,
    E100_CONVERGENCE_FAILURE,
    UNKNOWN,
)
from .exceptions import (
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

_ERROR_CLASS_BY_CODE = {
    E81_MATRIX_EMPTY: ShapeMismatchError,
    E82_MATRIX_SPARSE: MatrixStructureError,
    E83_NO_PIVOTS_FOUND: PersistenceError,
    E84_INSUFFICIENT_PIVOTS: PersistenceError,
    E85_MATRIX_STRUCTURE: MatrixStructureError,
    E86_NO_PERSISTENCE_PAIRS: PersistenceError,
    E87_INVALID_BETTI_NUMBERS: BettiError,
    E88_INVALID_SIMPLICES: InvalidSimplexError,
    E89_BOUNDARY_ERROR: InvalidSimplexError,
    E90_COMPLEX_ERROR: InvalidSimplexError,
    E91_REDUCED_HOMOLOGY: PersistenceError,
    E92_BETTI_MISMATCH: BettiError,
    E54_PH4_INVALID_INPUT: TypeMismatchError,
    E10_GPU_OOM: GPUMemoryError,
    E11_GPU_LAUNCH_FAIL: GPULaunchError,
    E41_RESOURCE_LIMIT: OutOfMemoryError,
    E20_NUM_NAN: NumericalError,
    E21_NUM_NO_CONVERGE: ConvergenceError,
    E71_PRECISION_LOSS: PrecisionError,
    E73_PRECISION_CATASTROPHIC: NumericalInstabilityError,
    E53_PH4_BUDGET_EXCEEDED: BudgetExceededError,
    E00_IO_TIMEOUT: NerveIOError,
    E01_IO_CORRUPT: NerveIOError,
    E30_DET_MISMATCH: DeterminismError,
    E31_SCHEMA_VERSION: DeterminismError,
    E60_NUMA_BIND_FAIL: NUMAError,
    E61_NUMA_AFFINITY_FAIL: NUMAError,
    E62_NUMA_MIGRATION_ERROR: NUMAError,
    E50_PH_ABORT: PersistenceError,
    E70_PRECISION_DOWNGRADE: PrecisionError,
    E72_PRECISION_UNDERFLOW: PrecisionError,
    E95_EMPTY_COMPLEX: ShapeMismatchError,
    E99_COMPUTATION_TIMEOUT: ConvergenceError,
    E100_CONVERGENCE_FAILURE: ConvergenceError,
}


def translate_cpp_exception(cpp_exception: Exception) -> NerveError:
    """Translate a C++ binding exception into the Python error taxonomy."""
    if not isinstance(cpp_exception, Exception):
        raise TypeError("cpp_exception must be an Exception")
    error_code = getattr(cpp_exception, "error_code", UNKNOWN)
    if error_code is None or isinstance(error_code, bool) or not isinstance(error_code, int):
        error_code = UNKNOWN
    message = str(cpp_exception)
    exc_class = _ERROR_CLASS_BY_CODE.get(error_code, NerveError)
    if not message:
        message = f"{exc_class.__name__} (code=0x{error_code:08X})"
    return cast(NerveError, exc_class(message, cpp_message=message))


__all__ = [
    "translate_cpp_exception",
]
