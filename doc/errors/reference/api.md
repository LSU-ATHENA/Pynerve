# API

## C++ API

```cpp
#include <nerve/errors/errors.hpp>

namespace nerve::errors {

// Error codes (component-prefixed hex)
enum class ErrorCode : uint32_t {
    SUCCESS                 = 0x00000000,
    // IO / infrastructure (0x000001xx)
    E00_IO_TIMEOUT          = 0x00000100,
    E01_IO_CORRUPT          = 0x00000101,
    // GPU (0x000002xx)
    E10_GPU_OOM             = 0x00000200,
    E11_GPU_LAUNCH_FAIL     = 0x00000201,
    // Numerical (0x000003xx)
    E20_NUM_NAN             = 0x00000300,
    E21_NUM_NO_CONVERGE     = 0x00000301,
    // Determinism (0x000004xx)
    E30_DET_MISMATCH        = 0x00000400,
    E31_SCHEMA_VERSION      = 0x00000401,
    // Capacity (0x000005xx)
    E40_CPU_OVERLOAD        = 0x00000500,
    E41_RESOURCE_LIMIT      = 0x00000501,
    // Algorithmic / PH (0x000006xx)
    E50_PH_ABORT            = 0x00000600,
    E51_LAPLACIAN_ABORT     = 0x00000601,
    E52_PH4_ABORT           = 0x00000602,
    E53_PH4_BUDGET_EXCEEDED = 0x00000603,
    E54_PH4_INVALID_INPUT   = 0x00000604,
    E55_PH4_CONVERGENCE_FAIL = 0x00000605,
    E56_PH4_WITNESS_ERROR   = 0x00000606,
    E11_PH5_OVERFLOW        = 0x00000607,
    E12_PH6_OVERFLOW        = 0x00000608,
    E13_PH_HIGHDIM_PRECISION = 0x00000609,
    // NUMA / affinity (0x000007xx)
    E60_NUMA_BIND_FAIL      = 0x00000700,
    E61_NUMA_AFFINITY_FAIL  = 0x00000701,
    E62_NUMA_MIGRATION_ERROR = 0x00000702,
    // Precision (0x000008xx)
    E70_PRECISION_DOWNGRADE = 0x00000800,
    E71_PRECISION_LOSS      = 0x00000801,
    E72_PRECISION_UNDERFLOW = 0x00000802,
    E73_PRECISION_CATASTROPHIC = 0x00000803,
    // Matrix / homology (0x000009xx)
    E81_MATRIX_EMPTY        = 0x00000900,
    E82_MATRIX_SPARSE       = 0x00000901,
    E83_NO_PIVOTS_FOUND     = 0x00000902,
    E84_INSUFFICIENT_PIVOTS = 0x00000903,
    E85_MATRIX_STRUCTURE    = 0x00000904,
    E86_NO_PERSISTENCE_PAIRS = 0x00000905,
    E87_INVALID_BETTI       = 0x00000906,
    E88_INVALID_SIMPLICES   = 0x00000907,
    E89_BOUNDARY_ERROR      = 0x00000908,
    E90_COMPLEX_ERROR       = 0x00000909,
    E91_REDUCED_HOMOLOGY    = 0x00000910,
    E92_BETTI_MISMATCH      = 0x00000911,
    E93_COMPUTATION_TIMEOUT = 0x0000090C,
    E94_CONVERGENCE_FAILURE = 0x0000090D,
    UNKNOWN                 = 0xFFFFFFFF,
};

// Categories
enum class ErrorCategory : uint8_t {
    SUCCESS, IO_INFRA, GPU_COMPUTE, NUMERICAL, DETERMINISM,
    CAPACITY, ALGORITHMIC, OPERATIONAL, PH4_RESEARCH,
    PH5_PH6_HIGHDIM, NUMA_AFFINITY, PRECISION, UNKNOWN_CATEGORY = 255,
};

// Severity
enum class ErrorSeverity : uint8_t {
    INFO = 0, WARNING = 1, ERROR = 2, CRITICAL = 3,
};

// Metadata for each error code
struct ErrorMetadata {
    ErrorCode code;
    ErrorCategory category;
    ErrorSeverity severity;
    std::string name;
    std::string description;
    std::string action_hint;
    bool isTransient;
    bool isUserError;
};

// Error context (operation tracking)
struct ErrorContext {
    std::string operation_name;
    std::string component_name;
    std::string session_id;
    std::string request_id;
    std::unordered_map<std::string, std::string> metadata;
    uint64_t timestampNs;
    double durationMs;
    std::string toJson() const;
};

// Error result type (monadic)
template <typename T>
class ErrorResult {
    static ErrorResult success(T&& value);
    static ErrorResult error(ErrorCode code, std::string_view msg = {});
    bool isSuccess() const;
    bool isError() const;
    bool isOk() const;
    bool isErr() const;
    const T& value() const;
    T& value();
    ErrorCode errorCode() const;
    std::string errorMessage() const;
    // -> use .value() after checking .isOk()
};

template <>
class ErrorResult<void> {
    // Same interface, no value accessor
};

// Error registry (singleton)
class ErrorRegistry {
    static ErrorRegistry& instance();
    const ErrorMetadata& getMetadata(ErrorCode) const;
    ErrorCategory getCategory(ErrorCode) const;
    ErrorSeverity getSeverity(ErrorCode) const;
    bool isTransient(ErrorCode) const;
    bool isUserError(ErrorCode) const;
    std::vector<ErrorCode> getAllCodes() const;
    std::vector<ErrorCode> getCodesByCategory(ErrorCategory) const;
    void reportError(ErrorCode code, const ErrorContext& ctx = {});
    bool hasOperationFailed(const std::string& op) const;
    void clearFailedOperation(const std::string& op);
};

// Configurable error system
class ConfigurableErrorSystemBase {
    static ConfigurableErrorSystemBase& instance();
    void setPolicy(const ErrorPolicy&);
    void report(const Error&);
    void setErrorEnabled(ErrorCode, bool);
    bool isErrorEnabled(ErrorCode) const;
};

}
```

### Python API

```python
from pynerve.exceptions import (
    # Error code constants
    SUCCESS, E00_IO_TIMEOUT, E01_IO_CORRUPT,
    E10_GPU_OOM, E11_GPU_LAUNCH_FAIL,
    E20_NUM_NAN, E21_NUM_NO_CONVERGE,
    E30_DET_MISMATCH, E31_SCHEMA_VERSION,
    E40_CPU_OVERLOAD, E41_RESOURCE_LIMIT,
    E50_PH_ABORT, E51_LAPLACIAN_ABORT,
    E52_PH4_ABORT, E53_PH4_BUDGET_EXCEEDED,
    E54_PH4_INVALID_INPUT, E55_PH4_CONVERGENCE_FAIL,
    E81_MATRIX_EMPTY, E85_MATRIX_STRUCTURE,
    E87_INVALID_BETTI, E88_INVALID_SIMPLICES,
    E89_EMPTY_COMPLEX, E93_COMPUTATION_TIMEOUT,
    E94_CONVERGENCE_FAILURE, UNKNOWN,

    # Exception classes
    NerveError,               # base (error_code = UNKNOWN)
    ShapeMismatchError,       # E81_MATRIX_EMPTY
    DimensionError,           # E88_INVALID_SIMPLICES
    TypeMismatchError,        # E54_PH4_INVALID_INPUT
    InvalidSimplexError,      # E88_INVALID_SIMPLICES
    MatrixStructureError,     # E85_MATRIX_STRUCTURE
    GPUError,                 # E10_GPU_OOM
    GPUMemoryError,           # E10_GPU_OOM
    GPULaunchError,           # E11_GPU_LAUNCH_FAIL
    NerveMemoryError,         # E41_RESOURCE_LIMIT
    OutOfMemoryError,         # E41_RESOURCE_LIMIT
    AllocationError,          # E41_RESOURCE_LIMIT
    NumericalError,           # E20_NUM_NAN
    ConvergenceError,         # E21_NUM_NO_CONVERGE
    PrecisionError,           # E71_PRECISION_LOSS
    NumericalInstabilityError, # E73_PRECISION_CATASTROPHIC
    InvalidArgumentError,     # E54_PH4_INVALID_INPUT
    BudgetExceededError,      # E53_PH4_BUDGET_EXCEEDED
    NerveIOError,             # E00_IO_TIMEOUT
    DeterminismError,         # E30_DET_MISMATCH
    NUMAError,                # E60_NUMA_BIND_FAIL
    PersistenceError,         # E50_PH_ABORT
    BettiError,               # E87_INVALID_BETTI

    # Translation
    translate_cpp_exception,  # C++ -> Python exception bridge
)
```


[Back to index](index.md)
