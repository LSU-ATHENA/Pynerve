
#include "nerve/errors/errors.hpp"

#include <sstream>
#include <unordered_map>

namespace nerve::errors
{
ErrorRegistry &ErrorRegistry::instance()
{
    static ErrorRegistry instance;
    return instance;
}
ErrorRegistry::ErrorRegistry()
{
    initializeMetadata();
}
const ErrorMetadata &ErrorRegistry::getMetadata(ErrorCode code) const
{
    auto it = metadata_.find(code);
    if (it != metadata_.end())
    {
        return it->second;
    }
    static const ErrorMetadata unknown_metadata = {ErrorCode::UNKNOWN,
                                                   ErrorCategory::UNKNOWN_CATEGORY,
                                                   ErrorSeverity::ERROR,
                                                   "UNKNOWN",
                                                   "Unknown or unregistered error code",
                                                   "Check error registration",
                                                   false,
                                                   false,
                                                   false};
    return unknown_metadata;
}
ErrorCategory ErrorRegistry::getCategory(ErrorCode code) const
{
    return getMetadata(code).category;
}
ErrorSeverity ErrorRegistry::getSeverity(ErrorCode code) const
{
    return getMetadata(code).severity;
}
bool ErrorRegistry::isTransient(ErrorCode code) const
{
    return getMetadata(code).isTransient;
}
bool ErrorRegistry::isUserError(ErrorCode code) const
{
    return getMetadata(code).isUserError;
}
std::vector<ErrorCode> ErrorRegistry::getAllCodes() const
{
    std::vector<ErrorCode> codes;
    codes.reserve(metadata_.size());
    for (const auto &pair : metadata_)
    {
        codes.push_back(pair.first);
    }
    return codes;
}
std::vector<ErrorCode> ErrorRegistry::getCodesByCategory(ErrorCategory category) const
{
    std::vector<ErrorCode> codes;
    for (const auto &pair : metadata_)
    {
        if (pair.second.category == category)
        {
            codes.push_back(pair.first);
        }
    }
    return codes;
}
void ErrorRegistry::initializeMetadata()
{
    registerError(ErrorCode::SUCCESS, ErrorCategory::SUCCESS, ErrorSeverity::INFO, "SUCCESS",
                  "Operation completed successfully", "No action needed", false, false, false);
    registerError(ErrorCode::E00_IO_TIMEOUT, ErrorCategory::IO_INFRA, ErrorSeverity::ERROR,
                  "E00_IO_TIMEOUT", "Underlying IO timed out (disk / network)",
                  "Check network connectivity and retry", true, false, false);
    registerError(ErrorCode::E01_IO_CORRUPT, ErrorCategory::IO_INFRA, ErrorSeverity::ERROR,
                  "E01_IO_CORRUPT", "Input message/corpus corrupted (schema mismatch)",
                  "Validate input format", false, true, false);
    registerError(ErrorCode::E10_GPU_OOM, ErrorCategory::GPU_COMPUTE, ErrorSeverity::CRITICAL,
                  "E10_GPU_OOM", "GPU out-of-memory", "Reduce batch size or use larger GPU", false,
                  false, true);
    registerError(ErrorCode::E11_GPU_LAUNCH_FAIL, ErrorCategory::GPU_COMPUTE, ErrorSeverity::ERROR,
                  "E11_GPU_LAUNCH_FAIL", "Kernel launch or driver failure",
                  "Check GPU drivers and kernel compatibility", false, false, true);
    registerError(ErrorCode::E20_NUM_NAN, ErrorCategory::NUMERICAL, ErrorSeverity::CRITICAL,
                  "E20_NUM_NAN", "NaN/Inf or catastrophic numerical instability",
                  "Check input data and algorithm parameters", false, false, false);
    registerError(ErrorCode::E21_NUM_NO_CONVERGE, ErrorCategory::NUMERICAL, ErrorSeverity::ERROR,
                  "E21_NUM_NO_CONVERGE", "Iterative solver failed to converge",
                  "Adjust convergence parameters or algorithm", false, false, false);
    registerError(ErrorCode::E30_DET_MISMATCH, ErrorCategory::DETERMINISM, ErrorSeverity::ERROR,
                  "E30_DET_MISMATCH", "Checksum/params_hash mismatch on replay",
                  "Verify deterministic replay setup", false, false, false);
    registerError(ErrorCode::E31_SCHEMA_VERSION, ErrorCategory::DETERMINISM, ErrorSeverity::ERROR,
                  "E31_SCHEMA_VERSION", "Incompatible schema / params version",
                  "Update schema or params version", false, false, false);
    registerError(ErrorCode::E40_CPU_OVERLOAD, ErrorCategory::CAPACITY, ErrorSeverity::ERROR,
                  "E40_CPU_OVERLOAD", "CPU queue saturated / deadline missed",
                  "Reduce load or increase resources", true, false, false);
    registerError(ErrorCode::E41_RESOURCE_LIMIT, ErrorCategory::CAPACITY, ErrorSeverity::CRITICAL,
                  "E41_RESOURCE_LIMIT", "Memory or file descriptor limits reached",
                  "Free resources or increase limits", false, false, false);
    registerError(ErrorCode::E50_PH_ABORT, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E50_PH_ABORT", "PH aborted due to time budget or internal abort",
                  "Check time budget and input parameters", false, false, false);
    registerError(ErrorCode::E51_LAPLACIAN_ABORT, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E51_LAPLACIAN_ABORT", "Laplacian solver aborted",
                  "Check matrix properties and solver parameters", false, false, false);
    registerError(ErrorCode::E52_PH4_ABORT, ErrorCategory::PH4_RESEARCH, ErrorSeverity::ERROR,
                  "E52_PH4_ABORT", "PH4 computation aborted",
                  "Check PH4 parameters and memory budget", false, false, false);
    registerError(ErrorCode::E53_PH4_BUDGET_EXCEEDED, ErrorCategory::PH4_RESEARCH,
                  ErrorSeverity::ERROR, "E53_PH4_BUDGET_EXCEEDED", "PH4 memory budget exceeded",
                  "Reduce PH4 complexity or increase memory budget", false, false, false);
    registerError(ErrorCode::E54_PH4_INVALID_INPUT, ErrorCategory::PH4_RESEARCH,
                  ErrorSeverity::ERROR, "E54_PH4_INVALID_INPUT", "Invalid PH4 input parameters",
                  "Validate PH4 input parameters", true, false, false);
    registerError(ErrorCode::E55_PH4_SPARSE_CONVERGENCE_FAIL, ErrorCategory::PH4_RESEARCH,
                  ErrorSeverity::ERROR, "E55_PH4_SPARSE_CONVERGENCE_FAIL",
                  "PH4 sparse solver failed to converge", "Adjust sparse solver parameters", false,
                  false, false);
    registerError(ErrorCode::E56_PH4_WITNESS_SAMPLING_ERROR, ErrorCategory::PH4_RESEARCH,
                  ErrorSeverity::ERROR, "E56_PH4_WITNESS_SAMPLING_ERROR",
                  "PH4 witness sampling failed", "Check witness sampling strategy", false, false,
                  false);
    registerError(ErrorCode::E11_PH5_OVERFLOW, ErrorCategory::PH5_PH6_HIGHDIM, ErrorSeverity::ERROR,
                  "E11_PH5_OVERFLOW", "PH5 computation overflow (dimension explosion)",
                  "Reduce input complexity or use sparse representations", false, false, false);
    registerError(ErrorCode::E12_PH6_OVERFLOW, ErrorCategory::PH5_PH6_HIGHDIM, ErrorSeverity::ERROR,
                  "E12_PH6_OVERFLOW", "PH6 computation overflow (dimension explosion)",
                  "Reduce input complexity or use hierarchical sampling", false, false, false);
    registerError(ErrorCode::E13_PH_HIGHDIM_PRECISION, ErrorCategory::PH5_PH6_HIGHDIM,
                  ErrorSeverity::WARNING, "E13_PH_HIGHDIM_PRECISION",
                  "Higher-dimensional PH precision loss detected", "Use a higher precision mode",
                  false, false, false);
    registerError(ErrorCode::E60_NUMA_BIND_FAIL, ErrorCategory::NUMA_AFFINITY, ErrorSeverity::ERROR,
                  "E60_NUMA_BIND_FAIL", "Failed to bind thread to NUMA node",
                  "Check NUMA configuration and thread affinity", false, false, false);
    registerError(ErrorCode::E61_NUMA_AFFINITY_FAIL, ErrorCategory::NUMA_AFFINITY,
                  ErrorSeverity::ERROR, "E61_NUMA_AFFINITY_FAIL",
                  "NUMA affinity configuration failed", "Verify NUMA topology and configuration",
                  false, false, false);
    registerError(ErrorCode::E62_NUMA_MIGRATION_ERROR, ErrorCategory::NUMA_AFFINITY,
                  ErrorSeverity::ERROR, "E62_NUMA_MIGRATION_ERROR", "NUMA page migration failed",
                  "Check NUMA memory allocation patterns", false, false, false);
    registerError(ErrorCode::E70_PRECISION_DOWNGRADE, ErrorCategory::PRECISION,
                  ErrorSeverity::WARNING, "E70_PRECISION_DOWNGRADE",
                  "Computation downgraded to lower precision", "Consider higher precision", false,
                  false, false);
    registerError(ErrorCode::E71_PRECISION_LOSS, ErrorCategory::PRECISION, ErrorSeverity::ERROR,
                  "E71_PRECISION_LOSS", "Precision loss detected (NaN, overflow)",
                  "Check numerical stability and input range", false, false, false);
    registerError(ErrorCode::E72_PRECISION_UNDERFLOW, ErrorCategory::PRECISION,
                  ErrorSeverity::WARNING, "E72_PRECISION_UNDERFLOW", "Numerical underflow detected",
                  "Adjust scaling or use higher precision", false, false, false);
    registerError(ErrorCode::E73_PRECISION_CATASTROPHIC, ErrorCategory::PRECISION,
                  ErrorSeverity::CRITICAL, "E73_PRECISION_CATASTROPHIC",
                  "Catastrophic precision loss", "Abort computation and use higher precision",
                  false, false, false);
    registerError(ErrorCode::E81_MATRIX_EMPTY, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E81_MATRIX_EMPTY", "Input matrix is empty", "Check input data", false, false,
                  false);
    registerError(ErrorCode::E82_MATRIX_SPARSE, ErrorCategory::ALGORITHMIC, ErrorSeverity::WARNING,
                  "E82_MATRIX_SPARSE", "Matrix is very sparse", "Consider sparse algorithms", false,
                  false, false);
    registerError(ErrorCode::E83_NO_PIVOTS_FOUND, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E83_NO_PIVOTS_FOUND", "No pivots found in reduction", "Check matrix structure",
                  false, false, false);
    registerError(ErrorCode::E84_INSUFFICIENT_PIVOTS, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::WARNING, "E84_INSUFFICIENT_PIVOTS",
                  "Insufficient pivots for analysis", "Increase data size", false, false, false);
    registerError(ErrorCode::E85_MATRIX_STRUCTURE, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E85_MATRIX_STRUCTURE", "Invalid matrix structure", "Verify matrix construction",
                  false, false, false);
    registerError(ErrorCode::E86_NO_PERSISTENCE_PAIRS, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::WARNING, "E86_NO_PERSISTENCE_PAIRS", "No persistence pairs found",
                  "Check filtration", false, false, false);
    registerError(ErrorCode::E87_INVALID_BETTI_NUMBERS, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::ERROR, "E87_INVALID_BETTI_NUMBERS",
                  "Betti numbers are mathematically inconsistent", "Validate computation", false,
                  true, true);
    registerError(ErrorCode::E88_INVALID_SIMPLICES, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::ERROR, "E88_INVALID_SIMPLICES", "Invalid simplex structure",
                  "Check simplex construction", false, false, false);
    registerError(ErrorCode::E89_BOUNDARY_ERROR, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E89_BOUNDARY_ERROR", "Boundary computation error", "Check boundary matrix",
                  false, false, false);
    registerError(ErrorCode::E90_COMPLEX_ERROR, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E90_COMPLEX_ERROR", "Complex computation error", "Check complex structure",
                  false, false, false);
    registerError(ErrorCode::E91_INVALID_SIMPLICES, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::ERROR, "E91_INVALID_SIMPLICES", "Invalid simplex structure",
                  "Check simplex construction", false, false, false);
    registerError(ErrorCode::E91_REDUCED_HOMOLOGY, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::WARNING, "E91_REDUCED_HOMOLOGY", "Homology reduction completed",
                  "Normal operation", false, false, false);
    registerError(ErrorCode::E92_BETTI_MISMATCH, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E92_BETTI_MISMATCH", "Betti numbers mismatch expectations",
                  "Validate computation", false, false, false);
    registerError(ErrorCode::E92_MEMORY_PRESSURE, ErrorCategory::CAPACITY, ErrorSeverity::ERROR,
                  "E92_MEMORY_PRESSURE", "Memory pressure detected",
                  "Reduce problem size or increase memory", true, false, false);
    registerError(ErrorCode::E93_COMPUTATION_TIMEOUT, ErrorCategory::CAPACITY, ErrorSeverity::ERROR,
                  "E93_COMPUTATION_TIMEOUT", "Computation timed out",
                  "Increase time budget or reduce problem size", true, false, false);
    registerError(ErrorCode::E94_CONVERGENCE_FAILURE, ErrorCategory::NUMERICAL,
                  ErrorSeverity::ERROR, "E94_CONVERGENCE_FAILURE", "Algorithm failed to converge",
                  "Adjust convergence parameters", false, false, false);
    registerError(ErrorCode::E74_RACE_CONDITION, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::CRITICAL, "E74_RACE_CONDITION", "Concurrent access race condition",
                  "Add proper synchronization", false, true, true);
    registerError(ErrorCode::E75_ATOMICITY_VIOLATION, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::CRITICAL, "E75_ATOMICITY_VIOLATION",
                  "Operation atomicity violated", "Ensure atomic operations", false, true, true);
    registerError(ErrorCode::E76_CONSISTENCY_ERROR, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::CRITICAL, "E76_CONSISTENCY_ERROR",
                  "Mathematical consistency violation", "Validate mathematical properties", false,
                  true, true);
    registerError(ErrorCode::E77_ISOLATION_VIOLATION, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::ERROR, "E77_ISOLATION_VIOLATION", "Operation isolation violated",
                  "Check operation boundaries", false, false, false);
    registerError(ErrorCode::E78_DURABILITY_ERROR, ErrorCategory::ALGORITHMIC, ErrorSeverity::ERROR,
                  "E78_DURABILITY_ERROR", "Result durability issue", "Check data persistence",
                  false, false, false);
    registerError(ErrorCode::E79_TRANSACTION_ABORT, ErrorCategory::ALGORITHMIC,
                  ErrorSeverity::WARNING, "E79_TRANSACTION_ABORT", "Operation transaction aborted",
                  "Check transaction conditions", true, false, false);
    registerError(ErrorCode::E80_CAPACITY_EXCEEDED, ErrorCategory::CAPACITY, ErrorSeverity::ERROR,
                  "E80_CAPACITY_EXCEEDED", "System capacity exceeded", "Increase system resources",
                  true, false, false);
    registerError(ErrorCode::E95_INPUT_VALIDATION, ErrorCategory::INPUT_VALIDATION,
                  ErrorSeverity::ERROR, "E95_INPUT_VALIDATION", "Input validation failed",
                  "Check input dimensions, types, and value ranges", true, true, false);
    registerError(ErrorCode::E96_TYPE_MISMATCH, ErrorCategory::INPUT_VALIDATION,
                  ErrorSeverity::ERROR, "E96_TYPE_MISMATCH", "Type mismatch in input data",
                  "Ensure input types match expected schema", true, true, false);
    registerError(ErrorCode::E97_SHAPE_MISMATCH, ErrorCategory::INPUT_VALIDATION,
                  ErrorSeverity::ERROR, "E97_SHAPE_MISMATCH", "Shape/dimension mismatch",
                  "Verify input tensor dimensions", true, true, false);
    registerError(ErrorCode::E98_DIMENSION_MISMATCH, ErrorCategory::INPUT_VALIDATION,
                  ErrorSeverity::ERROR, "E98_DIMENSION_MISMATCH", "Dimensionality mismatch",
                  "Check point cloud or matrix dimensions", true, true, false);
    registerError(ErrorCode::E99_NOT_IMPLEMENTED, ErrorCategory::FEATURE_FLAG, ErrorSeverity::ERROR,
                  "E99_NOT_IMPLEMENTED", "Feature not yet implemented",
                  "Use a different code path or wait for implementation", false, false, false);
    registerError(ErrorCode::E9A_UNSUPPORTED_OPERATION, ErrorCategory::FEATURE_FLAG,
                  ErrorSeverity::ERROR, "E9A_UNSUPPORTED_OPERATION", "Operation not supported",
                  "Use an alternative operation or configuration", false, true, false);
    registerError(ErrorCode::E9B_INTERNAL_STATE, ErrorCategory::OPERATIONAL,
                  ErrorSeverity::CRITICAL, "E9B_INTERNAL_STATE", "Internal state corruption",
                  "Restart computation or file bug report", false, false, true);
    registerError(ErrorCode::EA0_MPI_COMM_FAIL, ErrorCategory::DISTRIBUTED, ErrorSeverity::ERROR,
                  "EA0_MPI_COMM_FAIL", "MPI communication failure",
                  "Check MPI environment and network connectivity", true, false, false);
    registerError(ErrorCode::EA1_MPI_RANK_FAIL, ErrorCategory::DISTRIBUTED, ErrorSeverity::ERROR,
                  "EA1_MPI_RANK_FAIL", "MPI rank operation failure",
                  "Verify MPI rank count and topology", false, false, false);
    registerError(ErrorCode::EA2_MPI_BARRIER_FAIL, ErrorCategory::DISTRIBUTED, ErrorSeverity::ERROR,
                  "EA2_MPI_BARRIER_FAIL", "MPI barrier synchronization failure",
                  "Check for deadlocked or crashed ranks", true, false, true);
    registerError(ErrorCode::EB0_FEATURE_DISABLED, ErrorCategory::FEATURE_FLAG,
                  ErrorSeverity::WARNING, "EB0_FEATURE_DISABLED", "Feature is disabled via config",
                  "Enable the feature flag in configuration", false, true, false);
    registerError(ErrorCode::EB1_FEATURE_MISCONFIGURED, ErrorCategory::FEATURE_FLAG,
                  ErrorSeverity::ERROR, "EB1_FEATURE_MISCONFIGURED", "Feature is misconfigured",
                  "Check feature configuration parameters", false, true, false);
    registerError(ErrorCode::UNKNOWN, ErrorCategory::UNKNOWN_CATEGORY, ErrorSeverity::CRITICAL,
                  "UNKNOWN", "Unknown/uncategorized error occurred", "Check error code", false,
                  false, false);
}
void ErrorRegistry::registerError(ErrorCode code, ErrorCategory category, ErrorSeverity severity,
                                  const std::string &name, const std::string &description,
                                  const std::string &action_hint, bool isTransient,
                                  bool isUserError, bool requires_human_intervention)
{
    metadata_[code] = {code,        category,    severity,
                       name,        description, action_hint,
                       isTransient, isUserError, requires_human_intervention};
}
namespace utils
{
std::string toString(ErrorCode code)
{
    return ErrorRegistry::instance().getMetadata(code).name;
}
std::string getErrorName(ErrorCode code)
{
    return ErrorRegistry::instance().getMetadata(code).name;
}
std::string getErrorDescription(ErrorCode code)
{
    return ErrorRegistry::instance().getMetadata(code).description;
}
std::string getCategoryString(ErrorCategory category)
{
    switch (category)
    {
        case ErrorCategory::SUCCESS:
            return "SUCCESS";
        case ErrorCategory::IO_INFRA:
            return "IO_INFRA";
        case ErrorCategory::GPU_COMPUTE:
            return "GPU_COMPUTE";
        case ErrorCategory::NUMERICAL:
            return "NUMERICAL";
        case ErrorCategory::DETERMINISM:
            return "DETERMINISM";
        case ErrorCategory::CAPACITY:
            return "CAPACITY";
        case ErrorCategory::ALGORITHMIC:
            return "ALGORITHMIC";
        case ErrorCategory::OPERATIONAL:
            return "OPERATIONAL";
        case ErrorCategory::PH4_RESEARCH:
            return "PH4_RESEARCH";
        case ErrorCategory::PH5_PH6_HIGHDIM:
            return "PH5_PH6_HIGHDIM";
        case ErrorCategory::NUMA_AFFINITY:
            return "NUMA_AFFINITY";
        case ErrorCategory::PRECISION:
            return "PRECISION";
        case ErrorCategory::UNKNOWN_CATEGORY:
            return "UNKNOWN_CATEGORY";
        default:
            return "INVALID_CATEGORY";
    }
}
std::string getSeverityString(ErrorSeverity severity)
{
    switch (severity)
    {
        case ErrorSeverity::INFO:
            return "INFO";
        case ErrorSeverity::WARNING:
            return "WARNING";
        case ErrorSeverity::ERROR:
            return "ERROR";
        case ErrorSeverity::CRITICAL:
            return "CRITICAL";
        default:
            return "INVALID_SEVERITY";
    }
}
std::string getMetricLabel(ErrorCode code)
{
    std::ostringstream oss;
    oss << getErrorName(code);
    return oss.str();
}
} // namespace utils
} // namespace nerve::errors
