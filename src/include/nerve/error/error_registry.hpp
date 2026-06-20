
#pragma once
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>

namespace nerve::error
{

enum class TDAErrorCode : int
{
    Success = 0,

    SingularMatrix = 100,
    InvalidFieldOperation = 101,
    NegativeDistance = 102,
    NonMonotoneFiltration = 103,
    InvalidPivot = 104,
    ReductionFailed = 105,

    AllocationFailed = 200,
    BufferOverflow = 201,
    MemoryPressure = 202,

    DataRace = 300,
    DeadlockDetected = 301,
    AtomicityViolation = 302,
    DeterminismViolation = 303,

    EmptyPointCloud = 400,
    InvalidDimension = 401,
    NaNInInput = 402,
    InvalidSimplices = 403,
    InvalidInput = 404,

    InvalidFiltrationValue = 500,
    NonMonotoneFiltrationValue = 501,
    ComplexError = 502,

    InvalidPivotReduction = 600,
    ReductionFailedReduction = 601,
    NoPivotsFound = 602,
    MatrixEmpty = 603,

    PHAbort = 700,
    LaplacianAbort = 701,
    ConvergenceFailure = 702,
    ComputationTimeout = 703,

    ResourceLimit = 800,
    CapacityExceeded = 801,
    RecoveryActivated = 802,
    GpuKernelLaunchFailed = 803,
    GpuSyncFailed = 804,
    GpuInvalidDevice = 805,
    PrecisionInsufficient = 900,

    Unknown = 999
};

class TDAErrorCategory : public std::error_category
{
public:
    static const TDAErrorCategory &instance() noexcept
    {
        static TDAErrorCategory cat;
        return cat;
    }

    const char *name() const noexcept override { return "tda"; }

    std::string message(int ev) const override
    {
        switch (static_cast<TDAErrorCode>(ev))
        {
            case TDAErrorCode::Success:
                return "ok";
            case TDAErrorCode::SingularMatrix:
                return "Matrix is singular: try a different field (e.g., Z2 instead of Z3) or "
                       "check your boundary matrix construction";
            case TDAErrorCode::InvalidFieldOperation:
                return "Invalid field operation: check that the field characteristic supports the "
                       "requested arithmetic";
            case TDAErrorCode::NegativeDistance:
                return "Negative distance: ensure your metric is non-negative and input "
                       "coordinates are valid";
            case TDAErrorCode::NonMonotoneFiltration:
                return "Non-monotone filtration: ensure simplex values increase along inclusion "
                       "(check edge weights and vertex values)";
            case TDAErrorCode::InvalidPivot:
                return "Invalid pivot: verify reduction algorithm state or check for corrupted "
                       "input data";
            case TDAErrorCode::ReductionFailed:
                return "Matrix reduction failed: try increasing max dimension or reducing point "
                       "count";
            case TDAErrorCode::AllocationFailed:
                return "Memory allocation failed: reduce problem size, enable streaming, or use a "
                       "machine with more RAM/VRAM";
            case TDAErrorCode::BufferOverflow:
                return "Buffer overflow: increase buffer size parameters or reduce input size";
            case TDAErrorCode::MemoryPressure:
                return "Memory pressure: enable streaming, reduce max_dim, or try sparsification";
            case TDAErrorCode::DataRace:
                return "Data race: set deterministic=true or use single-threaded execution";
            case TDAErrorCode::DeadlockDetected:
                return "Deadlock: reduce concurrency or check for circular dependencies in "
                       "parallel tasks";
            case TDAErrorCode::AtomicityViolation:
                return "Atomicity violation: enable determinism checks and set deterministic=true";
            case TDAErrorCode::DeterminismViolation:
                return "Determinism violated: non-deterministic GPU execution detected; set "
                       "deterministic=true or use CPU path";
            case TDAErrorCode::EmptyPointCloud:
                return "Empty point cloud: provide at least 1 point with dimension >= 1";
            case TDAErrorCode::InvalidDimension:
                return "Invalid dimension: must be >= 1 (for VR) or appropriate for your "
                       "filtration type";
            case TDAErrorCode::NaNInInput:
                return "NaN detected: check your input data for missing or invalid values";
            case TDAErrorCode::InvalidSimplices:
                return "Invalid simplices: check simplex indices are valid and within vertex range";
            case TDAErrorCode::InvalidInput:
                return "Invalid input: check parameter ranges and data types";
            case TDAErrorCode::InvalidFiltrationValue:
                return "Invalid filtration: ensure values are finite and ordered correctly";
            case TDAErrorCode::NonMonotoneFiltrationValue:
                return "Non-monotone filtration value: ensure simplex values increase along "
                       "inclusion";
            case TDAErrorCode::ComplexError:
                return "Complex error: check for duplicate or invalid simplices in the complex";
            case TDAErrorCode::InvalidPivotReduction:
                return "Invalid pivot: check reduction state or restart with fresh input";
            case TDAErrorCode::ReductionFailedReduction:
                return "Reduction failed: try with fewer points or lower max dimension";
            case TDAErrorCode::NoPivotsFound:
                return "No pivots found: all features are essential; try increasing max_dim or "
                       "checking input";
            case TDAErrorCode::MatrixEmpty:
                return "Empty matrix: provide non-empty boundary matrix";
            case TDAErrorCode::PHAbort:
                return "Persistence aborted: computation was cancelled or hit a resource limit";
            case TDAErrorCode::LaplacianAbort:
                return "Laplacian aborted: computation exceeded resources; try a smaller complex";
            case TDAErrorCode::ConvergenceFailure:
                return "Convergence failure: increase iteration limit, adjust tolerance, or check "
                       "input stability";
            case TDAErrorCode::ComputationTimeout:
                return "Computation timed out: increase time_budget_ms, reduce problem size, or "
                       "enable acceleration";
            case TDAErrorCode::ResourceLimit:
                return "Resource limit: reduce input size, enable streaming, or increase memory "
                       "limits";
            case TDAErrorCode::CapacityExceeded:
                return "Capacity exceeded: input exceeds configured limits; increase capacity or "
                       "split workload";
            case TDAErrorCode::RecoveryActivated:
                return "Recovery activated: a fallback path was used due to hardware error; "
                       "results may be degraded";
            case TDAErrorCode::GpuKernelLaunchFailed:
                return "Kernel launch failed: reduce grid/block dimensions, check GPU "
                       "compatibility, or update drivers";
            case TDAErrorCode::GpuSyncFailed:
                return "GPU sync failed: kernel may have timed out; try smaller batches or disable "
                       "GPU acceleration";
            case TDAErrorCode::GpuInvalidDevice:
                return "Invalid GPU device: check cudaVisibleDevices, driver version, and device "
                       "count";
            case TDAErrorCode::PrecisionInsufficient:
                return "Precision insufficient: use float64 or increase precision parameters";
            default:
                return "Unknown error: check logs for details and consider filing a bug report";
        }
    }
};

inline std::error_code makeErrorCode(TDAErrorCode e) noexcept
{
    return {static_cast<int>(e), TDAErrorCategory::instance()};
}

template <typename T>
class Result
{
public:
    static Result ok(T value) { return Result(std::move(value)); }

    static Result err(TDAErrorCode code, std::string detail = {},
                      std::source_location loc = std::source_location::current())
    {
        return Result(makeErrorCode(code), std::move(detail), loc);
    }

    bool isOk() const noexcept { return !ec_; }
    bool isErr() const noexcept { return static_cast<bool>(ec_); }

    T &value()
    {
        if (!isOk())
        {
            throw std::runtime_error("Attempted to access value from error result: " +
                                     ec_.message());
        }
        return *val_;
    }

    const T &value() const
    {
        if (!isOk())
        {
            throw std::runtime_error("Attempted to access value from error result: " +
                                     ec_.message());
        }
        return *val_;
    }

    std::error_code error() const noexcept { return ec_; }
    std::string_view detail() const noexcept { return detail_; }
    std::source_location where() const noexcept { return loc_; }

    template <typename F>
    auto map(F &&f) -> Result<std::invoke_result_t<F, T>>
    {
        if (isErr())
        {
            return Result<std::invoke_result_t<F, T>>::err(
                static_cast<TDAErrorCode>(error().value()), std::string(detail_), loc_);
        }
        return Result<std::invoke_result_t<F, T>>::ok(f(value()));
    }

    template <typename F>
    auto andThen(F &&f) -> std::invoke_result_t<F, T>
    {
        if (isErr())
        {
            return std::invoke_result_t<F, T>::err(static_cast<TDAErrorCode>(error().value()),
                                                   std::string(detail_), loc_);
        }
        return f(value());
    }

private:
    std::optional<T> val_;
    std::error_code ec_;
    std::string detail_;
    std::source_location loc_;

    Result(T &&value)
        : val_(std::move(value))
        , ec_()
    {}
    Result(std::error_code ec, std::string detail, std::source_location loc)
        : ec_(ec)
        , detail_(std::move(detail))
        , loc_(loc)
    {}
};

template <>
class Result<void>
{
public:
    static Result ok() { return Result(); }

    static Result err(TDAErrorCode code, std::string detail = {},
                      std::source_location loc = std::source_location::current())
    {
        return Result(makeErrorCode(code), std::move(detail), loc);
    }

    bool isOk() const noexcept { return !ec_; }
    bool isErr() const noexcept { return static_cast<bool>(ec_); }

    void value() const
    {
        if (!isOk())
        {
            throw std::runtime_error("Attempted to access value from error result: " +
                                     ec_.message());
        }
    }

    std::error_code error() const noexcept { return ec_; }
    std::string_view detail() const noexcept { return detail_; }
    std::source_location where() const noexcept { return loc_; }

private:
    std::error_code ec_;
    std::string detail_;
    std::source_location loc_;

    Result()
        : ec_()
    {}
    Result(std::error_code ec, std::string detail, std::source_location loc)
        : ec_(ec)
        , detail_(std::move(detail))
        , loc_(loc)
    {}
};

// Convenience macros for error handling
#ifndef TRY_RESULT
#define TRY_RESULT(expr)                                                                           \
    do                                                                                             \
    {                                                                                              \
        auto _result = (expr);                                                                     \
        if (_result.isErr())                                                                       \
        {                                                                                          \
            return decltype(_result)::err(static_cast<TDAErrorCode>(_result.error().value()),      \
                                          std::string(_result.detail()), _result.where());         \
        }                                                                                          \
    } while (0)
#endif

#ifndef TRY_ASSIGN
#define TRY_ASSIGN(var, expr)                                                                      \
    do                                                                                             \
    {                                                                                              \
        auto _result = (expr);                                                                     \
        if (_result.isErr())                                                                       \
        {                                                                                          \
            return decltype(_result)::err(static_cast<TDAErrorCode>(_result.error().value()),      \
                                          std::string(_result.detail()), _result.where());         \
        }                                                                                          \
        var = _result.value();                                                                     \
    } while (0)
#endif

} // namespace nerve::error

// Enable implicit conversion for std::error_code
namespace std
{
template <>
struct is_error_code_enum<nerve::error::TDAErrorCode> : true_type
{};
} // namespace std
