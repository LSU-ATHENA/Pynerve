
#pragma once

#include "nerve/errors/errors.hpp"

#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nerve::errors
{

// ErrorResult template for handling operations that may fail.
template <typename T>
class ErrorResult
{
public:
    struct ErrorInfo
    {
        ErrorCode code;
        std::string message;
    };

    ErrorResult(T &&value)
        : value_(std::move(value))
        , error_code_(ErrorCode::SUCCESS)
    {}
    explicit ErrorResult(ErrorCode errorCode, std::string message = {})
        : error_code_(errorCode)
        , message_(std::move(message))
    {}
    bool isSuccess() const { return error_code_ == ErrorCode::SUCCESS; }
    bool isError() const { return error_code_ != ErrorCode::SUCCESS; }
    bool is_error() const { return isError(); }
    bool isOk() const { return isSuccess(); }
    bool isErr() const { return isError(); }
    const T &value() const
    {
        if (isError())
        {
            throw std::runtime_error("Attempted to access value from error result");
        }
        return *value_;
    }
    T &value()
    {
        if (isError())
        {
            throw std::runtime_error("Attempted to access value from error result");
        }
        return *value_;
    }
    T moveValue()
    {
        if (isError())
        {
            throw std::runtime_error("Attempted to move value from error result");
        }
        return std::move(*value_);
    }
    ErrorCode errorCode() const { return error_code_; }
    ErrorInfo error() const
    {
        const auto message =
            message_.empty() ? std::to_string(static_cast<uint32_t>(error_code_)) : message_;
        return ErrorInfo{error_code_, message};
    }
    static ErrorResult<T> success(T &&value) { return ErrorResult<T>(std::move(value)); }
    static ErrorResult<T> ok(T &&value) { return success(std::move(value)); }
    static ErrorResult<T> error(ErrorCode errorCode) { return ErrorResult<T>(errorCode); }
    static ErrorResult<T> error(ErrorCode errorCode, std::string_view message)
    {
        return ErrorResult<T>(errorCode, std::string(message));
    }

    std::string compactSummary() const
    {
        std::ostringstream oss;
        const auto &meta = ErrorRegistry::instance().getMetadata(error_code_);
        oss << meta.name << " [" << severityLabel(meta.severity) << "] "
            << categoryLabel(meta.category) << '\n'
            << "  " << meta.description << '\n';
        if (!message_.empty())
            oss << "  detail: " << message_ << '\n';
        oss << "  action: " << meta.action_hint << '\n';
        oss << "  transient: " << (meta.isTransient ? "yes" : "no")
            << " | user-fixable: " << (meta.isUserError ? "yes" : "no")
            << " | requires-human: " << (meta.requires_human_intervention ? "yes" : "no");
        return oss.str();
    }

    std::string nextSteps() const
    {
        const auto &meta = ErrorRegistry::instance().getMetadata(error_code_);
        std::ostringstream oss;
        oss << "Next steps for " << meta.name << ":\n";
        oss << "  1. " << meta.action_hint << '\n';
        if (meta.isTransient)
            oss << "  2. Retry the operation (error is transient)\n";
        if (!meta.isUserError)
            oss << "  2. File a bug report with the error code and detail message\n";
        switch (meta.category)
        {
            case ErrorCategory::GPU_COMPUTE:
                oss << "  3. Check: GPU memory, driver version, CUDA_VISIBLE_DEVICES\n";
                break;
            case ErrorCategory::NUMA_AFFINITY:
                oss << "  3. Check: NUMA topology (numactl --hardware), pinning policy\n";
                break;
            case ErrorCategory::DETERMINISM:
                oss << "  3. Re-run with --deterministic=true --seed=<same-value>\n";
                break;
            case ErrorCategory::INPUT_VALIDATION:
                oss << "  3. Validate input dimensions, types, and value ranges\n";
                break;
            case ErrorCategory::DISTRIBUTED:
                oss << "  3. Check: MPI environment, network connectivity, rank count\n";
                break;
            default:
                break;
        }
        if (meta.requires_human_intervention)
            oss << "  N. This error requires manual intervention\n";
        return oss.str();
    }

private:
    std::optional<T> value_;
    ErrorCode error_code_;
    std::string message_;

    static const char *severityLabel(ErrorSeverity s)
    {
        switch (s)
        {
            case ErrorSeverity::INFO:
                return "INFO";
            case ErrorSeverity::WARNING:
                return "WARN";
            case ErrorSeverity::ERROR:
                return "ERROR";
            case ErrorSeverity::CRITICAL:
                return "FATAL";
        }
        return "???";
    }

    static const char *categoryLabel(ErrorCategory c)
    {
        switch (c)
        {
            case ErrorCategory::IO_INFRA:
                return "IO";
            case ErrorCategory::GPU_COMPUTE:
                return "GPU";
            case ErrorCategory::NUMERICAL:
                return "NUM";
            case ErrorCategory::DETERMINISM:
                return "DET";
            case ErrorCategory::CAPACITY:
                return "CAP";
            case ErrorCategory::ALGORITHMIC:
                return "ALG";
            case ErrorCategory::NUMA_AFFINITY:
                return "NUMA";
            case ErrorCategory::PRECISION:
                return "PREC";
            case ErrorCategory::INPUT_VALIDATION:
                return "INPUT";
            case ErrorCategory::DISTRIBUTED:
                return "DIST";
            case ErrorCategory::FEATURE_FLAG:
                return "FEAT";
            case ErrorCategory::PH4_RESEARCH:
                return "PH4";
            case ErrorCategory::PH5_PH6_HIGHDIM:
                return "PH56";
            default:
                return "UNK";
        }
    }
};

template <>
class ErrorResult<void>
{
public:
    struct ErrorInfo
    {
        ErrorCode code;
        std::string message;
    };

    ErrorResult()
        : error_code_(ErrorCode::SUCCESS)
    {}
    explicit ErrorResult(ErrorCode errorCode, std::string message = {})
        : error_code_(errorCode)
        , message_(std::move(message))
    {}
    bool isSuccess() const { return error_code_ == ErrorCode::SUCCESS; }
    bool isError() const { return error_code_ != ErrorCode::SUCCESS; }
    bool is_error() const { return isError(); }
    bool isOk() const { return isSuccess(); }
    bool isErr() const { return isError(); }
    ErrorCode errorCode() const { return error_code_; }
    ErrorInfo error() const
    {
        const auto message =
            message_.empty() ? std::to_string(static_cast<uint32_t>(error_code_)) : message_;
        return ErrorInfo{error_code_, message};
    }
    static ErrorResult<void> success() { return ErrorResult<void>(); }
    static ErrorResult<void> ok() { return success(); }
    static ErrorResult<void> error(ErrorCode errorCode) { return ErrorResult<void>(errorCode); }
    static ErrorResult<void> error(ErrorCode errorCode, std::string_view message)
    {
        return ErrorResult<void>(errorCode, std::string(message));
    }

    std::string compactSummary() const
    {
        std::ostringstream oss;
        const auto &meta = ErrorRegistry::instance().getMetadata(error_code_);
        oss << meta.name << " [" << severityLabel(meta.severity) << "] "
            << categoryLabel(meta.category) << '\n'
            << "  " << meta.description << '\n';
        if (!message_.empty())
            oss << "  detail: " << message_ << '\n';
        oss << "  action: " << meta.action_hint << '\n';
        oss << "  transient: " << (meta.isTransient ? "yes" : "no")
            << " | user-fixable: " << (meta.isUserError ? "yes" : "no")
            << " | requires-human: " << (meta.requires_human_intervention ? "yes" : "no");
        return oss.str();
    }

    std::string nextSteps() const
    {
        const auto &meta = ErrorRegistry::instance().getMetadata(error_code_);
        std::ostringstream oss;
        oss << "Next steps for " << meta.name << ":\n";
        oss << "  1. " << meta.action_hint << '\n';
        if (meta.isTransient)
            oss << "  2. Retry (error is transient)\n";
        if (!meta.isUserError)
            oss << "  N. File a bug report with the error code above\n";
        return oss.str();
    }

private:
    ErrorCode error_code_;
    std::string message_;

    static const char *severityLabel(ErrorSeverity s)
    {
        switch (s)
        {
            case ErrorSeverity::INFO:
                return "INFO";
            case ErrorSeverity::WARNING:
                return "WARN";
            case ErrorSeverity::ERROR:
                return "ERROR";
            case ErrorSeverity::CRITICAL:
                return "FATAL";
        }
        return "???";
    }

    static const char *categoryLabel(ErrorCategory c)
    {
        switch (c)
        {
            case ErrorCategory::IO_INFRA:
                return "IO";
            case ErrorCategory::GPU_COMPUTE:
                return "GPU";
            case ErrorCategory::NUMERICAL:
                return "NUM";
            case ErrorCategory::DETERMINISM:
                return "DET";
            case ErrorCategory::CAPACITY:
                return "CAP";
            case ErrorCategory::ALGORITHMIC:
                return "ALG";
            case ErrorCategory::NUMA_AFFINITY:
                return "NUMA";
            case ErrorCategory::PRECISION:
                return "PREC";
            case ErrorCategory::INPUT_VALIDATION:
                return "INPUT";
            case ErrorCategory::DISTRIBUTED:
                return "DIST";
            case ErrorCategory::FEATURE_FLAG:
                return "FEAT";
            case ErrorCategory::PH4_RESEARCH:
                return "PH4";
            case ErrorCategory::PH5_PH6_HIGHDIM:
                return "PH56";
            default:
                return "UNK";
        }
    }
};

namespace tda
{

void reportMatrixIssue(const std::string &operation, std::size_t actual_rows,
                       std::size_t actual_cols, std::size_t expected_rows,
                       std::size_t expected_cols, std::size_t actual_nonzeros,
                       std::size_t expected_nonzeros, double actual_density,
                       double expected_density, const ErrorContext &context);

void reportPivotAnalysis(std::size_t total_columns, std::size_t found_pivots,
                         std::size_t expected_min_pivots, double pivot_ratio,
                         const std::string &algorithm_name);

void reportConnectivityAnalysis(std::size_t total_points, std::size_t connected_components,
                                std::size_t expected_components, double connectivity_threshold,
                                const std::string &filtration_type);

void reportHomologyValidation(const std::vector<int> &actual_betti,
                              const std::vector<int> &expected_betti,
                              const std::string &topology_type);

} // namespace tda

} // namespace nerve::errors

#ifndef TRY_RESULT
#define TRY_RESULT(expr)                                                                           \
    do                                                                                             \
    {                                                                                              \
        auto _result = (expr);                                                                     \
        if (_result.isError())                                                                     \
        {                                                                                          \
            return ::nerve::errors::ErrorResult<decltype(_result.value())>::error(                 \
                _result.errorCode());                                                              \
        }                                                                                          \
    } while (0)
#endif

#ifndef TRY_ASSIGN
#define TRY_ASSIGN(var, expr)                                                                      \
    do                                                                                             \
    {                                                                                              \
        auto _result = (expr);                                                                     \
        if (_result.isError())                                                                     \
        {                                                                                          \
            return ::nerve::errors::ErrorResult<decltype(_result.value())>::error(                 \
                _result.errorCode());                                                              \
        }                                                                                          \
        var = _result.moveValue();                                                                 \
    } while (0)
#endif
