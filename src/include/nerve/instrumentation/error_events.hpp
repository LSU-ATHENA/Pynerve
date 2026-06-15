
#pragma once
#include "../errors/errors.hpp"

#include <cstdint>
#include <functional>

namespace nerve::instrumentation
{
struct alignas(64) ErrorEvent
{
    errors::ErrorCode code;
    uint64_t timestamp_ns;
    uint64_t duration_ns;
    uint32_t session_id;
    uint32_t request_id;
    uint16_t line_number;
    uint8_t severity;
    uint8_t reserved[5];
    uint32_t max_dimension_attempted;
    uint32_t num_boundary_ops;
    bool truncated_by_budget;
    uint8_t precision_level;
    uint8_t reserved2[7];
    bool isValid() const { return timestamp_ns > 0 && session_id > 0; }
};
static_assert(sizeof(ErrorEvent) == 64, "ErrorEvent must be 64 bytes for cache efficiency");
using ErrorSinkFn = std::function<void(const ErrorEvent &)>;
class ErrorEventsRegistry
{
public:
    static ErrorEventsRegistry &instance();
    void registerSink(ErrorSinkFn sink);
    void unregisterSinks();
    bool hasSinks() const;
    void emit(const ErrorEvent &event);
    void emitError(errors::ErrorCode code, const char *component, const char *operation,
                   uint64_t duration_ns = 0, const char *message = nullptr, uint32_t session_id = 0,
                   uint32_t request_id = 0, uint16_t line_number = 0);

private:
    ErrorEventsRegistry() = default;
    static constexpr size_t MAX_SINKS = 4;
    ErrorSinkFn sinks_[MAX_SINKS];
    size_t sink_count_;
    mutable bool sinks_enabled_;
    uint64_t getTimestampNs() const;
};
class ErrorContext
{
public:
    ErrorContext(errors::ErrorCode code, const char *component, const char *operation,
                 const char *message = nullptr);
    ~ErrorContext();
    void setSuccess();
    void setError(errors::ErrorCode code, const char *message = nullptr);
    errors::ErrorCode getCode() const { return code_; }
    bool isSuccess() const { return success_; }

private:
    errors::ErrorCode code_;
    const char *component_;
    const char *operation_;
    const char *message_;
    uint64_t start_time_ns_;
    bool success_;
    uint64_t getTimestampNs() const;
};
#define EMIT_ERROR(code, component, operation)                                                     \
    nerve::instrumentation::ErrorEventsRegistry::instance().emitError(code, component, operation)
#define EMIT_ERROR_MESSAGE(code, component, operation, message)                                    \
    nerve::instrumentation::ErrorEventsRegistry::instance().emitError(code, component, operation,  \
                                                                      0, message)
#define EMIT_ERROR_DURATION(code, component, operation, duration_ns)                               \
    nerve::instrumentation::ErrorEventsRegistry::instance().emitError(code, component, operation,  \
                                                                      duration_ns)
#define ERROR_CONTEXT(code, component, operation)                                                  \
    nerve::instrumentation::ErrorContext _error_context(code, component, operation)
#define ERROR_CONTEXT_MESSAGE(code, component, operation, message)                                 \
    nerve::instrumentation::ErrorContext _error_context(code, component, operation, message)
enum class ErrorSeverity : uint8_t
{
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    CRITICAL = 3
};
inline uint8_t errorCodeToSeverity(errors::ErrorCode code)
{
    using errors::ErrorCode;
    switch (code)
    {
        case ErrorCode::SUCCESS:
            return static_cast<uint8_t>(ErrorSeverity::INFO);
        case ErrorCode::E00_IO_TIMEOUT:
        case ErrorCode::E01_IO_CORRUPT:
        case ErrorCode::E10_GPU_OOM:
        case ErrorCode::E11_GPU_LAUNCH_FAIL:
        case ErrorCode::E20_NUM_NAN:
        case ErrorCode::E21_NUM_NO_CONVERGE:
        case ErrorCode::E30_DET_MISMATCH:
        case ErrorCode::E31_SCHEMA_VERSION:
        case ErrorCode::E40_CPU_OVERLOAD:
        case ErrorCode::E41_RESOURCE_LIMIT:
        case ErrorCode::E81_MATRIX_EMPTY:
        case ErrorCode::E82_MATRIX_SPARSE:
        case ErrorCode::E83_NO_PIVOTS_FOUND:
        case ErrorCode::E84_INSUFFICIENT_PIVOTS:
        case ErrorCode::E85_MATRIX_STRUCTURE:
        case ErrorCode::E86_NO_PERSISTENCE_PAIRS:
        case ErrorCode::E87_INVALID_BETTI_NUMBERS:
        case ErrorCode::E88_INVALID_SIMPLICES:
        case ErrorCode::E89_BOUNDARY_ERROR:
        case ErrorCode::E90_COMPLEX_ERROR:
        case ErrorCode::E91_REDUCED_HOMOLOGY:
        case ErrorCode::E92_BETTI_MISMATCH:
        case ErrorCode::E74_RACE_CONDITION:
        case ErrorCode::E75_ATOMICITY_VIOLATION:
        case ErrorCode::E76_CONSISTENCY_ERROR:
        case ErrorCode::E77_ISOLATION_VIOLATION:
        case ErrorCode::E78_DURABILITY_ERROR:
        case ErrorCode::E79_TRANSACTION_ABORT:
        case ErrorCode::E80_CAPACITY_EXCEEDED:
        case ErrorCode::UNKNOWN:
            return static_cast<uint8_t>(ErrorSeverity::CRITICAL);
        default:
            return static_cast<uint8_t>(ErrorSeverity::ERROR);
    }
}
} // namespace nerve::instrumentation
