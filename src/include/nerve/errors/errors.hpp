
#pragma once

#ifndef NERVE_ENABLE_ERROR_REGISTRY
#define NERVE_ENABLE_ERROR_REGISTRY 0
#endif

#ifndef NERVE_ENABLE_ERROR_TRACKING
#define NERVE_ENABLE_ERROR_TRACKING 0
#endif

#ifndef NERVE_ENABLE_ERROR_METRICS
#define NERVE_ENABLE_ERROR_METRICS 0
#endif

#ifndef NERVE_ENABLE_EXTERNAL_LOGGING
#define NERVE_ENABLE_EXTERNAL_LOGGING 0
#endif

#include "nerve/common.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::errors
{

enum class ErrorCode : uint32_t
{
    SUCCESS = 0x00000000,
    E00_IO_TIMEOUT = 0x00000100,
    E01_IO_CORRUPT = 0x00000101,
    E10_GPU_OOM = 0x00000200,
    E11_GPU_LAUNCH_FAIL = 0x00000201,
    E20_NUM_NAN = 0x00000300,
    E21_NUM_NO_CONVERGE = 0x00000301,
    E30_DET_MISMATCH = 0x00000400,
    E31_SCHEMA_VERSION = 0x00000401,
    E40_CPU_OVERLOAD = 0x00000500,
    E41_RESOURCE_LIMIT = 0x00000501,
    E50_PH_ABORT = 0x00000600,
    E51_LAPLACIAN_ABORT = 0x00000601,
    E52_PH4_ABORT = 0x00000602,
    E53_PH4_BUDGET_EXCEEDED = 0x00000603,
    E54_PH4_INVALID_INPUT = 0x00000604,
    E51_PH_INPUT = E54_PH4_INVALID_INPUT,
    E52_PH_CONFIG = E54_PH4_INVALID_INPUT,
    E55_PH4_SPARSE_CONVERGENCE_FAIL = 0x00000605,
    E56_PH4_WITNESS_SAMPLING_ERROR = 0x00000606,
    E11_PH5_OVERFLOW = 0x00000607,
    E12_PH6_OVERFLOW = 0x00000608,
    E13_PH_HIGHDIM_PRECISION = 0x00000609,
    E60_NUMA_BIND_FAIL = 0x00000700,
    E61_NUMA_AFFINITY_FAIL = 0x00000701,
    E62_NUMA_MIGRATION_ERROR = 0x00000702,
    E70_PRECISION_DOWNGRADE = 0x00000800,
    E71_PRECISION_LOSS = 0x00000801,
    E72_PRECISION_UNDERFLOW = 0x00000802,
    E73_PRECISION_CATASTROPHIC = 0x00000803,

    E81_MATRIX_EMPTY = 0x00000900,
    E82_MATRIX_SPARSE = 0x00000901,
    E83_NO_PIVOTS_FOUND = 0x00000902,
    E84_INSUFFICIENT_PIVOTS = 0x00000903,
    E85_MATRIX_STRUCTURE = 0x00000904,
    E86_NO_PERSISTENCE_PAIRS = 0x00000905,
    E87_INVALID_BETTI_NUMBERS = 0x00000906,
    E88_INVALID_SIMPLICES = 0x00000907,
    E89_BOUNDARY_ERROR = 0x00000908,
    E90_COMPLEX_ERROR = 0x00000909,
    E91_REDUCED_HOMOLOGY = 0x00000910,
    E92_BETTI_MISMATCH = 0x00000911,
    E74_RACE_CONDITION = 0x00000912,
    E75_ATOMICITY_VIOLATION = 0x00000913,
    E76_CONSISTENCY_ERROR = 0x00000914,
    E77_ISOLATION_VIOLATION = 0x00000915,
    E78_DURABILITY_ERROR = 0x00000916,
    E79_TRANSACTION_ABORT = 0x00000917,
    E80_CAPACITY_EXCEEDED = 0x00000918,
    E88_HOMOLOGY_MISMATCH = 0x00000907,
    E89_EMPTY_COMPLEX = 0x00000908,
    E90_DISCONNECTED_COMPLEX = 0x00000909,
    E91_INVALID_SIMPLICES = 0x0000090A,
    E92_MEMORY_PRESSURE = 0x0000090B,
    E93_COMPUTATION_TIMEOUT = 0x0000090C,
    E94_CONVERGENCE_FAILURE = 0x0000090D,

    E95_INPUT_VALIDATION = 0x00000A00,
    E96_TYPE_MISMATCH = 0x00000A01,
    E97_SHAPE_MISMATCH = 0x00000A02,
    E98_DIMENSION_MISMATCH = 0x00000A03,
    E99_NOT_IMPLEMENTED = 0x00000A04,
    E9A_UNSUPPORTED_OPERATION = 0x00000A05,
    E9B_INTERNAL_STATE = 0x00000A06,

    EA0_MPI_COMM_FAIL = 0x00000B00,
    EA1_MPI_RANK_FAIL = 0x00000B01,
    EA2_MPI_BARRIER_FAIL = 0x00000B02,

    EB0_FEATURE_DISABLED = 0x00000C00,
    EB1_FEATURE_MISCONFIGURED = 0x00000C01,

    UNKNOWN = 0xFFFFFFFF
};
enum class ErrorCategory : uint8_t
{
    SUCCESS = 0,
    IO_INFRA = 1,
    GPU_COMPUTE = 2,
    NUMERICAL = 3,
    DETERMINISM = 4,
    CAPACITY = 5,
    ALGORITHMIC = 6,
    OPERATIONAL = 7,
    PH4_RESEARCH = 8,
    PH5_PH6_HIGHDIM = 9,
    INPUT_VALIDATION = 10,
    NUMA_AFFINITY = 11,
    PRECISION = 12,
    DISTRIBUTED = 13,
    FEATURE_FLAG = 14,
    UNKNOWN_CATEGORY = 255
};
enum class ErrorSeverity : uint8_t
{
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    CRITICAL = 3
};
struct ErrorMetadata
{
    ErrorCode code;
    ErrorCategory category;
    ErrorSeverity severity;
    std::string name;
    std::string description;
    std::string action_hint;
    bool isTransient;
    bool isUserError;
    bool requires_human_intervention;
};

struct ErrorContext
{
    std::string operation_name;
    std::string component_name;
    std::string session_id;
    std::string request_id;
    std::unordered_map<std::string, std::string> metadata;
    uint64_t timestampNs;
    double durationMs;
    ErrorContext()
        : timestampNs(0)
        , durationMs(0.0)
    {}
    void addMetadata(const std::string &key, const std::string &value) { metadata[key] = value; }

    static std::string escapeJsonString(std::string_view input)
    {
        std::ostringstream escaped;
        escaped << std::hex << std::setfill('0');
        for (unsigned char ch : input)
        {
            switch (ch)
            {
                case '"':
                    escaped << "\\\"";
                    break;
                case '\\':
                    escaped << "\\\\";
                    break;
                case '\b':
                    escaped << "\\b";
                    break;
                case '\f':
                    escaped << "\\f";
                    break;
                case '\n':
                    escaped << "\\n";
                    break;
                case '\r':
                    escaped << "\\r";
                    break;
                case '\t':
                    escaped << "\\t";
                    break;
                default:
                    if (ch < 0x20)
                    {
                        escaped << "\\u" << std::setw(4) << static_cast<int>(ch);
                    }
                    else
                    {
                        escaped << static_cast<char>(ch);
                    }
                    break;
            }
        }
        return escaped.str();
    }

    std::string toJson() const
    {
        const double safe_duration_ms =
            std::isfinite(durationMs) && durationMs >= 0.0 ? durationMs : 0.0;
        std::ostringstream oss;
        oss << "{"
            << "\"operation_name\":\"" << escapeJsonString(operation_name) << "\","
            << "\"component_name\":\"" << escapeJsonString(component_name) << "\","
            << "\"session_id\":\"" << escapeJsonString(session_id) << "\","
            << "\"request_id\":\"" << escapeJsonString(request_id) << "\","
            << "\"timestampNs\":" << timestampNs << ","
            << "\"durationMs\":" << std::fixed << std::setprecision(3) << safe_duration_ms;

        if (!metadata.empty())
        {
            oss << ",\"metadata\":{";
            bool first = true;
            for (const auto &pair : metadata)
            {
                if (!first)
                    oss << ",";
                oss << "\"" << escapeJsonString(pair.first) << "\":\""
                    << escapeJsonString(pair.second) << "\"";
                first = false;
            }
            oss << "}";
        }

        oss << "}";
        return oss.str();
    }
};

class ErrorRegistry
{
public:
    static ErrorRegistry &instance();
    const ErrorMetadata &getMetadata(ErrorCode code) const;
    ErrorCategory getCategory(ErrorCode code) const;
    ErrorSeverity getSeverity(ErrorCode code) const;
    bool isTransient(ErrorCode code) const;
    bool isUserError(ErrorCode code) const;
    std::vector<ErrorCode> getAllCodes() const;
    std::vector<ErrorCode> getCodesByCategory(ErrorCategory category) const;
    void reportError(ErrorCode code, const ErrorContext &context = {});
    void logErrorToExternalFile(ErrorCode code, const ErrorContext &context,
                                const ErrorMetadata &metadata, bool is_critical = false);

    bool hasOperationFailed(const std::string &operation_name) const
    {
#if NERVE_ENABLE_ERROR_TRACKING
        std::lock_guard<std::mutex> lock(failed_ops_mutex_);
        return failed_operations_.find(operation_name) != failed_operations_.end();
#else
        (void)operation_name;
        return false;
#endif
    }

    void clearFailedOperation(const std::string &operation_name)
    {
#if NERVE_ENABLE_ERROR_TRACKING
        std::lock_guard<std::mutex> lock(failed_ops_mutex_);
        failed_operations_.erase(operation_name);
#else
        (void)operation_name;
#endif
    }

private:
    ErrorRegistry();
    std::unordered_map<ErrorCode, ErrorMetadata> metadata_;

    void initializeMetadata();
    void registerError(ErrorCode code, ErrorCategory category, ErrorSeverity severity,
                       const std::string &name, const std::string &description,
                       const std::string &action_hint, bool isTransient, bool isUserError,
                       bool requires_human_intervention);

#if NERVE_ENABLE_ERROR_TRACKING
    struct ErrorRecord
    {
        ErrorCode code;
        ErrorContext context;
        std::chrono::steady_clock::time_point timestamp;
        size_t occurrenceCount;
        std::string stack_trace;

        ErrorRecord(ErrorCode c, const ErrorContext &ctx)
            : code(c)
            , context(ctx)
            , timestamp(std::chrono::steady_clock::now())
            , occurrenceCount(1)
        {}
    };

    std::vector<ErrorRecord> error_history_;
    std::unordered_map<ErrorCode, size_t> error_counts_;
    std::mutex tracking_mutex_;
    size_t max_history_size_;

#endif

#if NERVE_ENABLE_ERROR_METRICS
    struct PerformanceMetrics
    {
        std::chrono::steady_clock::time_point operation_start;
        std::string operation_name;
        std::unordered_map<std::string, double> metrics;
    };

    std::vector<PerformanceMetrics> performance_history_;
    std::mutex performance_mutex_;
#endif

#if NERVE_ENABLE_ERROR_TRACKING
    std::unordered_set<std::string> failed_operations_;
    mutable std::mutex failed_ops_mutex_;
#endif
};

struct ErrorConfig
{
    bool lightweight_mode = true;
    bool enable_monitoring = false;
    bool enable_performance_monitoring = false;
    bool enableExternalLogging = false;

    ErrorSeverity min_log_level = ErrorSeverity::WARNING;

    std::string external_log_file = "";
    size_t max_error_history = 0;

    double performance_threshold_multiplier = 2.0;

    static ErrorConfig lightweight() { return ErrorConfig{}; }

    static ErrorConfig debug()
    {
        ErrorConfig config;
        config.lightweight_mode = false;
        config.enable_monitoring = true;
        config.enable_performance_monitoring = true;
        config.max_error_history = 1000;
        return config;
    }
};

struct LightweightErrorRecord
{
    ErrorCode code;
    uint64_t timestamp_ns;
    uint32_t count;

    LightweightErrorRecord()
        : code(ErrorCode::SUCCESS)
        , timestamp_ns(0)
        , count(0)
    {}
    LightweightErrorRecord(ErrorCode c, uint64_t ts, uint32_t cnt = 1)
        : code(c)
        , timestamp_ns(ts)
        , count(cnt)
    {}
};

class LightweightErrorCounter
{
public:
    static constexpr size_t MAX_ERROR_CODES = 256;

    void increment(ErrorCode code)
    {
        auto idx = static_cast<uint32_t>(code) % MAX_ERROR_CODES;
        counts_[idx].fetch_add(1, std::memory_order_relaxed);
    }

    uint32_t getCount(ErrorCode code) const
    {
        auto idx = static_cast<uint32_t>(code) % MAX_ERROR_CODES;
        return counts_[idx].load(std::memory_order_relaxed);
    }

    uint32_t getTotalCount() const
    {
        uint32_t total = 0;
        for (const auto &c : counts_)
        {
            total += c.load(std::memory_order_relaxed);
        }
        return total;
    }

private:
    std::array<std::atomic<uint32_t>, MAX_ERROR_CODES> counts_{};
};

class ConfigurableErrorSystem
{
public:
    static void configure(const ErrorConfig &config);
    static ErrorConfig getConfig();
    static bool isMonitoringEnabled();
    static bool shouldLog(ErrorSeverity level);
    static void enableExternalLogging(const std::string &log_file);
    static void disableMonitoring();
    static bool isExternalLoggingEnabled();
    static std::string getExternalLogFile();

private:
    static ErrorConfig config_;
    static std::mutex config_mutex_;
};

} // namespace nerve::errors

#include "nerve/errors/detail/error_result.hpp"
#include "nerve/errors/detail/public_reexports.hpp"
