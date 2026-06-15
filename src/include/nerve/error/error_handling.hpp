
#pragma once
#include "nerve/optimization/component_optimizations.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
namespace nerve
{
namespace error
{
using ErrorCode = optimization::ErrorCode;
using CallContract = optimization::CallContract;
using CircuitBreakerConfig = optimization::CircuitBreakerConfig;

struct ErrorEvent
{
    ErrorCode error_code;
    std::string operation_name;
    uint64_t params_hash;
    int64_t window_start_ns;
    int64_t window_end_ns;
    double duration_ms;
    std::string error_message;
    std::string recovery_reason;
    std::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> metadata;
    std::string toJson() const;
    std::string toStructuredLog() const;
};
class CircuitBreaker
{
public:
    explicit CircuitBreaker(const CircuitBreakerConfig &config);
    enum class State
    {
        CLOSED,
        OPEN,
        HALF_OPEN
    };
    bool shouldAllowOperation(const std::string &operation_name);
    void recordSuccess(const std::string &operation_name);
    void recordFailure(const std::string &operation_name, ErrorCode error_code);
    State getState() const;
    bool isTripped() const;
    void reset();
    void forceOpen();
    void forceClose();
    struct CircuitBreakerStats
    {
        uint64_t total_operations;
        uint64_t successful_operations;
        uint64_t failed_operations;
        uint64_t consecutive_failures;
        State current_state;
        std::chrono::steady_clock::time_point last_failure_time;
        std::chrono::steady_clock::time_point last_reset_time;
        std::vector<std::string> failing_operations;
    };
    CircuitBreakerStats getStats() const;
    void resetStats();

private:
    CircuitBreakerConfig config_;
    std::atomic<State> state_{State::CLOSED};
    struct OperationState
    {
        std::atomic<uint64_t> consecutive_failures{0};
        std::atomic<uint64_t> total_operations{0};
        std::atomic<uint64_t> successful_operations{0};
        std::atomic<std::chrono::steady_clock::time_point> last_failure_time;
        std::atomic<std::chrono::steady_clock::time_point> last_success_time;
    };
    std::unordered_map<std::string, std::unique_ptr<OperationState>> operation_states_;
    mutable std::shared_mutex states_mutex_;
    std::thread recovery_thread_;
    std::atomic<bool> recovery_running_{false};
    void recoveryWorker();
    void checkAndUpdateState(const std::string &operation_name);
    bool shouldTripCircuit(const std::string &operation_name);
    void attemptRecovery();
};
class RetryBackoffManager
{
public:
    struct RetryConfig
    {
        size_t max_retries = 2;
        std::vector<uint64_t> backoff_ms = {50, 200};
        bool enable_exponential_backoff = true;
        bool enable_jitter = true;
        double jitter_factor = 0.1;
        std::vector<ErrorCode> transient_errors = {ErrorCode::GPU_OOM, ErrorCode::GPU_TIMEOUT,
                                                   ErrorCode::IO_TIMEOUT, ErrorCode::CPU_OVERLOAD};
    };
    explicit RetryBackoffManager(const RetryConfig &config);
    template <typename Func>
    std::pair<ErrorCode, bool> executeWithRetry(const std::string &operation_name, Func &&operation,
                                                const CallContract &contract);
    uint64_t calculateBackoff(size_t retry_attempt, const std::string &operation_name);
    bool isTransientError(ErrorCode error_code) const;
    bool isPermanentError(ErrorCode error_code) const;
    struct RetryStats
    {
        uint64_t total_retries;
        uint64_t successful_retries;
        uint64_t failed_retries;
        std::unordered_map<ErrorCode, uint64_t> error_counts;
        std::unordered_map<std::string, uint64_t> operation_retry_counts;
        double average_backoff_ms;
    };
    RetryStats getStats() const;
    void resetStats();

private:
    RetryConfig config_;
    std::unordered_map<std::string, uint64_t> operation_retry_counts_;
    mutable std::shared_mutex retry_mutex_;
    mutable std::atomic<uint64_t> total_retries_{0};
    mutable std::atomic<uint64_t> successful_retries_{0};
    mutable std::atomic<uint64_t> failed_retries_{0};
    mutable std::unordered_map<ErrorCode, std::atomic<uint64_t>> error_counts_;
    void updateRetryStats(const std::string &operation_name, ErrorCode error_code, bool success);
    uint64_t addJitter(uint64_t base_backoff);
};
class ErrorObservability
{
public:
    struct ObservabilityConfig
    {
        bool enable_structured_logging = true;
        bool enable_metric_increment = true;
        std::string log_format = "json";
        std::string metric_prefix = "nerve.tda";
        bool enable_performance_correlation = true;
    };
    explicit ErrorObservability(const ObservabilityConfig &config);
    void logErrorEvent(const ErrorEvent &event);
    void logErrorEvent(const std::string &operation_name, ErrorCode error_code,
                       const CallContract &contract, const std::string &error_message = "");
    void incrementMetric(const std::string &metric_name,
                         const std::unordered_map<std::string, std::string> &labels = {});
    void updateErrorHistogram(const std::string &operation_name, ErrorCode error_code,
                              double duration_ms);
    void correlatePerformanceWithErrors(const std::string &operation_name,
                                        double performance_metric, ErrorCode error_code);
    std::string exportErrorReport(const std::string &time_range = "1h");
    std::string exportMetrics(const std::string &format = "prometheus");
    struct ObservabilityStats
    {
        uint64_t total_error_events;
        std::unordered_map<ErrorCode, uint64_t> error_code_counts;
        std::unordered_map<std::string, uint64_t> operation_error_counts;
        std::unordered_map<std::string, std::vector<double>> duration_histograms;
        std::unordered_map<std::string, double> performance_correlations;
    };
    ObservabilityStats getStats() const;
    void resetStats();

private:
    ObservabilityConfig config_;
    std::unordered_map<std::string, std::vector<double>> operation_durations_;
    std::unordered_map<std::string, std::vector<ErrorCode>> operation_errors_;
    mutable std::shared_mutex tracking_mutex_;
    mutable std::atomic<uint64_t> total_error_events_{0};
    mutable std::unordered_map<ErrorCode, std::atomic<uint64_t>> error_code_counts_;
    mutable std::unordered_map<std::string, std::atomic<uint64_t>> operation_error_counts_;
    void writeStructuredLog(const ErrorEvent &event);
    void writeMetricIncrement(const std::string &metric_name,
                              const std::unordered_map<std::string, std::string> &labels);
    void updatePerformanceCorrelation(const std::string &operation_name, double performance_metric,
                                      ErrorCode error_code);
    std::string formatMetricName(const std::string &metric_name,
                                 const std::unordered_map<std::string, std::string> &labels);
};
class ErrorHandlingManager
{
public:
    static ErrorHandlingManager &instance();
    void setCircuitBreakerConfig(const CircuitBreakerConfig &config);
    void setRetryConfig(const RetryBackoffManager::RetryConfig &config);
    void setObservabilityConfig(const ErrorObservability::ObservabilityConfig &config);
    std::shared_ptr<CircuitBreaker> getCircuitBreaker();
    std::shared_ptr<RetryBackoffManager> getRetryManager();
    std::shared_ptr<ErrorObservability> getObservability();
    template <typename Func>
    std::pair<ErrorCode, bool> executeWithErrorHandling(const std::string &operation_name,
                                                        Func &&operation,
                                                        const CallContract &contract);
    void logError(const std::string &operation_name, ErrorCode error_code,
                  const CallContract &contract, const std::string &error_message = "");
    struct ErrorHandlingStats
    {
        CircuitBreaker::CircuitBreakerStats circuit_breaker_stats;
        RetryBackoffManager::RetryStats retry_stats;
        ErrorObservability::ObservabilityStats observability_stats;
    };
    ErrorHandlingStats getStats() const;
    void resetStats();
    bool isHealthy() const;
    std::vector<std::string> getHealthIssues() const;

private:
    ErrorHandlingManager() = default;
    CircuitBreakerConfig circuit_breaker_config_;
    RetryBackoffManager::RetryConfig retry_config_;
    ErrorObservability::ObservabilityConfig observability_config_;
    std::shared_ptr<CircuitBreaker> circuit_breaker_;
    std::shared_ptr<RetryBackoffManager> retry_manager_;
    std::shared_ptr<ErrorObservability> observability_;
    mutable std::shared_mutex mutex_;
    void initializeComponents();
    void updateErrorStats(ErrorCode error_code, const std::string &operation_name);
};
template <typename Func>
std::pair<ErrorCode, bool> RetryBackoffManager::executeWithRetry(const std::string &operation_name,
                                                                 Func &&operation,
                                                                 const CallContract &contract)
{
    ErrorCode last_error = ErrorCode::SUCCESS;
    size_t retry_count = 0;
    while (retry_count <= config_.max_retries)
    {
        try
        {
            auto start_time = std::chrono::steady_clock::now();
            auto result = operation();
            auto end_time = std::chrono::steady_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            if (contract.strict_time_enforcement && contract.time_budget_ms > 0.0 &&
                duration.count() > contract.time_budget_ms)
            {
                last_error = ErrorCode::PH_TIME_BUDGET_EXCEEDED;
                break;
            }
            updateRetryStats(operation_name, ErrorCode::SUCCESS, true);
            return {ErrorCode::SUCCESS, true};
        }
        catch (const std::exception &e)
        {
            last_error = ErrorCode::IO_READ_ERROR;
            updateRetryStats(operation_name, last_error, false);
        }
        if (retry_count < config_.max_retries && isTransientError(last_error))
        {
            retry_count++;
            uint64_t backoff_ms = calculateBackoff(retry_count, operation_name);
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            continue;
        }
        else
        {
            break;
        }
    }
    return {last_error, false};
}
template <typename Func>
std::pair<ErrorCode, bool>
ErrorHandlingManager::executeWithErrorHandling(const std::string &operation_name, Func &&operation,
                                               const CallContract &contract)
{
    if (!circuit_breaker_->shouldAllowOperation(operation_name))
    {
        logError(operation_name, ErrorCode::CPU_OVERLOAD, contract, "Circuit breaker open");
        return {ErrorCode::CPU_OVERLOAD, false};
    }
    auto [error_code, success] =
        retry_manager_->executeWithRetry(operation_name, std::forward<Func>(operation), contract);
    if (success)
    {
        circuit_breaker_->recordSuccess(operation_name);
    }
    else
    {
        circuit_breaker_->recordFailure(operation_name, error_code);
    }
    if (!success)
    {
        logError(operation_name, error_code, contract, "Operation failed after retries");
    }
    return {error_code, success};
}
} // namespace error
} // namespace nerve
