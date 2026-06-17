#include "nerve/error/error_handling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <numeric>
#include <sstream>
#include <thread>

namespace nerve::error
{

std::string ErrorEvent::toJson() const
{
    std::ostringstream oss;
    oss << "{";
    oss << "\"error_code\":" << static_cast<int>(error_code) << ",";
    oss << "\"operation_name\":\"" << operation_name << "\",";
    oss << "\"params_hash\":" << params_hash << ",";
    oss << "\"window_start_ns\":" << window_start_ns << ",";
    oss << "\"window_end_ns\":" << window_end_ns << ",";
    oss << "\"duration_ms\":" << duration_ms << ",";
    oss << "\"error_message\":\"" << error_message << "\",";
    oss << "\"recovery_reason\":\"" << recovery_reason << "\",";
    const auto ts_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count();
    oss << "\"timestamp_ns\":" << ts_ns << ",";
    oss << "\"metadata\":{";
    bool first = true;
    for (const auto &[k, v] : metadata)
    {
        if (!first)
        {
            oss << ",";
        }
        first = false;
        oss << "\"" << k << "\":\"" << v << "\"";
    }
    oss << "}";
    oss << "}";
    return oss.str();
}

std::string ErrorEvent::toStructuredLog() const
{
    std::ostringstream oss;
    oss << "error_code=" << static_cast<int>(error_code);
    oss << " operation=" << operation_name;
    oss << " duration_ms=" << duration_ms;
    oss << " error_message=\"" << error_message << "\"";
    if (!recovery_reason.empty())
    {
        oss << " recovery_reason=\"" << recovery_reason << "\"";
    }
    for (const auto &[k, v] : metadata)
    {
        oss << " " << k << "=\"" << v << "\"";
    }
    return oss.str();
}

CircuitBreaker::CircuitBreaker(const CircuitBreakerConfig &config)
    : config_(config)
    , state_(State::CLOSED)
{
    if (config_.enable_automatic_recovery)
    {
        recovery_running_ = true;
        recovery_thread_ = std::thread(&CircuitBreaker::recoveryWorker, this);
    }
}

bool CircuitBreaker::shouldAllowOperation(const std::string &operation_name)
{
    State current = state_.load(std::memory_order_acquire);

    if (current == State::OPEN)
    {
        return false;
    }

    if (current == State::HALF_OPEN)
    {
        std::shared_lock lock(states_mutex_);
        const auto it = operation_states_.find(operation_name);
        if (it != operation_states_.end())
        {
            const Size total = it->second->total_operations.load();
            const Size allowed =
                std::max<Size>(1, static_cast<Size>(std::ceil(static_cast<double>(total) *
                                                              config_.recovery_sample_ratio)));
            const Size recent =
                it->second->successful_operations.load() + it->second->consecutive_failures.load();
            return recent < allowed;
        }
        return true;
    }

    return true;
}

void CircuitBreaker::recordSuccess(const std::string &operation_name)
{
    std::unique_lock lock(states_mutex_);
    auto &op = operation_states_[operation_name];
    if (!op)
    {
        op = std::make_unique<OperationState>();
    }
    op->total_operations.fetch_add(1, std::memory_order_relaxed);
    op->successful_operations.fetch_add(1, std::memory_order_relaxed);
    op->consecutive_failures.store(0, std::memory_order_release);
    op->last_success_time.store(std::chrono::steady_clock::now(), std::memory_order_release);
}

void CircuitBreaker::recordFailure(const std::string &operation_name, ErrorCode error_code)
{
    (void)error_code;

    std::unique_lock lock(states_mutex_);
    auto &op = operation_states_[operation_name];
    if (!op)
    {
        op = std::make_unique<OperationState>();
    }
    op->total_operations.fetch_add(1, std::memory_order_relaxed);
    op->consecutive_failures.fetch_add(1, std::memory_order_relaxed);
    op->last_failure_time.store(std::chrono::steady_clock::now(), std::memory_order_release);

    checkAndUpdateState(operation_name);
}

CircuitBreaker::State CircuitBreaker::getState() const
{
    return state_.load(std::memory_order_acquire);
}

bool CircuitBreaker::isTripped() const
{
    return state_.load(std::memory_order_acquire) == State::OPEN;
}

void CircuitBreaker::reset()
{
    state_.store(State::CLOSED, std::memory_order_release);
    std::unique_lock lock(states_mutex_);
    for (auto &[name, op] : operation_states_)
    {
        (void)name;
        if (op)
        {
            op->consecutive_failures.store(0, std::memory_order_release);
        }
    }
}

void CircuitBreaker::forceOpen()
{
    state_.store(State::OPEN, std::memory_order_release);
}

void CircuitBreaker::forceClose()
{
    state_.store(State::CLOSED, std::memory_order_release);
    std::unique_lock lock(states_mutex_);
    for (auto &[name, op] : operation_states_)
    {
        (void)name;
        if (op)
        {
            op->consecutive_failures.store(0, std::memory_order_release);
        }
    }
}

CircuitBreaker::CircuitBreakerStats CircuitBreaker::getStats() const
{
    CircuitBreakerStats stats{};
    stats.current_state = state_.load(std::memory_order_acquire);

    std::shared_lock lock(states_mutex_);
    for (const auto &[name, op] : operation_states_)
    {
        if (!op)
        {
            continue;
        }
        stats.total_operations += op->total_operations.load();
        stats.successful_operations += op->successful_operations.load();
        stats.failed_operations += op->total_operations.load() - op->successful_operations.load();
        stats.consecutive_failures =
            std::max(stats.consecutive_failures, op->consecutive_failures.load());

        const auto last_fail = op->last_failure_time.load();
        if (last_fail.time_since_epoch().count() > 0)
        {
            if (stats.last_failure_time.time_since_epoch().count() == 0 ||
                last_fail > stats.last_failure_time)
            {
                stats.last_failure_time = last_fail;
            }
            stats.failing_operations.push_back(name);
        }
    }

    stats.last_reset_time = std::chrono::steady_clock::now();
    return stats;
}

void CircuitBreaker::resetStats()
{
    std::unique_lock lock(states_mutex_);
    operation_states_.clear();
    state_.store(State::CLOSED, std::memory_order_release);
}

void CircuitBreaker::recoveryWorker()
{
    while (recovery_running_.load(std::memory_order_acquire))
    {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(std::max<uint64_t>(100, config_.cooldown_ms / 10)));

        const State current = state_.load(std::memory_order_acquire);
        if (current == State::OPEN)
        {
            attemptRecovery();
        }
    }
}

void CircuitBreaker::checkAndUpdateState(const std::string &operation_name)
{
    if (shouldTripCircuit(operation_name))
    {
        state_.store(State::OPEN, std::memory_order_release);
    }
}

bool CircuitBreaker::shouldTripCircuit(const std::string &operation_name)
{
    const auto it = operation_states_.find(operation_name);
    if (it == operation_states_.end() || !it->second)
    {
        return false;
    }

    const uint64_t failures = it->second->consecutive_failures.load(std::memory_order_acquire);
    if (failures < config_.max_consecutive_failures)
    {
        return false;
    }

    Size total_failing = 0;
    for (const auto &[name, op] : operation_states_)
    {
        (void)name;
        if (op && op->consecutive_failures.load(std::memory_order_acquire) >=
                      config_.max_consecutive_failures)
        {
            ++total_failing;
        }
    }
    return total_failing > 0;
}

void CircuitBreaker::attemptRecovery()
{
    const auto now = std::chrono::steady_clock::now();
    {
        std::shared_lock lock(states_mutex_);
        for (const auto &[name, op] : operation_states_)
        {
            (void)name;
            if (!op)
            {
                continue;
            }
            const auto last_failure = op->last_failure_time.load();
            if (last_failure.time_since_epoch().count() == 0)
            {
                continue;
            }
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_failure);
            if (elapsed.count() >= static_cast<int64_t>(config_.cooldown_ms))
            {
                state_.store(State::HALF_OPEN, std::memory_order_release);
                return;
            }
        }
    }
}

RetryBackoffManager::RetryBackoffManager(const RetryConfig &config)
    : config_(config)
{}

uint64_t RetryBackoffManager::calculateBackoff(size_t retry_attempt,
                                               const std::string &operation_name)
{
    if (retry_attempt == 0)
    {
        return 0;
    }

    const size_t idx = std::min(retry_attempt - 1, config_.backoff_ms.size() - 1);
    uint64_t base = config_.backoff_ms[idx];

    if (config_.enable_exponential_backoff)
    {
        base = static_cast<uint64_t>(static_cast<double>(base) *
                                     std::pow(2.0, static_cast<double>(retry_attempt - 1)));
    }

    if (config_.enable_jitter)
    {
        base = addJitter(base);
    }

    std::shared_lock lock(retry_mutex_);
    const auto it = operation_retry_counts_.find(operation_name);
    if (it != operation_retry_counts_.end() && it->second > 0)
    {
        const double congestion_factor =
            1.0 + static_cast<double>(std::min(it->second, uint64_t{100})) * 0.01;
        base = static_cast<uint64_t>(static_cast<double>(base) * congestion_factor);
    }

    return base;
}

bool RetryBackoffManager::isTransientError(ErrorCode error_code) const
{
    return std::find(config_.transient_errors.begin(), config_.transient_errors.end(),
                     error_code) != config_.transient_errors.end();
}

bool RetryBackoffManager::isPermanentError(ErrorCode error_code) const
{
    return error_code != ErrorCode::SUCCESS && !isTransientError(error_code);
}

RetryBackoffManager::RetryStats RetryBackoffManager::getStats() const
{
    RetryStats stats{};
    stats.total_retries = total_retries_.load();
    stats.successful_retries = successful_retries_.load();
    stats.failed_retries = failed_retries_.load();

    {
        std::shared_lock lock(retry_mutex_);
        stats.operation_retry_counts = operation_retry_counts_;
    }

    if (stats.total_retries > 0)
    {
        stats.average_backoff_ms =
            static_cast<double>(std::accumulate(config_.backoff_ms.begin(),
                                                config_.backoff_ms.end(), uint64_t{0})) /
            static_cast<double>(config_.backoff_ms.size());
    }

    return stats;
}

void RetryBackoffManager::resetStats()
{
    total_retries_ = 0;
    successful_retries_ = 0;
    failed_retries_ = 0;
    std::unique_lock lock(retry_mutex_);
    operation_retry_counts_.clear();
}

void RetryBackoffManager::updateRetryStats(const std::string &operation_name, ErrorCode error_code,
                                           bool success)
{
    total_retries_.fetch_add(1, std::memory_order_relaxed);
    if (success)
    {
        successful_retries_.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        failed_retries_.fetch_add(1, std::memory_order_relaxed);
    }

    std::unique_lock lock(retry_mutex_);
    ++operation_retry_counts_[operation_name];
    lock.unlock();
}

uint64_t RetryBackoffManager::addJitter(uint64_t base_backoff)
{
    const double jitter_range = static_cast<double>(base_backoff) * config_.jitter_factor;
    const double random_offset =
        (static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX)) * jitter_range;
    return static_cast<uint64_t>(static_cast<double>(base_backoff) + random_offset);
}

ErrorObservability::ErrorObservability(const ObservabilityConfig &config)
    : config_(config)
{}

void ErrorObservability::logErrorEvent(const ErrorEvent &event)
{
    total_error_events_.fetch_add(1, std::memory_order_relaxed);

    {
        std::unique_lock lock(tracking_mutex_);
        ++error_code_counts_[event.error_code];
        ++operation_error_counts_[event.operation_name];
        operation_durations_[event.operation_name].push_back(event.duration_ms);
        operation_errors_[event.operation_name].push_back(event.error_code);
    }

    if (config_.enable_structured_logging)
    {
        writeStructuredLog(event);
    }
}

void ErrorObservability::logErrorEvent(const std::string &operation_name, ErrorCode error_code,
                                       const CallContract &contract,
                                       const std::string &error_message)
{
    ErrorEvent event{};
    event.error_code = error_code;
    event.operation_name = operation_name;
    event.params_hash = contract.params_hash;
    event.window_start_ns = contract.window_start_ns;
    event.window_end_ns = contract.window_end_ns;
    event.duration_ms = 0.0;
    event.error_message = error_message;
    event.timestamp = std::chrono::steady_clock::now();
    logErrorEvent(event);
}

void ErrorObservability::incrementMetric(const std::string &metric_name,
                                         const std::unordered_map<std::string, std::string> &labels)
{
    if (config_.enable_metric_increment)
    {
        writeMetricIncrement(metric_name, labels);
    }
}

void ErrorObservability::updateErrorHistogram(const std::string &operation_name,
                                              ErrorCode error_code, double duration_ms)
{
    std::unique_lock lock(tracking_mutex_);
    operation_durations_[operation_name].push_back(duration_ms);
    operation_errors_[operation_name].push_back(error_code);
}

void ErrorObservability::correlatePerformanceWithErrors(const std::string &operation_name,
                                                        double performance_metric,
                                                        ErrorCode error_code)
{
    if (config_.enable_performance_correlation)
    {
        updatePerformanceCorrelation(operation_name, performance_metric, error_code);
    }
}

std::string ErrorObservability::exportErrorReport(const std::string &time_range)
{
    (void)time_range;

    std::ostringstream oss;
    oss << "{";
    oss << "\"total_error_events\":" << total_error_events_.load() << ",";

    std::shared_lock lock(tracking_mutex_);
    oss << "\"error_code_counts\":{";
    bool first = true;
    for (const auto &[code, count] : error_code_counts_)
    {
        if (!first)
        {
            oss << ",";
        }
        first = false;
        oss << "\"" << static_cast<int>(code) << "\":" << count.load();
    }
    oss << "},";

    oss << "\"operation_error_counts\":{";
    first = true;
    for (const auto &[name, count] : operation_error_counts_)
    {
        if (!first)
        {
            oss << ",";
        }
        first = false;
        oss << "\"" << name << "\":" << count.load();
    }
    oss << "}";
    oss << "}";
    return oss.str();
}

std::string ErrorObservability::exportMetrics(const std::string &format)
{
    if (format == "prometheus")
    {
        std::ostringstream oss;
        std::shared_lock lock(tracking_mutex_);
        oss << "# HELP " << config_.metric_prefix << "_total_error_events Total error events\n";
        oss << "# TYPE " << config_.metric_prefix << "_total_error_events counter\n";
        oss << config_.metric_prefix << "_total_error_events " << total_error_events_.load()
            << "\n";

        for (const auto &[code, count] : error_code_counts_)
        {
            oss << "# HELP " << config_.metric_prefix << "_error_code_" << static_cast<int>(code)
                << " Error count by code\n";
            oss << "# TYPE " << config_.metric_prefix << "_error_code_" << static_cast<int>(code)
                << " counter\n";
            oss << config_.metric_prefix << "_error_code_" << static_cast<int>(code) << " "
                << count.load() << "\n";
        }
        return oss.str();
    }

    return exportErrorReport();
}

ErrorObservability::ObservabilityStats ErrorObservability::getStats() const
{
    ObservabilityStats stats{};
    stats.total_error_events = total_error_events_.load();

    std::shared_lock lock(tracking_mutex_);
    for (const auto &[code, count] : error_code_counts_)
    {
        stats.error_code_counts[code] = count.load();
    }
    for (const auto &[name, count] : operation_error_counts_)
    {
        stats.operation_error_counts[name] = count.load();
    }
    stats.duration_histograms = operation_durations_;

    return stats;
}

void ErrorObservability::resetStats()
{
    total_error_events_ = 0;
    std::unique_lock lock(tracking_mutex_);
    error_code_counts_.clear();
    operation_error_counts_.clear();
    operation_durations_.clear();
    operation_errors_.clear();
}

void ErrorObservability::writeStructuredLog(const ErrorEvent &event)
{
    const std::string log_entry =
        config_.log_format == "json" ? event.toJson() : event.toStructuredLog();
    (void)log_entry;
}

void ErrorObservability::writeMetricIncrement(
    const std::string &metric_name, const std::unordered_map<std::string, std::string> &labels)
{
    const std::string formatted = formatMetricName(metric_name, labels);
    (void)formatted;
}

void ErrorObservability::updatePerformanceCorrelation(const std::string &operation_name,
                                                      double performance_metric,
                                                      ErrorCode error_code)
{
    std::unique_lock lock(tracking_mutex_);
    operation_durations_[operation_name].push_back(performance_metric);
    operation_errors_[operation_name].push_back(error_code);
}

std::string
ErrorObservability::formatMetricName(const std::string &metric_name,
                                     const std::unordered_map<std::string, std::string> &labels)
{
    std::ostringstream oss;
    oss << config_.metric_prefix << "." << metric_name;
    if (!labels.empty())
    {
        oss << "{";
        bool first = true;
        for (const auto &[k, v] : labels)
        {
            if (!first)
            {
                oss << ",";
            }
            first = false;
            oss << k << "=\"" << v << "\"";
        }
        oss << "}";
    }
    return oss.str();
}

ErrorHandlingManager &ErrorHandlingManager::instance()
{
    static ErrorHandlingManager manager;
    return manager;
}

void ErrorHandlingManager::setCircuitBreakerConfig(const CircuitBreakerConfig &config)
{
    std::unique_lock lock(mutex_);
    circuit_breaker_config_ = config;
    if (circuit_breaker_)
    {
        circuit_breaker_.reset();
    }
    initializeComponents();
}

void ErrorHandlingManager::setRetryConfig(const RetryBackoffManager::RetryConfig &config)
{
    std::unique_lock lock(mutex_);
    retry_config_ = config;
    if (retry_manager_)
    {
        retry_manager_.reset();
    }
    initializeComponents();
}

void ErrorHandlingManager::setObservabilityConfig(
    const ErrorObservability::ObservabilityConfig &config)
{
    std::unique_lock lock(mutex_);
    observability_config_ = config;
    if (observability_)
    {
        observability_.reset();
    }
    initializeComponents();
}

std::shared_ptr<CircuitBreaker> ErrorHandlingManager::getCircuitBreaker()
{
    std::shared_lock lock(mutex_);
    return circuit_breaker_;
}

std::shared_ptr<RetryBackoffManager> ErrorHandlingManager::getRetryManager()
{
    std::shared_lock lock(mutex_);
    return retry_manager_;
}

std::shared_ptr<ErrorObservability> ErrorHandlingManager::getObservability()
{
    std::shared_lock lock(mutex_);
    return observability_;
}

void ErrorHandlingManager::logError(const std::string &operation_name, ErrorCode error_code,
                                    const CallContract &contract, const std::string &error_message)
{
    std::shared_lock lock(mutex_);
    if (observability_)
    {
        observability_->logErrorEvent(operation_name, error_code, contract, error_message);
    }
}

ErrorHandlingManager::ErrorHandlingStats ErrorHandlingManager::getStats() const
{
    ErrorHandlingStats stats{};
    std::shared_lock lock(mutex_);
    if (circuit_breaker_)
    {
        stats.circuit_breaker_stats = circuit_breaker_->getStats();
    }
    if (retry_manager_)
    {
        stats.retry_stats = retry_manager_->getStats();
    }
    if (observability_)
    {
        stats.observability_stats = observability_->getStats();
    }
    return stats;
}

void ErrorHandlingManager::resetStats()
{
    std::unique_lock lock(mutex_);
    if (circuit_breaker_)
    {
        circuit_breaker_->resetStats();
    }
    if (retry_manager_)
    {
        retry_manager_->resetStats();
    }
    if (observability_)
    {
        observability_->resetStats();
    }
}

bool ErrorHandlingManager::isHealthy() const
{
    std::shared_lock lock(mutex_);
    if (circuit_breaker_ && circuit_breaker_->isTripped())
    {
        return false;
    }
    return true;
}

std::vector<std::string> ErrorHandlingManager::getHealthIssues() const
{
    std::vector<std::string> issues;
    std::shared_lock lock(mutex_);
    if (circuit_breaker_ && circuit_breaker_->isTripped())
    {
        issues.push_back("Circuit breaker is tripped");
    }
    if (retry_manager_)
    {
        const auto retry_stats = retry_manager_->getStats();
        if (retry_stats.failed_retries > retry_stats.successful_retries &&
            retry_stats.total_retries > 10)
        {
            issues.push_back("High retry failure rate");
        }
    }
    return issues;
}

void ErrorHandlingManager::initializeComponents()
{
    if (!circuit_breaker_)
    {
        circuit_breaker_ = std::make_shared<CircuitBreaker>(circuit_breaker_config_);
    }
    if (!retry_manager_)
    {
        retry_manager_ = std::make_shared<RetryBackoffManager>(retry_config_);
    }
    if (!observability_)
    {
        observability_ = std::make_shared<ErrorObservability>(observability_config_);
    }
}

void ErrorHandlingManager::updateErrorStats(ErrorCode error_code, const std::string &operation_name)
{
    std::shared_lock lock(mutex_);
    if (observability_)
    {
        CallContract contract{};
        contract.operation_name = operation_name;
        contract.time_budget_ms = 0.0;
        contract.strict_time_enforcement = false;
        contract.params_hash = 0;
        contract.window_start_ns = 0;
        contract.window_end_ns = 0;
        observability_->logErrorEvent(operation_name, error_code, contract);
    }
}

} // namespace nerve::error
