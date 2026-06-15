#include "nerve/errors/errors.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

namespace nerve::errors
{

ErrorConfig ConfigurableErrorSystem::config_ = ErrorConfig::lightweight();
std::mutex ConfigurableErrorSystem::config_mutex_;

void ConfigurableErrorSystem::configure(const ErrorConfig &config)
{
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = config;
}

ErrorConfig ConfigurableErrorSystem::getConfig()
{
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

bool ConfigurableErrorSystem::isMonitoringEnabled()
{
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_.enable_monitoring;
}

bool ConfigurableErrorSystem::shouldLog(ErrorSeverity level)
{
    std::lock_guard<std::mutex> lock(config_mutex_);
    if (!config_.enable_monitoring)
    {
        return false;
    }
    return static_cast<int>(level) >= static_cast<int>(config_.min_log_level);
}

void ConfigurableErrorSystem::enableExternalLogging(const std::string &log_file)
{
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_.enableExternalLogging = !log_file.empty();
    config_.external_log_file = log_file;
}

void ConfigurableErrorSystem::disableMonitoring()
{
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_.enable_monitoring = false;
}

bool ConfigurableErrorSystem::isExternalLoggingEnabled()
{
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_.enableExternalLogging && !config_.external_log_file.empty();
}

std::string ConfigurableErrorSystem::getExternalLogFile()
{
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_.external_log_file;
}

void ErrorRegistry::reportError(ErrorCode code, const ErrorContext &context)
{
    const auto it = metadata_.find(code);
    const ErrorMetadata &metadata =
        (it != metadata_.end()) ? it->second : getMetadata(ErrorCode::UNKNOWN);

    if (!ConfigurableErrorSystem::isMonitoringEnabled())
    {
        return;
    }
    if (!ConfigurableErrorSystem::shouldLog(metadata.severity))
    {
        return;
    }

#if NERVE_ENABLE_ERROR_TRACKING
    {
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        ++error_counts_[code];

        if (max_history_size_ > 0)
        {
            error_history_.emplace_back(code, context);
            while (error_history_.size() > max_history_size_)
            {
                error_history_.erase(error_history_.begin());
            }
        }
    }

    if (!context.operation_name.empty())
    {
        std::lock_guard<std::mutex> lock(failed_ops_mutex_);
        failed_operations_.insert(context.operation_name);
    }
#endif

    const bool critical =
        metadata.severity == ErrorSeverity::CRITICAL || metadata.requires_human_intervention;

    if (ConfigurableErrorSystem::isExternalLoggingEnabled())
    {
        logErrorToExternalFile(code, context, metadata, critical);
    }
}

void ErrorRegistry::logErrorToExternalFile(ErrorCode code, const ErrorContext &context,
                                           const ErrorMetadata &metadata, bool is_critical)
{
    if (!ConfigurableErrorSystem::isExternalLoggingEnabled())
    {
        return;
    }

    const std::string log_file = ConfigurableErrorSystem::getExternalLogFile();
    if (log_file.empty())
    {
        return;
    }

    std::ofstream stream(log_file, std::ios::app);
    if (!stream.is_open())
    {
        return;
    }

    const auto now = std::chrono::system_clock::now();
    const auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_snapshot{};
#if defined(_WIN32)
    localtime_s(&tm_snapshot, &now_time_t);
#else
    localtime_r(&now_time_t, &tm_snapshot);
#endif

    stream << "[" << std::put_time(&tm_snapshot, "%Y-%m-%d %H:%M:%S") << "] "
           << (is_critical ? "[CRITICAL] " : "[ERROR] ") << "code=" << static_cast<uint32_t>(code)
           << " name=" << metadata.name << " operation=" << context.operation_name
           << " component=" << context.component_name << " details=" << context.toJson() << '\n';
}

} // namespace nerve::errors
