#pragma once

#include "nerve/errors/errors.hpp"

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::errors
{

enum class Severity
{
    Info,
    Warning,
    Error,
    Critical
};

struct ErrorPolicy
{
    bool throw_on_error = true;
    bool log_all_errors = true;
    Severity minimum_log_severity = Severity::Warning;
};

struct Error
{
    ErrorCode code;
    int severity;
    std::string message;
};

using ErrorHandler = std::function<void(const Error &)>;

class ConfigurableErrorSystemBase
{
public:
    static ConfigurableErrorSystemBase &instance()
    {
        static ConfigurableErrorSystemBase instance;
        return instance;
    }

    void setPolicy(const ErrorPolicy &policy) { policy_ = policy; }
    [[nodiscard]] ErrorPolicy getPolicy() const { return policy_; }

    void registerHandler(Severity severity, ErrorHandler handler)
    {
        handlers_[severity] = std::move(handler);
    }

    void report(const Error &error)
    {
        if (!isErrorEnabled(error.code))
        {
            return;
        }

        if (policy_.log_all_errors &&
            error.severity >= static_cast<int>(policy_.minimum_log_severity))
        {
            auto it = handlers_.find(static_cast<Severity>(error.severity));
            if (it != handlers_.end() && it->second)
            {
                it->second(error);
            }
        }

        if (policy_.throw_on_error && error.severity >= static_cast<int>(Severity::Error))
        {
            throw std::runtime_error(error.message);
        }
    }

    template <typename... Args>
    void reportError(ErrorCode code, Severity severity, const std::string &message, Args &&...)
    {
        static_assert(sizeof...(Args) == 0,
                      "ConfigurableErrorSystemBase::reportError does not accept format arguments");

        Error err;
        err.code = code;
        err.severity = static_cast<int>(severity);
        err.message = message;
        report(err);

        ErrorContext context;
        context.operation_name = "configurable_error_system";
        context.component_name = "ConfigurableErrorSystemBase";
        context.addMetadata("message", message);
        ErrorRegistry::instance().reportError(code, context);
    }

    void setErrorEnabled(ErrorCode code, bool enabled) { enabled_errors_[code] = enabled; }

    [[nodiscard]] bool isErrorEnabled(ErrorCode code) const
    {
        auto it = enabled_errors_.find(code);
        return it == enabled_errors_.end() || it->second;
    }

    void setGlobalConfig(const std::map<std::string, std::string> &config) { config_ = config; }

    [[nodiscard]] std::map<std::string, std::string> getGlobalConfig() const { return config_; }

    void handleTdaError(ErrorCode code, const std::string &context)
    {
        reportError(code, Severity::Error, "TDA Error in " + context);
    }

    void registerErrorCode(ErrorCode code, const std::string &description)
    {
        error_descriptions_[code] = description;
    }

    [[nodiscard]] std::string getErrorDescription(ErrorCode code) const
    {
        auto it = error_descriptions_.find(code);
        return it != error_descriptions_.end() ? it->second : "Unknown error";
    }

private:
    ConfigurableErrorSystemBase();
    ~ConfigurableErrorSystemBase();
    ConfigurableErrorSystemBase(const ConfigurableErrorSystemBase &) = delete;
    ConfigurableErrorSystemBase &operator=(const ConfigurableErrorSystemBase &) = delete;

    ErrorPolicy policy_;
    std::map<Severity, ErrorHandler> handlers_;
    std::map<ErrorCode, bool> enabled_errors_;
    std::map<std::string, std::string> config_;
    std::map<ErrorCode, std::string> error_descriptions_;
};

inline void configureErrorPolicy(const ErrorPolicy &policy)
{
    ConfigurableErrorSystemBase::instance().setPolicy(policy);
}

inline void reportSystemError(ErrorCode code, const std::string &message)
{
    ConfigurableErrorSystemBase::instance().reportError(code, Severity::Error, message);
}

} // namespace nerve::errors
