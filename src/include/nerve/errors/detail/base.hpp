
#pragma once

#include <chrono>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nerve::errors
{

struct ErrorContext;

class NerveError : public std::runtime_error
{
public:
    explicit NerveError(const std::string &message)
        : std::runtime_error(message)
        , file_("unknown")
        , line_(0)
        , function_("unknown")
        , timestamp_(std::chrono::steady_clock::now())
    {}

    NerveError(const std::string &message, const char *file, int line, const char *function)
        : std::runtime_error(message)
        , file_(file ? file : "unknown")
        , line_(line)
        , function_(function ? function : "unknown")
        , timestamp_(std::chrono::steady_clock::now())
    {}

    virtual ~NerveError() = default;

    NerveError &addContext(const std::string &key, const std::string &value)
    {
        context_values_.emplace_back(key, value);
        return *this;
    }

    template <typename T>
    NerveError &addContext(const std::string &key, const T &value)
    {
        std::ostringstream oss;
        oss << value;
        context_values_.emplace_back(key, oss.str());
        return *this;
    }

    template <typename T, typename U>
    NerveError &addExpectedActual(const std::string &what, const T &expected, const U &actual)
    {
        std::ostringstream oss;
        oss << what << " mismatch: expected " << expected << ", got " << actual;
        context_values_.emplace_back(what, oss.str());
        return *this;
    }

    NerveError &addSuggestion(const std::string &suggestion)
    {
        suggestions_.push_back(suggestion);
        return *this;
    }

    NerveError &setOperation(const std::string &operation)
    {
        operation_ = operation;
        return *this;
    }

    const std::string &file() const { return file_; }
    int line() const { return line_; }
    const std::string &function() const { return function_; }
    const std::string &operation() const { return operation_; }
    const std::vector<std::pair<std::string, std::string>> &contextValues() const
    {
        return context_values_;
    }
    const std::vector<std::string> &suggestions() const { return suggestions_; }

    std::string toString() const
    {
        std::ostringstream oss;

        oss << errorTypeName() << ": " << std::runtime_error::what() << "\n";

        if (!operation_.empty())
        {
            oss << "  Operation: " << operation_ << "\n";
        }

        if (line_ > 0)
        {
            oss << "  File: " << file_ << ":" << line_ << "\n";
            oss << "  Function: " << function_ << "\n";
        }

        if (!context_values_.empty())
        {
            oss << "\n  Context:\n";
            for (const auto &[key, value] : context_values_)
            {
                oss << "    " << key << ": " << value << "\n";
            }
        }

        if (!suggestions_.empty())
        {
            oss << "\n  Suggestions:\n";
            for (const auto &suggestion : suggestions_)
            {
                oss << "    - " << suggestion << "\n";
            }
        }

        return oss.str();
    }

    virtual const char *what() const noexcept override
    {
        if (formatted_message_.empty())
        {
            try
            {
                formatted_message_ = toString();
            }
            catch (...)
            {
                return std::runtime_error::what();
            }
        }
        return formatted_message_.c_str();
    }

    virtual std::string errorTypeName() const { return "NerveError"; }

    virtual uint32_t errorCode() const { return 0xFFFFFFFF; }

protected:
    void setBaseMessage(const std::string &message)
    {
        *static_cast<std::runtime_error *>(this) = std::runtime_error(message);
        formatted_message_.clear();
    }

    std::string file_;
    int line_;
    std::string function_;
    std::string operation_;
    std::vector<std::pair<std::string, std::string>> context_values_;
    std::vector<std::string> suggestions_;
    std::chrono::steady_clock::time_point timestamp_;
    mutable std::string formatted_message_;
};

template <typename Container>
std::string formatShape(const Container &shape)
{
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto &dim : shape)
    {
        if (!first)
            oss << ", ";
        oss << dim;
        first = false;
    }
    oss << "]";
    return oss.str();
}

inline std::string formatBytes(uint64_t bytes)
{
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4)
    {
        size /= 1024.0;
        unit_idx++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return oss.str();
}

inline std::string formatDurationMs(double ms)
{
    if (ms < 1.0)
    {
        return std::to_string(static_cast<int>(ms * 1000)) + " us";
    }
    else if (ms < 1000.0)
    {
        return std::to_string(static_cast<int>(ms)) + " ms";
    }
    else
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (ms / 1000.0) << " s";
        return oss.str();
    }
}

} // namespace nerve::errors
