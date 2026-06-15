
#pragma once

#include "nerve/errors/errors.hpp"

#include <chrono>
#include <string>

namespace nerve::core
{

using ErrorCode = errors::ErrorCode;

class ErrorEvent
{
public:
    ErrorEvent()
        : error_code_(ErrorCode::SUCCESS)
        , message_()
        , timestamp_(std::chrono::high_resolution_clock::now())
    {}

    ErrorEvent(ErrorCode code, std::string message)
        : error_code_(code)
        , message_(std::move(message))
        , timestamp_(std::chrono::high_resolution_clock::now())
    {}

    virtual ~ErrorEvent() = default;

    ErrorCode getErrorCode() const { return error_code_; }
    const std::string &getMessage() const { return message_; }
    const std::chrono::high_resolution_clock::time_point &getTimestamp() const
    {
        return timestamp_;
    }

private:
    ErrorCode error_code_;
    std::string message_;
    std::chrono::high_resolution_clock::time_point timestamp_;
};

} // namespace nerve::core
