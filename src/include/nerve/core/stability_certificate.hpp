
#pragma once

#include <cstddef>

namespace nerve
{

class StabilityCertificate
{
public:
    StabilityCertificate() = default;

    static StabilityCertificate createPh5Ph6Certificate(std::size_t problemSize,
                                                        std::size_t maxDimension,
                                                        std::size_t memoryLimitMb,
                                                        std::size_t timeLimitMs)
    {
        StabilityCertificate certificate;
        certificate.problem_size_ = problemSize;
        certificate.max_dimension_ = maxDimension;
        certificate.memory_limit_mb_ = memoryLimitMb;
        certificate.time_limit_ms_ = timeLimitMs;
        return certificate;
    }

    std::size_t problemSize() const { return problem_size_; }
    std::size_t maxDimension() const { return max_dimension_; }
    std::size_t memoryLimitMb() const { return memory_limit_mb_; }
    std::size_t timeLimitMs() const { return time_limit_ms_; }

private:
    std::size_t problem_size_ = 0;
    std::size_t max_dimension_ = 0;
    std::size_t memory_limit_mb_ = 0;
    std::size_t time_limit_ms_ = 0;
};

} // namespace nerve
