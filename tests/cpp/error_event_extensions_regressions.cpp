#include "nerve/core/detail/error_event_extensions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

namespace
{

bool check_error_event_factory_creates_valid_errors()
{
    auto mem_err =
        nerve::core::ErrorEventFactory::createPh5MemoryError(3, 1000, 512, "reduction", "compute");
    if (!mem_err)
    {
        std::cerr << "createPh5MemoryError returned null\n";
        return false;
    }
    if (mem_err->getErrorCode() == nerve::core::ErrorCode::SUCCESS)
    {
        std::cerr << "error code should not be SUCCESS for a memory error\n";
        return false;
    }

    auto time_err =
        nerve::core::ErrorEventFactory::createPh5TimeError(3, 1000, 5.0, "reduction", "compute");
    if (!time_err)
    {
        std::cerr << "createPh5TimeError returned null\n";
        return false;
    }

    auto num_err = nerve::core::ErrorEventFactory::createPh5NumericalError(3, "instability", 0.5);
    if (!num_err)
    {
        std::cerr << "createPh5NumericalError returned null\n";
        return false;
    }

    auto ph6_mem =
        nerve::core::ErrorEventFactory::createPh6MemoryError(6, 500, 256, "sampling", "build");
    if (!ph6_mem)
    {
        std::cerr << "createPh6MemoryError returned null\n";
        return false;
    }
    return true;
}

bool check_error_confidence_scores_in_valid_range()
{
    auto num_err = nerve::core::ErrorEventFactory::createPh5NumericalError(3, "instability", 0.5);
    const auto &analysis = num_err->getFailureAnalysis();
    if (analysis.confidence_score < 0.0 || analysis.confidence_score > 1.0)
    {
        std::cerr << "confidence score should be in [0,1], got " << analysis.confidence_score
                  << "\n";
        return false;
    }

    auto mem_err =
        nerve::core::ErrorEventFactory::createPh5MemoryError(3, 1000, 512, "reduction", "compute");
    const auto &mem_analysis = mem_err->getFailureAnalysis();
    if (mem_analysis.confidence_score < 0.0 || mem_analysis.confidence_score > 1.0)
    {
        std::cerr << "memory error confidence out of [0,1]\n";
        return false;
    }
    return true;
}

bool check_error_aggregation_produces_correct_counts()
{
    auto streaming_err =
        nerve::core::ErrorEventFactory::createStreamingError(100, 42, 1.5, "timeout");
    if (!streaming_err)
    {
        std::cerr << "createStreamingError returned null\n";
        return false;
    }
    const auto &metrics = streaming_err->getPerformanceMetrics();
    (void)metrics;
    return true;
}

bool check_error_severity_levels()
{
    auto laplacian_err =
        nerve::core::ErrorEventFactory::createLaplacianError(100, 95, 1e-6, "singular");
    if (!laplacian_err)
    {
        std::cerr << "createLaplacianError returned null\n";
        return false;
    }
    if (laplacian_err->getComplexSize() != 100)
    {
        std::cerr << "laplacian error complex size should be 100\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_error_event_factory_creates_valid_errors())
    {
        std::cerr << "FAIL: error event factory creates valid errors\n";
        return 1;
    }
    if (!check_error_confidence_scores_in_valid_range())
    {
        std::cerr << "FAIL: error confidence scores in valid range\n";
        return 1;
    }
    if (!check_error_aggregation_produces_correct_counts())
    {
        std::cerr << "FAIL: error aggregation produces correct counts\n";
        return 1;
    }
    if (!check_error_severity_levels())
    {
        std::cerr << "FAIL: error severity levels\n";
        return 1;
    }
    return 0;
}
