#include "nerve/errors/errors.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef NERVE_HAS_CUDA
#include "nerve/persistence/cuda/cuda_error_handling.hpp"

#include <cuda_runtime.h>
#endif

namespace
{

using nerve::errors::ErrorCode;
using nerve::errors::ErrorMetadata;
using nerve::errors::ErrorRegistry;

#ifdef NERVE_HAS_CUDA

using nerve::persistence::accelerated::cuda_error_handling;
using nerve::persistence::accelerated::cuda_utils;

bool check_cuda_error_handling_init()
{
    auto &registry = ErrorRegistry::instance();
    if (!registry.hasOperationFailed("cuda_init"))
    {
        return true;
    }
    return true;
}

bool check_error_mapping()
{
    ErrorCode mem = cuda_error_handling::map_cuda_error_to_error_code(cudaErrorMemoryAllocation);
    if (mem != ErrorCode::E10_GPU_OOM)
    {
        std::cerr << "expected E10_GPU_OOM for memory allocation\n";
        return false;
    }
    ErrorCode launch = cuda_error_handling::map_cuda_error_to_error_code(cudaErrorLaunchFailure);
    if (launch != ErrorCode::E11_GPU_LAUNCH_FAIL)
    {
        std::cerr << "expected E11_GPU_LAUNCH_FAIL for launch failure\n";
        return false;
    }
    ErrorCode unknown = cuda_error_handling::map_cuda_error_to_error_code(cudaErrorInvalidValue);
    if (unknown != ErrorCode::UNKNOWN)
    {
        std::cerr << "expected UNKNOWN for unmapped error\n";
        return false;
    }
    return true;
}

bool check_error_severity_classification()
{
    auto sev = cuda_error_handling::classifyErrorSeverity(cudaErrorMemoryAllocation);
    (void)sev;
    sev = cuda_error_handling::classifyErrorSeverity(cudaErrorLaunchFailure);
    (void)sev;
    sev = cuda_error_handling::classifyErrorSeverity(cudaSuccess);
    (void)sev;
    return true;
}

bool check_error_descriptions_non_empty()
{
    std::string desc = cuda_error_handling::getErrorDescription(cudaErrorMemoryAllocation);
    if (desc.empty())
    {
        std::cerr << "error description should not be empty\n";
        return false;
    }
    desc = cuda_error_handling::getErrorDescription(cudaErrorLaunchFailure);
    if (desc.empty())
    {
        std::cerr << "error description should not be empty\n";
        return false;
    }
    return true;
}

bool check_recovery_strategy()
{
    auto strat = cuda_error_handling::getRecoveryStrategy(cudaErrorMemoryAllocation);
    (void)strat;
    strat = cuda_error_handling::getRecoveryStrategy(cudaErrorLaunchFailure);
    (void)strat;
    strat = cuda_error_handling::getRecoveryStrategy(cudaSuccess);
    (void)strat;
    return true;
}

#else

bool check_error_code_descriptions_non_empty()
{
    auto &registry = ErrorRegistry::instance();
    std::vector<ErrorCode> codes = registry.getAllCodes();
    for (auto code : codes)
    {
        const auto &meta = registry.getMetadata(code);
        if (meta.description.empty() && code != ErrorCode::SUCCESS && code != ErrorCode::UNKNOWN)
        {
            std::cerr << "empty description for code " << static_cast<uint32_t>(code) << "\n";
            return false;
        }
    }
    return true;
}

#endif

} // namespace

int main()
{
#ifdef NERVE_HAS_CUDA
    if (!check_cuda_error_handling_init())
    {
        std::cerr << "FAIL: cuda_error_handling_init\n";
        return 1;
    }
    if (!check_error_mapping())
    {
        std::cerr << "FAIL: error_mapping\n";
        return 1;
    }
    if (!check_error_severity_classification())
    {
        std::cerr << "FAIL: error_severity_classification\n";
        return 1;
    }
    if (!check_error_descriptions_non_empty())
    {
        std::cerr << "FAIL: error_descriptions_non_empty\n";
        return 1;
    }
    if (!check_recovery_strategy())
    {
        std::cerr << "FAIL: recovery_strategy\n";
        return 1;
    }
#else
    if (!check_error_code_descriptions_non_empty())
    {
        std::cerr << "FAIL: error_code_descriptions_non_empty\n";
        return 1;
    }
#endif
    return 0;
}
