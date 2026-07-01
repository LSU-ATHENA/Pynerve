
#ifdef NERVE_HAS_CUDA
#include "nerve/core_types.hpp"
#include "nerve/persistence/cuda/detail/cuda_detail.hpp"

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace
{

bool check_gpu_clearing_construction()
{
    nerve::algebra::BoundaryMatrix matrix(0, 0);
    std::vector<int> dims;
    std::vector<double> filt;
    nerve::gpu::persistence::detail::GPUClearingEngine::ClearingResult result;
    auto status = nerve::gpu::persistence::detail::GPUClearingEngine::applyClearingOptimization(
        matrix, dims, filt, 0, 1.0, result);
    if (status.isError())
        return false;
    return true;
}

bool check_gpu_cohomology_configures()
{
    nerve::algebra::BoundaryMatrix matrix(0, 0);
    std::vector<nerve::Index> pivots;
    std::vector<bool> processed;
    std::vector<std::vector<nerve::Size>> coboundary;
    auto status = nerve::gpu::detail::GPUCohomologyEngine::performCohomologyReduction(
        matrix, pivots, processed, coboundary);
    if (status.isError())
        return false;
    return true;
}

bool check_gpu_reduction_configures()
{
    nerve::algebra::BoundaryMatrix matrix(0, 0);
    std::vector<nerve::Index> pivots;
    std::vector<std::pair<nerve::Size, nerve::Size>> pairs;
    auto status = nerve::gpu::persistence::ReductionEngine::computeReduction(matrix, pivots, pairs);
    if (status.isError())
        return false;
    return true;
}

bool check_multi_gpu_configuration()
{
    auto info = nerve::gpu::multi::detectMultiGpuConfiguration();
    if (info.num_gpus < 1)
        return false;
    auto indices = nerve::gpu::multi::distributeIndices(100, std::max(1, info.num_gpus));
    if (indices.empty())
        return false;
    return true;
}

bool check_error_mapping()
{
    namespace cu = nerve::persistence::accelerated::cuda_utils;
    if (cu::is_cuda_available())
    {
        auto props = cu::getDeviceProperties(0);
        if (props.isError())
            return false;
    }
    nerve::Size block = cu::getOptimalBlockSize(0, 0);
    if (block == 0)
        return false;
    nerve::Size grid = cu::getOptimalGridSize(1000, 256);
    if (grid == 0)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_gpu_clearing_construction())
        return 1;
    if (!check_gpu_cohomology_configures())
        return 1;
    if (!check_gpu_reduction_configures())
        return 1;
    if (!check_multi_gpu_configuration())
        return 1;
    if (!check_error_mapping())
        return 1;
    return 0;
}
#else
int main()
{
    return 0;
}
#endif
