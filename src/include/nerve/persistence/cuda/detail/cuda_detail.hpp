#pragma once
#include "nerve/algebra/complex.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/types.hpp"

#include <string>
#include <vector>

namespace nerve::persistence::accelerated::cuda_utils
{
bool is_cuda_available();
::nerve::errors::ErrorResult<::nerve::DeviceInfo> getDeviceProperties(int device_id);
::nerve::Size getOptimalBlockSize(int problem_size, int block_size);
::nerve::Size getOptimalGridSize(int problem_size, int block_size);
} // namespace nerve::persistence::accelerated::cuda_utils

#ifdef NERVE_HAS_CUDA
namespace nerve::gpu::persistence::detail
{
class GPUClearingEngine
{
public:
    struct ClearingResult
    {};
    static ::nerve::errors::ErrorResult<void>
    applyClearingOptimization(const ::nerve::algebra::BoundaryMatrix &matrix,
                              const std::vector<int> &dims, const std::vector<double> &filt,
                              int max_dim, double max_radius, ClearingResult &result);
};
} // namespace nerve::gpu::persistence::detail

namespace nerve::gpu::detail
{
class GPUCohomologyEngine
{
public:
    static ::nerve::errors::ErrorResult<void>
    performCohomologyReduction(const ::nerve::algebra::BoundaryMatrix &matrix,
                               std::vector<::nerve::Index> &pivots, std::vector<bool> &processed,
                               std::vector<std::vector<::nerve::Size>> &coboundary);
};
} // namespace nerve::gpu::detail

namespace nerve::gpu::persistence
{
class ReductionEngine
{
public:
    static ::nerve::errors::ErrorResult<void>
    computeReduction(const ::nerve::algebra::BoundaryMatrix &matrix,
                     std::vector<::nerve::Index> &pivots,
                     std::vector<std::pair<::nerve::Size, ::nerve::Size>> &pairs);
};
} // namespace nerve::gpu::persistence

namespace nerve::gpu::multi
{
struct MultiGPUConfig
{
    int num_gpus = 1;
    bool enabled = false;
};
MultiGPUConfig detectMultiGpuConfiguration();
std::vector<int> distributeIndices(size_t total, int num_gpus);
} // namespace nerve::gpu::multi
#endif

namespace nerve::persistence::cuda
{
std::string cudaErrorToString(int cuda_error);
int mapCudaErrorToNerveError(int cuda_error);
} // namespace nerve::persistence::cuda
