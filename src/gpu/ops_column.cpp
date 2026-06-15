#include "nerve/gpu/kernel_launcher.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <string_view>
#include <vector>

namespace nerve::gpu
{

#ifdef NERVE_HAS_CUDA
namespace detail
{
extern void launchSymmetricDifference(const int *col_a_data, int col_a_size, const int *col_b_data,
                                      int col_b_size, int *out_result, int *out_count,
                                      int max_result_size, cudaStream_t stream);
}
#endif

namespace
{

#ifdef NERVE_HAS_CUDA
bool checkedByteCount(std::size_t count, std::size_t element_size, std::size_t &out) noexcept
{
    if (count != 0 && element_size > std::numeric_limits<std::size_t>::max() / count)
    {
        return false;
    }
    out = count * element_size;
    return true;
}

bool checkedIntSize(std::size_t value, int &out) noexcept
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

errors::ErrorResult<void> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

void cleanupColumnBuffers(int *d_col_a, int *d_col_b, int *d_result, int *d_count,
                          cudaStream_t stream) noexcept
{
    if (d_col_a != nullptr)
    {
        cudaFree(d_col_a);
    }
    if (d_col_b != nullptr)
    {
        cudaFree(d_col_b);
    }
    if (d_result != nullptr)
    {
        cudaFree(d_result);
    }
    if (d_count != nullptr)
    {
        cudaFree(d_count);
    }
    if (stream != nullptr)
    {
        cudaStreamDestroy(stream);
    }
}
#endif

void symmetricDifferenceColumnsCpu(const std::vector<int> &column_a,
                                   const std::vector<int> &column_b, std::vector<int> &out_result)
{
    std::vector<int> result;
    result.reserve(column_a.size() + column_b.size());

    std::set_symmetric_difference(column_a.begin(), column_a.end(), column_b.begin(),
                                  column_b.end(), std::back_inserter(result));
    out_result = std::move(result);
}

} // namespace

errors::ErrorResult<void>
ComputeManager::symmetricDifferenceColumns(const std::vector<int> &column_a,
                                           const std::vector<int> &column_b,
                                           std::vector<int> &out_result)
{
    constexpr const char *operation = "symmetricDifferenceColumns";

    auto strategy = selectStrategy(OperationType::kColumnSymmetricDifference,
                                   column_a.size() + column_b.size());

    if (strategy == Strategy::kCPUOnly)
    {
        symmetricDifferenceColumnsCpu(column_a, column_b, out_result);
        recordSuccess(operation, 1.0);
        return errors::ErrorResult<void>::success();
    }

#ifndef NERVE_HAS_CUDA
    symmetricDifferenceColumnsCpu(column_a, column_b, out_result);
    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
#else
    int *d_col_a = nullptr;
    int *d_col_b = nullptr;
    int *d_result = nullptr;
    int *d_count = nullptr;

    if (column_a.size() > std::numeric_limits<std::size_t>::max() - column_b.size())
    {
        recordFailure(operation, "Column symmetric difference input size overflows");
        return resourceLimit("column symmetric difference input size overflows");
    }
    const size_t max_result = column_a.size() + column_b.size();
    if (max_result == 0)
    {
        out_result.clear();
        recordSuccess(operation, 1.0);
        return errors::ErrorResult<void>::success();
    }

    int col_a_size = 0;
    int col_b_size = 0;
    int max_result_size = 0;
    std::size_t col_a_bytes = 0;
    std::size_t col_b_bytes = 0;
    std::size_t result_bytes = 0;
    if (!checkedIntSize(column_a.size(), col_a_size) ||
        !checkedIntSize(column_b.size(), col_b_size) ||
        !checkedIntSize(max_result, max_result_size) ||
        !checkedByteCount(column_a.size(), sizeof(int), col_a_bytes) ||
        !checkedByteCount(column_b.size(), sizeof(int), col_b_bytes) ||
        !checkedByteCount(max_result, sizeof(int), result_bytes))
    {
        recordFailure(operation, "Column symmetric difference exceeds CUDA kernel limits");
        return resourceLimit("column symmetric difference exceeds CUDA kernel limits");
    }

    cudaError_t err = cudaMalloc(&d_col_a, col_a_bytes);
    if (err != cudaSuccess)
    {
        recordFailure(operation, "Failed to allocate device memory for column A");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    err = cudaMalloc(&d_col_b, col_b_bytes);
    if (err != cudaSuccess)
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, nullptr);
        recordFailure(operation, "Failed to allocate device memory for column B");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    err = cudaMalloc(&d_result, result_bytes);
    if (err != cudaSuccess)
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, nullptr);
        recordFailure(operation, "Failed to allocate device memory for result");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    err = cudaMalloc(&d_count, sizeof(int));
    if (err != cudaSuccess)
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, nullptr);
        recordFailure(operation, "Failed to allocate device memory for count");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    err = cudaMemcpy(d_col_a, column_a.data(), col_a_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(d_col_b, column_b.data(), col_b_bytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset(d_count, 0, sizeof(int));
    }
    if (err != cudaSuccess)
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, nullptr);
        recordFailure(operation, "Failed to initialize device memory for symmetric difference");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    cudaStream_t stream = nullptr;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, nullptr);
        recordFailure(operation, "Failed to create stream for symmetric difference");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    detail::launchSymmetricDifference(d_col_a, col_a_size, d_col_b, col_b_size, d_result, d_count,
                                      max_result_size, stream);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, stream);
        recordFailure(operation, "Failed to execute symmetric difference kernel");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    int result_count = 0;
    err = cudaMemcpy(&result_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess || result_count < 0 || result_count > max_result_size)
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, stream);
        recordFailure(operation, "Invalid symmetric difference result count");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    out_result.resize(static_cast<std::size_t>(result_count));
    std::size_t result_copy_bytes = 0;
    if (!checkedByteCount(out_result.size(), sizeof(int), result_copy_bytes))
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, stream);
        recordFailure(operation, "Symmetric difference result copy size overflows");
        return resourceLimit("symmetric difference result copy size overflows");
    }
    err = cudaMemcpy(out_result.data(), d_result, result_copy_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, stream);
        recordFailure(operation, "Failed to copy symmetric difference result");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    cleanupColumnBuffers(d_col_a, d_col_b, d_result, d_count, stream);

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
#endif
}

errors::ErrorResult<void>
ComputeManager::computeReduction(persistence::Reducer &reduction,
                                 const algebra::BoundaryMatrix &boundary_matrix)
{
    constexpr const char *operation = "computeReduction";

    if (reduction.getMatrix() != &boundary_matrix)
    {
        recordFailure(operation, "Reducer matrix does not match requested boundary matrix");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "reducer matrix does not match boundary matrix");
    }

    const bool previous_gpu_state = reduction.gpuEnabled();
    reduction.enableGPU(false);
    reduction.reduceMatrix();
    reduction.enableGPU(previous_gpu_state);

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void>
ComputeManager::computeCohomology(persistence::Reducer &reduction,
                                  const algebra::BoundaryMatrix &boundary_matrix)
{
    constexpr const char *operation = "computeCohomology";

    if (reduction.getMatrix() != &boundary_matrix)
    {
        recordFailure(operation, "Reducer matrix does not match requested boundary matrix");
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "reducer matrix does not match boundary matrix");
    }

    const bool previous_gpu_state = reduction.gpuEnabled();
    reduction.enableGPU(false);
    reduction.CohomologyReduction();
    reduction.enableGPU(previous_gpu_state);

    recordSuccess(operation, 1.0);
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::gpu
