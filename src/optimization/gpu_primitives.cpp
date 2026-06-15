// AcceleratedGpuPrimitives Implementation
// GPU primitive operations for optimization

#include "nerve/optimization/component_optimizations.hpp"
#include "nerve/optimization/gpu_primitives_helpers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <ranges>

#if defined(__CUDACC__)
#include "nerve/persistence/cuda/cuda_distance_matrix.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace nerve::optimization
{

#if defined(__CUDACC__)
__global__ void columnXorReductionKernel(const uint32_t *column, uint32_t *result, size_t size);
__global__ void tiledSpmvKernel(const float *matrix, const float *vector, float *result,
                                size_t rows, size_t cols);
#endif

AcceleratedGpuPrimitives::AcceleratedGpuPrimitives(const GPUConfig &config)
#if defined(__CUDACC__)
    : config_(config)
    , stream_pool_(nullptr)
{
#else
    : config_(config)
{
#endif
#if defined(__CUDACC__)
    if (config_.num_streams > 0)
    {
        stream_pool_ = createStreamPool(config_.num_streams);
    }
#endif
}

AcceleratedGpuPrimitives::~AcceleratedGpuPrimitives()
{
#if defined(__CUDACC__)
    if (stream_pool_ != nullptr)
    {
        destroyStreamPool(stream_pool_, config_.num_streams);
    }
#endif
}

void AcceleratedGpuPrimitives::computeDistanceMatrixBatch(float *d_points, size_t n_points,
                                                          size_t point_dim)
{
    if (d_points == nullptr || n_points == 0 || point_dim == 0)
    {
        return;
    }
    if (multiplyWouldOverflow(n_points, point_dim) || multiplyWouldOverflow(n_points, n_points))
    {
        return;
    }

    const size_t point_values = n_points * point_dim;
    const size_t distance_values = n_points * n_points;
    size_t point_bytes = 0;
    size_t distance_bytes = 0;
    if (!checkedByteCount(point_values, sizeof(float), point_bytes) ||
        !checkedByteCount(distance_values, sizeof(float), distance_bytes))
    {
        return;
    }
    std::vector<float> host_points(point_values, 0.0f);
    if (cudaMemcpy(host_points.data(), d_points, point_bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess)
    {
        return;
    }
    if (!finiteFloatValues(host_points.data(), host_points.size()))
    {
        return;
    }

    std::vector<float> host_distances(distance_values, 0.0f);
    ErrorCode distance_error = ErrorCode::SUCCESS;
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i + 1; j < n_points; ++j)
        {
            const size_t i_offset = i * point_dim;
            const size_t j_offset = j * point_dim;
            float dist = 0.0f;
            if (!checkedFiniteDistance(&host_points[i_offset], &host_points[j_offset], point_dim,
                                       dist, &distance_error))
            {
                return;
            }
            host_distances[i * n_points + j] = dist;
            host_distances[j * n_points + i] = dist;
        }
    }

    const size_t write_count = std::min(point_values, host_distances.size());
    size_t write_bytes = 0;
    if (write_count == 0 || !checkedByteCount(write_count, sizeof(float), write_bytes))
    {
        return;
    }
    if (cudaMemcpy(d_points, host_distances.data(), write_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess)
    {
        return;
    }
}

void AcceleratedGpuPrimitives::reduceColumnGpu(uint32_t *column_data, size_t size)
{
    if (column_data == nullptr || size == 0 || multiplyWouldOverflow(size, sizeof(uint32_t)))
    {
        return;
    }

#if defined(__CUDACC__)
    if (stream_pool_ == nullptr)
    {
        return;
    }
    size_t column_bytes = 0;
    if (!checkedByteCount(size, sizeof(uint32_t), column_bytes))
    {
        return;
    }
    uint32_t *d_column = nullptr;
    uint32_t *d_result = nullptr;
    if (cudaMalloc(&d_column, column_bytes) != cudaSuccess)
    {
        return;
    }
    if (cudaMalloc(&d_result, sizeof(uint32_t)) != cudaSuccess)
    {
        cudaFree(d_column);
        return;
    }
    if (cudaMemcpy(d_column, column_data, column_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_result, 0, sizeof(uint32_t)) != cudaSuccess)
    {
        cudaFree(d_column);
        cudaFree(d_result);
        return;
    }

    constexpr int kBlockSize = 256;
    int grid_size = 0;
    if (!checkedGridBlocks(size, static_cast<size_t>(kBlockSize), grid_size))
    {
        cudaFree(d_column);
        cudaFree(d_result);
        return;
    }

    columnXorReductionKernel<<<grid_size, kBlockSize, kBlockSize * sizeof(uint32_t),
                               stream_pool_[0]>>>(d_column, d_result, size);

    uint32_t result = 0;
    const cudaError_t status = (cudaGetLastError() == cudaSuccess)
                                   ? cudaStreamSynchronize(stream_pool_[0])
                                   : cudaErrorUnknown;
    if (status == cudaSuccess &&
        cudaMemcpy(&result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess)
    {
        column_data[0] = result;
    }
    cudaFree(d_column);
    cudaFree(d_result);
#else
    uint32_t folded = 0;
    for (size_t i = 0; i < size; ++i)
    {
        folded ^= column_data[i];
    }
    if (size > 0)
    {
        column_data[0] = folded;
    }
#endif
}

void AcceleratedGpuPrimitives::sparseMatrixVectorMultiply(float *d_matrix, float *d_vector,
                                                          float *d_result, size_t rows, size_t cols)
{
    if (d_matrix == nullptr || d_vector == nullptr || d_result == nullptr || rows == 0 ||
        cols == 0 || multiplyWouldOverflow(rows, cols))
    {
        return;
    }
    const size_t matrix_values = rows * cols;

#if defined(__CUDACC__)
    if (stream_pool_ == nullptr)
    {
        return;
    }
    constexpr int kTileSize = 32;
    constexpr int kTileRows = 8;
    dim3 block(kTileSize, kTileRows);
    int grid_rows = 0;
    if (!checkedGridBlocks(rows, static_cast<size_t>(kTileRows), grid_rows))
    {
        return;
    }
    dim3 grid(1, grid_rows);

    tiledSpmvKernel<<<grid, block, 0, stream_pool_[0]>>>(d_matrix, d_vector, d_result, rows, cols);
    if (cudaGetLastError() != cudaSuccess)
    {
        return;
    }
#else
    if (!finiteFloatValues(d_matrix, matrix_values) || !finiteFloatValues(d_vector, cols))
    {
        return;
    }
    ErrorCode dot_error = ErrorCode::SUCCESS;
    for (size_t row = 0; row < rows; ++row)
    {
        float sum = 0.0f;
        if (!checkedFiniteDotProduct(&d_matrix[row * cols], d_vector, cols, sum, &dot_error))
        {
            return;
        }
        d_result[row] = sum;
    }
#endif
}

ErrorCode AcceleratedGpuPrimitives::executeBatchedOperation(const BatchedOperation &operation,
                                                            const CallContract &contract)
{
    if (operation.input_buffers.size() != operation.output_buffers.size() ||
        operation.input_buffers.size() != operation.batch_sizes.size())
    {
        return ErrorCode::IO_READ_ERROR;
    }

    const auto start = std::chrono::steady_clock::now();
    for (size_t batch = 0; batch < operation.input_buffers.size(); ++batch)
    {
        const float *src = operation.input_buffers[batch];
        float *dst = operation.output_buffers[batch];
        const size_t count = operation.batch_sizes[batch];

        if (src == nullptr)
        {
            return ErrorCode::IO_READ_ERROR;
        }
        if (dst == nullptr)
        {
            return ErrorCode::IO_WRITE_ERROR;
        }
        if (multiplyWouldOverflow(count, sizeof(float)))
        {
            return ErrorCode::IO_READ_ERROR;
        }
        ErrorCode finite_error = ErrorCode::SUCCESS;
        if (!finiteFloatValues(src, count, &finite_error))
        {
            return finite_error;
        }
        if (count > 0)
        {
            std::memcpy(dst, src, count * sizeof(float));
        }

        if (contract.strict_time_enforcement && exceedsBudgetMs(start, contract.time_budget_ms))
        {
            return ErrorCode::PH_TIME_BUDGET_EXCEEDED;
        }
    }
    return ErrorCode::SUCCESS;
}

ErrorCode AcceleratedGpuPrimitives::executeOneKernelPipeline(
    const float *points_a, const float *points_b, float *distances, float *image, size_t batch_size,
    size_t num_points_a, size_t num_points_b, const CallContract &contract)
{
    if (points_a == nullptr || points_b == nullptr || distances == nullptr || batch_size == 0 ||
        num_points_a == 0 || num_points_b == 0)
    {
        return ErrorCode::IO_READ_ERROR;
    }
    if (multiplyWouldOverflow(num_points_a, batch_size) ||
        multiplyWouldOverflow(num_points_b, batch_size))
    {
        return ErrorCode::IO_READ_ERROR;
    }
    if (multiplyWouldOverflow(num_points_a, num_points_b))
    {
        return ErrorCode::IO_WRITE_ERROR;
    }
    const size_t points_a_values = num_points_a * batch_size;
    const size_t points_b_values = num_points_b * batch_size;
    ErrorCode finite_error = ErrorCode::SUCCESS;
    if (!finiteFloatValues(points_a, points_a_values, &finite_error) ||
        !finiteFloatValues(points_b, points_b_values, &finite_error))
    {
        return finite_error;
    }

    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < num_points_a; ++i)
    {
        for (size_t j = 0; j < num_points_b; ++j)
        {
            float dist = 0.0f;
            if (!checkedFiniteDistance(&points_a[i * batch_size], &points_b[j * batch_size],
                                       batch_size, dist, &finite_error))
            {
                return finite_error;
            }
            distances[i * num_points_b + j] = dist;
            if (image != nullptr)
            {
                image[i * num_points_b + j] = dist;
            }
        }
        if (contract.strict_time_enforcement && exceedsBudgetMs(start, contract.time_budget_ms))
        {
            return ErrorCode::PH_TIME_BUDGET_EXCEEDED;
        }
    }
    return ErrorCode::SUCCESS;
}

ErrorCode AcceleratedGpuPrimitives::executeWarpReduction(const float *input, float *output,
                                                         size_t size, const CallContract &contract)
{
    if (input == nullptr)
    {
        return ErrorCode::IO_READ_ERROR;
    }
    if (output == nullptr)
    {
        return ErrorCode::IO_WRITE_ERROR;
    }
    const auto start = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i)
    {
        const double value = static_cast<double>(input[i]);
        if (!std::isfinite(value))
        {
            return finiteErrorCode(value);
        }
        const double next = sum + value;
        if (!std::isfinite(next))
        {
            return ErrorCode::NUM_INF;
        }
        sum = next;
        if (contract.strict_time_enforcement && (i % 1024U == 0) &&
            exceedsBudgetMs(start, contract.time_budget_ms))
        {
            return ErrorCode::PH_TIME_BUDGET_EXCEEDED;
        }
    }
    float narrowed_sum = 0.0f;
    if (!checkedFloatResult(sum, narrowed_sum))
    {
        return ErrorCode::NUM_INF;
    }
    output[0] = narrowed_sum;
    return ErrorCode::SUCCESS;
}

bool AcceleratedGpuPrimitives::validatePerformance() const
{
    if (config_.num_streams <= 0)
    {
        return false;
    }
    if (config_.min_batch_size == 0 || config_.optimal_batch_size < config_.min_batch_size ||
        config_.max_batch_size < config_.optimal_batch_size)
    {
        return false;
    }
    if (config_.enable_mixed_precision &&
        (!std::isfinite(config_.mixed_precision_error_threshold) ||
         config_.mixed_precision_error_threshold <= 0.0))
    {
        return false;
    }
    return true;
}

size_t AcceleratedGpuPrimitives::getPeakMemoryUsage() const
{
    size_t stream_bytes = 0;
    if (config_.num_streams > 0 && !checkedByteCount(static_cast<size_t>(config_.num_streams),
                                                     sizeof(cudaStream_t), stream_bytes))
    {
        return std::numeric_limits<size_t>::max();
    }
    size_t pinned_pool_bytes = 0;
    if (config_.use_pinned_memory)
    {
        if (multiplyWouldOverflow(config_.max_batch_size, 2U) ||
            !checkedByteCount(config_.max_batch_size * 2U, sizeof(float), pinned_pool_bytes))
        {
            return std::numeric_limits<size_t>::max();
        }
    }
    if (stream_bytes > std::numeric_limits<size_t>::max() - pinned_pool_bytes)
    {
        return std::numeric_limits<size_t>::max();
    }
    return stream_bytes + pinned_pool_bytes;
}

void AcceleratedGpuPrimitives::setupAsyncOperations()
{
#if defined(__CUDACC__)
    if (stream_pool_ == nullptr && config_.num_streams > 0)
    {
        stream_pool_ = createStreamPool(config_.num_streams);
    }
#endif
}

void AcceleratedGpuPrimitives::cleanupAsyncOperations()
{
#if defined(__CUDACC__)
    if (stream_pool_ != nullptr)
    {
        destroyStreamPool(stream_pool_, config_.num_streams);
        stream_pool_ = nullptr;
    }
#endif
}

bool AcceleratedGpuPrimitives::validateMixedPrecision(const float *double_result,
                                                      const __half *fp16_result, size_t size)
{
    if (size == 0)
    {
        return true;
    }
    if (double_result == nullptr || fp16_result == nullptr)
    {
        return false;
    }

#if defined(__CUDACC__)
    const float threshold = static_cast<float>(config_.mixed_precision_error_threshold);
    if (!std::isfinite(threshold) || threshold < 0.0f)
    {
        return false;
    }
    for (size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(double_result[i]))
        {
            return false;
        }
        const float fp16_value = __half2float(fp16_result[i]);
        if (!std::isfinite(fp16_value))
        {
            return false;
        }
        const float abs_error = std::abs(double_result[i] - fp16_value);
        if (!std::isfinite(abs_error) || abs_error > threshold)
        {
            return false;
        }
    }
    return true;
#else
    return false;
#endif
}

void AcceleratedGpuPrimitives::measureKernelLatency()
{
    constexpr size_t kProbeSize = 32;
    float input[kProbeSize]{};
    float output[1]{};
    CallContract contract{};
    contract.strict_time_enforcement = false;
    if (executeWarpReduction(input, output, kProbeSize, contract) != ErrorCode::SUCCESS)
    {
        return;
    }
}

void AcceleratedGpuPrimitives::measureTransferLatency()
{
    constexpr size_t kProbeSize = 128;
    float src[kProbeSize]{};
    float dst[kProbeSize]{};
    std::memcpy(dst, src, sizeof(src));
}

} // namespace nerve::optimization
