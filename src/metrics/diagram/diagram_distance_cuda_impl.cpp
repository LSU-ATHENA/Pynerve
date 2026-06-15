
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/compute_manager.hpp"
#include "nerve/metrics/distances.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace nerve::gpu::detail
{

// Structure matching the GPU kernel definition
struct GPUPersistencePair
{
    double birth;
    double death;
    bool isInfinite;
};

namespace
{

bool checkedProduct(size_t lhs, size_t rhs, size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedByteCount(size_t count, size_t element_size, size_t &out) noexcept
{
    return checkedProduct(count, element_size, out);
}

errors::ErrorResult<void> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

errors::ErrorResult<void> numericError(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN, message);
}

bool isValidDiagramPair(const persistence::Pair &pair)
{
    const bool valid_death =
        std::isfinite(pair.death) || pair.death == std::numeric_limits<double>::infinity();
    return std::isfinite(pair.birth) && valid_death && pair.dimension >= 0;
}

bool finiteLifetime(const persistence::Pair &pair, double &lifetime)
{
    if (pair.isInfinite())
    {
        lifetime = std::numeric_limits<double>::infinity();
        return true;
    }
    lifetime = std::abs(pair.death - pair.birth);
    return std::isfinite(lifetime);
}

void cleanupCostMatrixGpu(GPUPersistencePair *d_pairs1, GPUPersistencePair *d_pairs2,
                          double *d_cost_matrix) noexcept
{
    cudaFree(d_pairs1);
    cudaFree(d_pairs2);
    cudaFree(d_cost_matrix);
}

} // namespace

// Forward declaration of CUDA kernel launcher
extern void launchDiagramCostMatrixKernel(const GPUPersistencePair *d_pairs1,
                                          const GPUPersistencePair *d_pairs2, double *d_cost_matrix,
                                          size_t n1, size_t n2, size_t n, double large_penalty,
                                          cudaStream_t stream);

// GPU-accelerated cost matrix computation
errors::ErrorResult<void> computeCostMatrixGpu(const std::vector<persistence::Pair> &pairs1,
                                               const std::vector<persistence::Pair> &pairs2,
                                               std::vector<std::vector<double>> &out_cost_matrix)
{
    size_t n1 = pairs1.size();
    size_t n2 = pairs2.size();
    if (n2 > std::numeric_limits<size_t>::max() - n1)
    {
        out_cost_matrix.clear();
        return resourceLimit("diagram cost matrix size overflows");
    }
    size_t n = n1 + n2;
    size_t matrix_entries = 0;
    size_t matrix_bytes = 0;
    size_t pairs1_bytes = 0;
    size_t pairs2_bytes = 0;
    if (!checkedProduct(n, n, matrix_entries) ||
        !checkedByteCount(matrix_entries, sizeof(double), matrix_bytes) ||
        !checkedByteCount(n1, sizeof(GPUPersistencePair), pairs1_bytes) ||
        !checkedByteCount(n2, sizeof(GPUPersistencePair), pairs2_bytes))
    {
        out_cost_matrix.clear();
        return resourceLimit("diagram cost matrix byte count overflows");
    }

    if (n == 0)
    {
        out_cost_matrix.clear();
        return errors::ErrorResult<void>::success();
    }

    // Convert pairs to GPU format
    std::vector<GPUPersistencePair> gpuPairs1(n1);
    std::vector<GPUPersistencePair> gpuPairs2(n2);

    double max_finite_cost = 1.0;

    for (size_t i = 0; i < n1; ++i)
    {
        if (!isValidDiagramPair(pairs1[i]))
        {
            out_cost_matrix.clear();
            return numericError("diagram GPU cost pair values must be finite");
        }
        gpuPairs1[i].birth = pairs1[i].birth;
        gpuPairs1[i].death = pairs1[i].death;
        gpuPairs1[i].isInfinite = pairs1[i].isInfinite();
        if (!pairs1[i].isInfinite())
        {
            double lifetime = 0.0;
            if (!finiteLifetime(pairs1[i], lifetime))
            {
                out_cost_matrix.clear();
                return numericError("diagram GPU cost pair lifetime overflow");
            }
            max_finite_cost = std::max(max_finite_cost, lifetime);
        }
    }

    for (size_t i = 0; i < n2; ++i)
    {
        if (!isValidDiagramPair(pairs2[i]))
        {
            out_cost_matrix.clear();
            return numericError("diagram GPU cost pair values must be finite");
        }
        gpuPairs2[i].birth = pairs2[i].birth;
        gpuPairs2[i].death = pairs2[i].death;
        gpuPairs2[i].isInfinite = pairs2[i].isInfinite();
        if (!pairs2[i].isInfinite())
        {
            double lifetime = 0.0;
            if (!finiteLifetime(pairs2[i], lifetime))
            {
                out_cost_matrix.clear();
                return numericError("diagram GPU cost pair lifetime overflow");
            }
            max_finite_cost = std::max(max_finite_cost, lifetime);
        }
    }

    double large_penalty = max_finite_cost * (static_cast<double>(n) + 1.0) + 1.0;
    if (!std::isfinite(large_penalty))
    {
        out_cost_matrix.clear();
        return numericError("diagram GPU cost penalty overflow");
    }

    // Allocate GPU memory
    GPUPersistencePair *d_pairs1 = nullptr;
    GPUPersistencePair *d_pairs2 = nullptr;
    double *d_cost_matrix = nullptr;

    cudaError_t err;

    if (pairs1_bytes != 0)
    {
        err = cudaMalloc(&d_pairs1, pairs1_bytes);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }
    }

    if (pairs2_bytes != 0)
    {
        err = cudaMalloc(&d_pairs2, pairs2_bytes);
        if (err != cudaSuccess)
        {
            cleanupCostMatrixGpu(d_pairs1, d_pairs2, d_cost_matrix);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }
    }

    err = cudaMalloc(&d_cost_matrix, matrix_bytes);
    if (err != cudaSuccess)
    {
        cleanupCostMatrixGpu(d_pairs1, d_pairs2, d_cost_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    // Copy data to GPU
    if (pairs1_bytes != 0)
    {
        err = cudaMemcpy(d_pairs1, gpuPairs1.data(), pairs1_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupCostMatrixGpu(d_pairs1, d_pairs2, d_cost_matrix);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
    }

    if (pairs2_bytes != 0)
    {
        err = cudaMemcpy(d_pairs2, gpuPairs2.data(), pairs2_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupCostMatrixGpu(d_pairs1, d_pairs2, d_cost_matrix);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
    }

    // Launch kernel
    launchDiagramCostMatrixKernel(d_pairs1, d_pairs2, d_cost_matrix, n1, n2, n, large_penalty,
                                  nullptr);

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cleanupCostMatrixGpu(d_pairs1, d_pairs2, d_cost_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    // Copy result back
    std::vector<double> flatCostMatrix;
    try
    {
        flatCostMatrix.resize(matrix_entries);
    }
    catch (const std::bad_alloc &)
    {
        cleanupCostMatrixGpu(d_pairs1, d_pairs2, d_cost_matrix);
        out_cost_matrix.clear();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM,
                                                "diagram cost matrix host allocation failed");
    }
    err = cudaMemcpy(flatCostMatrix.data(), d_cost_matrix, matrix_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cleanupCostMatrixGpu(d_pairs1, d_pairs2, d_cost_matrix);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    // Cleanup
    cleanupCostMatrixGpu(d_pairs1, d_pairs2, d_cost_matrix);

    // Convert flat matrix to 2D
    try
    {
        out_cost_matrix.assign(n, std::vector<double>(n));
    }
    catch (const std::bad_alloc &)
    {
        out_cost_matrix.clear();
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM,
                                                "diagram cost matrix output allocation failed");
    }
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            const double value = flatCostMatrix[i * n + j];
            if (!std::isfinite(value))
            {
                out_cost_matrix.clear();
                return numericError("diagram GPU cost matrix contains non-finite values");
            }
            out_cost_matrix[i][j] = value;
        }
    }

    return errors::ErrorResult<void>::success();
}

// CPU proxy for cost matrix computation
void computeCostMatrixCpu(const std::vector<persistence::Pair> &pairs1,
                          const std::vector<persistence::Pair> &pairs2,
                          std::vector<std::vector<double>> &out_cost_matrix)
{
    size_t n1 = pairs1.size();
    size_t n2 = pairs2.size();
    if (n2 > std::numeric_limits<size_t>::max() - n1)
    {
        throw std::length_error("diagram cost matrix size overflows");
    }
    size_t n = n1 + n2;
    size_t matrix_entries = 0;
    if (!checkedProduct(n, n, matrix_entries))
    {
        throw std::length_error("diagram cost matrix area overflows");
    }
    (void)matrix_entries;

    out_cost_matrix.assign(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));

    auto computePairDistance = [](const persistence::Pair &p1,
                                  const persistence::Pair &p2) -> double {
        if (p1.isInfinite() && p2.isInfinite())
        {
            const double cost = std::abs(p1.birth - p2.birth);
            return std::isfinite(cost) ? cost : std::numeric_limits<double>::infinity();
        }
        if (p1.isInfinite() != p2.isInfinite())
        {
            return std::numeric_limits<double>::infinity();
        }
        const double birth_cost = std::abs(p1.birth - p2.birth);
        const double death_cost = std::abs(p1.death - p2.death);
        const double cost = std::max(birth_cost, death_cost);
        return std::isfinite(cost) ? cost : std::numeric_limits<double>::infinity();
    };

    auto computeDiagonalCost = [](const persistence::Pair &p) -> double {
        if (p.isInfinite())
        {
            return std::numeric_limits<double>::infinity();
        }
        const double cost = std::abs(p.death - p.birth) * 0.5;
        return std::isfinite(cost) ? cost : std::numeric_limits<double>::infinity();
    };

    double max_cost = 1.0;

    // Fill pair-to-pair block
    for (size_t i = 0; i < n1; ++i)
    {
        for (size_t j = 0; j < n2; ++j)
        {
            double cost = computePairDistance(pairs1[i], pairs2[j]);
            out_cost_matrix[i][j] = cost;
            if (std::isfinite(cost))
            {
                max_cost = std::max(max_cost, cost);
            }
        }
    }

    // Fill diagonal for pairs1
    for (size_t i = 0; i < n1; ++i)
    {
        double cost = computeDiagonalCost(pairs1[i]);
        out_cost_matrix[i][n2 + i] = cost;
        if (std::isfinite(cost))
        {
            max_cost = std::max(max_cost, cost);
        }
    }

    // Fill diagonal for pairs2
    for (size_t j = 0; j < n2; ++j)
    {
        double cost = computeDiagonalCost(pairs2[j]);
        out_cost_matrix[n1 + j][j] = cost;
        if (std::isfinite(cost))
        {
            max_cost = std::max(max_cost, cost);
        }
    }

    // Fill auxiliary-to-auxiliary block
    for (size_t i = n1; i < n; ++i)
    {
        for (size_t j = n2; j < n; ++j)
        {
            out_cost_matrix[i][j] = 0.0;
        }
    }

    // Replace infinity with large penalty
    double large_penalty = max_cost * (static_cast<double>(n) + 1.0) + 1.0;
    if (!std::isfinite(large_penalty))
    {
        throw std::overflow_error("diagram cost matrix penalty overflows");
    }
    for (auto &row : out_cost_matrix)
    {
        for (double &val : row)
        {
            if (!std::isfinite(val))
            {
                val = large_penalty;
            }
        }
    }
}

} // namespace nerve::gpu::detail
