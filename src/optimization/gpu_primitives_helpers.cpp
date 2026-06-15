#include "nerve/optimization/gpu_primitives_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace nerve::optimization
{

bool exceedsBudgetMs(const std::chrono::steady_clock::time_point &start, double budget_ms)
{
    if (budget_ms <= 0.0)
    {
        return false;
    }
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return elapsed > budget_ms;
}

bool multiplyWouldOverflow(size_t lhs, size_t rhs)
{
    return lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs;
}

bool checkedByteCount(size_t count, size_t element_size, size_t &bytes)
{
    if (multiplyWouldOverflow(count, element_size))
    {
        return false;
    }
    bytes = count * element_size;
    return true;
}

ErrorCode finiteErrorCode(double value)
{
    return std::isnan(value) ? ErrorCode::NUM_NAN : ErrorCode::NUM_INF;
}

bool finiteFloatValues(const float *values, size_t count, ErrorCode *error_code)
{
    if (count == 0)
    {
        return true;
    }
    if (values == nullptr)
    {
        if (error_code != nullptr)
        {
            *error_code = ErrorCode::IO_READ_ERROR;
        }
        return false;
    }
    for (size_t i = 0; i < count; ++i)
    {
        const double value = static_cast<double>(values[i]);
        if (!std::isfinite(value))
        {
            if (error_code != nullptr)
            {
                *error_code = finiteErrorCode(value);
            }
            return false;
        }
    }
    return true;
}

bool checkedFloatResult(double value, float &result, ErrorCode *error_code)
{
    if (!std::isfinite(value))
    {
        if (error_code != nullptr)
        {
            *error_code = finiteErrorCode(value);
        }
        return false;
    }
    if (std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
    {
        if (error_code != nullptr)
        {
            *error_code = ErrorCode::NUM_INF;
        }
        return false;
    }
    result = static_cast<float>(value);
    if (!std::isfinite(result))
    {
        if (error_code != nullptr)
        {
            *error_code = finiteErrorCode(static_cast<double>(result));
        }
        return false;
    }
    return true;
}

bool checkedFiniteDistance(const float *lhs, const float *rhs, size_t dimension, float &distance,
                           ErrorCode *error_code)
{
    double dist_sq = 0.0;
    for (size_t d = 0; d < dimension; ++d)
    {
        const double lhs_value = static_cast<double>(lhs[d]);
        const double rhs_value = static_cast<double>(rhs[d]);
        if (!std::isfinite(lhs_value))
        {
            if (error_code != nullptr)
            {
                *error_code = finiteErrorCode(lhs_value);
            }
            return false;
        }
        if (!std::isfinite(rhs_value))
        {
            if (error_code != nullptr)
            {
                *error_code = finiteErrorCode(rhs_value);
            }
            return false;
        }
        const double diff = lhs_value - rhs_value;
        const double contribution = diff * diff;
        const double next = dist_sq + contribution;
        if (!std::isfinite(diff) || !std::isfinite(contribution) || !std::isfinite(next))
        {
            if (error_code != nullptr)
            {
                *error_code = ErrorCode::NUM_INF;
            }
            return false;
        }
        dist_sq = next;
    }
    return checkedFloatResult(std::sqrt(dist_sq), distance, error_code);
}

bool checkedFiniteDotProduct(const float *matrix_row, const float *vector, size_t cols,
                             float &result, ErrorCode *error_code)
{
    double sum = 0.0;
    for (size_t col = 0; col < cols; ++col)
    {
        const double matrix_value = static_cast<double>(matrix_row[col]);
        const double vector_value = static_cast<double>(vector[col]);
        if (!std::isfinite(matrix_value))
        {
            if (error_code != nullptr)
            {
                *error_code = finiteErrorCode(matrix_value);
            }
            return false;
        }
        if (!std::isfinite(vector_value))
        {
            if (error_code != nullptr)
            {
                *error_code = finiteErrorCode(vector_value);
            }
            return false;
        }
        const double product = matrix_value * vector_value;
        const double next = sum + product;
        if (!std::isfinite(product) || !std::isfinite(next))
        {
            if (error_code != nullptr)
            {
                *error_code = ErrorCode::NUM_INF;
            }
            return false;
        }
        sum = next;
    }
    return checkedFloatResult(sum, result, error_code);
}

#if defined(__CUDACC__)

bool checkedGridBlocks(size_t work_items, size_t block_size, int &blocks)
{
    if (block_size == 0)
    {
        return false;
    }
    const size_t block_count =
        (work_items / block_size) + ((work_items % block_size) != 0 ? 1U : 0U);
    if (block_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    blocks = static_cast<int>(block_count);
    return true;
}

cudaStream_t *createStreamPool(int stream_count)
{
    if (stream_count <= 0)
    {
        return nullptr;
    }
    cudaStream_t *streams = new cudaStream_t[stream_count]{};
    for (int i = 0; i < stream_count; ++i)
    {
        if (cudaStreamCreate(&streams[i]) != cudaSuccess)
        {
            for (int j = 0; j < i; ++j)
            {
                cudaStreamDestroy(streams[j]);
            }
            delete[] streams;
            return nullptr;
        }
    }
    return streams;
}

void destroyStreamPool(cudaStream_t *streams, int stream_count)
{
    if (streams == nullptr)
    {
        return;
    }
    for (int i = 0; i < stream_count; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

#endif // defined(__CUDACC__)

} // namespace nerve::optimization
