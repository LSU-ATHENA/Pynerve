#pragma once

#include "nerve/optimization/component_optimizations.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace nerve::optimization
{

bool exceedsBudgetMs(const std::chrono::steady_clock::time_point &start, double budget_ms);

bool multiplyWouldOverflow(size_t lhs, size_t rhs);

bool checkedByteCount(size_t count, size_t element_size, size_t &bytes);

ErrorCode finiteErrorCode(double value);

bool finiteFloatValues(const float *values, size_t count, ErrorCode *error_code = nullptr);

bool checkedFloatResult(double value, float &result, ErrorCode *error_code = nullptr);

bool checkedFiniteDistance(const float *lhs, const float *rhs, size_t dimension, float &distance,
                           ErrorCode *error_code = nullptr);

bool checkedFiniteDotProduct(const float *matrix_row, const float *vector, size_t cols,
                             float &result, ErrorCode *error_code = nullptr);

#if defined(__CUDACC__)
bool checkedGridBlocks(size_t work_items, size_t block_size, int &blocks);

cudaStream_t *createStreamPool(int stream_count);

void destroyStreamPool(cudaStream_t *streams, int stream_count);
#endif

} // namespace nerve::optimization
