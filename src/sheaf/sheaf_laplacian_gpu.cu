#include "nerve/gpu/gpu_error.hpp"
#include "nerve/sheaf/gpu_sheaf.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cg = cooperative_groups;

namespace nerve
{
namespace sheaf
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;
constexpr int SHEAF_MAX_ITERATIONS = 100;
constexpr float SHEAF_CONVERGENCE_TOLERANCE = 1e-6f;

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

inline void checkCuda(cudaError_t status, const char *context)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

inline void checkCusparse(cusparseStatus_t status, const char *context)
{
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(context) + ": cuSPARSE status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

inline size_t checkedBytes(size_t count, size_t element_size, const char *label)
{
    if (count != 0 && element_size > std::numeric_limits<size_t>::max() / count)
    {
        throw std::length_error(std::string(label) + " exceeds size_t limits");
    }
    return count * element_size;
}

inline int checkedIntSize(size_t value, const char *label)
{
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA int range");
    }
    return static_cast<int>(value);
}

inline int checkedGridBlocks(size_t count, const char *label)
{
    if (count == 0)
    {
        return 0;
    }
    const size_t blocks =
        (count + static_cast<size_t>(BLOCK_SIZE) - 1) / static_cast<size_t>(BLOCK_SIZE);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA grid range");
    }
    return static_cast<int>(blocks);
}

inline bool valuesAreFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

inline bool valuesAreFinite(const float *values, size_t count)
{
    return std::all_of(values, values + count, [](float value) { return std::isfinite(value); });
}

inline void requireFiniteInput(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFinite(values))
    {
        throw std::invalid_argument(label);
    }
}

inline void requireFiniteInput(const float *values, size_t count, const char *label)
{
    if (count != 0 && (values == nullptr || !valuesAreFinite(values, count)))
    {
        throw std::invalid_argument(label);
    }
}

inline void requireFiniteOutput(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFinite(values))
    {
        throw std::runtime_error(label);
    }
}

template <typename T>
inline void allocateDevice(T **ptr, size_t count, const char *label)
{
    *ptr = nullptr;
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMalloc(reinterpret_cast<void **>(ptr), checkedBytes(count, sizeof(T), label)),
              label);
}

template <typename T>
inline void copyToDevice(T *dst, const T *src, size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, checkedBytes(count, sizeof(T), label), cudaMemcpyHostToDevice),
              label);
}

template <typename T>
inline void copyToHost(T *dst, const T *src, size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, checkedBytes(count, sizeof(T), label), cudaMemcpyDeviceToHost),
              label);
}

struct SheafStalk
{
    int id;
    int dimension;
    float *data;
};

struct RestrictionMap
{
    int from_stalk = 0;
    int to_stalk = 0;
    int *indices = nullptr;
    float *values = nullptr;
    int nnz = 0;
};

// clang-format off
#include "sheaf_laplacian_gpu_kernels.inl"
#include "sheaf_laplacian_gpu_solver.inl"
#include "sheaf_laplacian_gpu_multi.inl"
// clang-format on

} // namespace gpu
} // namespace sheaf
} // namespace nerve
