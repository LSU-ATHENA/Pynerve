#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve
{
namespace specialized
{
namespace gpu
{

constexpr int MAX_COHOM_DIM = 4;
constexpr int CUP_PRODUCT_BLOCK_SIZE = 256;

inline void checkCuda(cudaError_t status, const char *context)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

inline size_t checkedMulSize(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        throw std::length_error(std::string(label) + " exceeds size_t limits");
    }
    return a * b;
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
    const size_t blocks = (count + static_cast<size_t>(CUP_PRODUCT_BLOCK_SIZE) - 1) /
                          static_cast<size_t>(CUP_PRODUCT_BLOCK_SIZE);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA grid range");
    }
    return static_cast<int>(blocks);
}

template <typename T>
inline void allocateDevice(T **ptr, size_t count, const char *label)
{
    *ptr = nullptr;
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMalloc(reinterpret_cast<void **>(ptr), checkedMulSize(count, sizeof(T), label)),
              label);
}

template <typename T>
inline void copyToDevice(T *dst, const T *src, size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, checkedMulSize(count, sizeof(T), label), cudaMemcpyHostToDevice),
              label);
}

template <typename T>
inline void copyToHost(T *dst, const T *src, size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, checkedMulSize(count, sizeof(T), label), cudaMemcpyDeviceToHost),
              label);
}

struct Cochain
{
    int dimension;
    std::vector<int> simplex_indices;
    std::vector<float> coefficients;
    int num_simplices = 0;
};

// clang-format off
#include "cup_product_kernels.inl"
#include "cup_product_computer.inl"
#include "cup_product_benchmark.inl"
// clang-format on

} // namespace gpu
} // namespace specialized
} // namespace nerve
