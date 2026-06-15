#include "nerve/encoders/gpu_encoders.hpp"
#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve
{
namespace encoders
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;
constexpr int FEATURE_EXTRACTION_BLOCK_SIZE = 128;
constexpr int GRID_ROUNDING_OFFSET = 255;
constexpr int GRID_DIVISOR = 256;

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

inline void requireFiniteValues(const std::vector<float> &values, const char *label)
{
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error(std::string(label) + " contains non-finite values");
        }
    }
}

inline void requireValidDiagramValues(const std::vector<float> &diagrams, const char *label)
{
    for (size_t i = 0; i + 1 < diagrams.size(); i += 2U)
    {
        const float birth = diagrams[i];
        const float death = diagrams[i + 1U];
        const bool finite_death = std::isfinite(death);
        const bool open_death = std::isinf(death) && death > 0.0f;
        if (!std::isfinite(birth) || (!finite_death && !open_death) ||
            (finite_death && death < birth))
        {
            throw std::invalid_argument(std::string(label) + " contains invalid persistence pairs");
        }
    }
}

inline void validateDeviceFinite(const float *d_values, size_t count, const char *label)
{
    std::vector<float> values(count);
    copyToHost(values.data(), d_values, values.size(), label);
    requireFiniteValues(values, label);
}

float deterministicEncoderWeight(int layer_idx, int in_idx, int out_idx)
{
    const int mixed = (layer_idx + 1) * 131 + in_idx * 17 + out_idx * 29;
    return (static_cast<float>(mixed % 23) - 11.0f) * 0.01f;
}

float deterministicEncoderBias(int layer_idx, int out_idx)
{
    const int mixed = (layer_idx + 1) * 43 + out_idx * 7;
    return (static_cast<float>(mixed % 11) - 5.0f) * 0.001f;
}

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

// clang-format off
#include "encoder_gpu_kernels_device.inl"
#include "encoder_gpu_kernels_api.inl"
#include "encoder_gpu_kernels_benchmark.inl"
// clang-format on

} // namespace gpu
} // namespace encoders
} // namespace nerve
