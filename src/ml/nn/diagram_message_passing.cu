#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve
{
namespace ml
{
namespace nn
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;
constexpr float EPSILON = 1e-8f;
constexpr int DIAGRAM_TILE_DIM = 16;
constexpr int DIAGRAM_TILE_MASK = DIAGRAM_TILE_DIM - 1;
constexpr int DIAGRAM_FIELDS_PER_PAIR = 3;
constexpr float DIFFERENT_DIMENSION_DISTANCE = 1000.0f;
constexpr float TOPOLOGICAL_BIAS_PENALTY = 0.1f;
constexpr float RBF_KERNEL_DENOMINATOR = 2.0f;
constexpr float DEFAULT_BANDWIDTH = 0.1f;

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
            throw std::invalid_argument(std::string(label) + " must be finite");
        }
    }
}

inline void requireFiniteDiagramValues(const std::vector<float> &diagrams, const char *label)
{
    for (size_t pair = 0; pair * DIAGRAM_FIELDS_PER_PAIR < diagrams.size(); ++pair)
    {
        const size_t base = pair * DIAGRAM_FIELDS_PER_PAIR;
        const float birth = diagrams[base];
        const float death = diagrams[base + 1];
        const float dim = diagrams[base + 2];
        if (!std::isfinite(birth) || !std::isfinite(death) || !std::isfinite(dim) ||
            death < birth || dim < 0.0f || std::floor(dim) != dim)
        {
            throw std::invalid_argument(std::string(label) + " must contain finite valid pairs");
        }
    }
}

inline void validateDeviceFinite(const float *d_values, size_t count, const char *label)
{
    std::vector<float> values(count);
    copyToHost(values.data(), d_values, values.size(), label);
    requireFiniteValues(values, label);
}

// clang-format off
#include "diagram_message_passing_kernels.inl"
#include "diagram_message_passing_layers.inl"
#include "diagram_message_passing_benchmark.inl"
// clang-format on

} // namespace gpu
} // namespace nn
} // namespace ml
} // namespace nerve
