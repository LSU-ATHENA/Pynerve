
#pragma once
#include "nerve/core_types.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(NERVE_CUDA_FP16_AVAILABLE) && !NERVE_CUDA_FP16_AVAILABLE
using half = __half;
#endif

#if defined(__has_include)
#if __has_include(<cuda_bf16.h>)
#include <cuda_bf16.h>
#else
struct __nv_bfloat16
{
    unsigned short x;
};
#endif
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#else
struct __nv_fp8_e4m3
{
    unsigned char x;
};
#endif
#else
struct __nv_bfloat16
{
    unsigned short x;
};
struct __nv_fp8_e4m3
{
    unsigned char x;
};
#endif

#include <concepts>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace nerve
{
namespace sheaf
{
namespace gpu
{
namespace tensorcore
{

inline constexpr int kWmmaM = 16;
inline constexpr int kWmmaN = 16;
inline constexpr int kWmmaK = 16;

template <typename T>
concept WMMACompatible =
    std::same_as<T, half> || std::same_as<T, float> || std::same_as<T, __nv_bfloat16>;

class TensorCoreSheafLaplacian
{
public:
    explicit TensorCoreSheafLaplacian(int num_gpus = 1);
    ~TensorCoreSheafLaplacian();

    [[nodiscard]] bool initialize();

    template <WMMACompatible InputPrecision>
    [[nodiscard]] bool compute(std::span<const InputPrecision> stalk_matrix,
                               std::span<const InputPrecision> restriction_matrix,
                               std::span<float> output_laplacian, int num_stalks,
                               int num_restrictions);

    [[nodiscard]] bool computeBlackwellTMA(std::span<const __nv_fp8_e4m3> stalk_data,
                                           std::span<const __nv_fp8_e4m3> restriction_maps,
                                           std::span<float> output_laplacian, int num_stalks);

    [[nodiscard]] double getLastComputeTime() const noexcept;
    [[nodiscard]] double getEffectiveTFLOPS() const noexcept;
    [[nodiscard]] bool hasTensorCoreSupport() const noexcept;

private:
    int num_gpus_;
    double last_compute_time_ms_ = 0.0;
    int last_num_stalks_ = 0;
    int last_num_restrictions_ = 0;
    bool initialized_ = false;
};

struct TensorCoreConfig
{
    bool use_fp16 = true;
    bool use_tf32 = false;
    bool use_fp8 = false;
    bool use_tma = false;
    int block_size = 256;
};

#ifndef __CUDACC__
inline TensorCoreSheafLaplacian::TensorCoreSheafLaplacian(int num_gpus)
    : num_gpus_(num_gpus)
{
    if (num_gpus <= 0)
    {
        throw std::invalid_argument("num_gpus must be positive");
    }
}

inline TensorCoreSheafLaplacian::~TensorCoreSheafLaplacian() = default;

inline bool TensorCoreSheafLaplacian::initialize()
{
    initialized_ = false;
    return false;
}

template <WMMACompatible InputPrecision>
inline bool TensorCoreSheafLaplacian::compute(std::span<const InputPrecision> stalk_matrix,
                                              std::span<const InputPrecision> restriction_matrix,
                                              std::span<float> output_laplacian, int num_stalks,
                                              int num_restrictions)
{
    if (num_stalks <= 0 || num_restrictions <= 0)
    {
        throw std::invalid_argument("num_stalks and num_restrictions must be positive");
    }
    const size_t n = static_cast<size_t>(num_stalks);
    const size_t r = static_cast<size_t>(num_restrictions);
    if (stalk_matrix.size() < n * n || restriction_matrix.size() < r * n ||
        output_laplacian.size() < n * n)
    {
        throw std::invalid_argument("TensorCoreSheafLaplacian buffers are too small");
    }
    return false;
}

inline bool
TensorCoreSheafLaplacian::computeBlackwellTMA(std::span<const __nv_fp8_e4m3> stalk_data,
                                              std::span<const __nv_fp8_e4m3> restriction_maps,
                                              std::span<float> output_laplacian, int num_stalks)
{
    if (num_stalks <= 0)
    {
        throw std::invalid_argument("num_stalks must be positive");
    }
    const size_t n = static_cast<size_t>(num_stalks);
    if (stalk_data.size() < n || restriction_maps.size() < n * n || output_laplacian.size() < n)
    {
        throw std::invalid_argument("FP8 sheaf buffers are too small");
    }
    return false;
}

inline double TensorCoreSheafLaplacian::getLastComputeTime() const noexcept
{
    return last_compute_time_ms_;
}

inline double TensorCoreSheafLaplacian::getEffectiveTFLOPS() const noexcept
{
    if (last_compute_time_ms_ <= 0.0 || last_num_stalks_ <= 0 || last_num_restrictions_ <= 0)
    {
        return 0.0;
    }
    const double n = static_cast<double>(last_num_stalks_);
    const double r = static_cast<double>(last_num_restrictions_);
    return (2.0 * n * n * r) / (last_compute_time_ms_ * 1.0e9);
}

inline bool TensorCoreSheafLaplacian::hasTensorCoreSupport() const noexcept
{
    return false;
}
#endif

} // namespace tensorcore
} // namespace gpu
} // namespace sheaf
} // namespace nerve
