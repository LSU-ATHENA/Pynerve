#include "nerve/gpu/gpu_error.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nerve
{
namespace spectral
{
namespace gpu
{

constexpr int EIGENSOLVER_BLOCK_SIZE = 256;

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
#include "eigensolver_gpu_core.inl"
#include "eigensolver_gpu_extensions.inl"
#include "eigensolver_gpu_benchmark.inl"
// clang-format on

} // namespace gpu
} // namespace spectral
} // namespace nerve
