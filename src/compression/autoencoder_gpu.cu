#include "nerve/compression/gpu_autoencoder.hpp"
#include "nerve/gpu/gpu_error.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve
{
namespace compression
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;

// clang-format off
#include "autoencoder_gpu_kernels.inl"
#include "autoencoder_gpu_api.inl"
#include "autoencoder_gpu_benchmark.inl"
// clang-format on

} // namespace gpu
} // namespace compression
} // namespace nerve
