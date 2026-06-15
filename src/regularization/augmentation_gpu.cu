#include "nerve/gpu/gpu_error.hpp"
#include "nerve/regularization/gpu_regularization.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

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
namespace regularization
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;

// clang-format off
#include "augmentation_gpu_kernels.inl"
#include "augmentation_gpu_engine.inl"
#include "augmentation_gpu_benchmark.inl"
// clang-format on

} // namespace gpu
} // namespace regularization
} // namespace nerve
