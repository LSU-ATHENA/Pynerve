#include "nerve/gpu/gpu_error.hpp"
#include "nerve/regularization/gpu_regularization.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve
{
namespace regularization
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr float EPSILON = 1e-6f;
constexpr unsigned int FULL_WARP_MASK = 0xFFFFFFFF;

// clang-format off
#include "loss_kernels_device.inl"
#include "loss_kernels_api.inl"
#include "loss_kernels_benchmark.inl"
// clang-format on

} // namespace gpu
} // namespace regularization
} // namespace nerve
