#include "nerve/gpu/gpu_error.hpp"
#include "nerve/runtime/gpu_calibration.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve
{
namespace runtime
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;

// clang-format off
#include "calibration_gpu_kernels.inl"
#include "calibration_gpu_model.inl"
#include "calibration_gpu_benchmark.inl"
// clang-format on

} // namespace gpu
} // namespace runtime
} // namespace nerve
