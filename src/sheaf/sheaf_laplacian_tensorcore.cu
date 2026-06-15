#include "nerve/sheaf/tensorcore_sheaf.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>

namespace nerve
{
namespace sheaf
{
namespace gpu
{
namespace tensorcore
{
namespace
{

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { reset(); }

    bool allocate(size_t count)
    {
        reset();
        if (count == 0)
        {
            return true;
        }
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
        {
            return false;
        }
        return cudaMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T)) == cudaSuccess;
    }

    void reset() noexcept
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    [[nodiscard]] T *get() const noexcept { return ptr_; }

private:
    T *ptr_ = nullptr;
};

template <typename T>
__device__ float asFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ float asFloat<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ float asFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename InputPrecision>
__global__ __launch_bounds__(256) void denseSheafLaplacianKernel(
    const InputPrecision *__restrict__ stalk_matrix,
    const InputPrecision *__restrict__ restriction_matrix, float *__restrict__ output_laplacian,
    int num_stalks, int num_restrictions)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_stalks * num_stalks;
    if (idx >= total)
    {
        return;
    }

    const int row = idx % num_stalks;
    const int col = idx / num_stalks;
    float value = asFloat(stalk_matrix[row + col * num_stalks]);

    for (int r = 0; r < num_restrictions; ++r)
    {
        const float left = asFloat(restriction_matrix[r + row * num_restrictions]);
        const float right = asFloat(restriction_matrix[r + col * num_restrictions]);
        value = fmaf(left, right, value);
    }

    output_laplacian[idx] = value;
}

__global__ __launch_bounds__(256) void fp8DiagonalSheafLaplacianKernel(
    const __nv_fp8_e4m3 *__restrict__ stalk_data,
    const __nv_fp8_e4m3 *__restrict__ restriction_maps, float *__restrict__ laplacian_output,
    int num_stalks)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_stalks)
    {
        return;
    }

    float stalk = static_cast<float>(stalk_data[tid]);
    float accum = stalk * stalk;
    for (int j = 0; j < num_stalks; ++j)
    {
        const int idx = tid * num_stalks + j;
        float r = static_cast<float>(restriction_maps[idx]);
        accum = fmaf(r, r, accum);
    }

    laplacian_output[tid] = accum;
}

[[nodiscard]] bool validCuda(cudaError_t err) noexcept
{
    return err == cudaSuccess;
}

size_t checkedMulSize(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        throw std::length_error(std::string(label) + " exceeds size_t limits");
    }
    return a * b;
}

template <typename T>
size_t checkedByteCount(size_t count, const char *label)
{
    return checkedMulSize(count, sizeof(T), label);
}

int checkedGridBlocks(size_t count, int block_size, const char *label)
{
    if (block_size <= 0)
    {
        throw std::invalid_argument("block size must be positive");
    }
    const size_t blocks =
        (count + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA grid range");
    }
    return static_cast<int>(blocks);
}

void validateDimensions(size_t stalk_size, size_t restriction_size, size_t output_size,
                        int num_stalks, int num_restrictions)
{
    if (num_stalks <= 0 || num_restrictions <= 0)
    {
        throw std::invalid_argument("num_stalks and num_restrictions must be positive");
    }
    const size_t n = static_cast<size_t>(num_stalks);
    const size_t r = static_cast<size_t>(num_restrictions);
    const size_t matrix_count = checkedMulSize(n, n, "sheaf tensor-core matrix size");
    const size_t restriction_count = checkedMulSize(r, n, "sheaf tensor-core restriction size");
    if (stalk_size < matrix_count)
    {
        throw std::invalid_argument("stalk_matrix must contain num_stalks * num_stalks values");
    }
    if (restriction_size < restriction_count)
    {
        throw std::invalid_argument(
            "restriction_matrix must contain num_restrictions * num_stalks values");
    }
    if (output_size < matrix_count)
    {
        throw std::invalid_argument("output_laplacian must contain num_stalks * num_stalks values");
    }
}

} // namespace

TensorCoreSheafLaplacian::TensorCoreSheafLaplacian(int num_gpus)
    : num_gpus_(num_gpus)
{
    if (num_gpus <= 0)
    {
        throw std::invalid_argument("num_gpus must be positive");
    }
}

TensorCoreSheafLaplacian::~TensorCoreSheafLaplacian() = default;

bool TensorCoreSheafLaplacian::initialize()
{
    int device_count = 0;
    if (!validCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0)
    {
        initialized_ = false;
        return false;
    }
    if (num_gpus_ > device_count)
    {
        initialized_ = false;
        return false;
    }
    initialized_ = validCuda(cudaSetDevice(0)) && validCuda(cudaFree(nullptr));
    return initialized_;
}

template <WMMACompatible InputPrecision>
bool TensorCoreSheafLaplacian::compute(std::span<const InputPrecision> stalk_matrix,
                                       std::span<const InputPrecision> restriction_matrix,
                                       std::span<float> output_laplacian, int num_stalks,
                                       int num_restrictions)
{
    validateDimensions(stalk_matrix.size(), restriction_matrix.size(), output_laplacian.size(),
                       num_stalks, num_restrictions);
    if (!initialized_ && !initialize())
    {
        return false;
    }

    const size_t stalk_count =
        checkedMulSize(static_cast<size_t>(num_stalks), static_cast<size_t>(num_stalks),
                       "sheaf tensor-core stalk count");
    const size_t restriction_count =
        checkedMulSize(static_cast<size_t>(num_restrictions), static_cast<size_t>(num_stalks),
                       "sheaf tensor-core restriction count");

    DeviceBuffer<InputPrecision> d_stalk;
    DeviceBuffer<InputPrecision> d_restriction;
    DeviceBuffer<float> d_output;
    if (!d_stalk.allocate(stalk_count) || !d_restriction.allocate(restriction_count) ||
        !d_output.allocate(stalk_count))
    {
        return false;
    }

    if (!validCuda(cudaMemcpy(
            d_stalk.get(), stalk_matrix.data(),
            checkedByteCount<InputPrecision>(stalk_count, "sheaf tensor-core stalk bytes"),
            cudaMemcpyHostToDevice)) ||
        !validCuda(cudaMemcpy(d_restriction.get(), restriction_matrix.data(),
                              checkedByteCount<InputPrecision>(
                                  restriction_count, "sheaf tensor-core restriction bytes"),
                              cudaMemcpyHostToDevice)))
    {
        return false;
    }

    const auto started = std::chrono::steady_clock::now();
    constexpr int block_size = 256;
    const int blocks = checkedGridBlocks(stalk_count, block_size, "sheaf tensor-core dense grid");
    denseSheafLaplacianKernel<<<blocks, block_size>>>(d_stalk.get(), d_restriction.get(),
                                                      d_output.get(), num_stalks, num_restrictions);
    if (!validCuda(cudaGetLastError()) || !validCuda(cudaDeviceSynchronize()))
    {
        return false;
    }

    if (!validCuda(
            cudaMemcpy(output_laplacian.data(), d_output.get(),
                       checkedByteCount<float>(stalk_count, "sheaf tensor-core output bytes"),
                       cudaMemcpyDeviceToHost)))
    {
        return false;
    }

    const auto elapsed = std::chrono::steady_clock::now() - started;
    last_compute_time_ms_ = std::chrono::duration<double, std::milli>(elapsed).count();
    last_num_stalks_ = num_stalks;
    last_num_restrictions_ = num_restrictions;
    return true;
}

bool TensorCoreSheafLaplacian::computeBlackwellTMA(std::span<const __nv_fp8_e4m3> stalk_data,
                                                   std::span<const __nv_fp8_e4m3> restriction_maps,
                                                   std::span<float> output_laplacian,
                                                   int num_stalks)
{
    if (num_stalks <= 0)
    {
        throw std::invalid_argument("num_stalks must be positive");
    }
    const size_t n = static_cast<size_t>(num_stalks);
    const size_t n_square = checkedMulSize(n, n, "FP8 sheaf restriction size");
    if (stalk_data.size() < n || restriction_maps.size() < n_square || output_laplacian.size() < n)
    {
        throw std::invalid_argument("FP8 sheaf buffers are smaller than the requested problem");
    }
    if (!initialized_ && !initialize())
    {
        return false;
    }

    DeviceBuffer<__nv_fp8_e4m3> d_stalk;
    DeviceBuffer<__nv_fp8_e4m3> d_restriction;
    DeviceBuffer<float> d_output;
    if (!d_stalk.allocate(n) || !d_restriction.allocate(n_square) || !d_output.allocate(n))
    {
        return false;
    }
    if (!validCuda(cudaMemcpy(d_stalk.get(), stalk_data.data(),
                              checkedByteCount<__nv_fp8_e4m3>(n, "FP8 sheaf stalk bytes"),
                              cudaMemcpyHostToDevice)) ||
        !validCuda(
            cudaMemcpy(d_restriction.get(), restriction_maps.data(),
                       checkedByteCount<__nv_fp8_e4m3>(n_square, "FP8 sheaf restriction bytes"),
                       cudaMemcpyHostToDevice)))
    {
        return false;
    }

    const auto started = std::chrono::steady_clock::now();
    constexpr int block_size = 256;
    const int blocks = checkedGridBlocks(n, block_size, "FP8 sheaf grid");
    fp8DiagonalSheafLaplacianKernel<<<blocks, block_size>>>(d_stalk.get(), d_restriction.get(),
                                                            d_output.get(), num_stalks);
    if (!validCuda(cudaGetLastError()) || !validCuda(cudaDeviceSynchronize()))
    {
        return false;
    }
    if (!validCuda(cudaMemcpy(output_laplacian.data(), d_output.get(),
                              checkedByteCount<float>(n, "FP8 sheaf output bytes"),
                              cudaMemcpyDeviceToHost)))
    {
        return false;
    }

    const auto elapsed = std::chrono::steady_clock::now() - started;
    last_compute_time_ms_ = std::chrono::duration<double, std::milli>(elapsed).count();
    last_num_stalks_ = num_stalks;
    last_num_restrictions_ = num_stalks;
    return true;
}

double TensorCoreSheafLaplacian::getLastComputeTime() const noexcept
{
    return last_compute_time_ms_;
}

double TensorCoreSheafLaplacian::getEffectiveTFLOPS() const noexcept
{
    if (last_compute_time_ms_ <= 0.0 || last_num_stalks_ <= 0 || last_num_restrictions_ <= 0)
    {
        return 0.0;
    }
    const double n = static_cast<double>(last_num_stalks_);
    const double r = static_cast<double>(last_num_restrictions_);
    const double operations = 2.0 * n * n * r;
    return operations / (last_compute_time_ms_ * 1.0e9);
}

bool TensorCoreSheafLaplacian::hasTensorCoreSupport() const noexcept
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return false;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
    {
        return false;
    }
    return prop.major >= 7;
}

template bool TensorCoreSheafLaplacian::compute<float>(std::span<const float>,
                                                       std::span<const float>, std::span<float>,
                                                       int, int);

template bool TensorCoreSheafLaplacian::compute<half>(std::span<const half>, std::span<const half>,
                                                      std::span<float>, int, int);

template bool TensorCoreSheafLaplacian::compute<__nv_bfloat16>(std::span<const __nv_bfloat16>,
                                                               std::span<const __nv_bfloat16>,
                                                               std::span<float>, int, int);

} // namespace tensorcore
} // namespace gpu
} // namespace sheaf
} // namespace nerve
