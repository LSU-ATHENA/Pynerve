#include "nerve/compression/gpu_autoencoder.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve
{
namespace compression
{
namespace gpu
{
namespace
{

const char *cublasStatusName(cublasStatus_t status) noexcept
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
#ifdef CUBLAS_STATUS_LICENSE_ERROR
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

void checkCuda(cudaError_t status, const char *context)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

void checkCudnn(cudnnStatus_t status, const char *context)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(context) + ": " + cudnnGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char *context)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(context) + ": " + cublasStatusName(status));
    }
}

std::size_t checkedMul(std::size_t a, std::size_t b, const char *label)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::length_error(std::string(label) + " exceeds size_t limits");
    }
    return a * b;
}

std::size_t positiveIntToSize(int value, const char *label)
{
    if (value <= 0)
    {
        throw std::invalid_argument(std::string(label) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checkedElements(int a, int b, const char *label)
{
    return checkedMul(positiveIntToSize(a, label), positiveIntToSize(b, label), label);
}

std::size_t checkedFloatBytes(std::size_t elements, const char *label)
{
    return checkedMul(elements, sizeof(float), label);
}

void requireFiniteValues(const std::vector<float> &values, const char *label)
{
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(std::string(label) + " must be finite");
        }
    }
}

void validateDeviceFloats(const float *d_values, std::size_t count, const char *label)
{
    if (d_values == nullptr)
    {
        throw std::invalid_argument(std::string(label) + " device pointer must not be null");
    }
    std::vector<float> values(count);
    if (!values.empty())
    {
        checkCuda(cudaMemcpy(values.data(), d_values, checkedFloatBytes(values.size(), label),
                             cudaMemcpyDeviceToHost),
                  label);
    }
    requireFiniteValues(values, label);
}

float checkedInitialScale(int input_dim, int output_dim, const char *label)
{
    const double fan = static_cast<double>(input_dim) + static_cast<double>(output_dim);
    const double scale = std::sqrt(2.0 / fan);
    if (!std::isfinite(scale) || scale > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::length_error(std::string(label) + " initialization scale is not finite");
    }
    return static_cast<float>(scale);
}

} // namespace

} // namespace gpu
} // namespace compression
} // namespace nerve
