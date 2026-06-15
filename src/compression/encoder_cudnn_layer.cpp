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

class CuDNNEncoderLayer
{
public:
    CuDNNEncoderLayer(cudnnHandle_t cudnn, cublasHandle_t cublas, int batch_size, int input_dim,
                      int output_dim)
        : cudnn_(cudnn)
        , cublas_(cublas)
        , batch_size_(batch_size)
        , input_dim_(input_dim)
        , output_dim_(output_dim)
    {
        if (cudnn_ == nullptr || cublas_ == nullptr)
        {
            throw std::invalid_argument("encoder layer requires valid cuDNN and cuBLAS handles");
        }
        if (batch_size_ <= 0 || input_dim_ <= 0 || output_dim_ <= 0)
        {
            throw std::invalid_argument("encoder layer dimensions must be positive");
        }

        try
        {
            checkCudnn(cudnnCreateTensorDescriptor(&input_desc_),
                       "create encoder input descriptor");
            checkCudnn(cudnnCreateTensorDescriptor(&output_desc_),
                       "create encoder output descriptor");
            checkCudnn(cudnnCreateTensorDescriptor(&bias_desc_), "create encoder bias descriptor");

            checkCudnn(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  batch_size_, input_dim_, 1, 1),
                       "configure encoder input descriptor");
            checkCudnn(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  batch_size_, output_dim_, 1, 1),
                       "configure encoder output descriptor");
            checkCudnn(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  1, output_dim_, 1, 1),
                       "configure encoder bias descriptor");

            const std::size_t weight_count =
                checkedElements(input_dim_, output_dim_, "encoder weight count");
            const std::size_t output_count =
                checkedElements(batch_size_, output_dim_, "encoder output count");
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_weights_),
                                 checkedFloatBytes(weight_count, "encoder weight bytes")),
                      "allocate encoder weights");
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_bias_),
                                 checkedFloatBytes(static_cast<std::size_t>(output_dim_),
                                                   "encoder bias bytes")),
                      "allocate encoder bias");
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_output_),
                                 checkedFloatBytes(output_count, "encoder output bytes")),
                      "allocate encoder output");
            initializeWeights();
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    CuDNNEncoderLayer(const CuDNNEncoderLayer &) = delete;
    CuDNNEncoderLayer &operator=(const CuDNNEncoderLayer &) = delete;

    CuDNNEncoderLayer(CuDNNEncoderLayer &&other) noexcept { takeFrom(other); }

    CuDNNEncoderLayer &operator=(CuDNNEncoderLayer &&other) noexcept
    {
        if (this != &other)
        {
            release();
            takeFrom(other);
        }
        return *this;
    }

    ~CuDNNEncoderLayer() { release(); }

    float *forward(const float *d_input, cudnnActivationDescriptor_t activation)
    {
        if (d_input == nullptr)
        {
            throw std::invalid_argument("encoder input device pointer must not be null");
        }

        float alpha = 1.0f, beta = 0.0f;
        checkCublas(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, output_dim_, batch_size_,
                                input_dim_, &alpha, d_weights_, input_dim_, d_input, input_dim_,
                                &beta, d_output_, output_dim_),
                    "run encoder GEMM");

        float bias_alpha = 1.0f;
        checkCudnn(cudnnAddTensor(cudnn_, &bias_alpha, bias_desc_, d_bias_, &alpha, output_desc_,
                                  d_output_),
                   "add encoder bias");

        const std::size_t output_count =
            checkedElements(batch_size_, output_dim_, "encoder layer output count");
        validateDeviceFloats(d_output_, output_count, "encoder affine output");

        if (activation != nullptr)
        {
            checkCudnn(cudnnActivationForward(cudnn_, activation, &alpha, output_desc_, d_output_,
                                              &beta, output_desc_, d_output_),
                       "run encoder activation");
            validateDeviceFloats(d_output_, output_count, "encoder activation output");
        }

        return d_output_;
    }

    void setWeights(const std::vector<float> &weights, const std::vector<float> &bias)
    {
        const std::size_t expected_weights =
            checkedElements(input_dim_, output_dim_, "encoder weight count");
        if (weights.size() != expected_weights)
        {
            throw std::runtime_error("Invalid weight dimensions");
        }
        if (bias.size() != static_cast<std::size_t>(output_dim_))
        {
            throw std::runtime_error("Invalid bias dimensions");
        }
        requireFiniteValues(weights, "encoder weights");
        requireFiniteValues(bias, "encoder bias");

        checkCuda(cudaMemcpy(d_weights_, weights.data(),
                             checkedFloatBytes(weights.size(), "encoder weight copy bytes"),
                             cudaMemcpyHostToDevice),
                  "copy encoder weights");
        checkCuda(cudaMemcpy(d_bias_, bias.data(),
                             checkedFloatBytes(bias.size(), "encoder bias copy bytes"),
                             cudaMemcpyHostToDevice),
                  "copy encoder bias");
    }

    float *getOutput() const { return d_output_; }
    float *getWeights() const { return d_weights_; }
    float *getBias() const { return d_bias_; }
    int getOutputSize() const noexcept { return output_dim_; }

private:
    void takeFrom(CuDNNEncoderLayer &other) noexcept
    {
        cudnn_ = other.cudnn_;
        cublas_ = other.cublas_;
        batch_size_ = other.batch_size_;
        input_dim_ = other.input_dim_;
        output_dim_ = other.output_dim_;
        input_desc_ = std::exchange(other.input_desc_, nullptr);
        output_desc_ = std::exchange(other.output_desc_, nullptr);
        bias_desc_ = std::exchange(other.bias_desc_, nullptr);
        d_weights_ = std::exchange(other.d_weights_, nullptr);
        d_bias_ = std::exchange(other.d_bias_, nullptr);
        d_output_ = std::exchange(other.d_output_, nullptr);
    }

    void release() noexcept
    {
        if (input_desc_)
            cudnnDestroyTensorDescriptor(input_desc_);
        if (output_desc_)
            cudnnDestroyTensorDescriptor(output_desc_);
        if (bias_desc_)
            cudnnDestroyTensorDescriptor(bias_desc_);
        if (d_weights_)
            cudaFree(d_weights_);
        if (d_bias_)
            cudaFree(d_bias_);
        if (d_output_)
            cudaFree(d_output_);
        input_desc_ = nullptr;
        output_desc_ = nullptr;
        bias_desc_ = nullptr;
        d_weights_ = nullptr;
        d_bias_ = nullptr;
        d_output_ = nullptr;
    }

    void initializeWeights()
    {
        std::mt19937 gen(42);
        float scale = checkedInitialScale(input_dim_, output_dim_, "encoder");
        std::normal_distribution<float> dist(0.0f, scale);

        const std::size_t weight_count =
            checkedElements(input_dim_, output_dim_, "encoder initial weight count");
        std::vector<float> weights(weight_count);
        std::vector<float> bias(static_cast<std::size_t>(output_dim_), 0.0f);

        for (auto &w : weights)
        {
            w = dist(gen);
        }

        setWeights(weights, bias);
    }

    cudnnHandle_t cudnn_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    int batch_size_ = 0;
    int input_dim_ = 0;
    int output_dim_ = 0;
    cudnnTensorDescriptor_t input_desc_ = nullptr;
    cudnnTensorDescriptor_t output_desc_ = nullptr;
    cudnnTensorDescriptor_t bias_desc_ = nullptr;
    float *d_weights_ = nullptr;
    float *d_bias_ = nullptr;
    float *d_output_ = nullptr;
};

} // namespace gpu
} // namespace compression
} // namespace nerve
