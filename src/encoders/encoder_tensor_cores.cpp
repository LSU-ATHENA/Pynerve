#include "nerve/encoders/gpu_encoders.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace nerve::encoders::tensorcore
{

namespace
{

constexpr int CUDA_BLOCK_SIZE = 256;
constexpr float DEFAULT_INIT_SCALE = 0.01f;
constexpr float MAX_LOSS_SCALE = 65536.0f;

void checkCuda(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(msg);
    }
}

void checkCublas(cublasStatus_t status, const char *msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(msg);
    }
}

void checkCudnn(cudnnStatus_t status, const char *msg)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error(msg);
    }
}

__global__ void convertFP32toFP16Kernel(const float *__restrict__ input,
                                        __half *__restrict__ output, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = __float2half_rn(input[idx]);
    }
}

__global__ void addBiasAndConvertKernel(const __half *__restrict__ data,
                                        const __half *__restrict__ bias, float *__restrict__ output,
                                        int n, int num_features, bool apply_relu)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const int feature = idx % num_features;
    float val = __half2float(data[idx]) + __half2float(bias[feature]);
    if (apply_relu)
    {
        val = fmaxf(val, 0.0f);
    }
    output[idx] = val;
}

__global__ void unscaleKernel(float *grads, int n, float inv_scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        grads[idx] *= inv_scale;
    }
}

__global__ void checkInfNanKernel(const float *data, int n, int *result)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const float val = data[idx];
        if (isinf(val) || isnan(val))
        {
            atomicExch(result, 1);
        }
    }
}

} // namespace

class TensorCoreMLPEncoder::Impl
{
public:
    Impl(int input_dim, const std::vector<int> &hidden_dims, int output_dim)
        : input_dim_(input_dim)
        , output_dim_(output_dim)
    {
        layer_dims_.push_back(input_dim);
        layer_dims_.insert(layer_dims_.end(), hidden_dims.begin(), hidden_dims.end());
        layer_dims_.push_back(output_dim);
        checkCublas(cublasCreate(&cublas_handle_), "Failed to initialize cuBLAS.");
        initWeights();
        initBuffers();
    }

    ~Impl()
    {
        for (auto *ptr : d_weights_)
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
        for (auto *ptr : d_bias_)
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
        for (auto *ptr : d_buffers_)
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
        if (cublas_handle_ != nullptr)
        {
            cublasDestroy(cublas_handle_);
        }
    }

    void encode(const std::vector<float> &input, std::vector<float> &output, int batch_size)
    {
        if (batch_size <= 0 || layer_dims_.empty())
        {
            output.clear();
            return;
        }
        const size_t in_size = static_cast<size_t>(batch_size) * layer_dims_[0];
        if (input.size() < in_size)
        {
            throw std::invalid_argument("TensorCoreMLPEncoder input buffer is too small.");
        }
        checkCuda(cudaMemcpy(d_buffers_[0], input.data(), in_size * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Failed to upload encoder input.");

        for (size_t i = 0; i + 1 < layer_dims_.size(); ++i)
        {
            const int in_dim = layer_dims_[i];
            const int out_dim = layer_dims_[i + 1];
            const int input_elems = batch_size * in_dim;
            const int output_elems = batch_size * out_dim;
            __half *d_input_fp16 = nullptr;
            __half *d_output_fp16 = nullptr;
            checkCuda(cudaMalloc(&d_input_fp16, static_cast<size_t>(input_elems) * sizeof(__half)),
                      "Failed to allocate FP16 input buffer.");
            checkCuda(
                cudaMalloc(&d_output_fp16, static_cast<size_t>(output_elems) * sizeof(__half)),
                "Failed to allocate FP16 output buffer.");

            const int input_blocks = (input_elems + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            convertFP32toFP16Kernel<<<input_blocks, CUDA_BLOCK_SIZE>>>(d_buffers_[i], d_input_fp16,
                                                                       input_elems);
            const cudaError_t convert_launch_status = cudaPeekAtLastError();
            if (convert_launch_status != cudaSuccess)
            {
                cudaFree(d_input_fp16);
                cudaFree(d_output_fp16);
                checkCuda(convert_launch_status, "Failed to launch FP32 to FP16 conversion.");
            }

            const __half alpha = __float2half_rn(1.0f);
            const __half beta = __float2half_rn(0.0f);
            checkCublas(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, out_dim, batch_size,
                                     in_dim, &alpha, d_weights_[i], CUDA_R_16F, out_dim,
                                     d_input_fp16, CUDA_R_16F, in_dim, &beta, d_output_fp16,
                                     CUDA_R_16F, out_dim, CUBLAS_COMPUTE_16F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "Tensor Core GEMM failed.");

            const int output_blocks = (output_elems + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            addBiasAndConvertKernel<<<output_blocks, CUDA_BLOCK_SIZE>>>(
                d_output_fp16, d_bias_[i], d_buffers_[i + 1], output_elems, out_dim,
                i + 1 < layer_dims_.size() - 1);
            const cudaError_t bias_launch_status = cudaPeekAtLastError();

            cudaFree(d_input_fp16);
            cudaFree(d_output_fp16);
            checkCuda(bias_launch_status, "Failed to launch bias conversion.");
        }

        output.resize(static_cast<size_t>(batch_size) * output_dim_);
        checkCuda(cudaMemcpy(output.data(), d_buffers_.back(), output.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Failed to download encoder output.");
    }

private:
    void initWeights()
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-DEFAULT_INIT_SCALE, DEFAULT_INIT_SCALE);
        for (size_t i = 0; i + 1 < layer_dims_.size(); ++i)
        {
            const int in_dim = layer_dims_[i];
            const int out_dim = layer_dims_[i + 1];
            __half *d_weights = nullptr;
            __half *d_bias = nullptr;
            checkCuda(
                cudaMalloc(&d_weights, static_cast<size_t>(in_dim) * out_dim * sizeof(__half)),
                "Failed to allocate FP16 weights.");
            checkCuda(cudaMalloc(&d_bias, static_cast<size_t>(out_dim) * sizeof(__half)),
                      "Failed to allocate FP16 bias.");
            std::vector<__half> h_weights(static_cast<size_t>(in_dim) * out_dim);
            std::vector<__half> h_bias(out_dim, __float2half_rn(0.0f));
            for (auto &w : h_weights)
            {
                w = __float2half_rn(dist(rng));
            }
            checkCuda(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(__half),
                                 cudaMemcpyHostToDevice),
                      "Failed to upload FP16 weights.");
            checkCuda(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(__half),
                                 cudaMemcpyHostToDevice),
                      "Failed to upload FP16 bias.");
            d_weights_.push_back(d_weights);
            d_bias_.push_back(d_bias);
        }
    }

    void initBuffers()
    {
        for (const int dim : layer_dims_)
        {
            float *d_buffer = nullptr;
            checkCuda(cudaMalloc(&d_buffer, static_cast<size_t>(dim) * sizeof(float)),
                      "Failed to allocate intermediate layer buffer.");
            d_buffers_.push_back(d_buffer);
        }
    }

    int input_dim_;
    int output_dim_;
    std::vector<int> layer_dims_;
    std::vector<__half *> d_weights_;
    std::vector<__half *> d_bias_;
    std::vector<float *> d_buffers_;
    cublasHandle_t cublas_handle_ = nullptr;
};

TensorCoreMLPEncoder::TensorCoreMLPEncoder(int input_dim, const std::vector<int> &hidden_dims,
                                           int output_dim)
    : impl_(std::make_unique<Impl>(input_dim, hidden_dims, output_dim))
{}
TensorCoreMLPEncoder::~TensorCoreMLPEncoder() = default;
void TensorCoreMLPEncoder::encode(const std::vector<float> &input, std::vector<float> &output,
                                  int batch_size)
{
    impl_->encode(input, output, batch_size);
}

class CUDNNTopologicalEncoder::Impl
{
public:
    Impl(int input_height, int input_width, int input_channels, int num_filters, int filter_size)
    {
        checkCudnn(cudnnCreate(&cudnn_handle_), "Failed to initialize cuDNN.");
        checkCudnn(cudnnCreateTensorDescriptor(&input_desc_), "Failed to create input descriptor.");
        checkCudnn(cudnnCreateFilterDescriptor(&filter_desc_),
                   "Failed to create filter descriptor.");
        checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc_),
                   "Failed to create convolution descriptor.");
        checkCudnn(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                                              input_channels, input_height, input_width),
                   "Failed to set input descriptor.");
        checkCudnn(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              num_filters, input_channels, filter_size,
                                              filter_size),
                   "Failed to set filter descriptor.");
        checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc_, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION,
                                                   CUDNN_DATA_FLOAT),
                   "Failed to set convolution descriptor.");

        int n = 0;
        int c = 0;
        int h = 0;
        int w = 0;
        checkCudnn(cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_, filter_desc_, &n,
                                                         &c, &h, &w),
                   "Failed to compute convolution output shape.");
        output_elements_ = static_cast<size_t>(n) * c * h * w;
        checkCudnn(cudnnCreateTensorDescriptor(&output_desc_),
                   "Failed to create output descriptor.");
        checkCudnn(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,
                                              c, h, w),
                   "Failed to set output descriptor.");

        const size_t input_size = static_cast<size_t>(input_channels) * input_height * input_width;
        const size_t filter_size_total =
            static_cast<size_t>(num_filters) * input_channels * filter_size * filter_size;
        checkCuda(cudaMalloc(&d_input_, input_size * sizeof(float)),
                  "Failed to allocate encoder input tensor.");
        checkCuda(cudaMalloc(&d_filters_, filter_size_total * sizeof(float)),
                  "Failed to allocate encoder filters.");
        checkCuda(cudaMalloc(&d_output_, output_elements_ * sizeof(float)),
                  "Failed to allocate encoder output tensor.");
    }

    ~Impl()
    {
        if (input_desc_ != nullptr)
            cudnnDestroyTensorDescriptor(input_desc_);
        if (output_desc_ != nullptr)
            cudnnDestroyTensorDescriptor(output_desc_);
        if (filter_desc_ != nullptr)
            cudnnDestroyFilterDescriptor(filter_desc_);
        if (conv_desc_ != nullptr)
            cudnnDestroyConvolutionDescriptor(conv_desc_);
        if (cudnn_handle_ != nullptr)
            cudnnDestroy(cudnn_handle_);
        if (d_input_ != nullptr)
            cudaFree(d_input_);
        if (d_filters_ != nullptr)
            cudaFree(d_filters_);
        if (d_output_ != nullptr)
            cudaFree(d_output_);
        if (d_workspace_ != nullptr)
            cudaFree(d_workspace_);
    }

    void encode(const std::vector<float> &input, std::vector<float> &output)
    {
        checkCuda(cudaMemcpy(d_input_, input.data(), input.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Failed to upload cuDNN input.");
        size_t workspace_size = 0;
        constexpr cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, input_desc_, filter_desc_,
                                                           conv_desc_, output_desc_, algo,
                                                           &workspace_size),
                   "Failed to query cuDNN workspace.");
        if (workspace_size > workspace_size_)
        {
            if (d_workspace_ != nullptr)
            {
                cudaFree(d_workspace_);
            }
            checkCuda(cudaMalloc(&d_workspace_, workspace_size),
                      "Failed to allocate cuDNN workspace.");
            workspace_size_ = workspace_size;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCudnn(cudnnConvolutionForward(cudnn_handle_, &alpha, input_desc_, d_input_,
                                           filter_desc_, d_filters_, conv_desc_, algo, d_workspace_,
                                           workspace_size_, &beta, output_desc_, d_output_),
                   "cuDNN convolution failed.");
        output.resize(output_elements_);
        checkCuda(cudaMemcpy(output.data(), d_output_, output.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Failed to download cuDNN output.");
    }

private:
    cudnnHandle_t cudnn_handle_ = nullptr;
    cudnnTensorDescriptor_t input_desc_ = nullptr;
    cudnnTensorDescriptor_t output_desc_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_ = nullptr;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    float *d_input_ = nullptr;
    float *d_filters_ = nullptr;
    float *d_output_ = nullptr;
    void *d_workspace_ = nullptr;
    size_t workspace_size_ = 0;
    size_t output_elements_ = 0;
};

CUDNNTopologicalEncoder::CUDNNTopologicalEncoder(int input_height, int input_width,
                                                 int input_channels, int num_filters,
                                                 int filter_size)
    : impl_(std::make_unique<Impl>(input_height, input_width, input_channels, num_filters,
                                   filter_size))
{}
CUDNNTopologicalEncoder::~CUDNNTopologicalEncoder() = default;
void CUDNNTopologicalEncoder::encode(const std::vector<float> &input, std::vector<float> &output)
{
    impl_->encode(input, output);
}

class MixedPrecisionEncoder::Impl
{
public:
    float loss_scale = MAX_LOSS_SCALE;
};

MixedPrecisionEncoder::MixedPrecisionEncoder()
    : impl_(std::make_unique<Impl>())
{}
float MixedPrecisionEncoder::scaleLoss(float loss)
{
    return loss * impl_->loss_scale;
}
void MixedPrecisionEncoder::unscaleGradients(float *gradients, int n)
{
    const float inv_scale = 1.0f / impl_->loss_scale;
    const int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    unscaleKernel<<<blocks, CUDA_BLOCK_SIZE>>>(gradients, n, inv_scale);
    checkCuda(cudaPeekAtLastError(), "Failed to launch gradient unscale kernel.");
}
bool MixedPrecisionEncoder::checkForInfNan(const float *gradients, int n)
{
    int *d_result = nullptr;
    checkCuda(cudaMalloc(&d_result, sizeof(int)), "Failed to allocate inf/nan status buffer.");
    checkCuda(cudaMemset(d_result, 0, sizeof(int)), "Failed to reset inf/nan status buffer.");
    const int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    checkInfNanKernel<<<blocks, CUDA_BLOCK_SIZE>>>(gradients, n, d_result);
    const cudaError_t launch_status = cudaPeekAtLastError();
    if (launch_status != cudaSuccess)
    {
        cudaFree(d_result);
        checkCuda(launch_status, "Failed to launch inf/nan check kernel.");
    }
    int result = 0;
    checkCuda(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost),
              "Failed to read inf/nan status.");
    cudaFree(d_result);
    return result != 0;
}
void MixedPrecisionEncoder::updateLossScale(bool found_inf)
{
    impl_->loss_scale =
        found_inf ? impl_->loss_scale * 0.5f : std::min(impl_->loss_scale * 2.0f, MAX_LOSS_SCALE);
}

} // namespace nerve::encoders::tensorcore
