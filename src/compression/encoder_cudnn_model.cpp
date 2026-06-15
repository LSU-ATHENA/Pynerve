#include "encoder_cudnn_layer.cpp"
#include "nerve/compression/gpu_autoencoder.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve
{
namespace compression
{
namespace gpu
{

class CuDNNEncoder
{
public:
    explicit CuDNNEncoder(const std::vector<int> &layer_dims, int batch_size = 32)
        : layer_dims_(layer_dims)
        , batch_size_(batch_size)
    {
        if (batch_size_ <= 0 || layer_dims_.size() < 2)
        {
            throw std::invalid_argument(
                "encoder requires a positive batch and at least two layer dimensions");
        }
        for (int dim : layer_dims_)
        {
            if (dim <= 0)
            {
                throw std::invalid_argument("encoder layer dimensions must be positive");
            }
        }

        try
        {
            checkCudnn(cudnnCreate(&cudnn_), "create encoder cuDNN handle");
            checkCublas(cublasCreate(&cublas_), "create encoder cuBLAS handle");

            checkCudnn(cudnnCreateActivationDescriptor(&relu_activation_),
                       "create encoder activation descriptor");
            checkCudnn(cudnnSetActivationDescriptor(relu_activation_, CUDNN_ACTIVATION_RELU,
                                                    CUDNN_PROPAGATE_NAN, 0.0),
                       "configure encoder activation descriptor");
            layers_.reserve(layer_dims_.size() - 1);
            for (std::size_t i = 0; i + 1 < layer_dims_.size(); ++i)
            {
                layers_.emplace_back(cudnn_, cublas_, batch_size_, layer_dims_[i],
                                     layer_dims_[i + 1]);
            }
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    ~CuDNNEncoder() { release(); }

    std::vector<float> encode(const std::vector<float> &input)
    {
        const std::size_t expected =
            checkedElements(batch_size_, layer_dims_.front(), "encoder input element count");
        if (input.size() != expected)
        {
            throw std::invalid_argument("encoder input must match batch_size * input_dim");
        }
        requireFiniteValues(input, "encoder input");

        float *d_input = nullptr;
        try
        {
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_input),
                                 checkedFloatBytes(input.size(), "encoder input bytes")),
                      "allocate encoder input");
            checkCuda(cudaMemcpy(d_input, input.data(),
                                 checkedFloatBytes(input.size(), "encoder input copy bytes"),
                                 cudaMemcpyHostToDevice),
                      "copy encoder input");

            float *current = d_input;
            for (std::size_t i = 0; i < layers_.size(); ++i)
            {
                cudnnActivationDescriptor_t act =
                    (i < layers_.size() - 1) ? relu_activation_ : nullptr;
                current = layers_[i].forward(current, act);
            }

            const std::size_t output_size = checkedElements(
                batch_size_, layers_.back().getOutputSize(), "encoder output element count");
            std::vector<float> output(output_size);
            checkCuda(cudaMemcpy(output.data(), current,
                                 checkedFloatBytes(output.size(), "encoder output copy bytes"),
                                 cudaMemcpyDeviceToHost),
                      "copy encoder output");
            requireFiniteValues(output, "encoder output");

            cudaFree(d_input);
            return output;
        }
        catch (...)
        {
            if (d_input)
                cudaFree(d_input);
            throw;
        }
    }

    void setLayerWeights(int layer_idx, const std::vector<float> &weights,
                         const std::vector<float> &bias)
    {
        if (layer_idx < 0 || static_cast<std::size_t>(layer_idx) >= layers_.size())
        {
            throw std::out_of_range("Invalid layer index");
        }
        layers_[static_cast<std::size_t>(layer_idx)].setWeights(weights, bias);
    }

private:
    void release() noexcept
    {
        layers_.clear();
        if (relu_activation_)
            cudnnDestroyActivationDescriptor(relu_activation_);
        if (cublas_)
            cublasDestroy(cublas_);
        if (cudnn_)
            cudnnDestroy(cudnn_);
        relu_activation_ = nullptr;
        cublas_ = nullptr;
        cudnn_ = nullptr;
    }

    cudnnHandle_t cudnn_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cudnnActivationDescriptor_t relu_activation_ = nullptr;
    std::vector<CuDNNEncoderLayer> layers_;
    std::vector<int> layer_dims_;
    int batch_size_ = 0;
};

} // namespace gpu
} // namespace compression
} // namespace nerve
