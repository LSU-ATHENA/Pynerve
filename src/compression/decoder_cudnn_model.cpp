#include "decoder_cudnn_layer.cpp"
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

class CuDNNDecoder
{
public:
    explicit CuDNNDecoder(const std::vector<int> &layer_dims, int batch_size = 32)
        : layer_dims_(layer_dims)
        , batch_size_(batch_size)
    {
        if (batch_size_ <= 0 || layer_dims_.size() < 2)
        {
            throw std::invalid_argument(
                "decoder requires a positive batch and at least two layer dimensions");
        }
        for (int dim : layer_dims_)
        {
            if (dim <= 0)
            {
                throw std::invalid_argument("decoder layer dimensions must be positive");
            }
        }

        try
        {
            checkCudnn(cudnnCreate(&cudnn_), "create decoder cuDNN handle");
            checkCublas(cublasCreate(&cublas_), "create decoder cuBLAS handle");

            checkCudnn(cudnnCreateActivationDescriptor(&relu_act_),
                       "create decoder ReLU descriptor");
            checkCudnn(cudnnSetActivationDescriptor(relu_act_, CUDNN_ACTIVATION_RELU,
                                                    CUDNN_PROPAGATE_NAN, 0.0),
                       "configure decoder ReLU descriptor");

            checkCudnn(cudnnCreateActivationDescriptor(&sigmoid_act_),
                       "create decoder sigmoid descriptor");
            checkCudnn(cudnnSetActivationDescriptor(sigmoid_act_, CUDNN_ACTIVATION_SIGMOID,
                                                    CUDNN_PROPAGATE_NAN, 0.0),
                       "configure decoder sigmoid descriptor");

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

    ~CuDNNDecoder() { release(); }

    std::vector<float> decode(const std::vector<float> &latent)
    {
        const std::size_t expected =
            checkedElements(batch_size_, layer_dims_.front(), "decoder input element count");
        if (latent.size() != expected)
        {
            throw std::invalid_argument("decoder input must match batch_size * latent_dim");
        }
        requireFiniteValues(latent, "decoder input");

        float *d_input = nullptr;
        try
        {
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_input),
                                 checkedFloatBytes(latent.size(), "decoder input bytes")),
                      "allocate decoder input");
            checkCuda(cudaMemcpy(d_input, latent.data(),
                                 checkedFloatBytes(latent.size(), "decoder input copy bytes"),
                                 cudaMemcpyHostToDevice),
                      "copy decoder input");

            float *current = d_input;
            for (std::size_t i = 0; i < layers_.size(); ++i)
            {
                cudnnActivationDescriptor_t act =
                    (i < layers_.size() - 1) ? relu_act_ : sigmoid_act_;
                current = layers_[i].forward(current, act);
            }

            const std::size_t output_size = checkedElements(
                batch_size_, layers_.back().getOutputSize(), "decoder output element count");
            std::vector<float> output(output_size);
            checkCuda(cudaMemcpy(output.data(), current,
                                 checkedFloatBytes(output.size(), "decoder output copy bytes"),
                                 cudaMemcpyDeviceToHost),
                      "copy decoder output");
            requireFiniteValues(output, "decoder output");

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
        if (sigmoid_act_)
            cudnnDestroyActivationDescriptor(sigmoid_act_);
        if (relu_act_)
            cudnnDestroyActivationDescriptor(relu_act_);
        if (cublas_)
            cublasDestroy(cublas_);
        if (cudnn_)
            cudnnDestroy(cudnn_);
        sigmoid_act_ = nullptr;
        relu_act_ = nullptr;
        cublas_ = nullptr;
        cudnn_ = nullptr;
    }

    cudnnHandle_t cudnn_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cudnnActivationDescriptor_t relu_act_ = nullptr;
    cudnnActivationDescriptor_t sigmoid_act_ = nullptr;
    std::vector<CuDNNDecoderLayer> layers_;
    std::vector<int> layer_dims_;
    int batch_size_ = 0;
};

} // namespace gpu
} // namespace compression
} // namespace nerve
