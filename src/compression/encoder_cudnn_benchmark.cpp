#include "encoder_cudnn_model.cpp"
#include "nerve/compression/gpu_autoencoder.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nerve
{
namespace compression
{
namespace gpu
{

float deterministicCuDNNWeight(int layer_idx, int in_idx, int out_idx)
{
    const int mixed = (layer_idx + 1) * 97 + in_idx * 19 + out_idx * 31;
    return (static_cast<float>(mixed % 29) - 14.0f) * 0.01f;
}

float deterministicCuDNNBias(int layer_idx, int out_idx)
{
    const int mixed = (layer_idx + 1) * 37 + out_idx * 5;
    return (static_cast<float>(mixed % 13) - 6.0f) * 0.001f;
}

std::vector<float> cpuDenseEncoderForward(std::vector<float> layer_input,
                                          const std::vector<int> &layers, int batch_size)
{
    for (std::size_t layer = 0; layer + 1 < layers.size(); ++layer)
    {
        const int in_dim = layers[layer];
        const int out_dim = layers[layer + 1];
        std::vector<float> layer_output(
            checkedElements(batch_size, out_dim, "CPU encoder layer output count"), 0.0f);
        for (int batch = 0; batch < batch_size; ++batch)
        {
            for (int out_idx = 0; out_idx < out_dim; ++out_idx)
            {
                float sum = deterministicCuDNNBias(static_cast<int>(layer), out_idx);
                for (int in_idx = 0; in_idx < in_dim; ++in_idx)
                {
                    sum += layer_input[static_cast<std::size_t>(batch) * in_dim + in_idx] *
                           deterministicCuDNNWeight(static_cast<int>(layer), in_idx, out_idx);
                }
                if (!std::isfinite(sum))
                {
                    throw std::runtime_error("CPU encoder reference produced non-finite output");
                }
                layer_output[static_cast<std::size_t>(batch) * out_dim + out_idx] =
                    (layer + 2 < layers.size()) ? std::max(sum, 0.0f) : sum;
            }
        }
        layer_input.swap(layer_output);
    }
    return layer_input;
}

CuDNNEncoderBenchmark benchmarkCuDNNEncoder(int batch_size, int input_dim, int latent_dim)
{
    if (batch_size <= 0 || input_dim <= 0 || latent_dim <= 0)
    {
        throw std::invalid_argument("CuDNN encoder benchmark dimensions must be positive");
    }

    CuDNNEncoderBenchmark bench;
    bench.batch_size = batch_size;
    bench.input_dim = input_dim;
    bench.latent_dim = latent_dim;

    std::vector<int> layers = {input_dim, std::max(1, input_dim / 2), latent_dim};

    std::vector<float> input(checkedElements(batch_size, input_dim, "encoder benchmark input"));
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<float>((i * 17) % 101) / 101.0f;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_output = cpuDenseEncoderForward(input, layers, batch_size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    CuDNNEncoder encoder(layers, batch_size);
    for (std::size_t layer = 0; layer + 1 < layers.size(); ++layer)
    {
        const int in_dim = layers[layer];
        const int out_dim = layers[layer + 1];
        std::vector<float> weights(checkedElements(in_dim, out_dim, "encoder benchmark weights"));
        std::vector<float> bias(static_cast<std::size_t>(out_dim));
        for (int in_idx = 0; in_idx < in_dim; ++in_idx)
        {
            for (int out_idx = 0; out_idx < out_dim; ++out_idx)
            {
                weights[static_cast<std::size_t>(in_idx) * out_dim + out_idx] =
                    deterministicCuDNNWeight(static_cast<int>(layer), in_idx, out_idx);
            }
        }
        for (int out_idx = 0; out_idx < out_dim; ++out_idx)
        {
            bias[static_cast<std::size_t>(out_idx)] =
                deterministicCuDNNBias(static_cast<int>(layer), out_idx);
        }
        encoder.setLayerWeights(static_cast<int>(layer), weights, bias);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto output = encoder.encode(input);
    auto end = std::chrono::high_resolution_clock::now();
    bench.cudnn_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (output.size() != cpu_output.size())
    {
        throw std::runtime_error("CuDNN encoder output size does not match CPU reference");
    }
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        if (!std::isfinite(output[i]) || !std::isfinite(cpu_output[i]) ||
            std::abs(output[i] - cpu_output[i]) > 1e-4f)
        {
            throw std::runtime_error("CuDNN encoder output does not match CPU reference");
        }
    }

    bench.fused_time_ms = 0.0;
    bench.speedup_cudnn = (std::isfinite(bench.cpu_time_ms) && std::isfinite(bench.cudnn_time_ms) &&
                           bench.cpu_time_ms >= 0.0 && bench.cudnn_time_ms > 0.0)
                              ? bench.cpu_time_ms / bench.cudnn_time_ms
                              : 1.0;
    bench.speedup_fused = 1.0;

    return bench;
}

} // namespace gpu
} // namespace compression
} // namespace nerve
