#include "decoder_cudnn_model.cpp"
#include "nerve/compression/gpu_autoencoder.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace nerve
{
namespace compression
{
namespace gpu
{

CuDNNDecoderBenchmark benchmarkCuDNNDecoder(int batch_size, int latent_dim, int output_dim)
{
    if (batch_size <= 0 || latent_dim <= 0 || output_dim <= 0)
    {
        throw std::invalid_argument("CuDNN decoder benchmark dimensions must be positive");
    }

    CuDNNDecoderBenchmark bench;
    bench.batch_size = batch_size;
    bench.latent_dim = latent_dim;
    bench.output_dim = output_dim;

    std::vector<int> layers = {latent_dim, std::max(1, output_dim / 2), output_dim};

    std::vector<float> latent(checkedElements(batch_size, latent_dim, "decoder benchmark input"));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : latent)
    {
        v = dist(gen);
    }

    CuDNNDecoder decoder(layers, batch_size);

    auto start = std::chrono::high_resolution_clock::now();
    auto output = decoder.decode(latent);
    auto end = std::chrono::high_resolution_clock::now();
    bench.cudnn_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (output.size() != checkedElements(batch_size, output_dim, "decoder benchmark output"))
    {
        throw std::runtime_error("CuDNN decoder output size does not match requested dimensions");
    }
    requireFiniteValues(output, "decoder benchmark output");

    // No CPU baseline or isolated fused-path benchmark is run here.
    bench.cpu_time_ms = 0.0;
    bench.fused_time_ms = 0.0;
    bench.speedup_cudnn = 1.0;
    bench.speedup_fused = 1.0;

    return bench;
}

} // namespace gpu
} // namespace compression
} // namespace nerve
