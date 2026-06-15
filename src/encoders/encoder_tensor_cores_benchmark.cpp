#include "nerve/encoders/gpu_encoders.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace nerve::encoders::tensorcore
{

namespace
{
constexpr float DEFAULT_BENCH_WEIGHT = 0.01f;

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

std::size_t checkedProduct(std::size_t lhs, std::size_t rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}
} // namespace

TensorCoreBenchmark benchmarkTensorCore(int batch_size, int input_dim, int output_dim)
{
    if (batch_size <= 0 || input_dim <= 0 || output_dim <= 0)
    {
        throw std::invalid_argument("tensor core benchmark dimensions must be positive");
    }

    TensorCoreBenchmark bench{};
    bench.batch_size = batch_size;
    bench.input_dim = input_dim;
    bench.output_dim = output_dim;

    const std::size_t batch = static_cast<std::size_t>(batch_size);
    const std::size_t input_width = static_cast<std::size_t>(input_dim);
    const std::size_t output_width = static_cast<std::size_t>(output_dim);
    std::vector<float> input(
        checkedProduct(batch, input_width, "tensor core benchmark input size overflows"));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float &v : input)
    {
        v = dist(rng);
    }

    const auto start_fp32 = std::chrono::high_resolution_clock::now();
    std::vector<float> output_fp32(
        checkedProduct(batch, output_width, "tensor core benchmark output size overflows"));
    for (int b = 0; b < batch_size; ++b)
    {
        for (int o = 0; o < output_dim; ++o)
        {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; ++i)
            {
                sum += input[static_cast<size_t>(b) * input_dim + i] * DEFAULT_BENCH_WEIGHT;
            }
            output_fp32[static_cast<size_t>(b) * output_dim + o] = sum;
        }
    }
    const auto end_fp32 = std::chrono::high_resolution_clock::now();
    bench.fp32_time_ms = std::chrono::duration<double, std::milli>(end_fp32 - start_fp32).count();

    TensorCoreMLPEncoder encoder(input_dim, {256, output_dim}, output_dim);
    std::vector<float> output_fp16;
    const auto start_fp16 = std::chrono::high_resolution_clock::now();
    encoder.encode(input, output_fp16, batch_size);
    const auto end_fp16 = std::chrono::high_resolution_clock::now();
    bench.fp16_time_ms = std::chrono::duration<double, std::milli>(end_fp16 - start_fp16).count();
    bench.speedup = finiteBenchmarkSpeedup(bench.fp32_time_ms, bench.fp16_time_ms);
    return bench;
}

} // namespace nerve::encoders::tensorcore
