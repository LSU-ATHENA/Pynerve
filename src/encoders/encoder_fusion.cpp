#include "nerve/encoders/gpu_encoders.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>

namespace nerve::encoders::fusion
{
namespace
{

constexpr int kFusedOutputSize = 64;
constexpr int kBasicFeatureCount = 8;
constexpr int kMlpOutputSize = 64;
constexpr int kQuantizationMaxValue = 255;
constexpr float kMlpWeightFactor = 0.01f;
constexpr float kVarianceEpsilon = 1e-8f;
constexpr int kCheckpointOutputSize = 256;
constexpr int kMaxCheckpointInputFeatures = 100;
constexpr float kBenchmarkLifetime = 10.0f;

float deterministicWeight(size_t out_idx, size_t in_idx) noexcept
{
    const uint32_t a = static_cast<uint32_t>(out_idx + 1U) * 2654435761U;
    const uint32_t b = static_cast<uint32_t>(in_idx + 1U) * 2246822519U;
    const uint32_t hash = a ^ b ^ (a >> 13U);
    return kMlpWeightFactor * (0.5f + static_cast<float>(hash & 0xFFU) / 255.0f);
}

bool finitePair(const std::pair<float, float> &pair)
{
    const bool finite_death = std::isfinite(pair.second);
    const bool infinite_death = pair.second == std::numeric_limits<float>::infinity();
    if (!std::isfinite(pair.first) || (!finite_death && !infinite_death) ||
        (finite_death && pair.second < pair.first))
    {
        throw std::invalid_argument("encoder fusion diagram contains an invalid pair");
    }
    return finite_death && pair.second > pair.first;
}

float checkedAdd(float lhs, float rhs, const char *context)
{
    const float result = lhs + rhs;
    if (!std::isfinite(result))
    {
        throw std::overflow_error(context);
    }
    return result;
}

float finiteLifetime(const std::pair<float, float> &pair)
{
    if (!finitePair(pair))
    {
        return 0.0f;
    }
    const float lifetime = pair.second - pair.first;
    if (!std::isfinite(lifetime))
    {
        throw std::overflow_error("encoder fusion lifetime overflow");
    }
    return lifetime;
}

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

void applyRelu(std::vector<float> &values)
{
    for (float &value : values)
    {
        value = std::max(value, 0.0f);
    }
}

void normalizeInPlace(std::vector<float> &values)
{
    if (values.empty())
    {
        return;
    }
    float total = 0.0f;
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("encoder fusion normalization input is non-finite");
        }
        total = checkedAdd(total, value, "encoder fusion normalization mean overflow");
    }
    const float mean = total / static_cast<float>(values.size());
    if (!std::isfinite(mean))
    {
        throw std::overflow_error("encoder fusion normalization mean overflow");
    }
    float variance = 0.0f;
    for (float value : values)
    {
        const float diff = value - mean;
        const float contribution = diff * diff;
        if (!std::isfinite(diff) || !std::isfinite(contribution))
        {
            throw std::overflow_error("encoder fusion normalization variance overflow");
        }
        variance =
            checkedAdd(variance, contribution, "encoder fusion normalization variance overflow");
    }
    variance /= static_cast<float>(values.size());
    const float scale = std::sqrt(variance + kVarianceEpsilon);
    if (!std::isfinite(scale) || scale <= 0.0f)
    {
        throw std::overflow_error("encoder fusion normalization scale overflow");
    }
    for (float &value : values)
    {
        value = (value - mean) / scale;
        if (!std::isfinite(value))
        {
            throw std::overflow_error("encoder fusion normalization produced a non-finite value");
        }
    }
}

} // namespace

class FusedEncoderPipeline::Impl
{
public:
    explicit Impl(const FusionConfig &config)
        : config_(config)
    {}

    void encodeFused(const std::vector<std::pair<float, float>> &diagram,
                     std::vector<float> &output)
    {
        if (config_.fuse_persistence)
        {
            fusedPersistenceToFeatures(diagram, output);
        }
        else
        {
            std::vector<float> features;
            extractFeatures(diagram, features);
            applyMlp(features, output);
        }
    }

    void encodePersistent(const std::vector<std::vector<std::pair<float, float>>> &diagrams,
                          std::vector<std::vector<float>> &outputs)
    {
        persistent_kernel_initialized_ =
            persistent_kernel_initialized_ || config_.use_persistent_kernels;
        outputs.resize(diagrams.size());
        for (size_t i = 0; i < diagrams.size(); ++i)
        {
            encodeFused(diagrams[i], outputs[i]);
        }
    }

private:
    FusionConfig config_;
    bool persistent_kernel_initialized_ = false;

    void fusedPersistenceToFeatures(const std::vector<std::pair<float, float>> &diagram,
                                    std::vector<float> &output) const
    {
        output.assign(kFusedOutputSize, 0.0f);
        size_t finite_count = 0;
        for (const auto &pair : diagram)
        {
            const float lifetime = finiteLifetime(pair);
            if (lifetime <= 0.0f)
            {
                continue;
            }
            output[0] = checkedAdd(output[0], lifetime, "encoder fusion lifetime overflow");
            output[1] = std::max(output[1], lifetime);
            output[2] = checkedAdd(output[2], 1.0f, "encoder fusion feature count overflow");
            for (size_t i = 3; i < output.size(); ++i)
            {
                const float contribution = lifetime * deterministicWeight(i, finite_count);
                if (!std::isfinite(contribution))
                {
                    throw std::overflow_error("encoder fusion weighted feature overflow");
                }
                output[i] =
                    checkedAdd(output[i], contribution, "encoder fusion weighted feature overflow");
            }
            ++finite_count;
        }
        if (config_.fuse_activation)
        {
            applyRelu(output);
        }
        if (config_.fuse_normalization)
        {
            normalizeInPlace(output);
        }
    }

    void extractFeatures(const std::vector<std::pair<float, float>> &diagram,
                         std::vector<float> &features) const
    {
        features.assign(kBasicFeatureCount, 0.0f);
        size_t finite_count = 0;
        for (const auto &pair : diagram)
        {
            const float lifetime = finiteLifetime(pair);
            if (lifetime <= 0.0f)
            {
                continue;
            }
            features[0] = checkedAdd(features[0], lifetime, "encoder fusion lifetime overflow");
            features[1] = std::max(features[1], lifetime);
            features[2] = checkedAdd(features[2], 1.0f, "encoder fusion feature count overflow");
            features[3] = checkedAdd(features[3], pair.first, "encoder fusion birth mean overflow");
            features[4] =
                checkedAdd(features[4], pair.second, "encoder fusion death mean overflow");
            ++finite_count;
        }
        if (finite_count > 0)
        {
            features[3] /= static_cast<float>(finite_count);
            features[4] /= static_cast<float>(finite_count);
        }
    }

    void applyMlp(const std::vector<float> &input, std::vector<float> &output) const
    {
        output.assign(kMlpOutputSize, 0.0f);
        for (size_t i = 0; i < output.size(); ++i)
        {
            float pre_activation = deterministicWeight(i, input.size());
            for (size_t j = 0; j < input.size(); ++j)
            {
                const float contribution = input[j] * deterministicWeight(i, j);
                if (!std::isfinite(contribution))
                {
                    throw std::overflow_error("encoder fusion MLP activation overflow");
                }
                pre_activation = checkedAdd(pre_activation, contribution,
                                            "encoder fusion MLP activation overflow");
            }
            output[i] = config_.fuse_activation ? std::max(pre_activation, 0.0f) : pre_activation;
        }
        if (config_.fuse_normalization)
        {
            normalizeInPlace(output);
        }
    }
};

FusedEncoderPipeline::FusedEncoderPipeline(const FusionConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

FusedEncoderPipeline::~FusedEncoderPipeline() = default;

void FusedEncoderPipeline::encodeFused(const std::vector<std::pair<float, float>> &diagram,
                                       std::vector<float> &output)
{
    impl_->encodeFused(diagram, output);
}

void FusedEncoderPipeline::encodePersistent(
    const std::vector<std::vector<std::pair<float, float>>> &diagrams,
    std::vector<std::vector<float>> &outputs)
{
    impl_->encodePersistent(diagrams, outputs);
}

FusionBenchmark benchmarkFusedEncoder(int num_diagrams, int features_per_diagram)
{
    if (num_diagrams < 0 || features_per_diagram < 0)
    {
        throw std::invalid_argument("benchmark dimensions must be non-negative");
    }

    std::vector<std::vector<std::pair<float, float>>> diagrams(static_cast<size_t>(num_diagrams));
    for (int i = 0; i < num_diagrams; ++i)
    {
        diagrams[static_cast<size_t>(i)].reserve(static_cast<size_t>(features_per_diagram));
        for (int j = 0; j < features_per_diagram; ++j)
        {
            const float birth = static_cast<float>(j);
            diagrams[static_cast<size_t>(i)].push_back({birth, birth + kBenchmarkLifetime});
        }
    }

    FusionConfig unfused_config;
    unfused_config.fuse_persistence = false;
    unfused_config.fuse_activation = false;
    unfused_config.fuse_normalization = false;
    FusedEncoderPipeline unfused(unfused_config);

    auto start_unfused = std::chrono::steady_clock::now();
    for (const auto &diagram : diagrams)
    {
        std::vector<float> output;
        unfused.encodeFused(diagram, output);
    }
    auto end_unfused = std::chrono::steady_clock::now();

    FusionConfig fused_config;
    FusedEncoderPipeline fused(fused_config);
    auto start_fused = std::chrono::steady_clock::now();
    for (const auto &diagram : diagrams)
    {
        std::vector<float> output;
        fused.encodeFused(diagram, output);
    }
    auto end_fused = std::chrono::steady_clock::now();

    FusionBenchmark bench{};
    bench.num_diagrams = num_diagrams;
    bench.features_per_diagram = features_per_diagram;
    bench.unfused_time_ms =
        std::chrono::duration<double, std::milli>(end_unfused - start_unfused).count();
    bench.fused_time_ms =
        std::chrono::duration<double, std::milli>(end_fused - start_fused).count();
    bench.speedup = finiteBenchmarkSpeedup(bench.unfused_time_ms, bench.fused_time_ms);
    return bench;
}

} // namespace nerve::encoders::fusion
