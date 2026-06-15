// Encoder fusion helper classes  --  AsyncEncoderExecutor and MemoryOptimizedEncoder.

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
constexpr int kCheckpointOutputSize = 256;
constexpr int kMaxCheckpointInputFeatures = 100;
constexpr int kQuantizationMaxValue = 255;

float deterministicWeight(size_t out_idx, size_t in_idx) noexcept
{
    const uint32_t a = static_cast<uint32_t>(out_idx + 1U) * 2654435761U;
    const uint32_t b = static_cast<uint32_t>(in_idx + 1U) * 2246822519U;
    const uint32_t hash = a ^ b ^ (a >> 13U);
    return 0.01f * (0.5f + static_cast<float>(hash & 0xFFU) / 255.0f);
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

} // namespace

class AsyncEncoderExecutor::Impl
{
public:
    explicit Impl(size_t batch_size)
        : batch_size_(std::max<size_t>(1, batch_size))
    {}

    void submit(const std::vector<std::pair<float, float>> &diagram)
    {
        pending_diagrams_.push_back(diagram);
    }

    std::vector<std::vector<float>> flush()
    {
        std::vector<std::vector<float>> results(pending_diagrams_.size());
        for (size_t i = 0; i < pending_diagrams_.size(); ++i)
        {
            encodeSingle(pending_diagrams_[i], results[i]);
        }
        pending_diagrams_.clear();
        return results;
    }

    std::future<std::vector<float>> submitAsync(const std::vector<std::pair<float, float>> &diagram)
    {
        return std::async(std::launch::async, [diagram]() {
            std::vector<float> output;
            encodeSingle(diagram, output);
            return output;
        });
    }

private:
    size_t batch_size_;
    std::vector<std::vector<std::pair<float, float>>> pending_diagrams_;

    static void encodeSingle(const std::vector<std::pair<float, float>> &diagram,
                             std::vector<float> &output)
    {
        output.assign(kFusedOutputSize, 0.0f);
        std::vector<float> summary(3, 0.0f);
        size_t finite_count = 0;
        for (const auto &pair : diagram)
        {
            const float lifetime = finiteLifetime(pair);
            if (lifetime <= 0.0f)
            {
                continue;
            }
            summary[0] = checkedAdd(summary[0], lifetime, "encoder async lifetime overflow");
            summary[1] = std::max(summary[1], lifetime);
            summary[2] = checkedAdd(summary[2], pair.first, "encoder async birth mean overflow");
            ++finite_count;
        }
        if (finite_count > 0)
        {
            summary[0] /= static_cast<float>(finite_count);
            summary[2] /= static_cast<float>(finite_count);
        }
        for (size_t i = 0; i < output.size(); ++i)
        {
            float pre_activation = 0.0f;
            for (size_t j = 0; j < summary.size(); ++j)
            {
                const float contribution = summary[j] * deterministicWeight(i, j);
                if (!std::isfinite(contribution))
                {
                    throw std::overflow_error("encoder async activation overflow");
                }
                pre_activation =
                    checkedAdd(pre_activation, contribution, "encoder async activation overflow");
            }
            output[i] = std::max(pre_activation, 0.0f);
        }
    }
};

AsyncEncoderExecutor::AsyncEncoderExecutor(size_t batch_size)
    : impl_(std::make_unique<Impl>(batch_size))
{}

AsyncEncoderExecutor::~AsyncEncoderExecutor() = default;

void AsyncEncoderExecutor::submit(const std::vector<std::pair<float, float>> &diagram)
{
    impl_->submit(diagram);
}

std::vector<std::vector<float>> AsyncEncoderExecutor::flush()
{
    return impl_->flush();
}

std::future<std::vector<float>>
AsyncEncoderExecutor::submitAsync(const std::vector<std::pair<float, float>> &diagram)
{
    return impl_->submitAsync(diagram);
}

class MemoryOptimizedEncoder::Impl
{
public:
    explicit Impl(const MemoryConfig &config)
        : config_(config)
    {
        if (config_.compression_bits <= 0 || config_.compression_bits > 8)
        {
            throw std::invalid_argument("compression_bits must be in [1, 8]");
        }
    }

    std::vector<float> encodeWithCheckpointing(const std::vector<float> &input)
    {
        std::vector<float> output;
        auto checkpoint = forward(input, output);
        if (config_.enable_checkpointing)
        {
            checkpoints_.push_back(std::move(checkpoint));
        }
        return output;
    }

    std::vector<float> backward(const std::vector<float> &grad_output)
    {
        if (checkpoints_.empty())
        {
            return {};
        }
        auto checkpoint = std::move(checkpoints_.back());
        checkpoints_.pop_back();

        const size_t active_inputs =
            std::min(checkpoint.input.size(), static_cast<size_t>(kMaxCheckpointInputFeatures));
        const size_t active_outputs =
            std::min(grad_output.size(), static_cast<size_t>(kCheckpointOutputSize));
        std::vector<float> grad_input(checkpoint.input.size(), 0.0f);

        for (size_t i = 0; i < active_outputs; ++i)
        {
            float pre_activation = 0.0f;
            for (size_t j = 0; j < active_inputs; ++j)
            {
                pre_activation += checkpoint.input[j] * deterministicWeight(i, j);
            }
            if (pre_activation <= 0.0f)
            {
                continue;
            }
            for (size_t j = 0; j < active_inputs; ++j)
            {
                grad_input[j] += grad_output[i] * deterministicWeight(i, j);
            }
        }
        return grad_input;
    }

    void clearCheckpoints() { checkpoints_.clear(); }

private:
    struct Checkpoint
    {
        std::vector<float> input;
        std::vector<uint8_t> compressed_activations;
    };

    MemoryConfig config_;
    std::vector<Checkpoint> checkpoints_;

    Checkpoint forward(const std::vector<float> &input, std::vector<float> &output) const
    {
        Checkpoint checkpoint;
        checkpoint.input = input;
        output.assign(kCheckpointOutputSize, 0.0f);

        const size_t active_inputs =
            std::min(input.size(), static_cast<size_t>(kMaxCheckpointInputFeatures));
        for (size_t i = 0; i < output.size(); ++i)
        {
            float sum = 0.0f;
            for (size_t j = 0; j < active_inputs; ++j)
            {
                sum += input[j] * deterministicWeight(i, j);
            }
            output[i] = std::max(sum, 0.0f);
        }
        if (config_.activation_compression)
        {
            checkpoint.compressed_activations.reserve(output.size());
            const float scale = static_cast<float>((1 << config_.compression_bits) - 1);
            for (float value : output)
            {
                checkpoint.compressed_activations.push_back(static_cast<uint8_t>(
                    std::clamp(value * scale, 0.0f, static_cast<float>(kQuantizationMaxValue))));
            }
        }
        return checkpoint;
    }
};

MemoryOptimizedEncoder::MemoryOptimizedEncoder(const MemoryConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

MemoryOptimizedEncoder::~MemoryOptimizedEncoder() = default;

std::vector<float> MemoryOptimizedEncoder::encodeWithCheckpointing(const std::vector<float> &input)
{
    return impl_->encodeWithCheckpointing(input);
}

std::vector<float> MemoryOptimizedEncoder::backward(const std::vector<float> &grad_output)
{
    return impl_->backward(grad_output);
}

void MemoryOptimizedEncoder::clearCheckpoints()
{
    impl_->clearCheckpoints();
}

} // namespace nerve::encoders::fusion
