#include "nerve/gpu/gpu_error.hpp"
#include "nerve/regularization/gpu_regularization.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nerve
{
namespace regularization
{
namespace gpu
{

constexpr int BLOCK_SIZE = 256;

__global__ __launch_bounds__(BLOCK_SIZE) void computeBettiKernel(
    const float *__restrict__ birth_times, const float *__restrict__ death_times,
    const int *__restrict__ dimensions, int num_pairs, float threshold,
    int *__restrict__ betti_numbers, int max_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs)
        return;

    float persistence = death_times[idx] - birth_times[idx];
    int dim = dimensions[idx];

    if (dim >= 0 && dim <= max_dim && persistence > threshold)
    {
        atomicAdd(&betti_numbers[dim], 1);
    }
}

__global__ __launch_bounds__(BLOCK_SIZE) void bettiDifferenceLossKernel(
    const int *__restrict__ target_betti, const int *__restrict__ predicted_betti, int max_dim,
    float *__restrict__ loss)
{
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= max_dim)
        return;

    float diff = static_cast<float>(target_betti[dim] - predicted_betti[dim]);
    atomicAdd(loss, diff * diff);
}

__global__ __launch_bounds__(BLOCK_SIZE) void persistenceRegularizationKernel(
    const float *__restrict__ persistence_values, int num_values, float target_persistence,
    float *__restrict__ regularization_term)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_values)
        return;

    float diff = persistence_values[idx] - target_persistence;
    float penalty = diff * diff;

    atomicAdd(regularization_term, penalty);
}

__global__ __launch_bounds__(BLOCK_SIZE) void topologyGradientKernel(
    const float *__restrict__ input, const float *__restrict__ grad_output,
    const int *__restrict__ pair_indices, float *__restrict__ grad_input, int num_pairs,
    int feature_dim, float regularization_strength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs)
        return;

    int i = pair_indices[idx * 2];
    int j = pair_indices[idx * 2 + 1];

    if (i >= 0 && j >= 0 && i < feature_dim && j < feature_dim)
    {
        float grad = grad_output[idx] * regularization_strength;
        atomicAdd(&grad_input[i], grad);
        atomicAdd(&grad_input[j], -grad);
    }
}

__global__ __launch_bounds__(BLOCK_SIZE) void topologicalAugmentationKernel(
    const float *__restrict__ input_data, float *__restrict__ augmented_data, int num_samples,
    int feature_dim, float noise_scale, float persistence_threshold, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples * feature_dim)
        return;

    // Random perturbation bounded by the topology threshold scale.
    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    float noise = curand_uniform(&state) * noise_scale - noise_scale / 2.0f;
    float max_delta = fmaxf(noise_scale, persistence_threshold);
    augmented_data[idx] = input_data[idx] + fminf(fmaxf(noise, -max_delta), max_delta);
}

namespace
{

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

template <typename T>
std::size_t checkedBytes(std::size_t count, const char *label)
{
    std::size_t bytes = 0;
    if (!checkedProduct(count, sizeof(T), bytes))
    {
        throw std::length_error(label);
    }
    return bytes;
}

int checkedIntSize(std::size_t value, const char *label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(label);
    }
    return static_cast<int>(value);
}

int checkedGridBlocks(std::size_t values, const char *label)
{
    const std::size_t blocks = (values / static_cast<std::size_t>(BLOCK_SIZE)) +
                               ((values % static_cast<std::size_t>(BLOCK_SIZE)) != 0 ? 1U : 0U);
    return checkedIntSize(blocks, label);
}

template <typename T>
void allocateDevice(T **ptr, std::size_t count, const char *label)
{
    GPU_CHECK(cudaMalloc(ptr, checkedBytes<T>(count, label)));
}

template <typename T>
void copyToDevice(T *dst, const std::vector<T> &src, const char *label)
{
    if (src.empty())
    {
        return;
    }
    GPU_CHECK(
        cudaMemcpy(dst, src.data(), checkedBytes<T>(src.size(), label), cudaMemcpyHostToDevice));
}

template <typename T>
void copyFromDevice(std::vector<T> &dst, const T *src, const char *label)
{
    if (dst.empty())
    {
        return;
    }
    GPU_CHECK(
        cudaMemcpy(dst.data(), src, checkedBytes<T>(dst.size(), label), cudaMemcpyDeviceToHost));
}

float readScalar(float *ptr)
{
    float value = 0.0f;
    GPU_CHECK(cudaMemcpy(&value, ptr, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

bool valuesAreFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool valuesAreFiniteAndNonNegative(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value) && value >= 0.0f; });
}

void requireFiniteValues(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFinite(values))
    {
        throw std::invalid_argument(label);
    }
}

void requireFiniteNonNegativeValues(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFiniteAndNonNegative(values))
    {
        throw std::invalid_argument(label);
    }
}

void requireFiniteOutput(const std::vector<float> &values, const char *label)
{
    if (!valuesAreFinite(values))
    {
        throw std::runtime_error(label);
    }
}

void requireFinitePersistencePairs(const std::vector<std::pair<float, float>> &pairs,
                                   const char *label)
{
    for (const auto &[birth, death] : pairs)
    {
        const float persistence = death - birth;
        if (!std::isfinite(birth) || !std::isfinite(death) || !std::isfinite(persistence) ||
            persistence < 0.0f)
        {
            throw std::invalid_argument(label);
        }
    }
}

void requireFiniteResult(float value, const char *label)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error(label);
    }
}

} // namespace

class GPUTopologyRegularizer
{
public:
    struct RegularizationConfig
    {
        float lambda_betti = 0.1f;
        float lambda_persistence = 0.05f;
        float persistence_threshold = 0.01f;
        int max_homology_dim = 3;
        bool use_augmentation = true;
        unsigned int deterministic_seed = 42U;
    };

    explicit GPUTopologyRegularizer(const RegularizationConfig &config)
        : config_(config)
        , d_target_betti_(nullptr)
        , d_predicted_betti_(nullptr)
        , d_loss_buffer_(nullptr)
        , d_regularization_(nullptr)
    {
        if (config_.max_homology_dim < 0 ||
            config_.max_homology_dim == std::numeric_limits<int>::max() ||
            config_.lambda_betti < 0.0f || config_.lambda_persistence < 0.0f ||
            config_.persistence_threshold < 0.0f || !std::isfinite(config_.lambda_betti) ||
            !std::isfinite(config_.lambda_persistence) ||
            !std::isfinite(config_.persistence_threshold))
        {
            throw std::invalid_argument("regularizer configuration is invalid");
        }
        try
        {
            const auto betti_count = static_cast<std::size_t>(config_.max_homology_dim) + 1U;
            allocateDevice(&d_target_betti_, betti_count, "target Betti buffer");
            allocateDevice(&d_predicted_betti_, betti_count, "predicted Betti buffer");
            allocateDevice(&d_loss_buffer_, 1, "loss buffer");
            allocateDevice(&d_regularization_, 1, "regularization buffer");
            GPU_CHECK(cudaMemset(d_target_betti_, 0,
                                 checkedBytes<int>(betti_count, "target Betti buffer")));
            GPU_CHECK(cudaMemset(d_predicted_betti_, 0,
                                 checkedBytes<int>(betti_count, "predicted Betti buffer")));
            GPU_CHECK(cudaMemset(d_loss_buffer_, 0, sizeof(float)));
            GPU_CHECK(cudaMemset(d_regularization_, 0, sizeof(float)));
        }
        catch (...)
        {
            releaseDeviceState();
            throw;
        }
    }

    ~GPUTopologyRegularizer() { releaseDeviceState(); }

    void setTargetTopology(const std::vector<int> &target_betti)
    {
        const size_t betti_count = static_cast<size_t>(config_.max_homology_dim) + 1U;
        const size_t copy_count = std::min(target_betti.size(), betti_count);
        for (std::size_t i = 0; i < copy_count; ++i)
        {
            if (target_betti[i] < 0)
            {
                throw std::invalid_argument("target Betti numbers must be non-negative");
            }
        }
        GPU_CHECK(
            cudaMemset(d_target_betti_, 0, checkedBytes<int>(betti_count, "target Betti buffer")));
        if (copy_count != 0)
        {
            GPU_CHECK(cudaMemcpy(d_target_betti_, target_betti.data(),
                                 checkedBytes<int>(copy_count, "target Betti copy"),
                                 cudaMemcpyHostToDevice));
        }
    }

    float computeBettiLoss(const std::vector<std::pair<float, float>> &persistence_pairs,
                           const std::vector<int> &dimensions)
    {
        if (persistence_pairs.empty())
            return 0.0f;
        if (dimensions.size() != persistence_pairs.size())
        {
            throw std::invalid_argument("dimensions must match persistence_pairs");
        }
        requireFinitePersistencePairs(
            persistence_pairs, "persistence pairs must be finite with non-negative persistence");
        for (int dimension : dimensions)
        {
            if (dimension < 0 || dimension > config_.max_homology_dim)
            {
                throw std::out_of_range(
                    "persistence pair dimension is outside the configured range");
            }
        }

        const int num_pairs =
            checkedIntSize(persistence_pairs.size(), "persistence pair count exceeds CUDA limits");
        float *d_birth = nullptr;
        float *d_death = nullptr;
        int *d_dims = nullptr;
        try
        {
            std::vector<float> births;
            std::vector<float> deaths;
            births.reserve(persistence_pairs.size());
            deaths.reserve(persistence_pairs.size());
            for (const auto &p : persistence_pairs)
            {
                births.push_back(p.first);
                deaths.push_back(p.second);
            }

            allocateDevice(&d_birth, births.size(), "birth values");
            allocateDevice(&d_death, deaths.size(), "death values");
            allocateDevice(&d_dims, dimensions.size(), "persistence dimensions");
            copyToDevice(d_birth, births, "birth values");
            copyToDevice(d_death, deaths, "death values");
            copyToDevice(d_dims, dimensions, "persistence dimensions");

            const auto betti_count = static_cast<std::size_t>(config_.max_homology_dim) + 1U;
            GPU_CHECK(cudaMemset(d_predicted_betti_, 0,
                                 checkedBytes<int>(betti_count, "predicted Betti buffer")));

            const int blocks = checkedGridBlocks(persistence_pairs.size(), "Betti pair grid");
            computeBettiKernel<<<blocks, BLOCK_SIZE>>>(
                d_birth, d_death, d_dims, num_pairs, config_.persistence_threshold,
                d_predicted_betti_, config_.max_homology_dim);
            GPU_CHECK(cudaPeekAtLastError());

            GPU_CHECK(cudaMemset(d_loss_buffer_, 0, sizeof(float)));
            const int betti_blocks = checkedGridBlocks(betti_count, "Betti loss grid");
            bettiDifferenceLossKernel<<<betti_blocks, BLOCK_SIZE>>>(
                d_target_betti_, d_predicted_betti_, config_.max_homology_dim + 1, d_loss_buffer_);
            GPU_CHECK(cudaPeekAtLastError());

            const float loss = readScalar(d_loss_buffer_);
            requireFiniteResult(loss, "Betti regularization produced non-finite loss");
            const float scaled_loss = loss * config_.lambda_betti;
            requireFiniteResult(scaled_loss, "Betti regularization produced non-finite loss");
            cudaFree(d_birth);
            cudaFree(d_death);
            cudaFree(d_dims);
            return scaled_loss;
        }
        catch (...)
        {
            cudaFree(d_birth);
            cudaFree(d_death);
            cudaFree(d_dims);
            throw;
        }
    }

    float computePersistenceLoss(const std::vector<float> &persistence_values,
                                 float target_persistence)
    {
        if (persistence_values.empty())
            return 0.0f;
        requireFiniteNonNegativeValues(persistence_values,
                                       "persistence values must be finite and non-negative");
        if (!std::isfinite(target_persistence) || target_persistence < 0.0f)
        {
            throw std::invalid_argument("target persistence must be finite and non-negative");
        }
        const int n = checkedIntSize(persistence_values.size(),
                                     "persistence value count exceeds CUDA limits");

        float *d_persistence = nullptr;
        try
        {
            allocateDevice(&d_persistence, persistence_values.size(), "persistence values");
            copyToDevice(d_persistence, persistence_values, "persistence values");
            GPU_CHECK(cudaMemset(d_regularization_, 0, sizeof(float)));

            const int blocks = checkedGridBlocks(persistence_values.size(), "persistence grid");
            persistenceRegularizationKernel<<<blocks, BLOCK_SIZE>>>(
                d_persistence, n, target_persistence, d_regularization_);
            GPU_CHECK(cudaPeekAtLastError());

            const float reg = readScalar(d_regularization_);
            requireFiniteResult(reg, "persistence regularization produced non-finite loss");
            const float scaled_loss = reg * config_.lambda_persistence / static_cast<float>(n);
            requireFiniteResult(scaled_loss, "persistence regularization produced non-finite loss");
            cudaFree(d_persistence);
            return scaled_loss;
        }
        catch (...)
        {
            cudaFree(d_persistence);
            throw;
        }
    }

    std::vector<float> augmentData(const std::vector<float> &data, int num_augmentations = 1,
                                   float noise_scale = 0.1f)
    {
        if (!config_.use_augmentation || data.empty())
        {
            return data;
        }
        requireFiniteValues(data, "regularizer augmentation input must be finite");
        if (!std::isfinite(noise_scale) || noise_scale < 0.0f)
        {
            throw std::invalid_argument("augmentation noise scale must be finite and non-negative");
        }

        const int num_samples =
            checkedIntSize(data.size(), "augmentation sample count exceeds CUDA limits");
        const int augmentations = num_augmentations > 0 ? num_augmentations : 1;
        std::size_t output_values = 0;
        if (!checkedProduct(data.size(), static_cast<std::size_t>(augmentations), output_values))
        {
            throw std::length_error("augmentation output size overflows");
        }

        float *d_input = nullptr;
        float *d_output = nullptr;
        try
        {
            allocateDevice(&d_input, data.size(), "augmentation input");
            allocateDevice(&d_output, data.size(), "augmentation output");
            copyToDevice(d_input, data, "augmentation input");

            const int blocks = checkedGridBlocks(data.size(), "augmentation grid");
            std::vector<float> augmented(output_values);
            for (int aug = 0; aug < augmentations; ++aug)
            {
                topologicalAugmentationKernel<<<blocks, BLOCK_SIZE>>>(
                    d_input, d_output, num_samples, 1, noise_scale, config_.persistence_threshold,
                    config_.deterministic_seed + static_cast<unsigned int>(aug));
                GPU_CHECK(cudaPeekAtLastError());
                GPU_CHECK(cudaMemcpy(augmented.data() + static_cast<size_t>(aug) * data.size(),
                                     d_output,
                                     checkedBytes<float>(data.size(), "augmentation output"),
                                     cudaMemcpyDeviceToHost));
            }

            cudaFree(d_input);
            cudaFree(d_output);
            requireFiniteOutput(augmented, "regularizer augmentation produced non-finite output");
            return augmented;
        }
        catch (...)
        {
            cudaFree(d_input);
            cudaFree(d_output);
            throw;
        }
    }

    std::vector<float>
    computeGradient(const std::vector<float> &input,
                    const std::vector<std::pair<float, float>> &persistence_pairs,
                    const std::vector<int> &pair_indices)
    {
        const int num_pairs =
            checkedIntSize(persistence_pairs.size(), "gradient pair count exceeds CUDA limits");
        const int feature_dim =
            checkedIntSize(input.size(), "gradient feature dimension exceeds CUDA limits");
        if (num_pairs == 0 || feature_dim == 0)
        {
            return std::vector<float>(static_cast<std::size_t>(feature_dim), 0.0f);
        }
        requireFiniteValues(input, "gradient input must be finite");
        requireFinitePersistencePairs(persistence_pairs,
                                      "gradient persistence pairs must be finite");
        std::size_t required_indices = 0;
        if (!checkedProduct(static_cast<std::size_t>(num_pairs), 2U, required_indices))
        {
            throw std::length_error("gradient pair index count overflows");
        }
        if (pair_indices.size() < required_indices)
        {
            throw std::invalid_argument("pair_indices must contain two entries per pair");
        }
        for (std::size_t index = 0; index < required_indices; ++index)
        {
            if (pair_indices[index] < 0 || pair_indices[index] >= feature_dim)
            {
                throw std::out_of_range("pair index is outside gradient input bounds");
            }
        }

        float *d_input = nullptr;
        float *d_grad_output = nullptr;
        float *d_grad_input = nullptr;
        int *d_indices = nullptr;
        try
        {
            allocateDevice(&d_input, input.size(), "gradient input");
            allocateDevice(&d_grad_output, persistence_pairs.size(), "gradient output weights");
            allocateDevice(&d_grad_input, input.size(), "gradient input output");
            allocateDevice(&d_indices, required_indices, "gradient pair indices");

            copyToDevice(d_input, input, "gradient input");
            GPU_CHECK(cudaMemcpy(d_indices, pair_indices.data(),
                                 checkedBytes<int>(required_indices, "gradient pair indices"),
                                 cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemset(d_grad_input, 0,
                                 checkedBytes<float>(input.size(), "gradient input output")));

            std::vector<float> ones(persistence_pairs.size(), 1.0f);
            copyToDevice(d_grad_output, ones, "gradient output weights");

            const int blocks = checkedGridBlocks(persistence_pairs.size(), "gradient pair grid");
            topologyGradientKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_grad_output, d_indices,
                                                           d_grad_input, num_pairs, feature_dim,
                                                           config_.lambda_betti);
            GPU_CHECK(cudaPeekAtLastError());

            std::vector<float> gradient(input.size());
            copyFromDevice(gradient, d_grad_input, "gradient result");
            requireFiniteOutput(gradient, "regularizer gradient produced non-finite output");
            cudaFree(d_input);
            cudaFree(d_grad_output);
            cudaFree(d_grad_input);
            cudaFree(d_indices);
            return gradient;
        }
        catch (...)
        {
            cudaFree(d_input);
            cudaFree(d_grad_output);
            cudaFree(d_grad_input);
            cudaFree(d_indices);
            throw;
        }
    }

private:
    void releaseDeviceState() noexcept
    {
        cudaFree(d_target_betti_);
        cudaFree(d_predicted_betti_);
        cudaFree(d_loss_buffer_);
        cudaFree(d_regularization_);
        d_target_betti_ = nullptr;
        d_predicted_betti_ = nullptr;
        d_loss_buffer_ = nullptr;
        d_regularization_ = nullptr;
    }

    RegularizationConfig config_;

    int *d_target_betti_;
    int *d_predicted_betti_;
    float *d_loss_buffer_;
    float *d_regularization_;
};

RegularizerBenchmark benchmarkRegularizer(int num_pairs, int feature_dim)
{
    if (num_pairs < 0 || feature_dim < 0)
    {
        throw std::invalid_argument("benchmarkRegularizer sizes must be non-negative");
    }
    RegularizerBenchmark bench;
    bench.num_pairs = num_pairs;
    bench.feature_dim = feature_dim;

    GPUTopologyRegularizer::RegularizationConfig config;
    config.max_homology_dim = 3;
    GPUTopologyRegularizer regularizer(config);

    std::vector<std::pair<float, float>> pairs;
    std::vector<int> dims;
    for (int i = 0; i < num_pairs; ++i)
    {
        const float birth = static_cast<float>((i * 13) % 100) * 0.01f;
        const float lifetime = 0.05f + static_cast<float>((i * 7) % 30) * 0.01f;
        pairs.emplace_back(birth, birth + lifetime);
        dims.push_back(i % (config.max_homology_dim + 1));
    }

    std::vector<int> target_betti = {10, 5, 2, 1};
    regularizer.setTargetTopology(target_betti);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<int> predicted_betti(config.max_homology_dim + 1, 0);
    for (int i = 0; i < num_pairs; ++i)
    {
        const int dim = dims[i];
        const float persistence = pairs[i].second - pairs[i].first;
        if (dim >= 0 && dim <= config.max_homology_dim &&
            persistence > config.persistence_threshold)
        {
            ++predicted_betti[dim];
        }
    }
    float cpu_loss = 0.0f;
    for (int dim = 0; dim <= config.max_homology_dim; ++dim)
    {
        const float diff = static_cast<float>(target_betti[dim] - predicted_betti[dim]);
        cpu_loss += diff * diff;
    }
    cpu_loss *= config.lambda_betti;
    volatile float cpu_loss_sink = cpu_loss;
    (void)cpu_loss_sink;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_betti_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    auto start = std::chrono::high_resolution_clock::now();
    float loss = regularizer.computeBettiLoss(pairs, dims);
    auto end = std::chrono::high_resolution_clock::now();
    bench.gpu_betti_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::vector<float> data(feature_dim);
    for (int i = 0; i < feature_dim; ++i)
    {
        data[i] = static_cast<float>((i * 19) % 113) / 113.0f;
    }

    auto start_cpu_aug = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_augmented(feature_dim);
    for (int i = 0; i < feature_dim; ++i)
    {
        const float noise_unit = static_cast<float>((i * 37 + 11) % 101) / 101.0f;
        cpu_augmented[i] = data[i] + (noise_unit * 0.1f - 0.05f);
    }
    volatile float cpu_augmented_sink = cpu_augmented.empty() ? 0.0f : cpu_augmented.back();
    (void)cpu_augmented_sink;
    auto end_cpu_aug = std::chrono::high_resolution_clock::now();
    bench.cpu_augment_ms =
        std::chrono::duration<double, std::milli>(end_cpu_aug - start_cpu_aug).count();

    auto start_aug = std::chrono::high_resolution_clock::now();
    auto augmented = regularizer.augmentData(data);
    auto end_aug = std::chrono::high_resolution_clock::now();
    bench.gpu_augment_ms = std::chrono::duration<double, std::milli>(end_aug - start_aug).count();

    bench.speedup_betti = bench.gpu_betti_ms > 0.0 ? bench.cpu_betti_ms / bench.gpu_betti_ms : 0.0;
    bench.speedup_augment =
        bench.gpu_augment_ms > 0.0 ? bench.cpu_augment_ms / bench.gpu_augment_ms : 0.0;

    return bench;
}

} // namespace gpu
} // namespace regularization
} // namespace nerve
