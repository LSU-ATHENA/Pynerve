#include "nerve/probabilistic/gpu_probabilistic.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve
{
namespace probabilistic
{
namespace gpu
{

using Point = std::vector<float>;

constexpr int BLOCK_SIZE = 256;

void throwCudaLaunchError(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
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

std::size_t checkedBytes(std::size_t count, std::size_t element_size, const char *label)
{
    return checkedMul(count, element_size, label);
}

int checkedIntSize(std::size_t value, const char *label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA int range");
    }
    return static_cast<int>(value);
}

int checkedGridBlocks(std::size_t count, const char *label)
{
    if (count == 0)
    {
        return 0;
    }
    const std::size_t blocks =
        (count + static_cast<std::size_t>(BLOCK_SIZE) - 1) / static_cast<std::size_t>(BLOCK_SIZE);
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds CUDA grid range");
    }
    return static_cast<int>(blocks);
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

void requireFinitePointCloud(const std::vector<Point> &points, const char *label)
{
    for (const auto &point : points)
    {
        if (!valuesAreFinite(point))
        {
            throw std::invalid_argument(label);
        }
    }
}

template <typename T>
void allocateDevice(T **ptr, std::size_t count, const char *label)
{
    *ptr = nullptr;
    if (count == 0)
    {
        return;
    }
    throwCudaLaunchError(
        cudaMalloc(reinterpret_cast<void **>(ptr), checkedBytes(count, sizeof(T), label)), label);
}

template <typename T>
void copyToDevice(T *dst, const T *src, std::size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    throwCudaLaunchError(
        cudaMemcpy(dst, src, checkedBytes(count, sizeof(T), label), cudaMemcpyHostToDevice), label);
}

template <typename T>
void copyToHost(T *dst, const T *src, std::size_t count, const char *label)
{
    if (count == 0)
    {
        return;
    }
    throwCudaLaunchError(
        cudaMemcpy(dst, src, checkedBytes(count, sizeof(T), label), cudaMemcpyDeviceToHost), label);
}

__global__ __launch_bounds__(BLOCK_SIZE) void randomSubsamplingKernel(
    const int *__restrict__ input_indices, int *__restrict__ output_indices,
    int *__restrict__ output_count, int n, int max_output, float sample_rate, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float rand_val = curand_uniform(&state);

    if (rand_val < sample_rate)
    {
        int out_idx = atomicAdd(output_count, 1);
        if (out_idx < max_output)
        {
            output_indices[out_idx] = input_indices[idx];
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE) void approximatePersistenceKernel(
    const float *__restrict__ distance_matrix, float *__restrict__ persistence_pairs,
    int *__restrict__ pair_count, int n, int max_pairs, float epsilon, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n * (n - 1) / 2)
        return;

    int i = 0, j = 0;
    int remaining = idx;

    for (int k = n - 1; k > 0; --k)
    {
        if (remaining < k)
        {
            j = i + 1 + remaining;
            break;
        }
        remaining -= k;
        i++;
    }

    if (i >= n || j >= n || i >= j)
        return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float dist = distance_matrix[i * n + j];
    if (!isfinite(dist) || dist < 0.0f)
        return;
    float prob = 1.0f - expf(-dist / epsilon);

    if (curand_uniform(&state) < prob)
    {
        int pair_idx = atomicAdd(pair_count, 1);
        if (pair_idx < max_pairs)
        {
            const float death = dist * (1.0f + epsilon);
            if (!isfinite(death) || death < dist)
                return;
            persistence_pairs[2 * pair_idx] = dist;
            persistence_pairs[2 * pair_idx + 1] = death;
        }
    }
}

class GPUProbabilisticSampler
{
public:
    GPUProbabilisticSampler(int max_samples, float confidence = 0.95f)
        : max_samples_(max_samples)
    {
        if (max_samples_ <= 0 || !std::isfinite(confidence) || confidence <= 0.0f ||
            confidence >= 1.0f)
        {
            throw std::invalid_argument("probabilistic sampler configuration is invalid");
        }
        try
        {
            allocateDevice(&d_sample_buffer_, static_cast<std::size_t>(max_samples_),
                           "allocate probabilistic sample buffer");
            allocateDevice(&d_sample_count_, 1, "allocate probabilistic sample count");
            allocateDevice(&d_persistence_pairs_,
                           checkedMul(static_cast<std::size_t>(2),
                                      static_cast<std::size_t>(max_samples_),
                                      "probabilistic persistence pair capacity"),
                           "allocate probabilistic persistence pairs");
            allocateDevice(&d_pair_count_, 1, "allocate probabilistic pair count");
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUProbabilisticSampler() { cleanup(); }

    std::vector<int> subsample(const std::vector<int> &indices, float sample_rate)
    {
        if (!std::isfinite(sample_rate) || sample_rate < 0.0f || sample_rate > 1.0f)
        {
            throw std::invalid_argument("sample_rate must be finite and in [0, 1]");
        }
        if (indices.empty())
        {
            return {};
        }
        const int n = checkedIntSize(indices.size(), "probabilistic input index count");
        int *d_indices = nullptr;
        try
        {
            allocateDevice(&d_indices, indices.size(), "allocate probabilistic input indices");
            copyToDevice(d_indices, indices.data(), indices.size(),
                         "copy probabilistic input indices");

            throwCudaLaunchError(cudaMemset(d_sample_count_, 0, sizeof(int)),
                                 "reset probabilistic sample count");

            const int blocks = checkedGridBlocks(indices.size(), "probabilistic subsample grid");
            randomSubsamplingKernel<<<blocks, BLOCK_SIZE>>>(
                d_indices, d_sample_buffer_, d_sample_count_, n, max_samples_, sample_rate, 42u);
            throwCudaLaunchError(cudaPeekAtLastError(), "randomSubsamplingKernel launch failed");
            throwCudaLaunchError(cudaDeviceSynchronize(),
                                 "randomSubsamplingKernel execution failed");

            int count = 0;
            copyToHost(&count, d_sample_count_, 1, "copy probabilistic sample count");
            count = std::clamp(count, 0, max_samples_);

            std::vector<int> result(static_cast<std::size_t>(count));
            copyToHost(result.data(), d_sample_buffer_, result.size(),
                       "copy probabilistic sampled indices");

            cudaFree(d_indices);
            return result;
        }
        catch (...)
        {
            if (d_indices)
                cudaFree(d_indices);
            throw;
        }
    }

    std::vector<std::pair<float, float>>
    approximatePersistence(const std::vector<float> &distance_matrix, int n, float epsilon)
    {
        if (n <= 1 || !std::isfinite(epsilon) || epsilon <= 0.0f)
        {
            throw std::invalid_argument("distance matrix shape or epsilon is invalid");
        }
        const std::size_t n_size = static_cast<std::size_t>(n);
        const std::size_t matrix_count =
            checkedMul(n_size, n_size, "probabilistic distance matrix count");
        if (matrix_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            distance_matrix.size() != matrix_count)
        {
            throw std::invalid_argument("distance matrix shape or epsilon is invalid");
        }
        const std::size_t pair_count_size =
            checkedMul(n_size, n_size - 1, "probabilistic pair count") / 2;
        const int num_pairs = checkedIntSize(pair_count_size, "probabilistic pair count");
        requireFiniteNonNegativeValues(distance_matrix,
                                       "distance matrix must be finite and non-negative");

        float *d_dist = nullptr;
        try
        {
            allocateDevice(&d_dist, distance_matrix.size(), "allocate probabilistic distances");
            copyToDevice(d_dist, distance_matrix.data(), distance_matrix.size(),
                         "copy probabilistic distances");

            throwCudaLaunchError(cudaMemset(d_pair_count_, 0, sizeof(int)),
                                 "reset probabilistic pair count");
            const int blocks = checkedGridBlocks(pair_count_size, "probabilistic persistence grid");

            approximatePersistenceKernel<<<blocks, BLOCK_SIZE>>>(
                d_dist, d_persistence_pairs_, d_pair_count_, n, max_samples_, epsilon, 1337u);
            throwCudaLaunchError(cudaPeekAtLastError(),
                                 "approximatePersistenceKernel launch failed");
            throwCudaLaunchError(cudaDeviceSynchronize(),
                                 "approximatePersistenceKernel execution failed");

            int count = 0;
            copyToHost(&count, d_pair_count_, 1, "copy probabilistic pair count");
            count = std::clamp(count, 0, max_samples_);

            std::vector<float> pairs_data(checkedMul(static_cast<std::size_t>(2),
                                                     static_cast<std::size_t>(count),
                                                     "probabilistic output pair floats"));
            copyToHost(pairs_data.data(), d_persistence_pairs_, pairs_data.size(),
                       "copy probabilistic pairs");

            std::vector<std::pair<float, float>> result;
            result.reserve(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i)
            {
                const float birth = pairs_data[static_cast<std::size_t>(2 * i)];
                const float death = pairs_data[static_cast<std::size_t>(2 * i + 1)];
                if (!std::isfinite(birth) || !std::isfinite(death) || death < birth)
                {
                    throw std::runtime_error("probabilistic persistence produced invalid pair");
                }
                result.push_back({birth, death});
            }

            cudaFree(d_dist);
            return result;
        }
        catch (...)
        {
            if (d_dist)
                cudaFree(d_dist);
            throw;
        }
    }

    std::pair<float, float> confidenceBounds(const std::vector<float> &samples,
                                             float confidence_level)
    {
        if (samples.empty())
        {
            return {0.0f, 0.0f};
        }
        requireFiniteValues(samples, "confidence bound samples must be finite");
        if (!std::isfinite(confidence_level))
        {
            throw std::invalid_argument("confidence level must be finite");
        }
        std::vector<float> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const float clamped_confidence = std::max(0.0f, std::min(confidence_level, 1.0f));
        const float alpha = (1.0f - clamped_confidence) / 2.0f;
        const std::size_t n = sorted.size();
        const std::size_t lower_idx = std::min(n - 1, static_cast<std::size_t>(alpha * n));
        const std::size_t upper_idx = std::min(n - 1, static_cast<std::size_t>((1.0f - alpha) * n));
        return {sorted[lower_idx], sorted[upper_idx]};
    }

private:
    void cleanup() noexcept
    {
        if (d_sample_buffer_)
            cudaFree(d_sample_buffer_);
        if (d_sample_count_)
            cudaFree(d_sample_count_);
        if (d_persistence_pairs_)
            cudaFree(d_persistence_pairs_);
        if (d_pair_count_)
            cudaFree(d_pair_count_);
        d_sample_buffer_ = nullptr;
        d_sample_count_ = nullptr;
        d_persistence_pairs_ = nullptr;
        d_pair_count_ = nullptr;
    }

    int max_samples_ = 0;
    int *d_sample_buffer_ = nullptr;
    int *d_sample_count_ = nullptr;
    float *d_persistence_pairs_ = nullptr;
    int *d_pair_count_ = nullptr;
};

class MonteCarloPersistence
{
public:
    struct Config
    {
        int num_iterations = 1000;
        float sample_rate = 0.1f;
        int seed = 42;
    };

    explicit MonteCarloPersistence(const Config &config)
        : config_(config)
    {
        if (config_.num_iterations < 0 || !std::isfinite(config_.sample_rate) ||
            config_.sample_rate < 0.0f || config_.sample_rate > 1.0f)
        {
            throw std::invalid_argument("Monte Carlo persistence configuration is invalid");
        }
    }

    float estimatePersistence(const std::vector<Point> &points, int dimension)
    {
        if (points.empty() || config_.num_iterations <= 0 || config_.sample_rate <= 0.0f)
        {
            return 0.0f;
        }
        requireFinitePointCloud(points, "Monte Carlo point cloud must be finite");
        if (points.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::length_error("Monte Carlo point count exceeds int range");
        }

        float total_persistence = 0.0f;
        std::mt19937 rng(config_.seed);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(points.size()) - 1);

        for (int iter = 0; iter < config_.num_iterations; ++iter)
        {
            std::vector<Point> subsample;
            const float bounded_rate = std::min(config_.sample_rate, 1.0f);
            const float requested_size = static_cast<float>(points.size()) * bounded_rate;
            if (requested_size > static_cast<float>(std::numeric_limits<int>::max()))
            {
                throw std::length_error("Monte Carlo sample size exceeds int range");
            }
            int sample_size = static_cast<int>(requested_size);
            subsample.reserve(static_cast<std::size_t>(sample_size));

            for (int i = 0; i < sample_size; ++i)
            {
                subsample.push_back(points[static_cast<std::size_t>(dist(rng))]);
            }

            float persistence = computeSamplePersistence(subsample, dimension);
            const float next = total_persistence + persistence;
            if (!std::isfinite(persistence) || !std::isfinite(next))
            {
                throw std::runtime_error("Monte Carlo persistence produced non-finite output");
            }
            total_persistence = next;
        }

        const float estimate = total_persistence / config_.num_iterations;
        if (!std::isfinite(estimate))
        {
            throw std::runtime_error("Monte Carlo persistence produced non-finite output");
        }
        return estimate;
    }

private:
    Config config_;

    static float pointDistance(const Point &a, const Point &b)
    {
        const size_t dims = std::min(a.size(), b.size());
        if (dims == 0)
        {
            return 0.0f;
        }
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dims; ++d)
        {
            const float diff = a[d] - b[d];
            const float contribution = diff * diff;
            const float next = sum_sq + contribution;
            if (!std::isfinite(diff) || !std::isfinite(contribution) || !std::isfinite(next))
            {
                return std::numeric_limits<float>::infinity();
            }
            sum_sq = next;
        }
        return std::sqrt(sum_sq);
    }

    float computeSamplePersistence(const std::vector<Point> &sample, int dim)
    {
        if (sample.size() <= 1 || dim < 0)
        {
            return 0.0f;
        }
        if (sample.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::length_error("sample size exceeds int range");
        }

        const int n = static_cast<int>(sample.size());
        std::vector<float> best_edge(static_cast<std::size_t>(n),
                                     std::numeric_limits<float>::infinity());
        std::vector<char> in_tree(static_cast<std::size_t>(n), 0);
        best_edge[0] = 0.0f;

        float total_lifetime = 0.0f;
        for (int iter = 0; iter < n; ++iter)
        {
            int u = -1;
            float best = std::numeric_limits<float>::infinity();
            for (int i = 0; i < n; ++i)
            {
                if (!in_tree[static_cast<std::size_t>(i)] &&
                    best_edge[static_cast<std::size_t>(i)] < best)
                {
                    best = best_edge[static_cast<std::size_t>(i)];
                    u = i;
                }
            }
            if (u < 0)
            {
                break;
            }
            in_tree[static_cast<std::size_t>(u)] = 1;
            total_lifetime += (iter == 0) ? 0.0f : best;

            for (int v = 0; v < n; ++v)
            {
                if (in_tree[static_cast<std::size_t>(v)])
                {
                    continue;
                }
                const float dist = pointDistance(sample[static_cast<std::size_t>(u)],
                                                 sample[static_cast<std::size_t>(v)]);
                if (!std::isfinite(dist))
                {
                    throw std::runtime_error("sample persistence distance overflow");
                }
                if (dist < best_edge[static_cast<std::size_t>(v)])
                {
                    best_edge[static_cast<std::size_t>(v)] = dist;
                }
            }
        }

        return total_lifetime / static_cast<float>(std::max(1, n - 1));
    }
};

class StreamingProbabilisticPersistence
{
public:
    StreamingProbabilisticPersistence(float decay_factor = 0.9f)
        : decay_factor_(decay_factor)
    {}

    void update(const std::vector<std::pair<float, float>> &new_pairs)
    {
        for (auto &pair : accumulated_pairs_)
        {
            pair.second *= decay_factor_;
        }

        for (const auto &[birth, death] : new_pairs)
        {
            accumulated_pairs_.push_back({birth, death});
        }
    }

    std::vector<std::pair<float, float>> getEstimate() const { return accumulated_pairs_; }

private:
    float decay_factor_;
    std::vector<std::pair<float, float>> accumulated_pairs_;
};

ProbabilisticBenchmark benchmarkProbabilistic(int num_samples)
{
    if (num_samples <= 0)
    {
        throw std::invalid_argument("num_samples must be positive");
    }
    ProbabilisticBenchmark bench;
    bench.num_samples = num_samples;

    std::vector<int> indices(static_cast<std::size_t>(num_samples));
    for (int i = 0; i < num_samples; ++i)
        indices[static_cast<std::size_t>(i)] = i;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<int> cpu_result;
    std::mt19937 rng(42);
    std::bernoulli_distribution keep(0.5);
    for (int i = 0; i < num_samples; ++i)
    {
        if (keep(rng))
        {
            cpu_result.push_back(i);
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    GPUProbabilisticSampler sampler(num_samples);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_result = sampler.subsample(indices, 0.5f);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    bench.speedup = finiteBenchmarkSpeedup(bench.cpu_time_ms, bench.gpu_time_ms);
    const float target_rate = 0.5f;
    const float actual_rate =
        num_samples > 0 ? static_cast<float>(gpu_result.size()) / static_cast<float>(num_samples)
                        : target_rate;
    bench.accuracy = 1.0f - std::min(1.0f, std::fabs(actual_rate - target_rate) / target_rate);

    return bench;
}

} // namespace gpu
} // namespace probabilistic
} // namespace nerve
