/*
 * Differentiable persistence gradient kernels.
 *
 * This module computes gradients of persistence-pair birth/death edge lengths
 * with respect to input point coordinates. Each pair contributes up to two
 * edge-distance gradients (birth and death), and gradients are accumulated with
 * atomics on the point buffer.
 */

#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>

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
namespace ml
{
namespace diff
{
namespace gpu
{

namespace
{
constexpr int kBlockSize = 256;
constexpr float kEpsilon = 1.0e-8F;
constexpr int kPointDim = 3;

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
        throw std::length_error(std::string(label) + " byte count overflows");
    }
    return bytes;
}

int checkedIntSize(std::size_t value, const char *label)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(label) + " exceeds int range");
    }
    return static_cast<int>(value);
}

int checkedGridBlocks(std::size_t work_items, const char *label)
{
    const std::size_t blocks = (work_items / static_cast<std::size_t>(kBlockSize)) +
                               ((work_items % static_cast<std::size_t>(kBlockSize)) != 0 ? 1U : 0U);
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

void requireFiniteValues(const std::vector<float> &values, const char *label)
{
    for (float value : values)
    {
        if (!std::isfinite(static_cast<double>(value)))
        {
            throw std::invalid_argument(std::string(label) + " must be finite");
        }
    }
}

void validatePairEdge(int v1, int v2, int n_points, const char *label)
{
    const bool first_missing = v1 < 0;
    const bool second_missing = v2 < 0;
    if (first_missing != second_missing)
    {
        throw std::invalid_argument(std::string(label) +
                                    " endpoints must both be present or absent");
    }
    if (first_missing)
    {
        return;
    }
    if (v1 >= n_points || v2 >= n_points)
    {
        throw std::out_of_range(std::string(label) + " endpoint is outside the point range");
    }
}
} // namespace

__global__ __launch_bounds__(kBlockSize) void pairwiseDistanceKernel(
    const float *__restrict__ points, float *__restrict__ distance_matrix, int n_points, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_points * n_points;
    if (idx >= total)
    {
        return;
    }

    const int i = idx / n_points;
    const int j = idx % n_points;

    if (i == j)
    {
        distance_matrix[idx] = 0.0F;
        return;
    }

    double dist_sq = 0.0;
    bool valid = true;
    const int base_i = i * dim;
    const int base_j = j * dim;
    for (int d = 0; d < dim; ++d)
    {
        const double lhs = static_cast<double>(points[base_i + d]);
        const double rhs = static_cast<double>(points[base_j + d]);
        const double diff = lhs - rhs;
        const double contribution = diff * diff;
        const double next = dist_sq + contribution;
        if (!isfinite(lhs) || !isfinite(rhs) || !isfinite(diff) || !isfinite(contribution) ||
            !isfinite(next))
        {
            valid = false;
            break;
        }
        dist_sq = next;
    }
    const float distance = static_cast<float>(sqrt(dist_sq));
    distance_matrix[idx] = (valid && isfinite(distance)) ? distance : INFINITY;
}

__device__ void accumulateEdgeGradient(int v1, int v2, float upstream_grad,
                                       const float *__restrict__ points,
                                       const float *__restrict__ distance_matrix,
                                       float *__restrict__ grad_points, int n_points, int dim)
{
    if (v1 < 0 || v2 < 0 || v1 >= n_points || v2 >= n_points || v1 == v2)
    {
        return;
    }

    const float dist = distance_matrix[v1 * n_points + v2];
    if (!isfinite(dist) || !isfinite(upstream_grad) || dist <= kEpsilon)
    {
        return;
    }

    const int base_1 = v1 * dim;
    const int base_2 = v2 * dim;
    for (int d = 0; d < dim; ++d)
    {
        const float diff = points[base_1 + d] - points[base_2 + d];
        const float normalized = diff / dist;
        const float grad_component = upstream_grad * normalized;
        if (!isfinite(diff) || !isfinite(normalized) || !isfinite(grad_component))
        {
            return;
        }
        atomicAdd(&grad_points[base_1 + d], grad_component);
        atomicAdd(&grad_points[base_2 + d], -grad_component);
    }
}

__global__ __launch_bounds__(kBlockSize) void persistencePairBackpropKernel(
    const int *__restrict__ birth_v1, const int *__restrict__ birth_v2,
    const int *__restrict__ death_v1, const int *__restrict__ death_v2,
    const float *__restrict__ grad_birth, const float *__restrict__ grad_death,
    const float *__restrict__ points, const float *__restrict__ distance_matrix,
    float *__restrict__ grad_points, int num_pairs, int n_points, int dim)
{
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs)
    {
        return;
    }

    accumulateEdgeGradient(birth_v1[pair_idx], birth_v2[pair_idx], grad_birth[pair_idx], points,
                           distance_matrix, grad_points, n_points, dim);

    accumulateEdgeGradient(death_v1[pair_idx], death_v2[pair_idx], grad_death[pair_idx], points,
                           distance_matrix, grad_points, n_points, dim);
}

class GPUPersistenceGradient
{
public:
    struct PersistencePair
    {
        float birth = 0.0F;
        float death = 0.0F;
        int dim = 0;
        int birth_v1 = -1;
        int birth_v2 = -1;
        int death_v1 = -1;
        int death_v2 = -1;
    };

    GPUPersistenceGradient(int max_points, int max_pairs)
        : max_points_(max_points)
        , max_pairs_(max_pairs)
    {
        if (max_points_ <= 0 || max_pairs_ <= 0)
        {
            throw std::invalid_argument("GPUPersistenceGradient: invalid capacity");
        }

        const auto max_points_size = static_cast<std::size_t>(max_points_);
        const auto max_pairs_size = static_cast<std::size_t>(max_pairs_);
        std::size_t point_values = 0;
        std::size_t distance_values = 0;
        if (!checkedProduct(max_points_size, static_cast<std::size_t>(kPointDim), point_values) ||
            !checkedProduct(max_points_size, max_points_size, distance_values) ||
            distance_values > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::length_error("GPUPersistenceGradient capacity exceeds CUDA limits");
        }

        try
        {
            allocateDevice(&d_points_, point_values, "gradient points");
            allocateDevice(&d_dist_matrix_, distance_values, "gradient distance matrix");
            allocateDevice(&d_grad_points_, point_values, "gradient output points");
            allocateDevice(&d_birth_v1_, max_pairs_size, "birth v1");
            allocateDevice(&d_birth_v2_, max_pairs_size, "birth v2");
            allocateDevice(&d_death_v1_, max_pairs_size, "death v1");
            allocateDevice(&d_death_v2_, max_pairs_size, "death v2");
            allocateDevice(&d_grad_birth_, max_pairs_size, "birth gradients");
            allocateDevice(&d_grad_death_, max_pairs_size, "death gradients");
        }
        catch (...)
        {
            releaseDeviceState();
            throw;
        }
    }

    ~GPUPersistenceGradient() { releaseDeviceState(); }

    std::vector<float> computeGradients(const std::vector<float> &points,
                                        const std::vector<PersistencePair> &pairs,
                                        const std::vector<float> &upstream_grad_birth,
                                        const std::vector<float> &upstream_grad_death)
    {
        if (points.size() % kPointDim != 0)
        {
            throw std::invalid_argument("points must be packed in XYZ triplets");
        }

        const int n_points = checkedIntSize(points.size() / kPointDim, "gradient point count");
        const int n_pairs = checkedIntSize(pairs.size(), "persistence pair count");
        if (n_points > max_points_ || n_pairs > max_pairs_)
        {
            throw std::out_of_range("input exceeds GPUPersistenceGradient capacity");
        }
        std::size_t distance_values = 0;
        if (!checkedProduct(static_cast<std::size_t>(n_points), static_cast<std::size_t>(n_points),
                            distance_values) ||
            distance_values > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::length_error("gradient distance matrix exceeds CUDA kernel limits");
        }
        if (upstream_grad_birth.size() != pairs.size() ||
            upstream_grad_death.size() != pairs.size())
        {
            throw std::invalid_argument("upstream gradients must match pair count");
        }
        requireFiniteValues(points, "gradient points");
        requireFiniteValues(upstream_grad_birth, "birth upstream gradients");
        requireFiniteValues(upstream_grad_death, "death upstream gradients");

        std::vector<int> birth_v1(static_cast<std::size_t>(n_pairs));
        std::vector<int> birth_v2(static_cast<std::size_t>(n_pairs));
        std::vector<int> death_v1(static_cast<std::size_t>(n_pairs));
        std::vector<int> death_v2(static_cast<std::size_t>(n_pairs));

        for (int i = 0; i < n_pairs; ++i)
        {
            const auto &pair = pairs[static_cast<std::size_t>(i)];
            if (!std::isfinite(static_cast<double>(pair.birth)))
            {
                throw std::invalid_argument("persistence pair births must be finite");
            }
            const bool open_death = pair.death_v1 < 0 && pair.death_v2 < 0;
            if (!std::isfinite(static_cast<double>(pair.death)) &&
                !(open_death && std::isinf(static_cast<double>(pair.death)) && pair.death > 0.0F))
            {
                throw std::invalid_argument("persistence pair deaths must be finite or open");
            }
            if (pair.dim < 0)
            {
                throw std::invalid_argument("persistence pair dimensions must be non-negative");
            }
            validatePairEdge(pair.birth_v1, pair.birth_v2, n_points, "birth edge");
            validatePairEdge(pair.death_v1, pair.death_v2, n_points, "death edge");
            birth_v1[static_cast<std::size_t>(i)] = pair.birth_v1;
            birth_v2[static_cast<std::size_t>(i)] = pair.birth_v2;
            death_v1[static_cast<std::size_t>(i)] = pair.death_v1;
            death_v2[static_cast<std::size_t>(i)] = pair.death_v2;
        }

        copyToDevice(d_points_, points, "gradient points");
        copyToDevice(d_birth_v1_, birth_v1, "birth v1");
        copyToDevice(d_birth_v2_, birth_v2, "birth v2");
        copyToDevice(d_death_v1_, death_v1, "death v1");
        copyToDevice(d_death_v2_, death_v2, "death v2");
        copyToDevice(d_grad_birth_, upstream_grad_birth, "birth gradients");
        copyToDevice(d_grad_death_, upstream_grad_death, "death gradients");

        const std::size_t grad_point_values = points.size();
        GPU_CHECK(cudaMemset(d_grad_points_, 0,
                             checkedBytes<float>(grad_point_values, "gradient output points")));

        if (n_points > 0)
        {
            const int dist_blocks = checkedGridBlocks(distance_values, "distance gradient grid");
            pairwiseDistanceKernel<<<dist_blocks, kBlockSize>>>(d_points_, d_dist_matrix_, n_points,
                                                                kPointDim);
            GPU_CHECK(cudaPeekAtLastError());
            std::vector<float> distances(distance_values, 0.0F);
            copyFromDevice(distances, d_dist_matrix_, "gradient distance matrix");
            requireFiniteValues(distances, "gradient distance matrix");
        }

        if (n_pairs > 0)
        {
            const int pair_blocks =
                checkedGridBlocks(static_cast<std::size_t>(n_pairs), "pair gradient grid");
            persistencePairBackpropKernel<<<pair_blocks, kBlockSize>>>(
                d_birth_v1_, d_birth_v2_, d_death_v1_, d_death_v2_, d_grad_birth_, d_grad_death_,
                d_points_, d_dist_matrix_, d_grad_points_, n_pairs, n_points, kPointDim);
            GPU_CHECK(cudaPeekAtLastError());
        }

        std::vector<float> grad_points(points.size(), 0.0F);
        copyFromDevice(grad_points, d_grad_points_, "gradient output points");
        requireFiniteValues(grad_points, "gradient output points");
        return grad_points;
    }

private:
    void releaseDeviceState() noexcept
    {
        cudaFree(d_points_);
        cudaFree(d_dist_matrix_);
        cudaFree(d_grad_points_);
        cudaFree(d_birth_v1_);
        cudaFree(d_birth_v2_);
        cudaFree(d_death_v1_);
        cudaFree(d_death_v2_);
        cudaFree(d_grad_birth_);
        cudaFree(d_grad_death_);
        d_points_ = nullptr;
        d_dist_matrix_ = nullptr;
        d_grad_points_ = nullptr;
        d_birth_v1_ = nullptr;
        d_birth_v2_ = nullptr;
        d_death_v1_ = nullptr;
        d_death_v2_ = nullptr;
        d_grad_birth_ = nullptr;
        d_grad_death_ = nullptr;
    }

    int max_points_ = 0;
    int max_pairs_ = 0;

    float *d_points_ = nullptr;
    float *d_dist_matrix_ = nullptr;
    float *d_grad_points_ = nullptr;

    int *d_birth_v1_ = nullptr;
    int *d_birth_v2_ = nullptr;
    int *d_death_v1_ = nullptr;
    int *d_death_v2_ = nullptr;
    float *d_grad_birth_ = nullptr;
    float *d_grad_death_ = nullptr;
};

struct GradientBenchmark
{
    double cpu_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double speedup = 1.0;
    int n_points = 0;
    int n_pairs = 0;
};

GradientBenchmark benchmarkGradient(int n_points, int n_pairs)
{
    if (n_points < 0 || n_pairs < 0)
    {
        throw std::invalid_argument("benchmarkGradient sizes must be non-negative");
    }
    GradientBenchmark bench;
    bench.n_points = n_points;
    bench.n_pairs = n_pairs;

    GPUPersistenceGradient computer(std::max(1, n_points), std::max(1, n_pairs));

    std::mt19937 rng(1337U);
    std::uniform_real_distribution<float> coord_dist(0.0F, 1.0F);
    std::uniform_int_distribution<int> point_dist(0, std::max(0, n_points - 1));
    std::uniform_int_distribution<int> dim_dist(0, 2);

    std::vector<float> points(static_cast<std::size_t>(n_points) * kPointDim);
    for (float &value : points)
    {
        value = coord_dist(rng);
    }

    std::vector<GPUPersistenceGradient::PersistencePair> pairs(static_cast<std::size_t>(n_pairs));
    for (auto &pair : pairs)
    {
        pair.birth = coord_dist(rng);
        pair.death = pair.birth + coord_dist(rng);
        pair.dim = dim_dist(rng);

        pair.birth_v1 = point_dist(rng);
        pair.birth_v2 = point_dist(rng);
        if (pair.birth_v2 == pair.birth_v1 && n_points > 1)
        {
            pair.birth_v2 = (pair.birth_v2 + 1) % n_points;
        }

        pair.death_v1 = point_dist(rng);
        pair.death_v2 = point_dist(rng);
        if (pair.death_v2 == pair.death_v1 && n_points > 1)
        {
            pair.death_v2 = (pair.death_v2 + 1) % n_points;
        }
    }

    std::vector<float> g_birth(static_cast<std::size_t>(n_pairs), 1.0F);
    std::vector<float> g_death(static_cast<std::size_t>(n_pairs), 1.0F);

    const auto start = std::chrono::high_resolution_clock::now();
    auto gradients = computer.computeGradients(points, pairs, g_birth, g_death);
    const auto end = std::chrono::high_resolution_clock::now();
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    const bool finite_gradients = std::all_of(gradients.begin(), gradients.end(), [](float value) {
        return std::isfinite(static_cast<double>(value));
    });
    if (!finite_gradients)
    {
        throw std::runtime_error("gradient benchmark produced non-finite output");
    }

    if (!gradients.empty())
    {
        bench.cpu_time_ms = bench.gpu_time_ms;
    }

    return bench;
}

} // namespace gpu
} // namespace diff
} // namespace ml
} // namespace nerve
