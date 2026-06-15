#include "nerve/core.hpp"
#include "nerve/persistence/vr/vr_distance_tiled_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(NERVE_HAS_CUDA)
#include "nerve/gpu/distance_fasted.cuh"
#include "nerve/gpu/distance_tedjoin.cuh"
#endif

#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif

#ifdef __AVX2__
#include "nerve/cpu/x86_intrinsics.hpp"
#endif

namespace nerve::persistence
{

namespace
{

struct CacheConfig
{
    static constexpr size_t L1_DEFAULT_BYTES = 32ULL * 1024ULL;
    static constexpr size_t L2_DEFAULT_BYTES = 256ULL * 1024ULL;
    static constexpr size_t L3_DEFAULT_BYTES = 8ULL * 1024ULL * 1024ULL;
    static constexpr size_t MIN_TILE = 16;
    static constexpr size_t MAX_TILE = 512;
};

template <int LEVEL>
inline void prefetch(const void *ptr)
{
#ifdef __GNUC__
    __builtin_prefetch(ptr, 0, LEVEL);
#elif defined(_MSC_VER) && NERVE_HAS_X86_INTRINSICS
    _mm_prefetch(reinterpret_cast<const char *>(ptr), LEVEL == 3   ? _MM_HINT_T0
                                                      : LEVEL == 2 ? _MM_HINT_T1
                                                                   : _MM_HINT_T2);
#else
    (void)ptr;
#endif
}

#ifdef __AVX2__
inline double distanceSimd(const double *a, const double *b, size_t dim)
{
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;

    prefetch<3>(a + 32);
    prefetch<3>(b + 32);

    for (; i + 4 <= dim; i += 4)
    {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d diff = _mm256_sub_pd(a_vec, b_vec);
        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(diff, diff));
    }

    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, sum_vec);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (; i < dim; ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}
#endif

inline double distanceScalar(const double *a, const double *b, size_t dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void computeTile(const double *points, size_t point_dim, size_t i_start, size_t i_end,
                 size_t j_start, size_t j_end, double *out_matrix, size_t n_points)
{
    for (size_t i = i_start; i < i_end; ++i)
    {
        const double *p_i = &points[i * point_dim];

        if (i + 1 < n_points)
        {
            prefetch<2>(&points[(i + 1) * point_dim]);
        }

        for (size_t j = std::max(j_start, i + 1); j < j_end; ++j)
        {
            const double *p_j = &points[j * point_dim];

            if (j + 1 < n_points)
            {
                prefetch<1>(&points[(j + 1) * point_dim]);
            }

#ifdef __AVX2__
            double dist = distanceSimd(p_i, p_j, point_dim);
#else
            double dist = distanceScalar(p_i, p_j, point_dim);
#endif

            out_matrix[i * n_points + j] = dist;
            out_matrix[j * n_points + i] = dist;
        }
    }
}

bool checkedSquareCount(size_t num_points, size_t &matrix_size);

void computeDistanceMatrixTiledParallel(const std::vector<double> &points, size_t point_dim,
                                        size_t num_points, std::vector<double> &distance_matrix,
                                        size_t tile_size)
{
    size_t matrix_size = 0;
    if (!checkedSquareCount(num_points, matrix_size))
    {
        distance_matrix.clear();
        return;
    }
    distance_matrix.assign(matrix_size, 0.0);

    // Initialize diagonal to 0
    for (size_t i = 0; i < num_points; ++i)
    {
        distance_matrix[i * num_points + i] = 0.0;
    }

    const size_t num_tiles = (num_points + tile_size - 1) / tile_size;

#pragma omp parallel for schedule(dynamic, 1)
    for (size_t tile_i = 0; tile_i < num_tiles; ++tile_i)
    {
        size_t i_start = tile_i * tile_size;
        size_t i_end = std::min(i_start + tile_size, num_points);

        for (size_t tile_j = tile_i; tile_j < num_tiles; ++tile_j)
        {
            size_t j_start = tile_j * tile_size;
            size_t j_end = std::min(j_start + tile_size, num_points);

            computeTile(points.data(), point_dim, i_start, i_end, j_start, j_end,
                        distance_matrix.data(), num_points);
        }
    }
}

bool checkedPointValueCount(const size_t point_dim, const size_t num_points, size_t *value_count)
{
    if (value_count == nullptr || point_dim == 0 || num_points == 0)
    {
        return false;
    }
    if (num_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        return false;
    }
    *value_count = num_points * point_dim;
    return true;
}

bool hasValidShape(const std::vector<double> &points, const size_t point_dim,
                   const size_t num_points)
{
    size_t expected = 0;
    return checkedPointValueCount(point_dim, num_points, &expected) && points.size() == expected;
}

bool hasFiniteSafePoints(const std::vector<double> &points, const size_t point_dim,
                         const size_t num_points)
{
    size_t expected = 0;
    if (!checkedPointValueCount(point_dim, num_points, &expected) || points.size() != expected)
    {
        return false;
    }
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max()) /
                  static_cast<long double>(point_dim)) /
        4.0L;
    return std::all_of(points.begin(), points.end(), [&](double value) {
        return std::isfinite(value) && std::abs(static_cast<long double>(value)) <= safe_abs;
    });
}

bool canMaterializeSquareMatrix(const size_t num_points)
{
    if (num_points == 0)
    {
        return false;
    }
    if (num_points > std::numeric_limits<size_t>::max() / num_points)
    {
        return false;
    }
    return num_points * num_points <= std::vector<double>().max_size();
}

bool checkedSquareCount(const size_t num_points, size_t &matrix_size)
{
    if (!canMaterializeSquareMatrix(num_points))
    {
        matrix_size = 0;
        return false;
    }
    matrix_size = num_points * num_points;
    return true;
}

#if defined(__i386__) || defined(__x86_64__)
size_t readCacheLeafBytes(const unsigned leaf, const unsigned cache_level)
{
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (!__get_cpuid_count(leaf, cache_level, &eax, &ebx, &ecx, &edx))
    {
        return 0;
    }
    const unsigned type = eax & 0x1FU;
    if (type == 0U)
    {
        return 0;
    }
    const unsigned line_size = (ebx & 0xFFFU) + 1U;
    const unsigned partitions = ((ebx >> 12U) & 0x3FFU) + 1U;
    const unsigned ways = ((ebx >> 22U) & 0x3FFU) + 1U;
    const unsigned sets = ecx + 1U;
    return static_cast<size_t>(line_size) * partitions * ways * sets;
}
#endif

size_t estimateTileFromL2(const size_t l2_bytes)
{
    if (l2_bytes == 0)
    {
        return 64;
    }
    // Distance kernel holds two point rows and accumulators. This conservative
    // estimator keeps the active tile comfortably within private cache.
    const size_t bytes_per_point = 3 * sizeof(double) * 64;
    size_t tile = static_cast<size_t>(
        std::sqrt(static_cast<double>(l2_bytes / std::max<size_t>(1, bytes_per_point))) * 64.0);
    tile = std::clamp(tile, CacheConfig::MIN_TILE, CacheConfig::MAX_TILE);
    // Keep tiles aligned to multiples of 16 for vector loops.
    tile = (tile / 16) * 16;
    return std::max<size_t>(16, tile);
}

} // namespace

CacheSizes detectCacheSizes()
{
    CacheSizes sizes{CacheConfig::L1_DEFAULT_BYTES, CacheConfig::L2_DEFAULT_BYTES,
                     CacheConfig::L3_DEFAULT_BYTES};
#if defined(__i386__) || defined(__x86_64__)
    const size_t l1 = readCacheLeafBytes(4U, 0U);
    const size_t l2 = readCacheLeafBytes(4U, 2U);
    const size_t l3 = readCacheLeafBytes(4U, 3U);
    if (l1 > 0)
        sizes.l1_data = l1;
    if (l2 > 0)
        sizes.l2 = l2;
    if (l3 > 0)
        sizes.l3 = l3;
#endif
    return sizes;
}

size_t getOptimalTileSize()
{
    const CacheSizes caches = detectCacheSizes();
    return estimateTileFromL2(caches.l2);
}

std::vector<double> computeDistanceMatrixTiled(const std::vector<double> &points, size_t point_dim,
                                               size_t num_points, size_t tile_size)
{
    if (!hasValidShape(points, point_dim, num_points) ||
        !hasFiniteSafePoints(points, point_dim, num_points) ||
        !canMaterializeSquareMatrix(num_points))
    {
        return {};
    }
#ifdef NERVE_HAS_CUDA
    if (num_points >= 128)
    {
        std::vector<double> distance_matrix(num_points * num_points);
        cudaError_t stat = nerve::gpu::tedjoin::launchFp64TensorDistance(
            points.data(), static_cast<int>(num_points), static_cast<int>(point_dim),
            distance_matrix.data(), static_cast<int>(num_points));
        if (stat == cudaSuccess)
        {
            return distance_matrix;
        }
    }
#endif
    if (tile_size == 0)
    {
        tile_size = getOptimalTileSize();
    }
    tile_size = std::max<size_t>(1, tile_size);

    std::vector<double> distance_matrix;
    computeDistanceMatrixTiledParallel(points, point_dim, num_points, distance_matrix, tile_size);

    return distance_matrix;
}

std::vector<double> computeDistanceMatrixNumaAware(const std::vector<double> &points,
                                                   size_t point_dim, size_t num_points,
                                                   int num_numa_nodes)
{
    if (!hasValidShape(points, point_dim, num_points) ||
        !hasFiniteSafePoints(points, point_dim, num_points) ||
        !canMaterializeSquareMatrix(num_points))
    {
        return {};
    }
    if (num_numa_nodes <= 0)
    {
        num_numa_nodes = 1;
    }

    size_t matrix_size = 0;
    if (!checkedSquareCount(num_points, matrix_size))
    {
        return {};
    }
    std::vector<double> distance_matrix(matrix_size, 0.0);

    size_t points_per_node = (num_points + num_numa_nodes - 1) / num_numa_nodes;

// Process contiguous row ranges per worker to keep writes and point-row
// accesses stable within each chunk. This improves locality on multi-socket
// hosts without depending on platform-specific NUMA pinning APIs.
#pragma omp parallel for schedule(static)
    for (int node = 0; node < num_numa_nodes; ++node)
    {
        size_t node_start = node * points_per_node;
        size_t node_end = std::min(node_start + points_per_node, num_points);

        for (size_t i = node_start; i < node_end; ++i)
        {
            const double *p_i = &points[i * point_dim];
            for (size_t j = i + 1; j < num_points; ++j)
            {
                const double *p_j = &points[j * point_dim];

                double dist = distanceScalar(p_i, p_j, point_dim);
                distance_matrix[i * num_points + j] = dist;
                distance_matrix[j * num_points + i] = dist;
            }
        }
    }

    return distance_matrix;
}

std::vector<double> computeDistanceMatrixHierarchical(const std::vector<double> &points,
                                                      size_t point_dim, size_t num_points)
{
    if (!hasValidShape(points, point_dim, num_points) ||
        !hasFiniteSafePoints(points, point_dim, num_points) ||
        !canMaterializeSquareMatrix(num_points))
    {
        return {};
    }

    size_t matrix_size = 0;
    if (!checkedSquareCount(num_points, matrix_size))
    {
        return {};
    }
    std::vector<double> distance_matrix(matrix_size, 0.0);

    const CacheSizes caches = detectCacheSizes();
    const size_t l3_block = std::max<size_t>(
        64, static_cast<size_t>(std::sqrt(static_cast<double>(caches.l3 / sizeof(double)))));

    for (size_t l3_i = 0; l3_i < num_points; l3_i += l3_block)
    {
        size_t l3_i_end = std::min(l3_i + l3_block, num_points);

        for (size_t l3_j = l3_i; l3_j < num_points; l3_j += l3_block)
        {
            size_t l3_j_end = std::min(l3_j + l3_block, num_points);

            const size_t l2_block = getOptimalTileSize();

            for (size_t l2_i = l3_i; l2_i < l3_i_end; l2_i += l2_block)
            {
                size_t l2_i_end = std::min(l2_i + l2_block, l3_i_end);

                for (size_t l2_j = std::max(l3_j, l2_i); l2_j < l3_j_end; l2_j += l2_block)
                {
                    size_t l2_j_end = std::min(l2_j + l2_block, l3_j_end);

                    computeTile(points.data(), point_dim, l2_i, l2_i_end, l2_j, l2_j_end,
                                distance_matrix.data(), num_points);
                }
            }
        }
    }

    return distance_matrix;
}

#ifdef NERVE_HAS_CUDA
std::vector<float> computeDistanceMatrixTiledF32(const std::vector<float> &points, size_t point_dim,
                                                 size_t num_points)
{
    if (num_points == 0 || points.size() < num_points * point_dim ||
        num_points > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};
    if (num_points >= 64)
    {
        std::vector<float> distance_matrix(num_points * num_points);
        auto config = nerve::gpu::FastedConfig{};
        config.use_l2_optimization = true;
        cudaError_t stat = nerve::gpu::launchDistanceFastEdAsync(
            points.data(), static_cast<int>(num_points), static_cast<int>(point_dim),
            distance_matrix.data(), static_cast<int>(num_points), nullptr, config);
        if (stat == cudaSuccess)
            return distance_matrix;
    }
    std::vector<float> distance_matrix(num_points * num_points);
    for (size_t i = 0; i < num_points; ++i)
        for (size_t j = 0; j < num_points; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < point_dim; ++k)
            {
                float d = points[i * point_dim + k] - points[j * point_dim + k];
                sum += d * d;
            }
            distance_matrix[i * num_points + j] = sum;
        }
    return distance_matrix;
}
#else
std::vector<float> computeDistanceMatrixTiledF32(const std::vector<float> &points, size_t point_dim,
                                                 size_t num_points)
{
    if (num_points == 0 || points.size() < num_points * point_dim ||
        num_points > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};
    std::vector<float> distance_matrix(num_points * num_points);
    for (size_t i = 0; i < num_points; ++i)
        for (size_t j = 0; j < num_points; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < point_dim; ++k)
            {
                float d = points[i * point_dim + k] - points[j * point_dim + k];
                sum += d * d;
            }
            distance_matrix[i * num_points + j] = sum;
        }
    return distance_matrix;
}
#endif

} // namespace nerve::persistence
