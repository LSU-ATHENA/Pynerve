// OpenMP-parallel stalk construction and SIMD stalk algebra.

#include "nerve/cpu/x86_intrinsics.hpp"
#include "nerve/sheaf/gpu_sheaf.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nerve::sheaf::parallel
{

namespace
{

constexpr float kMinimumCellSize = 1.0e-6f;
constexpr float kNormalizeEpsilon = 1.0e-8f;

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

[[nodiscard]] int floorCell(float value, float cell_size)
{
    return static_cast<int>(std::floor(value / cell_size));
}

[[nodiscard]] Point indexToPoint(int idx)
{
    return Point{static_cast<float>(idx % 10), static_cast<float>((idx / 10) % 10),
                 static_cast<float>(idx / 100)};
}

[[nodiscard]] int usableDimension(const StalkData &stalk)
{
    const int declared = std::max(stalk.dimension, 0);
    return std::min(declared, static_cast<int>(stalk.data.size()));
}

void requireNonNegative(int value, const char *name)
{
    if (value < 0)
    {
        throw std::invalid_argument(std::string(name) + " must be non-negative");
    }
}

[[nodiscard]] float deterministicStalkValue(int stalk_id, int component)
{
    const uint32_t i = static_cast<uint32_t>(stalk_id + 1);
    const uint32_t j = static_cast<uint32_t>(component + 1);
    const uint32_t mixed = (i * 2654435761u) ^ (j * 2246822519u);
    return static_cast<float>(mixed % 1000u) / 1000.0f;
}

} // namespace

class StalkSpatialHash::Impl
{
public:
    explicit Impl(float cell_size)
        : cell_size_(std::max(cell_size, kMinimumCellSize))
    {}

    void insert(int stalk_id, const Point &position)
    {
        cells_[keyFor(position)].push_back(stalk_id);
    }

    [[nodiscard]] std::vector<int> nearby(const Point &position) const
    {
        const int cx = floorCell(position.x, cell_size_);
        const int cy = floorCell(position.y, cell_size_);
        const int cz = floorCell(position.z, cell_size_);

        std::vector<int> result;
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    auto it = cells_.find(hashCell(cx + dx, cy + dy, cz + dz));
                    if (it != cells_.end())
                    {
                        result.insert(result.end(), it->second.begin(), it->second.end());
                    }
                }
            }
        }
        return result;
    }

private:
    [[nodiscard]] uint64_t keyFor(const Point &position) const
    {
        return hashCell(floorCell(position.x, cell_size_), floorCell(position.y, cell_size_),
                        floorCell(position.z, cell_size_));
    }

    /*
     * Spatial locality queries check the 3x3x3 neighborhood around a stalk.
     * Coordinates are floored before hashing so negative positions map to the
     * same cells that a geometric grid would use.
     */
    static uint64_t hashCell(int x, int y, int z)
    {
        const uint64_t ux = static_cast<uint32_t>(x);
        const uint64_t uy = static_cast<uint32_t>(y);
        const uint64_t uz = static_cast<uint32_t>(z);
        return (ux * 73856093ULL) ^ (uy * 19349663ULL) ^ (uz * 83492791ULL);
    }

    float cell_size_;
    std::unordered_map<uint64_t, std::vector<int>> cells_;
};

StalkSpatialHash::StalkSpatialHash(float cell_size)
    : impl_(std::make_unique<Impl>(cell_size))
{}

StalkSpatialHash::~StalkSpatialHash() = default;

void StalkSpatialHash::insertStalk(int stalk_id, const Point &position)
{
    impl_->insert(stalk_id, position);
}

std::vector<int> StalkSpatialHash::getNearbyStalks(const Point &position) const
{
    return impl_->nearby(position);
}

/*
 * SIMD stalk operations use AVX-512 when the translation unit is compiled for
 * it, then finish the tail with the same scalar loop used on portable builds.
 * The public dimension is treated as a bound, not trusted beyond data.size().
 */
void SIMDStalkOperations::addStalks(const StalkData &a, const StalkData &b, StalkData &result)
{
    const int n = std::min(usableDimension(a), usableDimension(b));
    StalkData target(result.id, n);

#if defined(__AVX512F__)
    constexpr int kWidth = 16;
    int i = 0;
    for (; i + kWidth <= n; i += kWidth)
    {
        const __m512 va = _mm512_loadu_ps(&a.data[static_cast<size_t>(i)]);
        const __m512 vb = _mm512_loadu_ps(&b.data[static_cast<size_t>(i)]);
        _mm512_storeu_ps(&target.data[static_cast<size_t>(i)], _mm512_add_ps(va, vb));
    }
    for (; i < n; ++i)
    {
        target.data[static_cast<size_t>(i)] =
            a.data[static_cast<size_t>(i)] + b.data[static_cast<size_t>(i)];
    }
#else
    for (int i = 0; i < n; ++i)
    {
        target.data[static_cast<size_t>(i)] =
            a.data[static_cast<size_t>(i)] + b.data[static_cast<size_t>(i)];
    }
#endif

    result = std::move(target);
}

void SIMDStalkOperations::scaleStalk(const StalkData &stalk, float scalar, StalkData &result)
{
    const int n = usableDimension(stalk);
    StalkData target(result.id, n);

#if defined(__AVX512F__)
    constexpr int kWidth = 16;
    const __m512 scale = _mm512_set1_ps(scalar);
    int i = 0;
    for (; i + kWidth <= n; i += kWidth)
    {
        const __m512 values = _mm512_loadu_ps(&stalk.data[static_cast<size_t>(i)]);
        _mm512_storeu_ps(&target.data[static_cast<size_t>(i)], _mm512_mul_ps(values, scale));
    }
    for (; i < n; ++i)
    {
        target.data[static_cast<size_t>(i)] = stalk.data[static_cast<size_t>(i)] * scalar;
    }
#else
    for (int i = 0; i < n; ++i)
    {
        target.data[static_cast<size_t>(i)] = stalk.data[static_cast<size_t>(i)] * scalar;
    }
#endif

    result = std::move(target);
}

float SIMDStalkOperations::dotProduct(const StalkData &a, const StalkData &b)
{
    const int n = std::min(usableDimension(a), usableDimension(b));
#if defined(__AVX512F__)
    constexpr int kWidth = 16;
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    for (; i + kWidth <= n; i += kWidth)
    {
        const __m512 va = _mm512_loadu_ps(&a.data[static_cast<size_t>(i)]);
        const __m512 vb = _mm512_loadu_ps(&b.data[static_cast<size_t>(i)]);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(va, vb));
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; i < n; ++i)
    {
        result += a.data[static_cast<size_t>(i)] * b.data[static_cast<size_t>(i)];
    }
    return result;
#else
    float result = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        result += a.data[static_cast<size_t>(i)] * b.data[static_cast<size_t>(i)];
    }
    return result;
#endif
}

void SIMDStalkOperations::normalizeStalk(StalkData &stalk)
{
    const float norm = std::sqrt(dotProduct(stalk, stalk));
    if (norm > kNormalizeEpsilon)
    {
        scaleStalk(stalk, 1.0f / norm, stalk);
    }
}

class ParallelSheafBuilder::Impl
{
public:
    explicit Impl(const SheafConfig &config)
        : config_(config)
    {
        requireNonNegative(config_.num_stalks, "num_stalks");
        requireNonNegative(config_.stalk_dimension, "stalk_dimension");
        requireNonNegative(config_.num_threads, "num_threads");
    }

    void build()
    {
#ifdef _OPENMP
        if (config_.num_threads > 0)
        {
            omp_set_num_threads(config_.num_threads);
        }
#endif
        initializeStalks();
        buildSpatialHash();
        countRestrictionEdges();
        computeLayoutChecksum();
    }

    [[nodiscard]] const std::vector<StalkData> &stalks() const { return stalks_; }

private:
    void initializeStalks()
    {
        const int count = config_.num_stalks;
        const int dim = config_.stalk_dimension;
        stalks_.assign(static_cast<size_t>(count), {});
        positions_.assign(static_cast<size_t>(count), {});

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < count; ++i)
        {
            StalkData stalk(i, dim);
            for (int d = 0; d < dim; ++d)
            {
                stalk.data[static_cast<size_t>(d)] = deterministicStalkValue(i, d);
            }
            stalks_[static_cast<size_t>(i)] = std::move(stalk);
            positions_[static_cast<size_t>(i)] = indexToPoint(i);
        }
    }

    void buildSpatialHash()
    {
        spatial_hash_ = std::make_unique<StalkSpatialHash>(1.0f);
        for (int i = 0; i < config_.num_stalks; ++i)
        {
            spatial_hash_->insertStalk(i, positions_[static_cast<size_t>(i)]);
        }
    }

    void countRestrictionEdges()
    {
        edge_count_.store(0, std::memory_order_relaxed);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < config_.num_stalks; ++i)
        {
            const auto nearby = spatial_hash_->getNearbyStalks(positions_[static_cast<size_t>(i)]);
            size_t local_edges = 0;
            for (int j : nearby)
            {
                local_edges += (i != j) ? 1U : 0U;
            }
            edge_count_.fetch_add(local_edges, std::memory_order_relaxed);
        }
    }

    void computeLayoutChecksum()
    {
        const int count = config_.num_stalks;
        const int dim = config_.stalk_dimension;
        if (count == 0 || dim == 0)
        {
            layout_checksum_ = 0.0f;
            return;
        }

        std::vector<float> soa(static_cast<size_t>(count) * static_cast<size_t>(dim));
        for (int d = 0; d < dim; ++d)
        {
            for (int i = 0; i < count; ++i)
            {
                soa[static_cast<size_t>(d) * static_cast<size_t>(count) + static_cast<size_t>(i)] =
                    stalks_[static_cast<size_t>(i)].data[static_cast<size_t>(d)];
            }
        }
        layout_checksum_ = std::accumulate(soa.begin(), soa.end(), 0.0f);
    }

    SheafConfig config_;
    std::vector<StalkData> stalks_;
    std::vector<Point> positions_;
    std::unique_ptr<StalkSpatialHash> spatial_hash_;
    std::atomic<size_t> edge_count_{0};
    float layout_checksum_ = 0.0f;
};

ParallelSheafBuilder::ParallelSheafBuilder(const SheafConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

ParallelSheafBuilder::~ParallelSheafBuilder() = default;

void ParallelSheafBuilder::build()
{
    impl_->build();
}

std::vector<StalkData> ParallelSheafBuilder::getStalks() const
{
    return impl_->stalks();
}

SheafParallelBenchmark benchmarkParallelSheaf(int num_stalks, int stalk_dim, int num_threads)
{
    requireNonNegative(num_stalks, "num_stalks");
    requireNonNegative(stalk_dim, "stalk_dim");
    requireNonNegative(num_threads, "num_threads");

    SheafParallelBenchmark bench;
    bench.num_stalks = num_stalks;
    bench.stalk_dim = stalk_dim;
    bench.num_threads = num_threads;

    const auto start_seq = std::chrono::high_resolution_clock::now();
    std::vector<StalkData> stalks_seq;
    stalks_seq.reserve(static_cast<size_t>(num_stalks));
    for (int i = 0; i < num_stalks; ++i)
    {
        stalks_seq.emplace_back(i, stalk_dim);
    }
    const auto end_seq = std::chrono::high_resolution_clock::now();
    bench.sequential_time_ms =
        std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

    ParallelSheafBuilder::SheafConfig config;
    config.num_stalks = num_stalks;
    config.stalk_dimension = stalk_dim;
    config.num_threads = num_threads;
    ParallelSheafBuilder builder(config);

    const auto start_par = std::chrono::high_resolution_clock::now();
    builder.build();
    const auto end_par = std::chrono::high_resolution_clock::now();
    bench.parallel_time_ms = std::chrono::duration<double, std::milli>(end_par - start_par).count();

    if (num_stalks >= 2 && stalk_dim > 0)
    {
        StalkData simd_out(0, stalk_dim);
        const auto start_simd = std::chrono::high_resolution_clock::now();
        SIMDStalkOperations::addStalks(stalks_seq[0], stalks_seq[1], simd_out);
        const auto end_simd = std::chrono::high_resolution_clock::now();
        bench.simd_time_ms =
            std::chrono::duration<double, std::milli>(end_simd - start_simd).count();
    }

    bench.speedup_parallel =
        finiteBenchmarkSpeedup(bench.sequential_time_ms, bench.parallel_time_ms);
    bench.speedup_simd = finiteBenchmarkSpeedup(bench.sequential_time_ms, bench.simd_time_ms);
    return bench;
}

} // namespace nerve::sheaf::parallel
