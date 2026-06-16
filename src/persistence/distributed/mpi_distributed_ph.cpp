#include "nerve/persistence/core/flood_complex.hpp"
#include "nerve/persistence/distributed/mpi_distributed_ph.hpp"

#if defined(NERVE_HAS_MPI)
#include <mpi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace nerve::persistence::distributed
{

namespace
{

int mpi_rank = 0;
int mpi_size = 1;
[[maybe_unused]] bool mpi_initialized = false;

int checkedMpiCount(size_t count, const char *context)
{
    if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(count);
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

int buildMpiDisplacements(const std::vector<int> &sizes, std::vector<int> *displacements,
                          const char *context)
{
    if (displacements == nullptr)
    {
        throw std::invalid_argument("MPI displacement output cannot be null");
    }
    displacements->assign(sizes.size(), 0);

    size_t total = 0;
    const auto max_count = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        const int size = sizes[i];
        if (size < 0)
        {
            throw std::length_error(context);
        }
        const auto size_count = static_cast<size_t>(size);
        if (total > max_count - size_count)
        {
            throw std::length_error(context);
        }
        (*displacements)[i] = static_cast<int>(total);
        total += size_count;
    }
    return static_cast<int>(total);
}

bool hasValidPointBuffer(const std::vector<double> &points, size_t point_dim,
                         const DistributedConfig &config)
{
    if (point_dim == 0 || points.size() % point_dim != 0 || !std::isfinite(config.max_radius) ||
        config.max_radius < 0.0 || !std::isfinite(config.overlap_ratio) ||
        config.overlap_ratio < 0.0 || config.overlap_ratio > 0.5 || config.use_cuda)
    {
        return false;
    }
    return std::all_of(points.begin(), points.end(),
                       [](double value) { return std::isfinite(value); });
}

int64_t quantizePairValue(double value)
{
    if (value == std::numeric_limits<double>::infinity())
    {
        return std::numeric_limits<int64_t>::max();
    }
    if (value == -std::numeric_limits<double>::infinity())
    {
        return std::numeric_limits<int64_t>::min();
    }
    if (!std::isfinite(value))
    {
        return 0;
    }
    const long double scaled = static_cast<long double>(value) * 1000.0L;
    if (scaled >= static_cast<long double>(std::numeric_limits<int64_t>::max()))
    {
        return std::numeric_limits<int64_t>::max();
    }
    if (scaled <= static_cast<long double>(std::numeric_limits<int64_t>::min()))
    {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(scaled);
}

bool isValidPersistencePair(const Pair &pair)
{
    const bool valid_death =
        std::isfinite(pair.death) || pair.death == std::numeric_limits<double>::infinity();
    return pair.dimension >= 0 && std::isfinite(pair.birth) && valid_death;
}

class SpatialCover
{
public:
    struct CoverRegion
    {
        int rank;                          // MPI rank handling this region
        std::vector<double> bounds;        // [min_x, max_x, min_y, max_y, ...]
        double overlap;                    // Overlap with adjacent regions
        std::vector<size_t> point_indices; // Points in this region
    };

    std::vector<CoverRegion> regions;

    void computeCover(const std::vector<double> &points, size_t point_dim, size_t num_points,
                      int num_ranks, double overlap_ratio = 0.1)
    {
        if (point_dim == 0 || num_points == 0 || num_ranks <= 0)
        {
            regions.clear();
            return;
        }
        std::vector<double> global_bounds = computeBounds(points, point_dim, 0, num_points);
        regions.clear();
        double range_x = global_bounds[1] - global_bounds[0];
        double region_size = range_x / num_ranks;
        double overlap = region_size * overlap_ratio;

        for (int r = 0; r < num_ranks; ++r)
        {
            CoverRegion region;
            region.rank = r;
            region.overlap = overlap;

            region.bounds = global_bounds;
            region.bounds[0] = global_bounds[0] + r * region_size - overlap;
            region.bounds[1] = global_bounds[0] + (r + 1) * region_size + overlap;

            for (size_t i = 0; i < num_points; ++i)
            {
                double x = points[i * point_dim];
                if (x >= region.bounds[0] && x <= region.bounds[1])
                {
                    region.point_indices.push_back(i);
                }
            }

            regions.push_back(region);
        }
    }

private:
    std::vector<double> computeBounds(const std::vector<double> &points, size_t point_dim,
                                      size_t start, size_t count)
    {
        std::vector<double> bounds(point_dim * 2);
        for (size_t d = 0; d < point_dim; ++d)
        {
            bounds[d * 2] = std::numeric_limits<double>::infinity();
            bounds[d * 2 + 1] = -std::numeric_limits<double>::infinity();
        }

        for (size_t i = start; i < start + count && i < points.size() / point_dim; ++i)
        {
            for (size_t d = 0; d < point_dim; ++d)
            {
                double val = points[i * point_dim + d];
                bounds[d * 2] = std::min(bounds[d * 2], val);
                bounds[d * 2 + 1] = std::max(bounds[d * 2 + 1], val);
            }
        }

        return bounds;
    }
};

class MayerVietorisSpectralSequence
{
public:
    static std::vector<Pair>
    computeSpectralSequence(const std::vector<std::vector<Pair>> &local_diagrams,
                            const SpatialCover &cover)
    {
        std::vector<Pair> global_diagram;
        for (std::size_t i = 0; i < local_diagrams.size(); ++i)
        {
            if (!cover.regions.empty())
            {
                if (i >= cover.regions.size())
                {
                    break;
                }
                if (cover.regions[i].point_indices.empty())
                {
                    continue;
                }
            }
            const auto &local = local_diagrams[i];
            for (const auto &pair : local)
            {
                if (isValidPersistencePair(pair))
                {
                    global_diagram.push_back(pair);
                }
            }
        }
        std::unordered_set<uint64_t> seen;
        std::vector<Pair> deduplicated;

        for (const auto &pair : global_diagram)
        {
            uint64_t hash = hashPair(pair);
            if (seen.insert(hash).second)
            {
                deduplicated.push_back(pair);
            }
        }

        std::sort(deduplicated.begin(), deduplicated.end(), [](const Pair &a, const Pair &b) {
            if (a.dimension != b.dimension)
                return a.dimension < b.dimension;
            if (a.birth != b.birth)
                return a.birth < b.birth;
            return a.death < b.death;
        });

        return deduplicated;
    }

private:
    static uint64_t hashPair(const Pair &pair)
    {
        uint64_t hash = static_cast<uint64_t>(pair.dimension);
        hash = hash * 1000003u + static_cast<uint64_t>(quantizePairValue(pair.birth));
        hash = hash * 1000003u + static_cast<uint64_t>(quantizePairValue(pair.death));
        return hash;
    }
};

} // namespace

void initializeDistributed()
{
#if defined(NERVE_HAS_MPI)
    int mpi_is_initialized = 0;
    int mpi_err = MPI_Initialized(&mpi_is_initialized);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Initialized failed in initializeDistributed" << std::endl;
        return;
    }
    if (!mpi_is_initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        mpi_err = MPI_Init(&argc, &argv);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Init failed in initializeDistributed" << std::endl;
            return;
        }
    }
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Comm_rank failed in initializeDistributed" << std::endl;
        return;
    }
    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Comm_size failed in initializeDistributed" << std::endl;
        return;
    }
    mpi_initialized = true;
#endif
}

void finalizeDistributed()
{
#if defined(NERVE_HAS_MPI)
    if (mpi_initialized)
    {
        int mpi_is_finalized = 0;
        int mpi_err = MPI_Finalized(&mpi_is_finalized);
        if (mpi_err != MPI_SUCCESS)
        {
            std::cerr << "MPI_Finalized failed in finalizeDistributed" << std::endl;
            return;
        }
        if (!mpi_is_finalized)
        {
            mpi_err = MPI_Finalize();
            if (mpi_err != MPI_SUCCESS)
            {
                std::cerr << "MPI_Finalize failed in finalizeDistributed" << std::endl;
                return;
            }
        }
        mpi_initialized = false;
    }
#endif
}

DistributedResult computeDistributedPH(const std::vector<double> &points, size_t point_dim,
                                       const DistributedConfig &config)
{
    DistributedResult result{};
#if defined(NERVE_HAS_MPI)
    if (!mpi_initialized)
    {
        initializeDistributed();
    }
#endif
    if (!hasValidPointBuffer(points, point_dim, config))
    {
        return result;
    }
    result.mpi_rank = mpi_rank;
    result.mpi_size = mpi_size;
    auto start_total = std::chrono::high_resolution_clock::now();
    const size_t num_points = points.size() / point_dim;
    auto start_cover = std::chrono::high_resolution_clock::now();
    SpatialCover cover;
    cover.computeCover(points, point_dim, num_points, mpi_size, config.overlap_ratio);
    auto end_cover = std::chrono::high_resolution_clock::now();
    result.cover_time_ms =
        std::chrono::duration<double, std::milli>(end_cover - start_cover).count();

    auto start_local = std::chrono::high_resolution_clock::now();
    if (cover.regions.empty())
    {
        return result;
    }
    const int region_index = std::clamp(mpi_rank, 0, static_cast<int>(cover.regions.size()) - 1);
    const auto &my_region = cover.regions[region_index];
    std::vector<double> local_points;
    local_points.reserve(my_region.point_indices.size() * point_dim);
    for (size_t idx : my_region.point_indices)
    {
        for (size_t d = 0; d < point_dim; ++d)
        {
            local_points.push_back(points[idx * point_dim + d]);
        }
    }

    FloodComplexConfig flood_config;
    flood_config.max_dim = config.max_dim;
    flood_config.max_radius = config.max_radius;
    flood_config.subset_ratio = 0.05;
    auto local_result =
        computeFloodComplex(local_points, point_dim, my_region.point_indices.size(), flood_config);
    auto end_local = std::chrono::high_resolution_clock::now();
    result.local_computation_time_ms =
        std::chrono::duration<double, std::milli>(end_local - start_local).count();

    auto start_gather = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Pair>> all_diagrams;
#if defined(NERVE_HAS_MPI)
    std::vector<double> local_serialized;
    for (const auto &pair : local_result.pairs)
    {
        if (!isValidPersistencePair(pair))
        {
            continue;
        }
        local_serialized.push_back(static_cast<double>(pair.dimension));
        local_serialized.push_back(pair.birth);
        local_serialized.push_back(pair.death);
    }

    const int local_size =
        checkedMpiCount(local_serialized.size(), "distributed PH local payload exceeds MPI range");
    std::vector<int> all_sizes(mpi_size);
    int mpi_err =
        MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Allgather failed in computeDistributedPH" << std::endl;
        return result;
    }
    std::vector<int> displacements(mpi_size);
    const int total_size = buildMpiDisplacements(
        all_sizes, &displacements, "distributed PH gathered payload exceeds MPI range");
    std::vector<double> all_serialized(static_cast<size_t>(total_size));
    const double *send_data = local_serialized.empty() ? nullptr : local_serialized.data();
    mpi_err = MPI_Allgatherv(send_data, local_size, MPI_DOUBLE,
                             all_serialized.empty() ? nullptr : all_serialized.data(),
                             all_sizes.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Allgatherv failed in computeDistributedPH" << std::endl;
        return result;
    }
    for (int r = 0; r < mpi_size; ++r)
    {
        std::vector<Pair> rank_pairs;
        int start = displacements[r];
        int count = all_sizes[r];
        if (count < 0 || count % 3 != 0)
        {
            all_diagrams.push_back(rank_pairs);
            continue;
        }
        for (int i = start; i + 2 < start + count; i += 3)
        {
            const double dimension_value = all_serialized[i];
            if (!std::isfinite(dimension_value) || dimension_value < 0.0 ||
                dimension_value > static_cast<double>(std::numeric_limits<Dimension>::max()) ||
                std::trunc(dimension_value) != dimension_value)
            {
                continue;
            }
            Pair pair;
            pair.dimension = static_cast<Dimension>(dimension_value);
            pair.birth = all_serialized[i + 1];
            pair.death = all_serialized[i + 2];
            if (isValidPersistencePair(pair))
            {
                rank_pairs.push_back(pair);
            }
        }
        all_diagrams.push_back(rank_pairs);
    }
#else
    all_diagrams.push_back(local_result.pairs);
#endif
    auto end_gather = std::chrono::high_resolution_clock::now();
    result.communication_time_ms =
        std::chrono::duration<double, std::milli>(end_gather - start_gather).count();

    auto start_mv = std::chrono::high_resolution_clock::now();
    result.pairs = MayerVietorisSpectralSequence::computeSpectralSequence(all_diagrams, cover);
    auto end_mv = std::chrono::high_resolution_clock::now();
    result.mv_spectral_time_ms =
        std::chrono::duration<double, std::milli>(end_mv - start_mv).count();

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();
    result.total_points = num_points;
    result.points_per_rank = my_region.point_indices.size();
    result.total_pairs = result.pairs.size();
    const double serial_estimate_ms =
        result.local_computation_time_ms * static_cast<double>(std::max(1, result.mpi_size));
    result.estimated_speedup = finiteBenchmarkSpeedup(serial_estimate_ms, result.total_time_ms);
    return result;
}

DistributedResult computeDistributedPHSingleNode(const std::vector<double> &points,
                                                 size_t point_dim, const DistributedConfig &config,
                                                 int num_threads)
{
    DistributedResult final_result{};
    if (!hasValidPointBuffer(points, point_dim, config))
    {
        return final_result;
    }
    const auto start_total = std::chrono::high_resolution_clock::now();
    const int threads = std::max(1, num_threads);
    const auto start_cover = std::chrono::high_resolution_clock::now();
    SpatialCover cover;
    cover.computeCover(points, point_dim, points.size() / point_dim, threads, config.overlap_ratio);
    const auto end_cover = std::chrono::high_resolution_clock::now();
    if (cover.regions.empty())
    {
        return final_result;
    }
    std::vector<std::vector<Pair>> all_results(threads);
    const auto start_local = std::chrono::high_resolution_clock::now();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) if (config.use_openmp)
#endif
    for (int t = 0; t < threads; ++t)
    {
        const auto &region = cover.regions[t];
        std::vector<double> local_points;
        local_points.reserve(region.point_indices.size() * point_dim);
        for (size_t idx : region.point_indices)
        {
            for (size_t d = 0; d < point_dim; ++d)
            {
                local_points.push_back(points[idx * point_dim + d]);
            }
        }
        FloodComplexConfig flood_config;
        flood_config.max_dim = config.max_dim;
        flood_config.max_radius = config.max_radius;
        auto result =
            computeFloodComplex(local_points, point_dim, region.point_indices.size(), flood_config);
        all_results[t] = result.pairs;
    }
    const auto end_local = std::chrono::high_resolution_clock::now();
    const auto start_mv = std::chrono::high_resolution_clock::now();
    final_result.pairs = MayerVietorisSpectralSequence::computeSpectralSequence(all_results, cover);
    const auto end_mv = std::chrono::high_resolution_clock::now();
    const auto end_total = std::chrono::high_resolution_clock::now();
    final_result.cover_time_ms =
        std::chrono::duration<double, std::milli>(end_cover - start_cover).count();
    final_result.local_computation_time_ms =
        std::chrono::duration<double, std::milli>(end_local - start_local).count();
    final_result.mv_spectral_time_ms =
        std::chrono::duration<double, std::milli>(end_mv - start_mv).count();
    final_result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();
    final_result.mpi_size = threads;
    final_result.total_points = points.size() / point_dim;
    final_result.total_pairs = final_result.pairs.size();
    return final_result;
}

// within the repository hard-cap while preserving exported symbols.
#include "detail/mpi_distributed_helpers.inl"

} // namespace nerve::persistence::distributed

extern "C"
{
    void initializeDistributed()
    {
        nerve::persistence::distributed::initializeDistributed();
    }

    void finalizeDistributed()
    {
        nerve::persistence::distributed::finalizeDistributed();
    }

    int isMpiAvailable()
    {
#if defined(NERVE_HAS_MPI)
        return 1;
#else
        return 0;
#endif
    }
}
