#include "nerve/persistence/vr/vr_algorithm_selector_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"
#include "nerve/persistence/vr/vr_large_witness_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <unistd.h>
#elif __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace nerve::persistence
{

namespace
{

constexpr double APPROXIMATION_FACTOR_EXACT = 1.0;
constexpr double APPROXIMATION_FACTOR_WITNESS = 3.0;
constexpr size_t BYTES_PER_MB = 1024ULL * 1024ULL;
constexpr double MILLIS_PER_SECOND = 1000.0;

size_t saturatedProduct(size_t lhs, size_t rhs)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return std::numeric_limits<size_t>::max();
    }
    return lhs * rhs;
}

size_t saturatedAdd(size_t lhs, size_t rhs)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs)
    {
        return std::numeric_limits<size_t>::max();
    }
    return lhs + rhs;
}

double boundedRadiusForHeuristic(double max_radius)
{
    if (max_radius >= 1.0)
    {
        return 1.0;
    }
    if (std::isfinite(max_radius) && max_radius > 0.0)
    {
        return max_radius;
    }
    return 0.0;
}

size_t saturatedScaledSize(size_t value, double scale)
{
    if (!std::isfinite(scale) || scale <= 0.0 || value == 0)
    {
        return 0;
    }
    const double scaled = static_cast<double>(value) * scale;
    const double max_size = static_cast<double>(std::numeric_limits<size_t>::max());
    if (!std::isfinite(scaled) || scaled >= max_size)
    {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(scaled);
}

struct Thresholds
{
    static constexpr size_t FAST_MAX = 1024;    // Use small exact fast path
    static constexpr size_t FAST_MAX_DIM = 16;  // Point-cache capacity in fast path
    static constexpr size_t MEDIUM_MIN = 512;   // Start considering hybrid
    static constexpr size_t MEDIUM_MAX = 10000; // Max for exact computation
    static constexpr size_t LARGE_MIN = 5000;   // Start considering witness
};

struct ProblemCharacteristicsLocal
{
    size_t num_points;
    size_t point_dim;
    double max_radius;
    bool gpu_available;
    bool requires_exact;
    double time_budget_seconds;
};

ProblemCharacteristicsLocal makeProblemCharacteristics(size_t num_points, size_t point_dim,
                                                       double max_radius, bool requires_exact,
                                                       double time_budget_seconds,
                                                       bool gpu_available)
{
    return {
        num_points, point_dim, max_radius, gpu_available, requires_exact, time_budget_seconds,
    };
}

bool canUseSmallFastPath(const ProblemCharacteristicsLocal &prob)
{
    return prob.num_points <= Thresholds::FAST_MAX && prob.point_dim <= Thresholds::FAST_MAX_DIM;
}

double estimateComputationTime(const ProblemCharacteristicsLocal &prob)
{
    double n = static_cast<double>(prob.num_points);
    double dim = static_cast<double>(prob.point_dim);
    double r = boundedRadiusForHeuristic(prob.max_radius);

    (void)prob.gpu_available;
    double distance_time = 1e-7 * n * n * dim;
    double clique_time = 1e-8 * std::pow(n, std::min(dim + 1, 4.0)) * std::pow(r, dim);

    return distance_time + clique_time;
}

VRAlgorithm selectAlgorithm(const ProblemCharacteristicsLocal &prob)
{
    if (prob.requires_exact && prob.num_points <= Thresholds::MEDIUM_MAX)
    {
        if (canUseSmallFastPath(prob))
        {
            return VRAlgorithm::FAST_SIMD;
        }
        return VRAlgorithm::EXACT_STANDARD;
    }

    if (prob.time_budget_seconds > 0)
    {
        double estimated_time = estimateComputationTime(prob);
        if (estimated_time > prob.time_budget_seconds)
        {
            if (prob.num_points >= Thresholds::LARGE_MIN)
            {
                return VRAlgorithm::LARGE_WITNESS;
            }
        }
    }

    if (canUseSmallFastPath(prob))
    {
        return VRAlgorithm::FAST_SIMD;
    }

    if (prob.num_points <= Thresholds::FAST_MAX)
    {
        return VRAlgorithm::EXACT_STANDARD;
    }

    if (prob.num_points <= Thresholds::MEDIUM_MAX)
    {
        return VRAlgorithm::EXACT_STANDARD;
    }

    if (prob.num_points >= Thresholds::LARGE_MIN)
    {
        return VRAlgorithm::LARGE_WITNESS;
    }

    return VRAlgorithm::EXACT_STANDARD;
}

void updateConfigForAlgorithm(VRConfig &config, VRAlgorithm algo)
{
    switch (algo)
    {
        case VRAlgorithm::FAST_SIMD:
            config.enable_approximation = false;
            break;

        case VRAlgorithm::MEDIUM_HYBRID:
            config.acceleration.gpu_work_ratio = 0.0;
            config.enable_approximation = false;
            break;

        case VRAlgorithm::LARGE_WITNESS:
            config.enable_approximation = true;
            break;

        case VRAlgorithm::EXACT_STANDARD:
        default:
            break;
    }
}

} // namespace

// Get available system memory in GB
double getAvailableMemoryGB()
{
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<double>(status.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
#elif __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0)
    {
        return 16.0;
    }
    return (static_cast<double>(pages) * static_cast<double>(page_size)) /
           (1024.0 * 1024.0 * 1024.0);
#elif __APPLE__
    int64_t mem;
    size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    return static_cast<double>(mem) / (1024.0 * 1024.0 * 1024.0);
#else
    return 16.0; // Default value
#endif
}

// Memory usage estimation
size_t estimateMemoryUsage(size_t num_points, size_t point_dim, double max_radius)
{
    // Distance matrix: n^2 * sizeof(double)
    const size_t distance_values = saturatedProduct(num_points, num_points);
    size_t distance_matrix_mb = saturatedProduct(distance_values, sizeof(double)) / BYTES_PER_MB;

    // Points storage
    const size_t point_values = saturatedProduct(num_points, point_dim);
    size_t points_mb = saturatedProduct(point_values, sizeof(double)) / BYTES_PER_MB;

    // Simplices (heuristic: depends on radius and dimension)
    const double radius = boundedRadiusForHeuristic(max_radius);
    double sparsity = radius * radius; // Approximation
    size_t estimated_simplices = saturatedScaledSize(num_points, sparsity * 10.0);
    size_t simplices_mb = saturatedProduct(estimated_simplices, 32) / BYTES_PER_MB;

    return saturatedAdd(saturatedAdd(points_mb, distance_matrix_mb), simplices_mb);
}

// Main API: Auto-selected optimal VR algorithm
std::vector<Pair> computeVrPersistenceAuto(core::BufferView<const double> points, Size point_dim,
                                           const VRConfig &base_config)
{
    if (point_dim == 0 || points.size() == 0 || (points.size() % point_dim) != 0)
    {
        return {};
    }

    const Size num_points = points.size() / point_dim;

    const auto prob = makeProblemCharacteristics(
        num_points, point_dim, base_config.max_radius, !base_config.enable_approximation,
        base_config.time_budget_ms > 0 ? base_config.time_budget_ms / MILLIS_PER_SECOND : 0.0,
        false);

    VRAlgorithm selected = selectAlgorithm(prob);
    VRConfig config = base_config;
    updateConfigForAlgorithm(config, selected);

    switch (selected)
    {
        case VRAlgorithm::FAST_SIMD:
            return computeVrPersistenceFastSimd(points, point_dim, config);

        case VRAlgorithm::MEDIUM_HYBRID:
            return computeVrPersistenceMediumHybrid(points, point_dim, config);

        case VRAlgorithm::LARGE_WITNESS:
        {
            WitnessComplexConfig witness_config = getOptimalWitnessConfig(num_points, point_dim);
            return computeVrPersistenceLargeWitness(points, point_dim, config,
                                                    witness_config.num_landmarks);
        }

        case VRAlgorithm::EXACT_STANDARD:
        default:
            return computeVrPersistenceFast(points, point_dim, config);
    }
}

// Explicit algorithm selection API
std::vector<Pair> computeVrPersistenceWithAlgorithm(core::BufferView<const double> points,
                                                    Size point_dim, const VRConfig &config,
                                                    VRAlgorithm algorithm)
{
    if (point_dim == 0 || points.size() == 0 || (points.size() % point_dim) != 0)
    {
        return {};
    }

    const Size num_points = points.size() / point_dim;

    switch (algorithm)
    {
        case VRAlgorithm::FAST_SIMD:
            if (num_points > Thresholds::FAST_MAX || point_dim > Thresholds::FAST_MAX_DIM)
            {
                return computeVrPersistenceFast(points, point_dim, config);
            }
            return computeVrPersistenceFastSimd(points, point_dim, config);

        case VRAlgorithm::MEDIUM_HYBRID:
            if (num_points < Thresholds::MEDIUM_MIN)
            {
                return point_dim <= Thresholds::FAST_MAX_DIM
                           ? computeVrPersistenceFastSimd(points, point_dim, config)
                           : computeVrPersistenceFast(points, point_dim, config);
            }
            if (num_points > Thresholds::MEDIUM_MAX)
            {
                return computeVrPersistenceLargeWitness(points, point_dim, config);
            }
            return computeVrPersistenceMediumHybrid(points, point_dim, config);

        case VRAlgorithm::LARGE_WITNESS:
            if (num_points < Thresholds::LARGE_MIN)
            {
                return computeVrPersistenceFast(points, point_dim, config);
            }
            return computeVrPersistenceLargeWitness(points, point_dim, config);

        case VRAlgorithm::EXACT_STANDARD:
        case VRAlgorithm::AUTO:
        default:
            return computeVrPersistenceFast(points, point_dim, config);
    }
}

// Algorithm recommendation with explanation
AlgorithmRecommendation recommendAlgorithm(size_t num_points, size_t point_dim, double max_radius,
                                           bool require_exact, double time_budget_seconds)
{
    AlgorithmRecommendation rec;
    rec.problem_size = num_points;
    rec.point_dim = point_dim;

    const auto prob = makeProblemCharacteristics(num_points, point_dim, max_radius, require_exact,
                                                 time_budget_seconds, false);

    rec.recommended = selectAlgorithm(prob);
    rec.estimated_time_seconds = estimateComputationTime(prob);
    rec.memory_estimate_mb = estimateMemoryUsage(num_points, point_dim, max_radius);

    switch (rec.recommended)
    {
        case VRAlgorithm::FAST_SIMD:
            rec.description = isAvx512Available()
                                  ? "Small exact fast path with AVX-512 distance kernel. Best for "
                                    "<1K points and dimensions <=16."
                                  : "Small exact fast path with scalar FMA distance kernel. Best "
                                    "for <1K points and dimensions <=16.";
            rec.approximation_factor = APPROXIMATION_FACTOR_EXACT;
            break;

        case VRAlgorithm::MEDIUM_HYBRID:
            rec.description = "Tiled + parallel exact computation. Best for 1K-10K points.";
            rec.approximation_factor = APPROXIMATION_FACTOR_EXACT;
            break;

        case VRAlgorithm::LARGE_WITNESS:
            rec.description = "Witness complex approximation (epsilon-net landmarks). Best "
                              "for >10K points. "
                              "3-approximation.";
            rec.approximation_factor = APPROXIMATION_FACTOR_WITNESS;
            break;

        case VRAlgorithm::EXACT_STANDARD:
            rec.description = "Standard exact computation. Exact option.";
            rec.approximation_factor = APPROXIMATION_FACTOR_EXACT;
            break;

        default:
            rec.description = "Unknown algorithm";
            rec.approximation_factor = APPROXIMATION_FACTOR_EXACT;
    }

    return rec;
}

// Benchmark and compare all algorithms
std::vector<AlgorithmBenchmark> benchmarkAllAlgorithms(core::BufferView<const double> points,
                                                       Size point_dim, const VRConfig &config)
{
    std::vector<AlgorithmBenchmark> results;

    const std::vector<VRAlgorithm> algorithms = {VRAlgorithm::FAST_SIMD, VRAlgorithm::MEDIUM_HYBRID,
                                                 VRAlgorithm::LARGE_WITNESS,
                                                 VRAlgorithm::EXACT_STANDARD};

    for (auto algo : algorithms)
    {
        AlgorithmBenchmark bench{};
        bench.algorithm = algo;

        auto start = std::chrono::high_resolution_clock::now();

        try
        {
            auto pairs = computeVrPersistenceWithAlgorithm(points, point_dim, config, algo);

            auto end = std::chrono::high_resolution_clock::now();
            bench.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            bench.num_pairs = pairs.size();
            bench.success = true;

            if (algo == VRAlgorithm::LARGE_WITNESS)
            {
                bench.approximation_error = -1.0;
            }
            else
            {
                bench.approximation_error = 0.0;
            }
        }
        catch (...)
        {
            bench.success = false;
            bench.time_ms = -1.0;
        }

        results.push_back(bench);
    }

    return results;
}

} // namespace nerve::persistence
