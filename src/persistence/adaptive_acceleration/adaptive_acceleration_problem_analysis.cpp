
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_problem_analysis.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_system_capabilities.hpp"
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <thread>
#include <vector>

namespace nerve::persistence::adaptive_acceleration
{

namespace
{

constexpr double kMaxFiniteEstimate = 1.0e300;

double finiteNonnegativeEstimate(long double value)
{
    if (!std::isfinite(value))
    {
        return kMaxFiniteEstimate;
    }
    if (value <= 0.0L)
    {
        return 0.0;
    }
    return static_cast<double>(std::min(value, static_cast<long double>(kMaxFiniteEstimate)));
}

double finiteUnitRatio(double value, double fallback)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::clamp(value, 0.0, 1.0);
}

SystemCapabilities cpuOnlySystemCapabilities()
{
    SystemCapabilities system;
    system.cuda_available = false;
    system.num_cpu_cores =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
    return system;
}

bool shouldUseStreamingForMemory(double memory_requirement_mb, const SystemCapabilities &system)
{
    if (system.available_memory == 0)
    {
        return false;
    }
    return memory_requirement_mb > static_cast<double>(system.available_memory) / (1024.0 * 1024.0);
}

std::size_t binomialCoefficientBounded(std::size_t n, std::size_t k)
{
    if (k > n)
    {
        return 0;
    }
    k = std::min(k, n - k);
    long double value = 1.0L;
    for (std::size_t i = 1; i <= k; ++i)
    {
        value *= static_cast<long double>(n - k + i);
        value /= static_cast<long double>(i);
        if (value >= static_cast<long double>(std::numeric_limits<std::size_t>::max()))
        {
            return std::numeric_limits<std::size_t>::max();
        }
    }
    return static_cast<std::size_t>(value);
}

} // namespace

ProblemCharacteristics ProblemAnalyzer::analyzeProblem(const core::BufferView<const double> &points,
                                                       std::size_t point_dim)
{
    ProblemCharacteristics characteristics;
    if (point_dim == 0 || points.size() == 0 || (points.size() % point_dim) != 0)
    {
        return characteristics;
    }
    for (double value : points)
    {
        if (!std::isfinite(value))
        {
            return characteristics;
        }
    }

    characteristics.point_dim = point_dim;
    characteristics.num_points = points.size() / point_dim;
    characteristics.max_simplex_size =
        estimateMaxSimplexSize(characteristics.num_points, point_dim, 1.0);
    characteristics.estimated_columns =
        estimateNumColumns(characteristics.num_points, characteristics.max_simplex_size);
    characteristics.sparsity_ratio =
        estimateSparsityRatio(characteristics.num_points, characteristics.max_simplex_size);
    characteristics.density = computeDensity(points, point_dim);
    characteristics.apparent_pair_ratio = estimateApparentPairRatio(characteristics);
    characteristics.is_dense = characteristics.sparsity_ratio > 0.5;
    characteristics.is_sparse = characteristics.sparsity_ratio < 0.1;
    characteristics.isHighDimensional = isHighDimensional(point_dim);
    characteristics.hasRegularStructure = hasRegularStructure(points);
    characteristics.estimated_complexity = estimateComplexity(characteristics);
    characteristics.memory_requirement_mb = estimateMemoryRequirement(characteristics);

    const SystemCapabilities system = cpuOnlySystemCapabilities();
    characteristics.suitable_for_matrix_multiplication =
        shouldUseMatrixMultiplication(characteristics);
    characteristics.suitable_for_sparsification = shouldUseSparsification(characteristics);
    characteristics.suitable_for_lockfree_multicore =
        should_use_lockfree_multicore(characteristics, system);
    characteristics.suitable_for_gpu = shouldUseGpu(characteristics, system);
    characteristics.suitable_for_streaming =
        shouldUseStreamingForMemory(characteristics.memory_requirement_mb, system);

    return characteristics;
}

ProblemCharacteristics ProblemAnalyzer::analyzeMatrix(const SparseMatrix &matrix)
{
    ProblemCharacteristics characteristics;
    characteristics.num_points = matrix.numRows();
    characteristics.point_dim = matrix.numCols();
    characteristics.max_simplex_size = 2;
    characteristics.estimated_columns = matrix.numCols();
    characteristics.sparsity_ratio = 1.0 - matrix.sparsityRatio();
    characteristics.density = matrix.sparsityRatio();
    characteristics.is_dense = characteristics.density > 0.5;
    characteristics.is_sparse = characteristics.density < 0.1;
    characteristics.isHighDimensional = isHighDimensional(characteristics.point_dim);
    characteristics.hasRegularStructure = false;
    characteristics.estimated_complexity = estimateComplexity(characteristics);
    characteristics.memory_requirement_mb =
        static_cast<double>(matrix.memoryUsage()) / (1024.0 * 1024.0);
    characteristics.apparent_pair_ratio = estimateApparentPairRatio(characteristics);

    const SystemCapabilities system = cpuOnlySystemCapabilities();
    characteristics.suitable_for_matrix_multiplication =
        shouldUseMatrixMultiplication(characteristics);
    characteristics.suitable_for_sparsification = shouldUseSparsification(characteristics);
    characteristics.suitable_for_lockfree_multicore =
        should_use_lockfree_multicore(characteristics, system);
    characteristics.suitable_for_gpu = shouldUseGpu(characteristics, system);
    characteristics.suitable_for_streaming =
        shouldUseStreamingForMemory(characteristics.memory_requirement_mb, system);

    return characteristics;
}

double ProblemAnalyzer::estimateComplexity(const ProblemCharacteristics &problem)
{
    const long double n =
        static_cast<long double>(std::max<std::size_t>(1, problem.estimated_columns));
    long double complexity = n * std::log2(n + 1.0L);
    if (problem.is_dense)
    {
        complexity *= 1.25L;
    }
    if (problem.isHighDimensional)
    {
        complexity *= 1.30L;
    }
    if (problem.hasRegularStructure)
    {
        complexity *= 0.9L;
    }
    return finiteNonnegativeEstimate(complexity);
}

double ProblemAnalyzer::estimateMemoryRequirement(const ProblemCharacteristics &problem)
{
    const long double n =
        static_cast<long double>(std::max<std::size_t>(1, problem.estimated_columns));
    const long double dense_bytes = n * n * static_cast<long double>(sizeof(double));
    const long double sparse_factor = static_cast<long double>(
        std::clamp(finiteUnitRatio(problem.sparsity_ratio, 1.0), 0.01, 1.0));
    const long double effective_bytes = dense_bytes * sparse_factor;
    return finiteNonnegativeEstimate((effective_bytes * 1.15L) / (1024.0L * 1024.0L));
}

double ProblemAnalyzer::estimateApparentPairRatio(const ProblemCharacteristics &problem)
{
    const double base = 0.35 + 0.5 * finiteUnitRatio(problem.sparsity_ratio, 0.0);
    const double dim_boost = problem.isHighDimensional ? 0.1 : 0.0;
    return std::clamp(base + dim_boost, 0.0, 1.0);
}

bool ProblemAnalyzer::shouldUseMatrixMultiplication(const ProblemCharacteristics &problem)
{
    return problem.estimated_columns >= 2048 && problem.is_dense;
}

bool ProblemAnalyzer::shouldUseSparsification(const ProblemCharacteristics &problem)
{
    return problem.is_sparse ||
           (problem.sparsity_ratio < 0.35 && problem.estimated_columns >= 1024);
}

bool ProblemAnalyzer::should_use_lockfree_multicore(const ProblemCharacteristics &problem,
                                                    const SystemCapabilities &system)
{
    return system.num_cpu_cores >= 4 && problem.estimated_columns >= 512;
}

bool ProblemAnalyzer::shouldUseGpu(const ProblemCharacteristics &problem,
                                   const SystemCapabilities &system)
{
    (void)problem;
    (void)system;
    return false;
}

double ProblemAnalyzer::computeDensity(const core::BufferView<const double> &points,
                                       std::size_t point_dim)
{
    if (points.size() == 0 || point_dim == 0)
    {
        return 0.0;
    }
    const std::size_t num_points = points.size() / point_dim;
    if (num_points == 0)
    {
        return 0.0;
    }

    std::vector<double> mins(point_dim, 0.0);
    std::vector<double> maxs(point_dim, 0.0);
    for (std::size_t d = 0; d < point_dim; ++d)
    {
        const double value = points[d];
        if (!std::isfinite(value))
        {
            return 0.0;
        }
        mins[d] = value;
        maxs[d] = value;
    }
    for (std::size_t i = 0; i < num_points; ++i)
    {
        for (std::size_t d = 0; d < point_dim; ++d)
        {
            const double value = points[(i * point_dim) + d];
            if (!std::isfinite(value))
            {
                return 0.0;
            }
            mins[d] = std::min(mins[d], value);
            maxs[d] = std::max(maxs[d], value);
        }
    }

    long double volume = 1.0L;
    for (std::size_t d = 0; d < point_dim; ++d)
    {
        const long double span =
            static_cast<long double>(maxs[d]) - static_cast<long double>(mins[d]);
        const long double bounded_span =
            std::isfinite(span) && span > 0.0L ? std::max(1.0e-9L, span) : 1.0e-9L;
        volume *= bounded_span;
        if (!std::isfinite(volume))
        {
            return 0.0;
        }
    }
    if (volume <= 0.0L)
    {
        return 0.0;
    }
    return finiteNonnegativeEstimate(static_cast<long double>(num_points) / volume);
}

std::size_t ProblemAnalyzer::estimateMaxSimplexSize(std::size_t num_points, std::size_t point_dim,
                                                    double max_radius)
{
    if (num_points == 0 || point_dim == 0)
    {
        return 0;
    }
    const std::size_t dim_cap =
        point_dim == std::numeric_limits<std::size_t>::max() ? point_dim : point_dim + 1;
    const std::size_t geometric_cap = std::min<std::size_t>(dim_cap, num_points);
    const double radius_scale = finiteUnitRatio(max_radius, 0.0);
    const std::size_t scaled =
        static_cast<std::size_t>(std::max(1.0, static_cast<double>(geometric_cap) * radius_scale));
    return std::max<std::size_t>(1, std::min(geometric_cap, scaled));
}

std::size_t ProblemAnalyzer::estimateNumColumns(std::size_t num_points,
                                                std::size_t max_simplex_size)
{
    if (num_points == 0 || max_simplex_size == 0)
    {
        return 0;
    }
    std::size_t total = 0;
    const std::size_t max_k = std::min(max_simplex_size, num_points);
    for (std::size_t k = 1; k <= max_k; ++k)
    {
        const std::size_t term = binomialCoefficientBounded(num_points, k);
        if (term > std::numeric_limits<std::size_t>::max() - total)
        {
            return std::numeric_limits<std::size_t>::max();
        }
        total += term;
    }
    return total;
}

double ProblemAnalyzer::estimateSparsityRatio(std::size_t num_points, std::size_t max_simplex_size)
{
    const std::size_t columns = estimateNumColumns(num_points, max_simplex_size);
    if (columns == 0)
    {
        return 1.0;
    }
    double non_zero = 0.0;
    for (std::size_t k = 1; k <= max_simplex_size; ++k)
    {
        non_zero += static_cast<double>(k + 1);
    }
    return std::clamp(non_zero / static_cast<double>(columns), 0.0, 1.0);
}

bool ProblemAnalyzer::isHighDimensional(std::size_t point_dim)
{
    return point_dim > 10;
}

bool ProblemAnalyzer::hasRegularStructure(const core::BufferView<const double> &points)
{
    if (points.size() < 8)
    {
        return false;
    }
    std::vector<double> sorted(points.begin(), points.end());
    std::ranges::sort(sorted);
    std::vector<double> deltas;
    deltas.reserve(sorted.size() - 1);
    for (std::size_t i = 1; i < sorted.size(); ++i)
    {
        const double delta = sorted[i] - sorted[i - 1];
        if (!std::isfinite(delta))
        {
            return false;
        }
        deltas.push_back(delta);
    }
    long double sum = 0.0L;
    for (double delta : deltas)
    {
        sum += static_cast<long double>(delta);
        if (!std::isfinite(sum))
        {
            return false;
        }
    }
    const double mean = static_cast<double>(sum / static_cast<long double>(deltas.size()));
    if (!std::isfinite(mean))
    {
        return false;
    }
    if (mean == 0.0)
    {
        return true;
    }
    long double variance = 0.0L;
    for (double delta : deltas)
    {
        const double diff = delta - mean;
        const long double square = static_cast<long double>(diff) * static_cast<long double>(diff);
        variance += square;
        if (!std::isfinite(variance))
        {
            return false;
        }
    }
    variance /= static_cast<long double>(deltas.size());
    const double cv = static_cast<double>(std::sqrt(variance) / std::abs(mean));
    if (!std::isfinite(cv))
    {
        return false;
    }
    return cv < 0.15;
}

} // namespace nerve::persistence::adaptive_acceleration
