
#include "nerve/persistence/adaptive_acceleration/streaming/representative_cycles.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include <utility>

namespace nerve::persistence::adaptive_acceleration::representative
{
namespace
{

Dimension inferDimensionFromColumn(const std::vector<double> &column)
{
    std::size_t nnz = 0;
    for (double value : column)
    {
        if (std::abs(value) > 0.0)
        {
            ++nnz;
        }
    }
    if (nnz == 0)
    {
        return 0;
    }
    return static_cast<Dimension>(std::max<std::size_t>(0, nnz - 1));
}

void updateStats(RepresentativeStats &stats, const std::vector<Cycle> &cycles)
{
    stats.total_cycles_computed = cycles.size();
    std::fill(std::begin(stats.cycles_per_dimension), std::end(stats.cycles_per_dimension), 0);
    if (cycles.empty())
    {
        stats.average_cycle_size = 0.0;
        stats.max_cycle_size = 0.0;
        stats.min_cycle_size = 0.0;
        return;
    }
    std::size_t total_size = 0;
    std::size_t max_size = 0;
    std::size_t min_size = std::numeric_limits<std::size_t>::max();
    for (const Cycle &cycle : cycles)
    {
        total_size += cycle.size();
        max_size = std::max(max_size, cycle.size());
        min_size = std::min(min_size, cycle.size());
        if (cycle.dimension >= 0 && cycle.dimension < 4)
        {
            stats.cycles_per_dimension[cycle.dimension] += 1;
        }
    }
    stats.average_cycle_size = static_cast<double>(total_size) / static_cast<double>(cycles.size());
    stats.max_cycle_size = static_cast<double>(max_size);
    stats.min_cycle_size = static_cast<double>(min_size);
}

} // namespace

class RepresentativeCycleComputer::Impl
{
public:
    explicit Impl(const RepresentativeConfig &config)
        : config(config)
    {}

    RepresentativeConfig config;
};

errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
RepresentativeCycleComputer::create(const RepresentativeConfig &config)
{
    std::unique_ptr<RepresentativeCycleComputer> computer(new RepresentativeCycleComputer(config));
    return errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>::success(
        std::move(computer));
}

errors::ErrorResult<std::vector<Cycle>>
RepresentativeCycleComputer::computeRepresentativesFast(const SparseMatrix &reduced_matrix)
{
    const auto start = std::chrono::steady_clock::now();
    auto result = config_.use_matrix_multiplication
                      ? computeCyclesMatrixMultiplication(reduced_matrix)
                      : computeCyclesStandard(reduced_matrix);
    if (result.isError())
    {
        return errors::ErrorResult<std::vector<Cycle>>::error(result.errorCode());
    }
    const auto end = std::chrono::steady_clock::now();
    computation_stats_.computation_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    computation_stats_.used_matrix_multiplication = config_.use_matrix_multiplication;
    computation_stats_.used_dual_cohomology = false;
    updateStats(computation_stats_, result.value());
    computation_stats_.computation_details = "representative cycles from reduced matrix";
    return result;
}

errors::ErrorResult<std::vector<Cycle>>
RepresentativeCycleComputer::computeWithDualCohomology(const SparseMatrix &boundary_matrix,
                                                       const SparseMatrix &reduced_matrix)
{
    auto cohomology = DualCohomologyComputer::create(config_);
    if (cohomology.isError())
    {
        return errors::ErrorResult<std::vector<Cycle>>::error(cohomology.errorCode());
    }
    auto cochains =
        cohomology.value()->computeCohomologyRepresentatives(boundary_matrix, reduced_matrix);
    if (cochains.isError())
    {
        return errors::ErrorResult<std::vector<Cycle>>::error(cochains.errorCode());
    }
    auto homology = cohomology.value()->convertToHomology(cochains.value(), boundary_matrix);
    if (homology.isError())
    {
        return errors::ErrorResult<std::vector<Cycle>>::error(homology.errorCode());
    }
    computation_stats_.used_dual_cohomology = true;
    updateStats(computation_stats_, homology.value());
    return homology;
}

errors::ErrorResult<std::vector<Cycle>>
RepresentativeCycleComputer::computeForDimension(const SparseMatrix &reduced_matrix,
                                                 Dimension dimension)
{
    auto all_cycles = computeRepresentativesFast(reduced_matrix);
    if (all_cycles.isError())
    {
        return errors::ErrorResult<std::vector<Cycle>>::error(all_cycles.errorCode());
    }
    std::vector<Cycle> filtered;
    for (const Cycle &cycle : all_cycles.value())
    {
        if (cycle.dimension == dimension)
        {
            filtered.push_back(cycle);
        }
    }
    return errors::ErrorResult<std::vector<Cycle>>::success(std::move(filtered));
}

errors::ErrorResult<std::vector<Cycle>>
RepresentativeCycleComputer::computeParallel(const SparseMatrix &reduced_matrix,
                                             std::size_t num_threads)
{
    const std::size_t threads = std::max<std::size_t>(1, num_threads);
    if (threads == 1)
    {
        return computeRepresentativesFast(reduced_matrix);
    }
    std::vector<std::future<errors::ErrorResult<std::vector<Cycle>>>> futures;
    futures.reserve(4);
    for (Dimension dim = 0; dim < 4; ++dim)
    {
        futures.push_back(std::async(std::launch::async, [this, &reduced_matrix, dim]() {
            return computeForDimension(reduced_matrix, dim);
        }));
    }
    std::vector<Cycle> merged;
    for (auto &future : futures)
    {
        auto partial = future.get();
        if (partial.isError())
        {
            return errors::ErrorResult<std::vector<Cycle>>::error(partial.errorCode());
        }
        const auto &value = partial.value();
        merged.insert(merged.end(), value.begin(), value.end());
    }
    updateStats(computation_stats_, merged);
    return errors::ErrorResult<std::vector<Cycle>>::success(std::move(merged));
}

RepresentativeCycleComputer::RepresentativeCycleComputer(const RepresentativeConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
    , computation_stats_()
{}

RepresentativeCycleComputer::~RepresentativeCycleComputer() = default;

errors::ErrorResult<std::vector<Cycle>>
RepresentativeCycleComputer::computeCyclesStandard(const SparseMatrix &reduced_matrix)
{
    std::vector<Cycle> cycles;
    cycles.reserve(
        std::min<std::size_t>(reduced_matrix.numCols(), config_.max_cycles_per_dimension * 4));
    for (std::size_t column = 0; column < reduced_matrix.numCols(); ++column)
    {
        std::vector<double> values = reduced_matrix.getColumn(column);
        const Dimension dimension = inferDimensionFromColumn(values);
        auto cycle = extractSingleCycle(reduced_matrix, column, dimension);
        if (cycle.isError())
        {
            continue;
        }
        if (cycle.value().persistence() < config_.min_persistence)
        {
            continue;
        }
        cycles.push_back(cycle.value());
        if (cycles.size() >= config_.max_cycles_per_dimension * 4)
        {
            break;
        }
    }
    return errors::ErrorResult<std::vector<Cycle>>::success(std::move(cycles));
}

errors::ErrorResult<std::vector<Cycle>>
RepresentativeCycleComputer::computeCyclesMatrixMultiplication(const SparseMatrix &reduced_matrix)
{
    return computeCyclesStandard(reduced_matrix);
}

errors::ErrorResult<std::vector<Cycle>>
RepresentativeCycleComputer::computeCyclesDualCohomology(const SparseMatrix &boundary_matrix,
                                                         const SparseMatrix &reduced_matrix)
{
    return computeWithDualCohomology(boundary_matrix, reduced_matrix);
}

errors::ErrorResult<Cycle>
RepresentativeCycleComputer::extractSingleCycle(const SparseMatrix &matrix,
                                                std::size_t column_index, Dimension dimension)
{
    if (column_index >= matrix.numCols())
    {
        return errors::ErrorResult<Cycle>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    const std::vector<double> column = matrix.getColumn(column_index);
    Cycle cycle;
    cycle.dimension = dimension;
    cycle.birth_time = static_cast<double>(column_index);
    cycle.death_time = static_cast<double>(column_index);
    for (std::size_t row = 0; row < column.size(); ++row)
    {
        if (std::abs(column[row]) > 0.0)
        {
            cycle.vertices.push_back(static_cast<int>(row));
            cycle.coefficients.push_back(column[row]);
            cycle.death_time = static_cast<double>(row);
        }
    }
    auto valid = validateCycle(cycle);
    if (valid.isError())
    {
        return errors::ErrorResult<Cycle>::error(valid.errorCode());
    }
    return errors::ErrorResult<Cycle>::success(std::move(cycle));
}

errors::ErrorResult<void> RepresentativeCycleComputer::validateCycle(const Cycle &cycle)
{
    if (!cycle.isValid())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    return errors::ErrorResult<void>::ok();
}

class DualCohomologyComputer::Impl
{
public:
    explicit Impl(const RepresentativeConfig &config)
        : config(config)
    {}

    RepresentativeConfig config;
};

errors::ErrorResult<std::unique_ptr<DualCohomologyComputer>>
DualCohomologyComputer::create(const RepresentativeConfig &config)
{
    std::unique_ptr<DualCohomologyComputer> computer(new DualCohomologyComputer(config));
    return errors::ErrorResult<std::unique_ptr<DualCohomologyComputer>>::success(
        std::move(computer));
}

errors::ErrorResult<std::vector<Cycle>>
DualCohomologyComputer::computeCohomologyRepresentatives(const SparseMatrix &boundary_matrix,
                                                         const SparseMatrix &reduced_matrix)
{
    if (boundary_matrix.numCols() != reduced_matrix.numCols())
    {
        return errors::ErrorResult<std::vector<Cycle>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::vector<Cycle> cycles;
    for (std::size_t column = 0; column < reduced_matrix.numCols(); ++column)
    {
        std::vector<double> values = reduced_matrix.getColumn(column);
        Cycle cycle;
        cycle.dimension = inferDimensionFromColumn(values) + 1;
        cycle.birth_time = static_cast<double>(column);
        cycle.death_time = static_cast<double>(column);
        for (std::size_t row = 0; row < values.size(); ++row)
        {
            if (std::abs(values[row]) > 0.0)
            {
                cycle.vertices.push_back(static_cast<int>(row));
                cycle.coefficients.push_back(values[row]);
            }
        }
        if (cycle.isValid())
        {
            cycles.push_back(std::move(cycle));
        }
    }
    return errors::ErrorResult<std::vector<Cycle>>::success(std::move(cycles));
}

DualCohomologyComputer::DualCohomologyComputer(const RepresentativeConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
{}

DualCohomologyComputer::~DualCohomologyComputer() = default;

errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
RepresentativeCycleFactory::createFast(const RepresentativeConfig &config)
{
    RepresentativeConfig configured = config;
    configured.use_matrix_multiplication = true;
    return RepresentativeCycleComputer::create(configured);
}

errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
RepresentativeCycleFactory::createDualCohomology(const RepresentativeConfig &config)
{
    RepresentativeConfig configured = config;
    configured.use_dual_cohomology = true;
    return RepresentativeCycleComputer::create(configured);
}

errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
RepresentativeCycleFactory::createParallel(const RepresentativeConfig &config)
{
    RepresentativeConfig configured = config;
    configured.enable_parallel_computation = true;
    return RepresentativeCycleComputer::create(configured);
}

errors::ErrorResult<std::unique_ptr<RepresentativeCycleComputer>>
RepresentativeCycleFactory::createForVisualization(const RepresentativeConfig &config)
{
    RepresentativeConfig configured = config;
    configured.enable_visualization_data = true;
    return RepresentativeCycleComputer::create(configured);
}

} // namespace nerve::persistence::adaptive_acceleration::representative
