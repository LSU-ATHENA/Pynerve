
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_system_capabilities.hpp"
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

namespace nerve::persistence::adaptive_acceleration::gpu
{
namespace
{

struct WorkItem
{
    std::size_t column = 0;
    std::size_t non_zero_count = 0;
    double weight = 0.0;
};

struct WorkSplit
{
    std::vector<WorkItem> gpu_items;
    std::vector<WorkItem> cpu_items;
};

constexpr double kZeroTolerance = 1e-12;

std::size_t countNonZero(const std::vector<double> &column_values)
{
    std::size_t non_zero_count = 0;
    for (const double value : column_values)
    {
        if (std::abs(value) > kZeroTolerance)
        {
            ++non_zero_count;
        }
    }
    return non_zero_count;
}

std::vector<WorkItem> buildWorkItems(const SparseMatrix &matrix)
{
    std::vector<WorkItem> items;
    items.reserve(matrix.numCols());
    for (std::size_t column = 0; column < matrix.numCols(); ++column)
    {
        const std::vector<double> values = matrix.getColumn(column);
        const std::size_t non_zero_count = countNonZero(values);
        WorkItem item;
        item.column = column;
        item.non_zero_count = non_zero_count;
        item.weight = static_cast<double>(non_zero_count);
        items.push_back(item);
    }
    std::ranges::sort(items, {}, [](const WorkItem &w) { return std::pair(-w.weight, w.column); });
    return items;
}

double computeGpuWorkRatio(const ProblemCharacteristics &problem, const SystemCapabilities &system,
                           std::size_t required_bytes)
{
    if (!system.cuda_available || !problem.suitable_for_gpu)
    {
        return 0.0;
    }
    const std::size_t available_bytes =
        system.available_memory > 0 ? system.available_memory : system.total_memory;
    if (available_bytes == 0)
    {
        return 0.0;
    }

    const double memory_ratio =
        std::min(1.0, static_cast<double>(available_bytes) /
                          static_cast<double>(std::max<std::size_t>(required_bytes, 1)));
    const double gpu_compute = std::max(1.0, system.theoretical_gflops);
    const double cpu_compute =
        std::max(1.0, static_cast<double>(std::max<std::size_t>(1, system.num_cpu_cores)) * 32.0);
    const double compute_ratio = std::clamp(gpu_compute / (gpu_compute + cpu_compute), 0.0, 1.0);
    const double sparsity_penalty = problem.is_sparse ? 0.15 : 0.0;
    const double tensor_bonus = system.supports_tensor_cores ? 0.08 : 0.0;

    return std::clamp(
        (0.55 * memory_ratio) + (0.45 * compute_ratio) + tensor_bonus - sparsity_penalty, 0.0, 1.0);
}

WorkSplit splitWork(const std::vector<WorkItem> &items, const ProblemCharacteristics &problem,
                    const SystemCapabilities &system)
{
    WorkSplit split;
    if (items.empty())
    {
        return split;
    }

    const std::size_t estimated_bytes = std::accumulate(
        items.begin(), items.end(), std::size_t{0}, [](std::size_t acc, const WorkItem &item) {
            return acc + item.non_zero_count * sizeof(double);
        });
    const double gpu_ratio = computeGpuWorkRatio(problem, system, estimated_bytes);
    const std::size_t gpu_count =
        static_cast<std::size_t>(std::round(gpu_ratio * static_cast<double>(items.size())));

    split.gpu_items.reserve(gpu_count);
    split.cpu_items.reserve(items.size() - std::min(items.size(), gpu_count));

    for (std::size_t index = 0; index < items.size(); ++index)
    {
        if (index < gpu_count)
        {
            split.gpu_items.push_back(items[index]);
        }
        else
        {
            split.cpu_items.push_back(items[index]);
        }
    }
    return split;
}

Pair buildPairFromColumn(const SparseMatrix &matrix, const WorkItem &item)
{
    const std::vector<double> column_values = matrix.getColumn(item.column);
    std::size_t birth_index = 0;
    std::size_t death_index = item.column;
    bool seen_non_zero = false;

    for (std::size_t row = 0; row < column_values.size(); ++row)
    {
        if (std::abs(column_values[row]) <= kZeroTolerance)
        {
            continue;
        }
        if (!seen_non_zero)
        {
            birth_index = row;
            seen_non_zero = true;
        }
        death_index = row;
    }

    Pair pair;
    pair.birth = static_cast<double>(birth_index);
    pair.death = seen_non_zero ? static_cast<double>(death_index) : pair.birth;
    pair.dimension = static_cast<Dimension>(seen_non_zero ? 1 : 0);
    return pair;
}

std::vector<Pair> executePhase(const SparseMatrix &matrix, const std::vector<WorkItem> &items)
{
    std::vector<Pair> result;
    result.reserve(items.size());
    for (const WorkItem &item : items)
    {
        result.push_back(buildPairFromColumn(matrix, item));
    }
    return result;
}

} // namespace

errors::ErrorResult<std::vector<Pair>> executeHybridReduction(const SparseMatrix &boundary_matrix,
                                                              const ProblemCharacteristics &problem,
                                                              const SystemCapabilities &system)
{
    if (!boundary_matrix.isValid())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E85_MATRIX_STRUCTURE);
    }
    if (boundary_matrix.numCols() == 0)
    {
        return errors::ErrorResult<std::vector<Pair>>::success({});
    }
    if (problem.estimated_columns != 0 && boundary_matrix.numCols() != problem.estimated_columns)
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    const std::vector<WorkItem> items = buildWorkItems(boundary_matrix);
    const WorkSplit split = splitWork(items, problem, system);

    std::vector<Pair> output = executePhase(boundary_matrix, split.gpu_items);
    std::vector<Pair> cpu_output = executePhase(boundary_matrix, split.cpu_items);
    output.insert(output.end(), cpu_output.begin(), cpu_output.end());

    std::ranges::sort(output, {}, [](const Pair &p) { return std::pair(p.birth, p.death); });

    return errors::ErrorResult<std::vector<Pair>>::success(std::move(output));
}

} // namespace nerve::persistence::adaptive_acceleration::gpu
